#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 RLTrainer 在线 RL 训练流程：
  - TrajectoryStep 收集
  - end_episode() 触发梯度更新
  - 统计字段完整性
  - checkpoint 保存与加载
  - BCDataset + train_bc_pretrain 核心逻辑（smoke test）

运行方式：
    python -m pytest mol_evo/tests/test_rl_training.py -v
    # 或直接运行
    python mol_evo/tests/test_rl_training.py
"""

import json
import os
import sys
import tempfile
import unittest

# 项目根路径
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..", "..")
sys.path.insert(0, _root)


class TestRLTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from mol_evo.core.models.astar_rl import (
                PolicyNet,
                ValueNet,
                RLTrainer,
                PPORLTrainer,
            )
            from mol_evo.core.models.astar_rl.reward import RewardConfig
            from mol_evo.core.models.astar_rl.rl_trainer import TrajectoryStep
            from mol_evo.core.data.rl_demo_processing import STATE_DIM, ACTION_DIM

            cls.torch = torch
            cls.PolicyNet = PolicyNet
            cls.ValueNet = ValueNet
            cls.RLTrainer = RLTrainer
            cls.PPORLTrainer = PPORLTrainer
            cls.RewardConfig = RewardConfig
            cls.TrajectoryStep = TrajectoryStep
            cls.STATE_DIM = STATE_DIM
            cls.ACTION_DIM = ACTION_DIM
            cls._imports_ok = True
        except ImportError as e:
            cls._imports_ok = False
            cls._import_error = str(e)

    def _require_imports(self):
        if not self._imports_ok:
            self.skipTest(f"导入失败，跳过测试: {self._import_error}")

    def _make_trainer(self, checkpoint_dir=None, algo="reinforce"):
        """创建一个在线 RL 训练器实例（CPU）。"""
        policy_net = self.PolicyNet(state_dim=self.STATE_DIM, action_dim=self.ACTION_DIM)
        value_net = self.ValueNet(state_dim=self.STATE_DIM)
        reward_config = self.RewardConfig(direction="decrease")
        if algo == "ppo":
            trainer = self.PPORLTrainer(
                policy_net=policy_net,
                value_net=value_net,
                reward_config=reward_config,
                lr_policy=1e-4,
                lr_value=1e-3,
                gamma=0.99,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=10,
                clip_ratio=0.2,
                gae_lambda=0.95,
                update_epochs=2,
                minibatch_size=2,
                normalize_advantage=True,
            )
        else:
            trainer = self.RLTrainer(
                policy_net=policy_net,
                value_net=value_net,
                reward_config=reward_config,
                lr_policy=1e-4,
                lr_value=1e-3,
                gamma=0.99,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=10,
            )
        return trainer, policy_net, value_net

    def _make_step(self, n_actions=4, selected_action_idx=0, matches_policy_greedy=None):
        """创建一个随机 TrajectoryStep。"""
        torch = self.torch
        state = torch.randn(self.STATE_DIM)
        actions = torch.randn(n_actions, self.ACTION_DIM)
        selected_action_idx = min(selected_action_idx, n_actions - 1)
        policy_net = self.PolicyNet(state_dim=self.STATE_DIM, action_dim=self.ACTION_DIM)
        policy_net.eval()
        log_prob = policy_net.get_log_probs(state, actions, selected_action_idx)
        next_state = torch.randn(self.STATE_DIM)
        if matches_policy_greedy is None:
            matches_policy_greedy = (selected_action_idx == 0)
        return self.TrajectoryStep(
            state_tensor=state,
            action_tensors=actions,
            selected_action_idx=selected_action_idx,
            log_prob=log_prob,
            step_reward=0.3,
            next_state_tensor=next_state,
            value_estimate=0.1,
            done=False,
            selection_mode="sample",
            selected_action_rank=selected_action_idx,
            policy_entropy=1.234,
            selected_action_prob=0.42,
            greedy_action_idx=0,
            matches_policy_greedy=matches_policy_greedy,
            metadata={"source": "unit_test"},
        )

    # ------------------------------------------------------------------

    def test_collect_and_end_episode(self):
        """collect_step + end_episode 应返回有效统计字典。"""
        self._require_imports()
        trainer, _, _ = self._make_trainer()
        trainer.reset_episode()

        for _ in range(5):
            trainer.collect_step(self._make_step())

        stats = trainer.end_episode()

        self.assertIn("policy_loss", stats)
        self.assertIn("value_loss", stats)
        self.assertIn("episode_return", stats)
        self.assertIn("episode_steps", stats)
        self.assertEqual(stats["episode_steps"], 5)
        self.assertAlmostEqual(stats["episode_return"], 0.3 * 5, places=3)

    def test_episode_counter_increments(self):
        """每次 end_episode 应使 episode_count +1。"""
        self._require_imports()
        trainer, _, _ = self._make_trainer()

        for _ in range(3):
            trainer.reset_episode()
            trainer.collect_step(self._make_step())
            trainer.end_episode()

        self.assertEqual(trainer.episode_count, 3)

    def test_empty_episode_returns_empty(self):
        """空 episode 的 end_episode 应返回空 dict，不报错。"""
        self._require_imports()
        trainer, _, _ = self._make_trainer()
        trainer.reset_episode()
        stats = trainer.end_episode()
        self.assertEqual(stats, {})

    def test_checkpoint_save_and_load(self):
        """save_checkpoint / load_checkpoint 应能正确恢复状态。"""
        self._require_imports()
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer, _, _ = self._make_trainer(checkpoint_dir=tmpdir)
            trainer.reset_episode()
            trainer.collect_step(self._make_step())
            trainer.end_episode()

            ckpt_path = trainer.save_checkpoint(tag="test")
            self.assertTrue(os.path.isfile(ckpt_path))

            trainer2, _, _ = self._make_trainer(checkpoint_dir=tmpdir)
            trainer2.load_checkpoint(ckpt_path)
            self.assertEqual(trainer2.episode_count, 1)
            self.assertEqual(trainer2.total_steps, trainer.total_steps)

    def test_policy_loss_is_finite(self):
        """更新后 policy_loss 应为有限值。"""
        self._require_imports()
        import math

        trainer, _, _ = self._make_trainer()
        trainer.reset_episode()
        for _ in range(3):
            trainer.collect_step(self._make_step())
        stats = trainer.end_episode()
        self.assertTrue(
            math.isfinite(stats["policy_loss"]),
            f"policy_loss 不是有限值: {stats['policy_loss']}",
        )
        self.assertTrue(
            math.isfinite(stats["value_loss"]),
            f"value_loss 不是有限值: {stats['value_loss']}",
        )

    def test_policy_select_action_greedy_matches_argmax(self):
        """greedy 选择应与 logits argmax 保持一致。"""
        self._require_imports()
        policy_net = self.PolicyNet(state_dim=self.STATE_DIM, action_dim=self.ACTION_DIM)
        policy_net.eval()
        state = self.torch.randn(self.STATE_DIM)
        actions = self.torch.randn(5, self.ACTION_DIM)

        logits = policy_net(state, actions)
        expected_idx = int(self.torch.argmax(logits).item())
        selection = policy_net.select_action(state, actions, mode="greedy")

        self.assertEqual(selection["selected_idx"], expected_idx)
        selected_log_prob = self.torch.as_tensor(selection["log_prob"])
        expected_log_prob = policy_net.get_log_probs(state, actions, expected_idx)
        self.assertAlmostEqual(
            float(selected_log_prob.item()),
            float(expected_log_prob.item()),
            places=6,
        )

    def test_episode_stats_include_action_selection_metrics(self):
        """episode 统计应包含动作采样相关诊断字段。"""
        self._require_imports()
        trainer, _, _ = self._make_trainer()
        trainer.reset_episode()
        trainer.collect_step(self._make_step(selected_action_idx=2, matches_policy_greedy=False))
        trainer.collect_step(self._make_step(selected_action_idx=0, matches_policy_greedy=True))

        stats = trainer.end_episode()

        self.assertEqual(stats["action_selection_mode"], "sample")
        self.assertIn("avg_policy_entropy", stats)
        self.assertIn("avg_selected_action_prob", stats)
        self.assertIn("avg_selected_action_rank", stats)
        self.assertIn("policy_greedy_match_rate", stats)
        self.assertAlmostEqual(stats["avg_selected_action_rank"], 1.0, places=6)
        self.assertAlmostEqual(stats["policy_greedy_match_rate"], 0.5, places=6)

    def test_ppo_end_episode_reports_diagnostics(self):
        """PPO 更新后应输出 clip/KL/adv 等关键诊断字段。"""
        self._require_imports()
        import math

        trainer, _, _ = self._make_trainer(algo="ppo")
        trainer.reset_episode()
        trainer.collect_step(self._make_step(selected_action_idx=2, matches_policy_greedy=False))
        trainer.collect_step(self._make_step(selected_action_idx=1, matches_policy_greedy=False))
        trainer.collect_step(self._make_step(selected_action_idx=0, matches_policy_greedy=True))

        stats = trainer.end_episode()

        self.assertEqual(stats["algo"], "ppo")
        for key in (
            "policy_loss",
            "value_loss",
            "policy_entropy",
            "approx_kl",
            "clip_fraction",
            "adv_mean",
            "adv_std",
            "adv_min",
            "adv_max",
        ):
            self.assertIn(key, stats)
            self.assertTrue(
                math.isfinite(float(stats[key])),
                f"{key} 不是有限值: {stats[key]}",
            )
        self.assertGreaterEqual(stats["clip_fraction"], 0.0)
        self.assertLessEqual(stats["clip_fraction"], 1.0)


# ---------------------------------------------------------------------------
# train_astar_rl_demo helper smoke test
# ---------------------------------------------------------------------------

class TestTrainAstarRlDemoHelpers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from mol_evo.train_astar_rl_demo import (
                resolve_holdout_eval_every,
                build_holdout_priority,
                flatten_holdout_metrics,
                resolve_run_dir,
                load_training_state,
            )

            cls.resolve_holdout_eval_every = staticmethod(resolve_holdout_eval_every)
            cls.build_holdout_priority = staticmethod(build_holdout_priority)
            cls.flatten_holdout_metrics = staticmethod(flatten_holdout_metrics)
            cls.resolve_run_dir = staticmethod(resolve_run_dir)
            cls.load_training_state = staticmethod(load_training_state)
            cls._imports_ok = True
        except ImportError as e:
            cls._imports_ok = False
            cls._import_error = str(e)

    def _require_imports(self):
        if not self._imports_ok:
            self.skipTest(f"导入失败，跳过测试: {self._import_error}")

    def test_resolve_holdout_eval_every_disabled_without_holdout(self):
        """未提供 holdout 时，不应启用周期评估。"""
        self._require_imports()
        self.assertEqual(
            self.resolve_holdout_eval_every(None, holdout_eval_every=0, checkpoint_every=25),
            0,
        )

    def test_resolve_holdout_eval_every_falls_back_to_checkpoint_every(self):
        """提供 holdout 但未显式设频率时，应回退到 checkpoint_every。"""
        self._require_imports()
        self.assertEqual(
            self.resolve_holdout_eval_every("holdout.csv", holdout_eval_every=0, checkpoint_every=25),
            25,
        )
        self.assertEqual(
            self.resolve_holdout_eval_every("holdout.csv", holdout_eval_every=7, checkpoint_every=25),
            7,
        )

    def test_build_holdout_priority_follows_doc_order(self):
        """priority 应按 median → trimmed_mean → win_rate → mean 排序。"""
        self._require_imports()
        payload = {
            "summary": {
                "top1_median": 1.2,
                "trimmed_mean": 0.8,
                "top1_mean": 0.9,
            },
            "comparison_vs_reference": {
                "win_rate": 0.6,
            },
        }
        self.assertEqual(
            self.build_holdout_priority(payload),
            (1.2, 0.8, 0.6, 0.9),
        )

    def test_flatten_holdout_metrics_handles_optional_comparison(self):
        """无 reference 时也应能稳定产出基础 holdout 指标。"""
        self._require_imports()
        payload = {
            "summary": {
                "mols": 50,
                "nonempty_topk": 49,
                "top1_mean": 0.7,
                "top1_median": 0.6,
                "trimmed_mean": 0.55,
                "actual_expansions_mean": 12.0,
                "ofo_calls_mean": 20.0,
            },
            "comparison_vs_reference": None,
        }
        metrics = self.flatten_holdout_metrics(payload)
        self.assertEqual(metrics["holdout_mols"], 50)
        self.assertEqual(metrics["holdout_nonempty_topk"], 49)
        self.assertAlmostEqual(metrics["holdout_top1_median"], 0.6, places=6)
        self.assertNotIn("holdout_win_rate", metrics)

    def test_resolve_run_dir_prefers_checkpoint_parent(self):
        """resume 未显式指定 run_dir 时，应从 checkpoints/ 下的 ckpt 反推 run_dir。"""
        self._require_imports()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "rl_20260412_220000")
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "rl_ckpt_ep000050.pth")
            with open(ckpt_path, "w", encoding="utf-8") as f:
                f.write("placeholder")

            resolved_run_dir, is_resumed = self.resolve_run_dir(
                output_dir=os.path.join(tmpdir, "unused"),
                resume_checkpoint=ckpt_path,
                resume_run_dir=None,
            )
            self.assertTrue(is_resumed)
            self.assertEqual(resolved_run_dir, os.path.abspath(run_dir))

    def test_load_training_state_restores_best_resume_metadata(self):
        """恢复 run 状态时，应带回 history、best_return、holdout priority 和 last_episode_index。"""
        self._require_imports()
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = os.path.join(tmpdir, "rl_history.json")
            config_path = os.path.join(tmpdir, "rl_config.json")
            best_holdout_path = os.path.join(tmpdir, "best_holdout_summary.json")

            history = [
                {"episode": 3, "episode_return": 0.3},
                {"episode": 5, "episode_return": 0.8},
            ]
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False)

            payload = {
                "summary": {
                    "top1_median": 1.1,
                    "trimmed_mean": 0.9,
                    "top1_mean": 0.95,
                },
                "comparison_vs_reference": {
                    "win_rate": 0.7,
                },
            }
            with open(best_holdout_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_episode_return": 1.25,
                        "best_holdout_priority": [1.1, 0.9, 0.7, 0.95],
                        "last_episode_index": 6,
                    },
                    f,
                    ensure_ascii=False,
                )

            state = self.load_training_state(tmpdir)
            self.assertEqual(len(state["history"]), 2)
            self.assertAlmostEqual(state["best_episode_return"], 1.25, places=6)
            self.assertEqual(state["best_holdout_priority"], (1.1, 0.9, 0.7, 0.95))
            self.assertEqual(state["last_episode_index"], 6)
            self.assertEqual(state["best_holdout_payload"]["summary"]["top1_median"], 1.1)


# ---------------------------------------------------------------------------
# BC 数据集 smoke test
# ---------------------------------------------------------------------------

class TestBCDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from mol_evo.core.data.rl_demo_processing import BCDataset, STATE_DIM, ACTION_DIM
            import torch

            cls.BCDataset = BCDataset
            cls.STATE_DIM = STATE_DIM
            cls.ACTION_DIM = ACTION_DIM
            cls.torch = torch
            cls._imports_ok = True
        except ImportError as e:
            cls._imports_ok = False
            cls._import_error = str(e)

    def _require_imports(self):
        if not self._imports_ok:
            self.skipTest(f"导入失败，跳过测试: {self._import_error}")

    def _make_samples(self, n=10):
        """生成最小合法 BC 样本列表。"""
        return [
            {
                "smiles_from": "c1ccccc1",
                "smiles_to": "c1ccc(N)cc1",
                "operation": {"type": "add_functional_group", "params": {"position": 1}},
                "property_change": -0.2,
                "accumulated_from": 0.0,
                "reward": 0.2,
                "future_best_gain": 0.5,
                "remaining_depth": 3,
                "logp_in_range": True,
                "direction": "decrease",
            }
        ] * n

    def test_dataset_len(self):
        """BCDataset 长度应与样本数一致。"""
        self._require_imports()
        samples = self._make_samples(8)
        ds = self.BCDataset(samples, max_depth=4, direction="decrease")
        self.assertEqual(len(ds), 8)

    def test_item_shapes(self):
        """__getitem__ 返回的张量形状应符合预期。"""
        self._require_imports()
        samples = self._make_samples(4)
        ds = self.BCDataset(samples, max_depth=4, direction="decrease")
        item = ds[0]

        self.assertEqual(item["state"].shape, (self.STATE_DIM,))
        self.assertEqual(item["action"].shape, (self.ACTION_DIM,))
        self.assertEqual(item["next_state"].shape, (self.STATE_DIM,))
        self.assertEqual(item["reward"].shape, ())
        self.assertEqual(item["future_best_gain"].shape, ())

    def test_dataloader_batch(self):
        """DataLoader 批量加载应返回正确形状。"""
        self._require_imports()
        from torch.utils.data import DataLoader

        samples = self._make_samples(16)
        ds = self.BCDataset(samples, max_depth=4, direction="decrease")
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        self.assertEqual(batch["state"].shape, (4, self.STATE_DIM))
        self.assertEqual(batch["action"].shape, (4, self.ACTION_DIM))


# ---------------------------------------------------------------------------
# RewardConfig / compute_step_reward smoke test
# ---------------------------------------------------------------------------

class TestRewardFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from mol_evo.core.models.astar_rl.reward import (
                RewardConfig,
                compute_step_reward,
                compute_episode_reward,
                compute_discounted_returns,
                normalize_returns,
            )

            cls.RewardConfig = RewardConfig
            cls.compute_step_reward = staticmethod(compute_step_reward)
            cls.compute_episode_reward = staticmethod(compute_episode_reward)
            cls.compute_discounted_returns = staticmethod(compute_discounted_returns)
            cls.normalize_returns = staticmethod(normalize_returns)
            cls._imports_ok = True
        except ImportError as e:
            cls._imports_ok = False
            cls._import_error = str(e)

    def _require_imports(self):
        if not self._imports_ok:
            self.skipTest(f"导入失败，跳过测试: {self._import_error}")

    def test_step_reward_decrease_improvement(self):
        """decrease 方向下，负 property_change 应得到正奖励。"""
        self._require_imports()
        cfg = self.RewardConfig(direction="decrease")
        r = self.compute_step_reward(
            property_change=-0.5,
            logp_in_range=True,
            stagnation_count=0,
            config=cfg,
        )
        self.assertGreater(r, 0.0)

    def test_step_reward_logp_penalty(self):
        """logP 超出范围时奖励应低于无惩罚情形。"""
        self._require_imports()
        cfg = self.RewardConfig(direction="decrease")
        r_ok = self.compute_step_reward(
            -0.3,
            logp_in_range=True,
            stagnation_count=0,
            config=cfg,
        )
        r_bad = self.compute_step_reward(
            -0.3,
            logp_in_range=False,
            stagnation_count=0,
            config=cfg,
        )
        self.assertGreater(r_ok, r_bad)

    def test_discounted_returns_length(self):
        """compute_discounted_returns 输出长度应与输入一致。"""
        self._require_imports()
        rewards = [1.0, 0.5, -0.2, 0.8]
        returns = self.compute_discounted_returns(rewards, gamma=0.99)
        self.assertEqual(len(returns), len(rewards))

    def test_normalize_returns(self):
        """normalize_returns 输出均值应接近 0。"""
        self._require_imports()
        returns = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = self.normalize_returns(returns)
        mean = sum(normalized) / len(normalized)
        self.assertAlmostEqual(mean, 0.0, places=5)


# ---------------------------------------------------------------------------
# 直接运行
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
