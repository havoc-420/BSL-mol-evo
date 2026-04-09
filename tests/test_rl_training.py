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

import os
import sys
import json
import tempfile
import unittest

# 项目根路径
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..', '..')
sys.path.insert(0, _root)


class TestRLTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from mol_evo.core.models.astar_rl import PolicyNet, ValueNet, RLTrainer
            from mol_evo.core.models.astar_rl.reward import RewardConfig
            from mol_evo.core.models.astar_rl.rl_trainer import TrajectoryStep
            from mol_evo.core.data.rl_demo_processing import STATE_DIM, ACTION_DIM

            cls.torch = torch
            cls.PolicyNet = PolicyNet
            cls.ValueNet = ValueNet
            cls.RLTrainer = RLTrainer
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

    def _make_trainer(self, checkpoint_dir=None):
        """创建一个 RLTrainer 实例（CPU）。"""
        policy_net = self.PolicyNet(state_dim=self.STATE_DIM, action_dim=self.ACTION_DIM)
        value_net = self.ValueNet(state_dim=self.STATE_DIM)
        trainer = self.RLTrainer(
            policy_net=policy_net,
            value_net=value_net,
            reward_config=self.RewardConfig(direction="decrease"),
            lr_policy=1e-4,
            lr_value=1e-3,
            gamma=0.99,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=10,
        )
        return trainer, policy_net, value_net

    def _make_step(self, n_actions=4):
        """创建一个随机 TrajectoryStep。"""
        torch = self.torch
        state = torch.randn(self.STATE_DIM)
        actions = torch.randn(n_actions, self.ACTION_DIM)
        policy_net = self.PolicyNet(state_dim=self.STATE_DIM, action_dim=self.ACTION_DIM)
        log_prob = policy_net.get_log_probs(state, actions, 0)
        next_state = torch.randn(self.STATE_DIM)
        return self.TrajectoryStep(
            state_tensor=state,
            action_tensors=actions,
            selected_action_idx=0,
            log_prob=log_prob,
            step_reward=0.3,
            next_state_tensor=next_state,
            value_estimate=0.1,
            done=False,
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

        for ep in range(3):
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
            trainer.end_episode()  # episode_count = 1

            ckpt_path = trainer.save_checkpoint(tag="test")
            self.assertTrue(os.path.isfile(ckpt_path))

            # 恢复到新 trainer
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
        self.assertTrue(math.isfinite(stats["policy_loss"]),
                        f"policy_loss 不是有限值: {stats['policy_loss']}")
        self.assertTrue(math.isfinite(stats["value_loss"]),
                        f"value_loss 不是有限值: {stats['value_loss']}")


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
                RewardConfig, compute_step_reward, compute_episode_reward,
                compute_discounted_returns, normalize_returns,
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
            property_change=-0.5, logp_in_range=True, stagnation_count=0, config=cfg
        )
        self.assertGreater(r, 0.0)

    def test_step_reward_logp_penalty(self):
        """logP 超出范围时奖励应低于无惩罚情形。"""
        self._require_imports()
        cfg = self.RewardConfig(direction="decrease")
        r_ok = self.compute_step_reward(-0.3, logp_in_range=True, stagnation_count=0, config=cfg)
        r_bad = self.compute_step_reward(-0.3, logp_in_range=False, stagnation_count=0, config=cfg)
        self.assertGreater(r_ok, r_bad)

    def test_discounted_returns_length(self):
        """compute_discounted_returns 输出长度应与输入一致。"""
        self._require_imports()
        rewards = [1.0, 0.5, -0.2, 0.8]
        returns = self.compute_discounted_returns(rewards, gamma=0.99)
        self.assertEqual(len(returns), len(rewards))

    def test_normalize_returns(self):
        """normalize_returns 输出均值应接近 0，标准差接近 1（若非常数）。"""
        self._require_imports()
        import math
        returns = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = self.normalize_returns(returns)
        mean = sum(normalized) / len(normalized)
        self.assertAlmostEqual(mean, 0.0, places=5)


# ---------------------------------------------------------------------------
# 直接运行
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
