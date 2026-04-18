#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在线 RL 训练器（A* RL Demo）

封装 REINFORCE / PPO + ValueNet 在线更新逻辑，
供 astar_demo 搜索在 episode 结束后调用，实现在线训练闭环。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from mol_evo.core.models.astar_rl.policy_network import PolicyNet
from mol_evo.core.models.astar_rl.value_network import ValueNet
from mol_evo.core.models.astar_rl.reward import (
    RewardConfig,
    compute_discounted_returns,
    normalize_returns,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory step dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryStep:
    """单步轨迹数据（在线 RL 收集）。"""

    state_tensor: torch.Tensor
    action_tensors: torch.Tensor
    selected_action_idx: int
    log_prob: torch.Tensor
    step_reward: float
    next_state_tensor: torch.Tensor
    value_estimate: float
    done: bool = False
    selection_mode: Optional[str] = None
    selected_action_rank: Optional[int] = None
    policy_entropy: Optional[float] = None
    selected_action_prob: Optional[float] = None
    greedy_action_idx: Optional[int] = None
    matches_policy_greedy: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RLTrainer
# ---------------------------------------------------------------------------

class RLTrainer:
    """REINFORCE + ValueNet 在线训练器。"""

    def __init__(
        self,
        policy_net: PolicyNet,
        value_net: ValueNet,
        reward_config: Optional[RewardConfig] = None,
        lr_policy: float = 1e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 5.0,
        device: str = "cpu",
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 50,
        action_selection_mode: str = "sample",
        action_selection_temperature: float = 1.0,
        action_selection_epsilon: float = 0.0,
    ) -> None:
        self.policy_net = policy_net.to(device)
        self.value_net = value_net.to(device)
        self.reward_config = reward_config or RewardConfig()
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.action_selection_mode = action_selection_mode
        self.action_selection_temperature = action_selection_temperature
        self.action_selection_epsilon = action_selection_epsilon
        self.algo_name = "reinforce"

        self.policy_optimizer = torch.optim.Adam(  # pyright: ignore[reportPrivateImportUsage]
            policy_net.parameters(),
            lr=lr_policy,
        )
        self.value_optimizer = torch.optim.Adam(  # pyright: ignore[reportPrivateImportUsage]
            value_net.parameters(),
            lr=lr_value,
        )

        # 训练统计
        self.episode_count: int = 0
        self.total_steps: int = 0
        self.history: List[Dict[str, Any]] = []
        self._last_update_metrics: Dict[str, Any] = {}

        # 当前 episode 轨迹缓冲
        self._trajectory: List[TrajectoryStep] = []

    # ------------------------------------------------------------------
    # Trajectory collection API（搜索侧调用）
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """每个 episode 开始前重置轨迹缓冲。"""
        self._trajectory.clear()

    def collect_step(self, step: TrajectoryStep) -> None:
        """记录一步轨迹数据。由 generate_expansion_tree_astar_demo 在每步扩展后调用。"""
        self._trajectory.append(step)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _set_last_update_metrics(self, metrics: Optional[Dict[str, Any]] = None) -> None:
        self._last_update_metrics = {
            key: value
            for key, value in (metrics or {}).items()
            if value is not None
        }

    def _compute_advantage_stats(self, advantages: torch.Tensor) -> Dict[str, float]:
        if advantages.numel() == 0:
            return {}
        adv = advantages.detach().float()
        adv_std = float(adv.std(unbiased=False).item()) if adv.numel() > 1 else 0.0
        return {
            "adv_mean": float(adv.mean().item()),
            "adv_std": adv_std,
            "adv_min": float(adv.min().item()),
            "adv_max": float(adv.max().item()),
        }

    def _enter_deterministic_update_mode(self) -> Tuple[bool, bool]:
        """在线 RL 更新时关闭 dropout，避免策略/价值评估抖动。"""
        policy_was_training = self.policy_net.training
        value_was_training = self.value_net.training
        self.policy_net.eval()
        self.value_net.eval()
        return policy_was_training, value_was_training

    def _restore_module_mode(self, policy_was_training: bool, value_was_training: bool) -> None:
        if policy_was_training:
            self.policy_net.train()
        if value_was_training:
            self.value_net.train()

    def _evaluate_step(
        self,
        step: TrajectoryStep,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = step.state_tensor.to(self.device)
        actions = step.action_tensors.to(self.device)
        dist_temperature = (
            self.action_selection_temperature
            if temperature is None
            else float(temperature)
        )
        _, probs, log_probs = self.policy_net.get_action_distribution(
            state,
            actions,
            temperature=dist_temperature,
        )
        selected_log_prob = log_probs[step.selected_action_idx]
        entropy = -(probs * log_probs).sum()
        value_pred = self.value_net(state)
        return selected_log_prob, entropy, value_pred

    def _get_extra_checkpoint_state(self) -> Dict[str, Any]:
        return {}

    def _load_extra_checkpoint_state(self, _ckpt: Dict[str, Any]) -> None:
        return None

    # ------------------------------------------------------------------
    # Update API（episode 结束后调用）
    # ------------------------------------------------------------------

    def end_episode(self) -> Dict[str, Any]:
        """episode 结束，计算回报并更新 policy/value。"""
        if not self._trajectory:
            return {}

        self.episode_count += 1
        self.total_steps += len(self._trajectory)

        step_rewards = [s.step_reward for s in self._trajectory]

        # REINFORCE 默认使用 MC return；PPO 子类可在 update 中自行重算 GAE。
        returns = compute_discounted_returns(step_rewards, self.gamma)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns_normalized = normalize_returns(returns)
        normalized_returns_tensor = torch.tensor(
            returns_normalized,
            dtype=torch.float32,
            device=self.device,
        )

        policy_loss, value_loss = self.update_policy_and_value(
            returns_tensor=returns_tensor,
            normalized_returns_tensor=normalized_returns_tensor,
        )

        policy_entropies = [
            float(s.policy_entropy) for s in self._trajectory
            if s.policy_entropy is not None
        ]
        selected_probs = [
            float(s.selected_action_prob) for s in self._trajectory
            if s.selected_action_prob is not None
        ]
        selected_ranks = [
            int(s.selected_action_rank) for s in self._trajectory
            if s.selected_action_rank is not None
        ]
        greedy_matches = [
            1.0 if s.matches_policy_greedy else 0.0 for s in self._trajectory
            if s.matches_policy_greedy is not None
        ]

        stats: Dict[str, Any] = {
            "episode": self.episode_count,
            "algo": self.algo_name,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "episode_return": sum(step_rewards),
            "episode_steps": len(self._trajectory),
            "action_selection_mode": self.action_selection_mode,
        }
        if policy_entropies:
            stats["avg_policy_entropy"] = sum(policy_entropies) / len(policy_entropies)
        if selected_probs:
            stats["avg_selected_action_prob"] = sum(selected_probs) / len(selected_probs)
        if selected_ranks:
            stats["avg_selected_action_rank"] = sum(selected_ranks) / len(selected_ranks)
        if greedy_matches:
            stats["policy_greedy_match_rate"] = sum(greedy_matches) / len(greedy_matches)
        if self._last_update_metrics:
            stats.update(self._last_update_metrics)
        self.history.append(stats)

        if self.checkpoint_dir and self.episode_count % self.checkpoint_every == 0:
            self.save_checkpoint()

        logger.info(
            f"[RLTrainer] ep={self.episode_count} "
            f"algo={self.algo_name} "
            f"return={stats['episode_return']:.4f} "
            f"policy_loss={policy_loss:.6f} "
            f"value_loss={value_loss:.6f}"
        )

        self._trajectory.clear()
        return stats

    def update_policy_and_value(
        self,
        returns_tensor: torch.Tensor,
        normalized_returns_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """REINFORCE 梯度更新 + ValueNet MC 更新。"""
        if normalized_returns_tensor is None:
            normalized_returns_tensor = returns_tensor

        self._set_last_update_metrics()
        policy_was_training, value_was_training = self._enter_deterministic_update_mode()

        try:
            log_probs = []
            entropies = []
            value_preds = []

            for step in self._trajectory:
                log_prob, entropy, value_pred = self._evaluate_step(step)
                log_probs.append(log_prob)
                entropies.append(entropy)
                value_preds.append(value_pred)

            log_probs_tensor = torch.stack(log_probs)
            entropies_tensor = torch.stack(entropies)
            value_preds_tensor = torch.stack(value_preds)

            advantages = normalized_returns_tensor - value_preds_tensor.detach()
            policy_loss = -(log_probs_tensor * advantages).mean()
            entropy_bonus = entropies_tensor.mean()
            value_loss = nn.functional.mse_loss(value_preds_tensor, returns_tensor)
            total_loss = (
                policy_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_bonus
            )

            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()

            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)

            self.policy_optimizer.step()
            self.value_optimizer.step()

            metrics = {
                "policy_entropy": float(entropy_bonus.detach().item()),
            }
            metrics.update(self._compute_advantage_stats(advantages))
            self._set_last_update_metrics(metrics)
            return float(policy_loss.item()), float(value_loss.item())
        finally:
            self._restore_module_mode(policy_was_training, value_was_training)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: Optional[str] = None) -> str:
        """保存 policy/value 权重 + 优化器状态 + 训练统计。"""
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir 未设置，无法保存 checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        tag = tag or f"ep{self.episode_count:06d}"
        path = os.path.join(self.checkpoint_dir, f"rl_ckpt_{tag}.pth")

        payload: Dict[str, Any] = {
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "algo_name": self.algo_name,
            "policy_net": self.policy_net.state_dict(),
            "value_net": self.value_net.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "reward_config": self.reward_config,
            "history": self.history[-200:],
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "value_loss_coef": self.value_loss_coef,
            "max_grad_norm": self.max_grad_norm,
            "action_selection_mode": self.action_selection_mode,
            "action_selection_temperature": self.action_selection_temperature,
            "action_selection_epsilon": self.action_selection_epsilon,
        }
        payload.update(self._get_extra_checkpoint_state())

        torch.save(payload, path)
        logger.info(f"[RLTrainer] Checkpoint saved → {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """加载 checkpoint，支持断点续训。"""
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)

        self.episode_count = ckpt.get("episode_count", 0)
        self.total_steps = ckpt.get("total_steps", 0)
        self.algo_name = ckpt.get("algo_name", self.algo_name)
        self.gamma = ckpt.get("gamma", self.gamma)
        self.entropy_coef = ckpt.get("entropy_coef", self.entropy_coef)
        self.value_loss_coef = ckpt.get("value_loss_coef", self.value_loss_coef)
        self.max_grad_norm = ckpt.get("max_grad_norm", self.max_grad_norm)
        self.reward_config = ckpt.get("reward_config", self.reward_config)

        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.value_net.load_state_dict(ckpt["value_net"])
        self.policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
        self.value_optimizer.load_state_dict(ckpt["value_optimizer"])
        self.history = ckpt.get("history", [])
        self.action_selection_mode = ckpt.get(
            "action_selection_mode",
            self.action_selection_mode,
        )
        self.action_selection_temperature = ckpt.get(
            "action_selection_temperature",
            self.action_selection_temperature,
        )
        self.action_selection_epsilon = ckpt.get(
            "action_selection_epsilon",
            self.action_selection_epsilon,
        )
        self._load_extra_checkpoint_state(ckpt)
        logger.info(
            f"[RLTrainer] Checkpoint loaded ← {path} "
            f"(ep={self.episode_count}, steps={self.total_steps}, algo={self.algo_name})"
        )


# ---------------------------------------------------------------------------
# PPO trainer
# ---------------------------------------------------------------------------

class PPORLTrainer(RLTrainer):
    """PPO + GAE 在线训练器。"""

    def __init__(
        self,
        *args,
        clip_ratio: float = 0.2,
        gae_lambda: float = 0.95,
        update_epochs: int = 4,
        minibatch_size: int = 0,
        normalize_advantage: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.action_selection_mode != "sample":
            raise ValueError(
                "PPORLTrainer 目前要求 action_selection_mode='sample'，"
                f"实际为 {self.action_selection_mode!r}"
            )
        self.algo_name = "ppo"
        self.clip_ratio = float(clip_ratio)
        self.gae_lambda = float(gae_lambda)
        self.update_epochs = int(update_epochs)
        self.minibatch_size = int(minibatch_size)
        self.normalize_advantage = bool(normalize_advantage)

    def _get_extra_checkpoint_state(self) -> Dict[str, Any]:
        return {
            "clip_ratio": self.clip_ratio,
            "gae_lambda": self.gae_lambda,
            "update_epochs": self.update_epochs,
            "minibatch_size": self.minibatch_size,
            "normalize_advantage": self.normalize_advantage,
        }

    def _load_extra_checkpoint_state(self, ckpt: Dict[str, Any]) -> None:
        self.clip_ratio = float(ckpt.get("clip_ratio", self.clip_ratio))
        self.gae_lambda = float(ckpt.get("gae_lambda", self.gae_lambda))
        self.update_epochs = int(ckpt.get("update_epochs", self.update_epochs))
        self.minibatch_size = int(ckpt.get("minibatch_size", self.minibatch_size))
        self.normalize_advantage = bool(
            ckpt.get("normalize_advantage", self.normalize_advantage)
        )

    def _compute_gae_returns_and_advantages(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rewards = torch.tensor(
            [float(step.step_reward) for step in self._trajectory],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [1.0 if step.done else 0.0 for step in self._trajectory],
            dtype=torch.float32,
            device=self.device,
        )
        values = torch.tensor(
            [float(step.value_estimate) for step in self._trajectory],
            dtype=torch.float32,
            device=self.device,
        )

        next_values: List[float] = []
        value_was_training = self.value_net.training
        self.value_net.eval()
        try:
            with torch.no_grad():
                for step in self._trajectory:
                    if step.done:
                        next_values.append(0.0)
                        continue
                    next_state = step.next_state_tensor.to(self.device)
                    next_values.append(float(self.value_net(next_state).item()))
        finally:
            if value_was_training:
                self.value_net.train()

        next_values_tensor = torch.tensor(
            next_values,
            dtype=torch.float32,
            device=self.device,
        )

        deltas = rewards + self.gamma * next_values_tensor * (1.0 - dones) - values
        advantages = torch.zeros_like(deltas)
        gae = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        for idx in range(len(self._trajectory) - 1, -1, -1):
            mask = 1.0 - dones[idx]
            gae = deltas[idx] + self.gamma * self.gae_lambda * mask * gae
            advantages[idx] = gae

        returns = advantages + values
        policy_advantages = advantages.clone()
        if self.normalize_advantage:
            if policy_advantages.numel() > 1:
                adv_std = policy_advantages.std(unbiased=False)
                policy_advantages = (
                    (policy_advantages - policy_advantages.mean())
                    / (adv_std + 1e-8)
                )
            else:
                policy_advantages = policy_advantages - policy_advantages.mean()

        return returns.detach(), policy_advantages.detach(), advantages.detach()

    def _iter_minibatch_indices(self, total_steps: int):
        if total_steps <= 0:
            return
        effective_batch_size = total_steps
        if self.minibatch_size > 0:
            effective_batch_size = min(self.minibatch_size, total_steps)
        permutation = torch.randperm(total_steps, device=self.device)
        for start in range(0, total_steps, effective_batch_size):
            yield permutation[start : start + effective_batch_size]

    def update_policy_and_value(
        self,
        returns_tensor: torch.Tensor,
        normalized_returns_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """PPO clipped objective + GAE。"""
        del returns_tensor, normalized_returns_tensor

        self._set_last_update_metrics()
        total_steps = len(self._trajectory)
        if total_steps == 0:
            return 0.0, 0.0

        ppo_returns, policy_advantages, raw_advantages = (
            self._compute_gae_returns_and_advantages()
        )
        old_log_probs = torch.stack(
            [
                torch.as_tensor(
                    step.log_prob,
                    dtype=torch.float32,
                    device=self.device,
                )
                for step in self._trajectory
            ]
        ).detach()

        policy_was_training, value_was_training = self._enter_deterministic_update_mode()

        policy_loss_values: List[float] = []
        value_loss_values: List[float] = []
        entropy_values: List[float] = []
        approx_kl_values: List[float] = []
        clip_fraction_values: List[float] = []

        try:
            for _ in range(max(self.update_epochs, 1)):
                for batch_indices in self._iter_minibatch_indices(total_steps):
                    if batch_indices.numel() == 0:
                        continue

                    new_log_probs = []
                    entropies = []
                    value_preds = []
                    for batch_idx in batch_indices.tolist():
                        step = self._trajectory[batch_idx]
                        log_prob, entropy, value_pred = self._evaluate_step(step)
                        new_log_probs.append(log_prob)
                        entropies.append(entropy)
                        value_preds.append(value_pred)

                    new_log_probs_tensor = torch.stack(new_log_probs)
                    entropies_tensor = torch.stack(entropies)
                    value_preds_tensor = torch.stack(value_preds)

                    old_log_probs_batch = old_log_probs[batch_indices]
                    advantages_batch = policy_advantages[batch_indices]
                    returns_batch = ppo_returns[batch_indices]

                    log_ratio = new_log_probs_tensor - old_log_probs_batch
                    ratio = torch.exp(log_ratio)
                    clipped_ratio = torch.clamp(
                        ratio,
                        1.0 - self.clip_ratio,
                        1.0 + self.clip_ratio,
                    )

                    surrogate_unclipped = ratio * advantages_batch
                    surrogate_clipped = clipped_ratio * advantages_batch
                    policy_loss = -torch.minimum(
                        surrogate_unclipped,
                        surrogate_clipped,
                    ).mean()
                    entropy_bonus = entropies_tensor.mean()
                    value_loss = nn.functional.mse_loss(value_preds_tensor, returns_batch)
                    total_loss = (
                        policy_loss
                        + self.value_loss_coef * value_loss
                        - self.entropy_coef * entropy_bonus
                    )

                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()
                    total_loss.backward()

                    nn.utils.clip_grad_norm_(
                        self.policy_net.parameters(),
                        self.max_grad_norm,
                    )
                    nn.utils.clip_grad_norm_(
                        self.value_net.parameters(),
                        self.max_grad_norm,
                    )

                    self.policy_optimizer.step()
                    self.value_optimizer.step()

                    approx_kl = (old_log_probs_batch - new_log_probs_tensor.detach()).mean()
                    clip_fraction = (
                        (torch.abs(ratio.detach() - 1.0) > self.clip_ratio)
                        .float()
                        .mean()
                    )

                    policy_loss_values.append(float(policy_loss.detach().item()))
                    value_loss_values.append(float(value_loss.detach().item()))
                    entropy_values.append(float(entropy_bonus.detach().item()))
                    approx_kl_values.append(float(approx_kl.detach().item()))
                    clip_fraction_values.append(float(clip_fraction.detach().item()))

            metrics: Dict[str, Any] = {
                "policy_entropy": (
                    sum(entropy_values) / len(entropy_values)
                    if entropy_values else 0.0
                ),
                "approx_kl": (
                    sum(approx_kl_values) / len(approx_kl_values)
                    if approx_kl_values else 0.0
                ),
                "clip_fraction": (
                    sum(clip_fraction_values) / len(clip_fraction_values)
                    if clip_fraction_values else 0.0
                ),
                "gae_lambda": self.gae_lambda,
                "clip_ratio": self.clip_ratio,
                "update_epochs": self.update_epochs,
                "minibatch_size": (
                    min(self.minibatch_size, total_steps)
                    if self.minibatch_size > 0 else total_steps
                ),
                "normalize_advantage": self.normalize_advantage,
            }
            metrics.update(self._compute_advantage_stats(raw_advantages))
            self._set_last_update_metrics(metrics)

            avg_policy_loss = (
                sum(policy_loss_values) / len(policy_loss_values)
                if policy_loss_values else 0.0
            )
            avg_value_loss = (
                sum(value_loss_values) / len(value_loss_values)
                if value_loss_values else 0.0
            )
            return avg_policy_loss, avg_value_loss
        finally:
            self._restore_module_mode(policy_was_training, value_was_training)
