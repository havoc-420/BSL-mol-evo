#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在线 RL 训练器（A* RL Demo）

封装 REINFORCE 策略梯度更新 + ValueNet TD/MC 更新逻辑，
供 astar_demo 搜索在 episode 结束后调用，实现在线训练闭环。

核心方法：
  - collect_step()     : 记录单步轨迹数据
  - end_episode()      : episode 结束，计算奖励 / 回报，触发梯度更新
  - update_policy()    : REINFORCE 梯度更新 PolicyNet
  - update_value()     : MC/TD 更新 ValueNet
  - save_checkpoint()  : 保存权重 + 训练状态
  - load_checkpoint()  : 恢复权重 + 训练状态（支持断点续训）

PPO 作为进阶子类（PPORLTrainer）预留接口，demo 阶段不强制实现。
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

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
    """单步轨迹数据（在线 RL 收集）。

    Fields
    ------
    state_tensor:
        当前状态向量 ``(state_dim,)``，由 encode_state() 生成。
    action_tensors:
        候选动作矩阵 ``(n_actions, action_dim)``，policy 从中选择。
    selected_action_idx:
        被选中动作的索引。
    log_prob:
        选定动作的 log_prob（保留计算图，用于策略梯度）。
    step_reward:
        本步标量奖励（由 compute_step_reward 计算）。
    next_state_tensor:
        下一步状态向量 ``(state_dim,)``。
    value_estimate:
        ValueNet 对当前状态的估计值（float，推理时已 detach）。
    done:
        是否为终止步（达到 max_depth 或无合法后继）。
    """
    state_tensor: torch.Tensor
    action_tensors: torch.Tensor
    selected_action_idx: int
    log_prob: torch.Tensor
    step_reward: float
    next_state_tensor: torch.Tensor
    value_estimate: float
    done: bool = False


# ---------------------------------------------------------------------------
# RLTrainer
# ---------------------------------------------------------------------------

class RLTrainer:
    """REINFORCE + ValueNet 在线训练器。

    Parameters
    ----------
    policy_net:
        PolicyNet 实例。
    value_net:
        ValueNet 实例。
    reward_config:
        奖励函数超参。
    lr_policy / lr_value:
        学习率。
    gamma:
        折扣因子。
    entropy_coef:
        策略熵正则系数（减少策略过早收敛）。
    value_loss_coef:
        ValueNet 损失权重（相对于策略损失）。
    max_grad_norm:
        梯度裁剪范数。
    device:
        'cpu' 或 'cuda'。
    checkpoint_dir:
        checkpoint 保存目录。
    checkpoint_every:
        每多少个 episode 保存一次 checkpoint。
    """

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

        self.policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(value_net.parameters(), lr=lr_value)

        # 训练统计
        self.episode_count: int = 0
        self.total_steps: int = 0
        self.history: List[Dict] = []

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
    # Update API（episode 结束后调用）
    # ------------------------------------------------------------------

    def end_episode(self) -> Dict:
        """episode 结束，计算回报并更新 policy/value。

        Returns
        -------
        dict
            含 policy_loss, value_loss, episode_return, episode_steps 等统计。
        """
        if not self._trajectory:
            return {}

        self.episode_count += 1
        self.total_steps += len(self._trajectory)

        step_rewards = [s.step_reward for s in self._trajectory]

        # 计算折扣回报并标准化
        returns = compute_discounted_returns(step_rewards, self.gamma)
        returns_normalized = normalize_returns(returns)
        returns_tensor = torch.tensor(returns_normalized, dtype=torch.float32, device=self.device)

        policy_loss, value_loss = self.update_policy_and_value(returns_tensor)

        stats = {
            "episode": self.episode_count,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "episode_return": sum(step_rewards),
            "episode_steps": len(self._trajectory),
        }
        self.history.append(stats)

        # checkpoint
        if self.checkpoint_dir and self.episode_count % self.checkpoint_every == 0:
            self.save_checkpoint()

        logger.info(
            f"[RLTrainer] ep={self.episode_count} "
            f"return={stats['episode_return']:.4f} "
            f"policy_loss={policy_loss:.6f} "
            f"value_loss={value_loss:.6f}"
        )

        self._trajectory.clear()
        return stats

    def update_policy_and_value(
        self,
        returns_tensor: torch.Tensor,
    ) -> Tuple[float, float]:
        """REINFORCE 梯度更新 + ValueNet MC 更新。

        Parameters
        ----------
        returns_tensor:
            标准化后的折扣回报 G_t，形状 ``(T,)``。

        Returns
        -------
        (policy_loss_value, value_loss_value) : Tuple[float, float]
        """
        self.policy_net.train()
        self.value_net.train()

        # --- 重新计算 log_probs 和 values（保留计算图）---
        log_probs = []
        value_preds = []

        for step in self._trajectory:
            s = step.state_tensor.to(self.device)
            a = step.action_tensors.to(self.device)

            # policy log_prob
            lp = self.policy_net.get_log_probs(s, a, step.selected_action_idx)
            log_probs.append(lp)

            # value estimate
            v = self.value_net(s)
            value_preds.append(v)

        log_probs_tensor = torch.stack(log_probs)    # (T,)
        value_preds_tensor = torch.stack(value_preds) # (T,)

        # --- REINFORCE 策略 loss ---
        advantages = returns_tensor - value_preds_tensor.detach()
        policy_loss = -(log_probs_tensor * advantages).mean()

        # --- 熵正则（鼓励探索）---
        # 用 log_prob 近似熵：H ≈ -E[log_prob]
        entropy_loss = log_probs_tensor.mean()  # 值越大越不随机
        policy_loss = policy_loss + self.entropy_coef * entropy_loss

        # --- ValueNet MC loss ---
        value_loss = nn.functional.mse_loss(value_preds_tensor, returns_tensor)

        # --- 联合更新 ---
        total_loss = policy_loss + self.value_loss_coef * value_loss

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()

        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)

        self.policy_optimizer.step()
        self.value_optimizer.step()

        return policy_loss.item(), value_loss.item()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: Optional[str] = None) -> str:
        """保存 policy/value 权重 + 优化器状态 + 训练统计。

        Parameters
        ----------
        tag:
            文件名后缀，默认用 episode 数。

        Returns
        -------
        str
            保存的文件路径。
        """
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir 未设置，无法保存 checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        tag = tag or f"ep{self.episode_count:06d}"
        path = os.path.join(self.checkpoint_dir, f"rl_ckpt_{tag}.pth")

        torch.save(
            {
                "episode_count": self.episode_count,
                "total_steps": self.total_steps,
                "policy_net": self.policy_net.state_dict(),
                "value_net": self.value_net.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
                "reward_config": self.reward_config,
                "history": self.history[-200:],  # 只保留最近 200 条记录
            },
            path,
        )
        logger.info(f"[RLTrainer] Checkpoint saved → {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """加载 checkpoint，支持断点续训。

        Parameters
        ----------
        path:
            checkpoint 文件路径（.pth）。
        """
        ckpt = torch.load(path, map_location=self.device)
        self.episode_count = ckpt.get("episode_count", 0)
        self.total_steps = ckpt.get("total_steps", 0)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.value_net.load_state_dict(ckpt["value_net"])
        self.policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
        self.value_optimizer.load_state_dict(ckpt["value_optimizer"])
        self.history = ckpt.get("history", [])
        logger.info(
            f"[RLTrainer] Checkpoint loaded ← {path} "
            f"(ep={self.episode_count}, steps={self.total_steps})"
        )


# ---------------------------------------------------------------------------
# PPO placeholder（进阶扩展，demo 阶段仅保留骨架）
# ---------------------------------------------------------------------------

class PPORLTrainer(RLTrainer):
    """PPO 训练器（进阶扩展，当前为占位骨架）。

    继承 RLTrainer，重写 update_policy_and_value() 以实现 clip 目标函数。
    Demo 阶段不强制使用，优先用 REINFORCE（父类）验证方向。
    """

    def __init__(self, *args, clip_eps: float = 0.2, n_epochs: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs

    def update_policy_and_value(
        self,
        returns_tensor: torch.Tensor,
    ) -> Tuple[float, float]:
        """PPO clip 目标（TODO: 实现完整版本）。

        当前 fallback 到父类 REINFORCE 实现，待 Phase 3-b 补全。
        """
        logger.warning("[PPORLTrainer] PPO 完整实现待补全，当前使用 REINFORCE fallback")
        return super().update_policy_and_value(returns_tensor)
