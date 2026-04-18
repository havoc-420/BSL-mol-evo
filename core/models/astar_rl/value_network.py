#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ValueNet：未来收益估计模型（A* RL Demo）

功能：
  - 输入当前状态（分子 + 累计属性变化 + 剩余深度 + 目标属性信息），
    输出预期未来收益（future_best_gain）作为启发式 value bonus
  - 支持 BC(MSE) 预训练（回归 BFS/MCTS 树中的 future_best_gain 标签）
  - 支持在线 TD/MC 更新（TD: V(s) ← r + γ·V(s')，MC: V(s) ← G_t）

ValueNet 的作用：
  1. A* demo 中作为 h_score（启发式估计剩余收益），与 g_score 合成 f_score
  2. REINFORCE 中作为 baseline 减小梯度方差：advantage = G_t - V(s_t)

状态特征维度（state_dim 默认 64）：
  与 PolicyNet 使用相同的 encode_state() 输出，保持一致
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueNet(nn.Module):
    """未来收益估计模型。

    Parameters
    ----------
    state_dim:
        状态向量维度（与 PolicyNet 保持一致）。
    hidden_dim:
        隐藏层宽度。
    num_layers:
        MLP 层数。
    dropout:
        Dropout 比例（训练时启用）。
    """

    def __init__(
        self,
        state_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        layers: list = []
        in_dim = state_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        # 最终输出标量 value（无激活，允许负值）
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """估计状态的未来收益。

        Parameters
        ----------
        state:
            形状 ``(state_dim,)`` 或 ``(batch, state_dim)``。

        Returns
        -------
        torch.Tensor
            形状 ``()``（标量）或 ``(batch,)`` 的 value 估计。
        """
        squeezed = state.dim() == 1
        if squeezed:
            state = state.unsqueeze(0)  # (1, state_dim)
        out = self.net(state)           # (batch, 1)
        out = out.squeeze(-1)           # (batch,)
        if squeezed:
            out = out.squeeze(0)        # ()
        return out

    def estimate(self, state: torch.Tensor) -> float:
        """推理阶段：无梯度，返回 Python float。"""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                v = self.forward(state)
            return v.item()
        finally:
            if was_training:
                self.train()

    # ------------------------------------------------------------------
    # Loss helpers（供 RLTrainer 使用）
    # ------------------------------------------------------------------

    def bc_loss(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """BC 预训练 MSE loss。

        Parameters
        ----------
        states:
            形状 ``(batch, state_dim)``。
        targets:
            形状 ``(batch,)``，来自 export_rl_demo_transitions 的 future_best_gain。

        Returns
        -------
        torch.Tensor
            标量 MSE loss。
        """
        preds = self.forward(states)
        return nn.functional.mse_loss(preds, targets)

    def td_loss(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        """TD(0) 更新 loss（在线 RL 阶段）。

        Parameters
        ----------
        states:
            形状 ``(batch, state_dim)``。
        rewards:
            形状 ``(batch,)``。
        next_states:
            形状 ``(batch, state_dim)``。
        dones:
            形状 ``(batch,)``，0/1 标记是否终止状态。
        gamma:
            折扣因子。

        Returns
        -------
        torch.Tensor
            标量 TD loss（MSE of Bellman residual）。
        """
        v_s = self.forward(states)
        with torch.no_grad():
            v_next = self.forward(next_states)
        td_target = rewards + gamma * v_next * (1.0 - dones)
        return nn.functional.mse_loss(v_s, td_target)

    def mc_loss(
        self,
        states: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """Monte-Carlo 更新 loss（在线 RL 阶段，全 episode G_t 回报）。

        Parameters
        ----------
        states:
            形状 ``(batch, state_dim)``。
        returns:
            形状 ``(batch,)``，折扣累计回报 G_t。

        Returns
        -------
        torch.Tensor
            标量 MC loss（MSE）。
        """
        v_s = self.forward(states)
        return nn.functional.mse_loss(v_s, returns)
