#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PolicyNet：候选动作排序模型（A* RL Demo）

功能：
  - 输入当前分子状态表示 + 候选动作特征，输出每个候选的优先级分数
  - 支持 BC 预训练（cross-entropy 模仿 BFS/MCTS 的展开选择分布）
  - 支持在线 REINFORCE 微调（通过 log_prob 反向传播策略梯度）

状态特征维度（state_dim 默认 64）：
  [mol_fingerprint(50) | accumulated_change(1) | remaining_depth(1) |
   direction_flag(1) | logp_in_range(1) | stagnation_count(1) | ...]
  → 实际由调用方通过 encode_state() 工具函数构造，保持松耦合

动作特征维度（action_dim 默认 16）：
  [op_type_onehot(10) | position_norm(1) | fragment_size_norm(1) |
   ofo_predicted_change(1) | ...] （调用方可灵活扩展）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PolicyNet(nn.Module):
    """候选动作排序模型。

    给定当前状态向量和一组候选动作向量，输出每个候选的 logit 分数。
    训练阶段：将 logit 经 softmax 后与 BC 目标分布做 cross-entropy；
              或在 REINFORCE 中对选定动作计算 log_prob。
    推理阶段：取 top-N logit 对应的候选做 OFO 精评。

    Parameters
    ----------
    state_dim:
        状态向量维度（由 encode_state() 决定）。
    action_dim:
        单个动作特征向量维度（由 encode_action() 决定）。
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
        action_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 状态编码分支
        state_layers = []
        in_dim = state_dim
        for _ in range(num_layers - 1):
            state_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        self.state_encoder = nn.Sequential(*state_layers)

        # 动作编码分支
        action_layers = []
        in_dim = action_dim
        for _ in range(num_layers - 1):
            action_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        self.action_encoder = nn.Sequential(*action_layers)

        # 融合层 → scalar logit
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """计算候选动作的 logit 分数。

        Parameters
        ----------
        state:
            形状 ``(state_dim,)`` 或 ``(batch, state_dim)``。
            若为单状态，内部自动 expand 到 ``(n_actions, state_dim)``。
        actions:
            形状 ``(n_actions, action_dim)`` 或 ``(batch, n_actions, action_dim)``。

        Returns
        -------
        torch.Tensor
            形状 ``(n_actions,)`` 或 ``(batch, n_actions)`` 的 logit 分数。
        """
        # --- 单状态/多动作：expand state ---
        if state.dim() == 1:
            # state: (state_dim,), actions: (n_actions, action_dim)
            n = actions.shape[0]
            state_exp = state.unsqueeze(0).expand(n, -1)  # (n, state_dim)
            s_feat = self.state_encoder(state_exp)          # (n, hidden_dim)
            a_feat = self.action_encoder(actions)            # (n, hidden_dim)
            logits = self.fusion(torch.cat([s_feat, a_feat], dim=-1))  # (n, 1)
            return logits.squeeze(-1)  # (n,)

        # --- batch 模式：(batch, n_actions, ...) ---
        batch, n = actions.shape[:2]
        state_exp = state.unsqueeze(1).expand(-1, n, -1)   # (B, n, state_dim)
        s_feat = self.state_encoder(state_exp.reshape(batch * n, -1))
        a_feat = self.action_encoder(actions.reshape(batch * n, -1))
        logits = self.fusion(torch.cat([s_feat, a_feat], dim=-1))
        return logits.reshape(batch, n)

    def get_log_probs(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        selected_idx: int,
    ) -> torch.Tensor:
        """返回选定动作的 log probability（用于 REINFORCE 梯度）。

        Parameters
        ----------
        state:
            形状 ``(state_dim,)``。
        actions:
            形状 ``(n_actions, action_dim)``。
        selected_idx:
            被选中动作的索引。

        Returns
        -------
        torch.Tensor
            标量 log_prob（保留计算图，用于 loss.backward()）。
        """
        logits = self.forward(state, actions)   # (n_actions,)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs[selected_idx]

    def sample_action(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[int, torch.Tensor]:
        """从策略分布中采样一个动作（用于在线 RL 探索）。

        Parameters
        ----------
        temperature:
            温度系数，越大越随机，越小越贪婪。

        Returns
        -------
        (selected_idx, log_prob)
        """
        logits = self.forward(state, actions) / temperature
        probs = F.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1).item()
        log_prob = F.log_softmax(logits, dim=-1)[idx]
        return int(idx), log_prob

    def top_k_actions(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 top-K 动作的索引和对应 logit（用于推理阶段预筛）。

        Returns
        -------
        (top_k_indices, top_k_logits)
        """
        with torch.no_grad():
            logits = self.forward(state, actions)
            k = min(k, logits.shape[0])
            top_logits, top_idx = torch.topk(logits, k)
        return top_idx, top_logits
