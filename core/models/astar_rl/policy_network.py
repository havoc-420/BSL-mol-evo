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
from typing import Dict, Tuple


class PolicyNet(nn.Module):
    """候选动作排序模型。"""

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
        """计算候选动作的 logit 分数。"""
        if state.dim() == 1:
            n = actions.shape[0]
            state_exp = state.unsqueeze(0).expand(n, -1)
            s_feat = self.state_encoder(state_exp)
            a_feat = self.action_encoder(actions)
            logits = self.fusion(torch.cat([s_feat, a_feat], dim=-1))
            return logits.squeeze(-1)

        batch, n = actions.shape[:2]
        state_exp = state.unsqueeze(1).expand(-1, n, -1)
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
        """返回选定动作的 log probability（用于 REINFORCE 梯度）。"""
        logits = self.forward(state, actions)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs[selected_idx]

    def get_action_distribution(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回动作 logits / probs / log_probs。"""
        if temperature <= 0:
            raise ValueError(f"temperature 必须大于 0，实际为 {temperature}")

        logits = self.forward(state, actions)
        scaled_logits = logits / max(float(temperature), 1e-6)
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        probs = log_probs.exp()
        return logits, probs, log_probs

    def select_action(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        mode: str = "sample",
        temperature: float = 1.0,
        epsilon: float = 0.0,
    ) -> Dict[str, object]:
        """按指定模式选择动作，并返回训练所需统计。"""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                logits, probs, log_probs = self.get_action_distribution(
                    state,
                    actions,
                    temperature=temperature,
                )
                greedy_idx = int(torch.argmax(logits).item())

                if mode == "greedy":
                    selected_idx = greedy_idx
                elif mode == "sample":
                    selected_idx = int(torch.multinomial(probs, num_samples=1).item())
                elif mode == "epsilon_greedy":
                    if not 0.0 <= epsilon <= 1.0:
                        raise ValueError(f"epsilon 必须位于 [0, 1]，实际为 {epsilon}")
                    random_draw = torch.rand(1, device=probs.device).item()
                    if random_draw < epsilon:
                        selected_idx = int(
                            torch.randint(0, probs.shape[0], (1,), device=probs.device).item()
                        )
                    else:
                        selected_idx = greedy_idx
                else:
                    raise ValueError(f"不支持的动作选择模式: {mode}")

                entropy = -(probs * log_probs).sum()
                return {
                    "selected_idx": selected_idx,
                    "log_prob": log_probs[selected_idx],
                    "prob": probs[selected_idx],
                    "entropy": entropy,
                    "greedy_idx": greedy_idx,
                    "logits": logits,
                    "probs": probs,
                }
        finally:
            if was_training:
                self.train()

    def sample_action(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[int, torch.Tensor]:
        """从策略分布中采样一个动作（兼容旧接口）。"""
        selection = self.select_action(
            state,
            actions,
            mode="sample",
            temperature=temperature,
        )
        log_prob = torch.as_tensor(selection["log_prob"])
        selected_idx = int(torch.as_tensor(selection["selected_idx"]).item())
        return selected_idx, log_prob

    def top_k_actions(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 top-K 动作的索引和对应 logit（用于推理阶段预筛）。"""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                logits = self.forward(state, actions)
                k = min(k, logits.shape[0])
                top_logits, top_idx = torch.topk(logits, k)
            return top_idx, top_logits
        finally:
            if was_training:
                self.train()
