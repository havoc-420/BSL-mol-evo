#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0.3 长链路路径基座模型 — VisNet 节点编码 + Step Token 融合 + Path Transformer 编码

核心设计：
- 节点编码层：复用 VisNetMoleculeFeatureExtractor 编码每个合法节点
  非法中间节点使用可学习的 invalid_node_embedding
- 步骤编码层：每步组合 from_feat / to_feat / diff_feat / edge_feat → step token
- 路径编码层：Transformer Encoder 建模长链路上下文 (带 padding mask)
- 输出头：
  - path_head: 路径总属性变化回归（主输出）
  - step_head: 每步属性变化回归（可选辅助，默认关闭）
  - validity_head: 每步合法性预测（可选辅助，利用节点合法性白送的监督信号）

输入约定 (由 path_collate 提供的 batch dict):
- node_batch_list: List[Batch], 长度 = max_num_nodes
- node_valid_mask: (B, max_num_nodes) bool
- edge_features: (B, max_steps, edge_dim)
- step_valid_mask: (B, max_steps) bool
- path_padding_mask: (B, max_steps) bool

输出:
- path_pred: (B, output_dim)
- step_pred: (B, max_steps, output_dim) 或 None
- validity_pred: (B, max_steps, 1) 或 None  — 每步合法性 logit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from ..v0.molecule_feature_extractors import VisNetMoleculeFeatureExtractor
from ..v0.edge_feature_extractors import LinearEdgeFeatureExtractor
from ..v0 import register_model


@register_model(
    display_name="VisNet Path 长链路模型 v0.3",
    save_dir_name="visnet_path_v0_3"
)
class MoleculeEvolutionVisnetPathPredictorV03(nn.Module):
    """
    v0.3 长链路路径基座模型。
    
    架构：
        1. VisNet 节点编码（共享权重）→ 每节点 512 维
        2. 非法节点 → 可学习 invalid_node_embedding (512 维)
        3. Step Encoder: [from_feat, to_feat, diff_feat, edge_feat] → step_dim
        4. Path Transformer Encoder: step_dim → 上下文 step 表示
        5. path_head: 首尾特征 + 路径全局表示 → 总变化预测（主输出）
        6. step_head: 每步上下文表示 → 步级变化预测（可选辅助，默认关闭）
        7. validity_head: 每步上下文表示 → 步级合法性预测（可选辅助）
    """

    def __init__(
        self,
        node_feature_dim: int = 11,
        edge_feature_dim: int = 15,
        hidden_dims: list = None,
        output_dim: int = 1,
        # 路径 Transformer 参数
        path_nhead: int = 4,
        path_num_layers: int = 3,
        path_dim_feedforward: int = 512,
        path_dropout: float = 0.1,
        # 步骤编码参数
        step_hidden_dim: int = 256,
        # 辅助步骤头开关（默认关闭，与路径级主监督设计一致）
        enable_step_head: bool = False,
        # 辅助合法性预测头开关
        enable_validity_head: bool = False,
    ):
        """
        Args:
            node_feature_dim: 不直接用于 VisNet (VisNet 固定 64 维输出), 保留兼容
            edge_feature_dim: 操作边特征维度
            hidden_dims: VisNet FC 层维度列表, 默认 [128, 256, 256]
            output_dim: 输出维度 (通常 1)
            path_nhead: Transformer 注意力头数
            path_num_layers: Transformer 层数
            path_dim_feedforward: Transformer FFN 中间维度
            path_dropout: Transformer dropout
            step_hidden_dim: 步骤编码器输出维度
            enable_step_head: 是否启用步骤级辅助输出头（默认关闭）
            enable_validity_head: 是否启用步骤合法性辅助预测头
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 256]
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.enable_step_head = enable_step_head
        self.enable_validity_head = enable_validity_head
        
        # ====== 1. 节点编码器 (共享权重) ======
        # VisNet 输出 = hidden_dims[-1] * 2 = 512
        self.node_encoder = VisNetMoleculeFeatureExtractor(node_feature_dim, hidden_dims)
        self.node_feat_dim = hidden_dims[-1] * 2  # 512
        
        # 非法节点可学习占位 embedding
        self.invalid_node_embedding = nn.Parameter(
            torch.randn(self.node_feat_dim) * 0.01
        )
        
        # ====== 2. 边特征编码器 ======
        self.edge_encoder = LinearEdgeFeatureExtractor(edge_feature_dim, hidden_dims[-1])
        self.edge_feat_dim = hidden_dims[-1]  # 256
        
        # ====== 3. 步骤编码器 ======
        # 每步输入: [from_feat(512), to_feat(512), diff(512), edge_feat(256)] = 1792
        step_input_dim = self.node_feat_dim * 3 + self.edge_feat_dim
        self.step_encoder = nn.Sequential(
            nn.Linear(step_input_dim, step_hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(step_hidden_dim * 2),
            nn.Linear(step_hidden_dim * 2, step_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(step_hidden_dim),
        )
        self.step_dim = step_hidden_dim
        
        # ====== 4. 路径 Transformer 编码器 ======
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=step_hidden_dim,
            nhead=path_nhead,
            dim_feedforward=path_dim_feedforward,
            dropout=path_dropout,
            batch_first=True,  # (B, S, D) 格式
            activation='gelu',
        )
        self.path_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=path_num_layers,
        )
        
        # 可学习的步骤位置编码
        # 用 nn.Embedding，最大支持 128 步
        self.max_steps = 128
        self.step_position_encoding = nn.Embedding(self.max_steps, step_hidden_dim)
        
        # ====== 5. 路径级输出头 ======
        # 输入: start_feat(512) + end_feat(512) + path_pool(step_hidden_dim) = 1280
        path_head_input_dim = self.node_feat_dim * 2 + step_hidden_dim
        self.path_head = nn.Sequential(
            nn.Linear(path_head_input_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(path_dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )
        
        # ====== 6. 步骤级辅助输出头 (可选，默认关闭) ======
        if enable_step_head:
            self.step_head = nn.Sequential(
                nn.Linear(step_hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
            )
        else:
            self.step_head = None
        
        # ====== 7. 步骤合法性辅助预测头 (可选) ======
        if enable_validity_head:
            self.validity_head = nn.Sequential(
                nn.Linear(step_hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        else:
            self.validity_head = None
    
    def forward(self, batch_dict: dict) -> dict:
        """
        前向传播。
        
        Args:
            batch_dict: path_collate 返回的 dict，包含：
                - node_batch_list: List[Batch]
                - node_valid_mask: (B, max_num_nodes) bool
                - edge_features: (B, max_steps, edge_dim)
                - step_valid_mask: (B, max_steps) bool
                - path_padding_mask: (B, max_steps) bool
                
        Returns:
            dict:
                - path_pred: (B, output_dim) 路径总变化预测
                - step_pred: (B, max_steps, output_dim) 步骤变化预测, 或 None
                - validity_pred: (B, max_steps, 1) 步骤合法性 logit, 或 None
        """
        node_batch_list = batch_dict['node_batch_list']
        node_valid_mask = batch_dict['node_valid_mask']
        edge_features = batch_dict['edge_features']
        path_padding_mask = batch_dict['path_padding_mask']
        
        device = edge_features.device
        batch_size = batch_dict['batch_size']
        max_num_nodes = len(node_batch_list)
        max_steps = max_num_nodes - 1
        
        # ====== Step 1: 编码所有节点 ======
        # node_features: (B, max_num_nodes, node_feat_dim)
        node_features = self._encode_nodes(node_batch_list, node_valid_mask, batch_size, device)
        
        # ====== Step 2: 编码步骤边特征 ======
        # edge_features: (B, max_steps, edge_dim) → (B, max_steps, edge_feat_dim)
        B, S, E = edge_features.shape
        edge_flat = edge_features.reshape(B * S, E)
        edge_encoded = self.edge_encoder(edge_flat)  # (B*S, edge_feat_dim)
        edge_encoded = edge_encoded.reshape(B, S, self.edge_feat_dim)  # (B, S, edge_feat_dim)
        
        # ====== Step 3: 构建 step tokens ======
        from_feats = node_features[:, :max_steps, :]   # (B, S, 512)
        to_feats = node_features[:, 1:, :]              # (B, S, 512)
        diff_feats = to_feats - from_feats              # (B, S, 512)
        
        step_input = torch.cat([from_feats, to_feats, diff_feats, edge_encoded], dim=-1)
        # (B, S, 1792)
        
        step_tokens = self.step_encoder(step_input)  # (B, S, step_dim)
        
        # 加位置编码
        positions = torch.arange(max_steps, device=device)
        positions = positions.clamp(max=self.max_steps - 1)
        pos_enc = self.step_position_encoding(positions)  # (S, step_dim)
        step_tokens = step_tokens + pos_enc.unsqueeze(0)  # broadcast to (B, S, step_dim)
        
        # ====== Step 4: 路径 Transformer 编码 ======
        # Transformer 的 src_key_padding_mask: True 表示需要被忽略的位置
        # 我们的 path_padding_mask: True 表示有效位置
        # 所以取反
        transformer_padding_mask = ~path_padding_mask  # (B, S)
        
        path_encoded = self.path_transformer(
            step_tokens,
            src_key_padding_mask=transformer_padding_mask,
        )  # (B, S, step_dim)
        
        # ====== Step 5: 路径级预测 ======
        # 全局池化 (仅有效位置)
        mask_for_pool = path_padding_mask.unsqueeze(-1).float()  # (B, S, 1)
        path_sum = (path_encoded * mask_for_pool).sum(dim=1)     # (B, step_dim)
        path_count = mask_for_pool.sum(dim=1).clamp(min=1.0)     # (B, 1)
        path_pool = path_sum / path_count                         # (B, step_dim)
        
        # 首尾节点特征
        start_feat = node_features[:, 0, :]   # (B, 512)
        end_feat_indices = [min(ns, max_num_nodes - 1) for ns in batch_dict['num_steps']]
        end_feat = torch.stack([
            node_features[b, end_feat_indices[b], :]
            for b in range(batch_size)
        ])  # (B, 512)
        
        path_head_input = torch.cat([start_feat, end_feat, path_pool], dim=-1)
        path_pred = self.path_head(path_head_input)  # (B, output_dim)
        
        # ====== Step 6: 步骤级预测 (可选) ======
        step_pred = None
        if self.enable_step_head and self.step_head is not None:
            step_pred = self.step_head(path_encoded)  # (B, S, output_dim)
        
        # ====== Step 7: 步骤合法性预测 (可选) ======
        validity_pred = None
        if self.enable_validity_head and self.validity_head is not None:
            validity_pred = self.validity_head(path_encoded)  # (B, S, 1)
        
        return {
            'path_pred': path_pred,
            'step_pred': step_pred,
            'validity_pred': validity_pred,
        }
    
    def _encode_nodes(
        self,
        node_batch_list: list,
        node_valid_mask: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        编码所有节点位置的分子特征。
        
        Args:
            node_batch_list: List[Batch], 长度 = max_num_nodes
            node_valid_mask: (B, max_num_nodes) bool
            batch_size: B
            device: 目标设备
            
        Returns:
            node_features: (B, max_num_nodes, node_feat_dim)
        """
        max_num_nodes = len(node_batch_list)
        node_features = torch.zeros(batch_size, max_num_nodes, self.node_feat_dim, device=device)
        
        invalid_emb = self.invalid_node_embedding.to(device)
        
        for pos_i in range(max_num_nodes):
            batch_graph = node_batch_list[pos_i].to(device)
            valid_at_pos = node_valid_mask[:, pos_i]  # (B,)
            
            if valid_at_pos.any():
                # VisNet 编码整个 batch（包含占位图）
                # 输出 shape: (B, node_feat_dim)
                all_feats = self.node_encoder(batch_graph)
                
                # 对有效位置使用 VisNet 特征，对无效位置使用 invalid embedding
                for b_i in range(batch_size):
                    if valid_at_pos[b_i]:
                        node_features[b_i, pos_i] = all_feats[b_i]
                    else:
                        node_features[b_i, pos_i] = invalid_emb
            else:
                # 该位置全部无效
                node_features[:, pos_i] = invalid_emb
        
        return node_features
    
    def get_model_config(self) -> dict:
        """返回模型配置用于保存。"""
        return {
            'model_version': 'v0.3',
            'node_feature_dim': self.node_feature_dim,
            'edge_feature_dim': self.edge_feature_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'node_feat_dim': self.node_feat_dim,
            'edge_feat_dim': self.edge_feat_dim,
            'step_dim': self.step_dim,
            'enable_step_head': self.enable_step_head,
            'enable_validity_head': self.enable_validity_head,
        }
