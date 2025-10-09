#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGCNConv 版分子进化属性变化预测器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data

from .base import BaseGNNModel


class MoleculeEvolutionRGCNPredictor(BaseGNNModel):
    """
    使用 RGCNConv（关系图卷积）处理分子进化边。
    edge_type 必须是 LongTensor，形状为 [num_edges]，取值范围 [0, num_relations-1]。
    """

    def __init__(self,
                 node_feature_dim: int = 2048,
                 edge_feature_dim: int = 11,   # 原始边特征维度（不含 relation id）
                 hidden_dim: int = 128,
                 output_dim: int = 15,
                 num_relations: int = 6,      # 例如 6 种 op 类型
                 num_layers: int = 3):
        super().__init__(node_feature_dim,
                         edge_feature_dim,
                         hidden_dim,
                         output_dim)

        # ---------- 节点嵌入 ----------
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)

        # ---------- 边特征编码 ----------
        # 将原始数值特征映射到 hidden_dim，随后与 relation id 一起使用
        self.edge_feature_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU()
        )

        # ---------- RGCN 层 ----------
        self.rgcn_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = RGCNConv(in_channels=hidden_dim,
                           out_channels=hidden_dim,
                           num_relations=num_relations,
                           aggr='mean')
            self.rgcn_layers.append(conv)

        # ---------- 预测头 ----------
        self.property_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.num_layers = num_layers

    def forward(self, data: Data) -> torch.Tensor:
        """
        data 必须包含：
            x          : [num_nodes, node_feature_dim]
            edge_index : [2, num_edges]
            edge_attr  : [num_edges, edge_feature_dim]
            edge_type  : [num_edges] (LongTensor)   ← 关系 id
        """
        x, edge_index, edge_attr, edge_type = (
            data.x, data.edge_index, data.edge_attr, data.edge_type)

        # 节点嵌入
        x = F.relu(self.node_embedding(x))

        # 编码边特征（用于后续拼接）
        encoded_edge = self.edge_feature_encoder(edge_attr)   # [E, hidden_dim]

        # 多层 RGCN 消息传递
        for i in range(self.num_layers):
            x = self.rgcn_layers[i](x, edge_index, edge_type)
            if i < self.num_layers - 1:
                x = F.relu(x)

        # 取源/目标节点特征
        src = x[edge_index[0]]
        dst = x[edge_index[1]]

        # 拼接 (src, dst, encoded_edge) → 预测 Δattr
        edge_feat = torch.cat([src, dst, encoded_edge], dim=1)
        out = self.property_predictor(edge_feat)   # [E, output_dim]
        return out
