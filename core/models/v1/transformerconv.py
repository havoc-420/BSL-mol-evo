#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TransformerConv 版分子进化属性变化预测器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data

from .base import BaseGNNModel


class MoleculeEvolutionTransformerPredictor(BaseGNNModel):
    """
    使用 TransformerConv，边特征作为 attention bias。
    适合捕获跨多跳的进化关系。
    """

    def __init__(self,
                 node_feature_dim: int = 2048,
                 edge_feature_dim: int = 11,
                 hidden_dim: int = 128,
                 output_dim: int = 15,
                 heads: int = 4,
                 num_layers: int = 3):
        super().__init__(node_feature_dim,
                         edge_feature_dim,
                         hidden_dim,
                         output_dim)

        # 节点嵌入
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)

        # 边特征编码（作为 bias）
        self.edge_feature_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU()
        )

        # 多层 TransformerConv
        self.trans_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = TransformerConv(in_channels=hidden_dim,
                                   out_channels=hidden_dim // heads,
                                   heads=heads,
                                   edge_dim=hidden_dim,   # 边特征维度
                                   dropout=0.1,
                                   concat=True)
            self.trans_layers.append(conv)

        # 预测头
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
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 节点嵌入
        x = F.relu(self.node_embedding(x))

        # 编码边特征 → 作为 attention bias
        encoded_edge = self.edge_feature_encoder(edge_attr)   # [E, hidden_dim]

        # 多层 TransformerConv
        for i, conv in enumerate(self.trans_layers):
            x = conv(x, edge_index, edge_attr=encoded_edge)
            if i < self.num_layers - 1:
                x = F.relu(x)

        # 取源/目标节点特征
        src = x[edge_index[0]]
        dst = x[edge_index[1]]

        # 拼接并预测
        edge_feat = torch.cat([src, dst, encoded_edge], dim=1)
        out = self.property_predictor(edge_feat)
        return out
