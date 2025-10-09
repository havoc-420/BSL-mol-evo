#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GATv2Conv 版分子进化属性变化预测器（边特征参与注意力）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

from .base import BaseGNNModel


class MoleculeEvolutionGATv2Predictor(BaseGNNModel):
    """
    使用 GATv2Conv，边特征 (edge_attr) 直接参与注意力计算。
    适合属性变化、atom、op 等连续或离散特征的联合建模。
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

        # 边特征编码（保持与节点维度相同，供注意力使用）
        self.edge_feature_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU()
        )

        # 多层 GATv2（每层的输出维度 = hidden_dim // heads）
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            conv = GATv2Conv(in_channels=hidden_dim,
                             out_channels=hidden_dim // heads,
                             heads=heads,
                             edge_dim=hidden_dim,   # 告诉层 edge_attr 的维度
                             dropout=0.1,
                             concat=True)           # 拼接多头特征
            self.gat_layers.append(conv)

        # 预测头（同前）
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

        # 编码边特征（供注意力使用 & 预测头拼接）
        encoded_edge = self.edge_feature_encoder(edge_attr)   # [E, hidden_dim]

        # 多层 GATv2 消息传递
        for i, conv in enumerate(self.gat_layers):
            x = conv(x, edge_index, edge_attr=encoded_edge)
            if i < self.num_layers - 1:
                x = F.elu(x)   # GATv2 常用激活

        # 取源/目标节点特征
        src = x[edge_index[0]]
        dst = x[edge_index[1]]

        # 拼接并预测
        edge_feat = torch.cat([src, dst, encoded_edge], dim=1)
        out = self.property_predictor(edge_feat)
        return out
