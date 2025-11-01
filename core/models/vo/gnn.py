#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子进化图神经网络模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, GATConv, global_mean_pool, BatchNorm
from torch_geometric.data import Data
from typing import List

from .base import BaseGNNModel


class MoleculeGNN(BaseGNNModel):
    """
    基于分子进化的图神经网络
    
    该网络使用NNConv层，将分子作为节点，进化关系作为边，
    边特征包含位置、原子类型、操作类型和属性变化信息。
    """
    
    def __init__(self, node_feature_dim: int = 15, edge_feature_dim: int = 30,
                 hidden_dim: int = 128, output_dim: int = 64, num_layers: int = 3):
        """
        初始化分子GNN

        Args:
            node_feature_dim: 节点特征维度（默认15，对应QM9的15个量子化学属性）
            edge_feature_dim: 边特征维度（30 = 5个原子类型 + 6个操作类型 + 15个属性变化 + 3个位置特征 + 1个相似性特征）
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: GNN层数
        """
        super(MoleculeGNN, self).__init__(node_feature_dim, edge_feature_dim, hidden_dim, output_dim)
        
        # 节点嵌入层
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        
        # 边网络（用于NNConv）
        self.edge_networks = nn.ModuleList()
        self.nnconv_layers = nn.ModuleList()
        
        # 第一层
        edge_net = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        self.edge_networks.append(edge_net)
        self.nnconv_layers.append(NNConv(hidden_dim, hidden_dim, edge_net, aggr='mean'))
        
        # 中间层
        for _ in range(num_layers - 2):
            edge_net = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim)
            )
            self.edge_networks.append(edge_net)
            self.nnconv_layers.append(NNConv(hidden_dim, hidden_dim, edge_net, aggr='mean'))
        
        # 最后一层
        if num_layers > 1:
            edge_net = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * output_dim)
            )
            self.edge_networks.append(edge_net)
            self.nnconv_layers.append(NNConv(hidden_dim, output_dim, edge_net, aggr='mean'))
        
        # 输出层
        self.num_layers = num_layers

    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: 包含节点特征、边索引和边特征的图数据
            
        Returns:
            节点表示张量
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 节点嵌入
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # 多层NNConv
        for i in range(self.num_layers):
            x = self.nnconv_layers[i](x, edge_index, edge_attr)
            if i < self.num_layers - 1:  # 最后一层不加激活函数
                x = F.relu(x)
        
        # 全局池化
        if hasattr(data, 'batch'):
            x = global_mean_pool(x, data.batch)
        
        return x


class EnhancedMoleculeGNN(BaseGNNModel):
    """
    增强的分子进化图神经网络

    结合NNConv和GAT，添加残差连接和归一化层
    """

    def __init__(self, node_feature_dim: int = 15, edge_feature_dim: int = 30,
                 hidden_dim: int = 128, output_dim: int = 64, num_layers: int = 3,
                 heads: int = 4, dropout: float = 0.1):
        """
        初始化增强GNN

        Args:
            node_feature_dim: 节点特征维度
            edge_feature_dim: 边特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: GNN层数
            heads: 注意力头数
            dropout: Dropout率
        """
        super(EnhancedMoleculeGNN, self).__init__(node_feature_dim, edge_feature_dim, hidden_dim, output_dim)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # 节点嵌入层
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)

        # NNConv层（处理边特征）
        self.nnconv_layers = nn.ModuleList()
        self.nnconv_bns = nn.ModuleList()

        # GAT层（处理节点注意力）
        self.gat_layers = nn.ModuleList()
        self.gat_bns = nn.ModuleList()

        # 边网络
        self.edge_networks = nn.ModuleList()

        # 构建多层网络
        for i in range(num_layers):
            # 边网络
            edge_net = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim)
            )
            self.edge_networks.append(edge_net)

            # NNConv层
            self.nnconv_layers.append(NNConv(hidden_dim, hidden_dim, edge_net, aggr='mean'))
            self.nnconv_bns.append(BatchNorm(hidden_dim))

            # GAT层
            if i == num_layers - 1:
                # 最后一层输出到目标维度
                self.gat_layers.append(GATConv(hidden_dim, output_dim, heads=1, concat=False, dropout=dropout))
            else:
                self.gat_layers.append(GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout))
                self.gat_bns.append(BatchNorm(hidden_dim))

        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # 残差连接
        self.residual_connections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])

    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播

        Args:
            data: 图数据

        Returns:
            节点表示张量
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 节点嵌入
        x = self.node_embedding(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 多层消息传递
        for i in range(self.num_layers):
            # 保存残差连接
            residual = x

            # NNConv处理边特征
            x = self.nnconv_layers[i](x, edge_index, edge_attr)
            x = self.nnconv_bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # GAT处理节点注意力
            x = self.gat_layers[i](x, edge_index)

            if i < self.num_layers - 1:
                x = self.gat_bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            # 残差连接
            if i > 0 and residual.shape == x.shape:
                x = x + self.residual_connections[i-1](residual)

        # 全局池化
        if hasattr(data, 'batch'):
            x = global_mean_pool(x, data.batch)

        return x


class MoleculeGNNWithFingerprint(BaseGNNModel):
    """
    基于分子指纹的图神经网络
    
    使用Morgan指纹作为节点特征的GNN实现
    """
    
    def __init__(self, node_feature_dim: int = 2048, edge_feature_dim: int = 30,
                 hidden_dim: int = 128, output_dim: int = 64, num_layers: int = 3):
        """
        初始化分子GNN

        Args:
            node_feature_dim: 节点特征维度（默认2048，对应Morgan指纹）
            edge_feature_dim: 边特征维度（30维）
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: GNN层数
        """
        super(MoleculeGNNWithFingerprint, self).__init__(node_feature_dim, edge_feature_dim, hidden_dim, output_dim)
        
        # 节点嵌入层
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        
        # 边网络（用于NNConv）
        self.edge_networks = nn.ModuleList()
        self.nnconv_layers = nn.ModuleList()
        
        # 第一层
        edge_net = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        self.edge_networks.append(edge_net)
        self.nnconv_layers.append(NNConv(hidden_dim, hidden_dim, edge_net, aggr='mean'))
        
        # 中间层
        for _ in range(num_layers - 2):
            edge_net = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim)
            )
            self.edge_networks.append(edge_net)
            self.nnconv_layers.append(NNConv(hidden_dim, hidden_dim, edge_net, aggr='mean'))
        
        # 最后一层
        if num_layers > 1:
            edge_net = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * output_dim)
            )
            self.edge_networks.append(edge_net)
            self.nnconv_layers.append(NNConv(hidden_dim, output_dim, edge_net, aggr='mean'))
        
        # 输出层
        self.num_layers = num_layers

    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: 包含节点特征、边索引和边特征的图数据
            
        Returns:
            节点表示张量
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 节点嵌入
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # 多层NNConv
        for i in range(self.num_layers):
            x = self.nnconv_layers[i](x, edge_index, edge_attr)
            if i < self.num_layers - 1:  # 最后一层不加激活函数
                x = F.relu(x)
        
        # 全局池化
        if hasattr(data, 'batch'):
            x = global_mean_pool(x, data.batch)
        
        return x