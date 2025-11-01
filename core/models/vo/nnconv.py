#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于NNConv的分子进化属性变化预测器模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv
from torch_geometric.data import Data

from .base import BaseGNNModel


class MoleculeEvolutionNNConvPredictor(BaseGNNModel):
    """
    基于NNConv的分子进化属性变化预测器
    
    使用图神经网络处理分子进化关系，预测属性变化值
    """
    
    def __init__(self, node_feature_dim: int = 2048, edge_feature_dim: int = 11,
                 hidden_dim: int = 128, output_dim: int = 15, num_layers: int = 3):
        """
        初始化NNConv预测器

        Args:
            node_feature_dim: 节点特征维度（默认2048，对应Morgan指纹）
            edge_feature_dim: 边特征维度（11维，不含属性变化）
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（默认15，对应15个属性变化值）
            num_layers: GNN层数
        """
        super(MoleculeEvolutionNNConvPredictor, self).__init__(
            node_feature_dim, edge_feature_dim, hidden_dim, output_dim)
        
        # 节点嵌入层
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        
        # NNConv层
        self.nnconv_layers = nn.ModuleList()
        self.edge_networks = nn.ModuleList()
        
        # 构建多层NNConv网络
        for i in range(num_layers):
            # 边网络：将边特征映射到节点特征变换矩阵
            edge_net = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim)
            )
            self.edge_networks.append(edge_net)
            
            # NNConv层
            self.nnconv_layers.append(
                NNConv(hidden_dim, hidden_dim, edge_net, aggr='mean')
            )
        
        # 编码原始边特征
        self.edge_feature_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.output_dim = output_dim  # 显式保存输出维度
        
        # 属性变化预测头
        self.property_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim * 2),  # 拼接源节点、目标节点和边特征
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)  # 使用实例变量
        )
        
        self.num_layers = num_layers

    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: 包含节点特征、边索引和边特征的图数据
            
        Returns:
            属性变化预测值 [num_edges, output_dim]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 节点嵌入
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # 多层NNConv消息传递
        for i in range(self.num_layers):
            x = self.nnconv_layers[i](x, edge_index, edge_attr)
            if i < self.num_layers - 1:  # 最后一层不加激活函数
                x = F.relu(x)
        
        # 为每条边预测属性变化
        # 获取边的源节点和目标节点特征
        source_nodes = x[edge_index[0]]  # 源节点特征
        target_nodes = x[edge_index[1]]  # 目标节点特征
        
        # 编码原始边特征
        encoded_edge_attr = self.edge_feature_encoder(edge_attr)  # [num_edges, hidden_dim]
        
        # 拼接源节点、目标节点和边特征
        edge_features = torch.cat([source_nodes, target_nodes, encoded_edge_attr], dim=1)
        
        # 预测属性变化
        property_changes = self.property_predictor(edge_features)
        
        return property_changes