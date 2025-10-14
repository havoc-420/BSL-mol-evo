#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础GCN分子特征提取器

这是一个通用的GCN模型，用于从分子图中提取特征表示。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data


class GCNMoleculeFeatureExtractor(nn.Module):
    """
    基础GCN分子特征提取器
    
    该模型将SMILES转换为图结构后，通过固定三层GCN层提取分子特征表示。
    """
    
    def __init__(self, node_feature_dim: int = 1, hidden_dims: list = None):
        """
        初始化特征提取器，固定三层GCN结构 [128, 256, 256]

        Args:
            node_feature_dim: 节点特征维度
        """
        super(GCNMoleculeFeatureExtractor, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        # 固定三层GCN结构
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 256, 256]
        
        # GCN网络层
        self.gcn_layers = nn.ModuleList()
        
        # 构建固定的三层GCN结构
        input_dim = node_feature_dim
        for hidden_dim in self.hidden_dims:
            self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播，提取分子特征

        Args:
            data: 包含节点特征和边索引的图数据

        Returns:
            分子特征表示 (512维: mean和max池化的合并结果)
        """
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
        
        # 通过GCN层
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, edge_index)
            if i < len(self.gcn_layers) - 1:  # 最后一层不加激活函数
                x = F.relu(x)
        
        # 全局池化获取图表示 - 分别进行mean和max池化
        if batch is not None:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
        else:
            x_mean = torch.mean(x, dim=0, keepdim=True)
            x_max = torch.max(x, dim=0, keepdim=True)[0]
        
        # 合并mean和max池化结果
        x = torch.cat([x_mean, x_max], dim=1)
        
        return x