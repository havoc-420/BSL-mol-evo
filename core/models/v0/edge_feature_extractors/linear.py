#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础线性边特征编码器

这是一个通用的线性模型，用于对边特征（操作信息）进行编码。
"""

import torch
import torch.nn as nn
from typing import List


class LinearEdgeFeatureExtractor(nn.Module):
    """
    基础线性边特征编码器
    
    该模型通过线性层将边特征从原始维度映射到目标维度。
    """
    
    def __init__(self, edge_feature_dim: int = 11, hidden_dim: int = 128, hidden_dims: List[int] = None):
        """
        初始化边特征编码器

        Args:
            edge_feature_dim: 边特征维度
            hidden_dim: 输出隐藏层维度
            hidden_dims: 中间层维度列表，默认为[64, 128]
        """
        super(LinearEdgeFeatureExtractor, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128]
        
        # 构建网络层
        dims = [edge_feature_dim] + hidden_dims + [hidden_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # 除了最后一层，其他层都添加激活函数
                layers.append(nn.ReLU())
        
        # 边特征编码器
        # 从edge_feature_dim维演变到hidden_dim维
        self.edge_encoder = nn.Sequential(*layers)
    
    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        前向传播，编码边特征

        Args:
            edge_attr: 边特征 (batch_size, edge_feature_dim)

        Returns:
            编码后的边特征 (batch_size, hidden_dim)
        """
        return self.edge_encoder(edge_attr)