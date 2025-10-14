#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于MLP的特征融合预测器

使用多层感知机对起始分子、目标分子和边特征进行融合，
通过全连接层逐步降维，最后输出预测结果。
"""

import torch
import torch.nn as nn


class MLPFusionPredictor(nn.Module):
    """
    使用MLP的特征融合预测器
    
    将起始分子、目标分子和边特征拼接后通过多层感知机进行特征融合，
    最后输出属性变化预测值。
    """
    
    def __init__(self, node_dim: int = 512, edge_dim: int = 256,
                 hidden_dims: list = [512, 256, 128], output_dim: int = 15):
        """
        初始化融合预测器

        Args:
            node_dim: 节点特征维度 (来自GCN提取器的输出)
            edge_dim: 边特征维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
        """
        super(MLPFusionPredictor, self).__init__()
        
        # 总输入维度
        total_input_dim = node_dim * 2 + edge_dim
        
        # 构建MLP层
        layers = []
        input_dim = total_input_dim
        
        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        # 添加输出层
        layers.append(nn.Linear(input_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, from_feat: torch.Tensor, to_feat: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            from_feat: 起始分子特征 [B, node_dim]
            to_feat: 目标分子特征 [B, node_dim]
            edge_feat: 边特征 [B, edge_dim]

        Returns:
            属性变化预测值 [B, output_dim]
        """
        # 拼接所有特征
        x = torch.cat([from_feat, to_feat, edge_feat], dim=-1)
        
        # MLP前向传播
        output = self.mlp(x)
        
        return output