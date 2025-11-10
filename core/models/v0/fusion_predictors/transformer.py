#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Transformer的特征融合预测器
"""

import torch
import torch.nn as nn


class TransformerFusionPredictor(nn.Module):
    """
    使用Transformer的特征融合预测器
    """
    
    def __init__(self, node_dim: int = 512, edge_dim: int = 128,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 3,
                 output_dim: int = 15):
        """
        初始化融合预测器
        """
        super(TransformerFusionPredictor, self).__init__()
        
        # 总输入维度
        total_input_dim = node_dim * 2 + edge_dim
        
        # 渐进维度扩展投影
        self.input_projection = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, d_model)
        )
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        
        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 回归预测器
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, from_feat: torch.Tensor, to_feat: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        # 拼接特征
        x = torch.cat([from_feat, to_feat, edge_feat], dim=-1)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加序列维度
        x = x.unsqueeze(1)
        
        # Transformer编码
        x = self.transformer(x)
        
        # 去除序列维度，直接取唯一元素
        x = x.squeeze(1)
        
        # 回归预测
        output = self.regressor(x)
        
        return output