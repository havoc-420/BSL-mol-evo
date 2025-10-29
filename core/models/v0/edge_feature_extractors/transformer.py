#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Transformer的边特征编码器

使用Transformer架构对边特征（操作信息）进行编码。
"""

import torch
import torch.nn as nn


class TransformerEdgeFeatureExtractor(nn.Module):
    """
    Transformer边特征编码器
    
    使用Transformer架构对边特征（操作信息）进行编码，替代原有的线性层。
    """
    
    def __init__(self, input_dim: int = 11, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 2, dim_feedforward: int = 512, dropout: float = 0.1):
        """
        初始化Transformer编码器

        Args:
            input_dim: 输入特征维度
            d_model: Transformer模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout概率
        """
        super(TransformerEdgeFeatureExtractor, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 输入线性投影层，将输入维度映射到d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出投影层，将d_model映射到目标维度
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            edge_attr: 边特征 (batch_size, edge_feature_dim)

        Returns:
            编码后的边特征 (batch_size, d_model)
        """
        # 添加序列维度: (batch_size, 1, edge_feature_dim)
        x = edge_attr.unsqueeze(1)
        
        # 输入投影
        x = self.input_projection(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 全局平均池化，将序列维度合并
        x = x.mean(dim=1)    # TODO ...
        
        
        # 输出投影
        x = self.output_projection(x)
        
        return x