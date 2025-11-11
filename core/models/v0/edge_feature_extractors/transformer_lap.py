#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Transformer的边特征编码器

使用Transformer架构对边特征（操作信息）进行编码。
"""

import torch
import torch.nn as nn


class TransformerEdgeFeatureExtractorLap(nn.Module):
    """
    Transformer边特征编码器
    
    使用Transformer架构对边特征（操作信息）进行编码，替代原有的线性层。
    支持可选的位置编码输入，以增强模型对原子位置的感知能力。
    """
    
    def __init__(self, input_dim: int = 11, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 2, dim_feedforward: int = 512, dropout: float = 0.1,
                 pe_dim: int = 0):
        """
        初始化Transformer编码器

        Args:
            input_dim: 输入特征维度
            d_model: Transformer模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout概率
            pe_dim: 位置编码维度，如果为0则不使用位置编码
        """
        super(TransformerEdgeFeatureExtractorLap, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.pe_dim = pe_dim
        self.use_position_encoding = pe_dim > 0
        self.nhead = nhead
        self.num_layers = num_layers
        
        if self.use_position_encoding:
            # 基础特征投影
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, d_model)
            )
            
            # 位置编码投影
            self.pe_projection = nn.Linear(pe_dim, d_model)
            
            # Transformer编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model * 2,  # 两倍维度，因为要拼接基础特征和位置特征
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )
            
            # 输出投影
            self.output_projection = nn.Linear(d_model * 2, d_model)
        else:
            # 渐进维度扩展: 11 -> 64 -> 128 -> d_model
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, d_model)
            )
            
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
            
    def forward(self, edge_attr: torch.Tensor, position_encoding: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            edge_attr: 边特征 (batch_size, edge_feature_dim)
            position_encoding: 位置编码 (batch_size, pe_dim)，仅在pe_dim>0时使用

        Returns:
            编码后的边特征 (batch_size, d_model)
        """
        print("edge_attr:", edge_attr.shape)
        print("position_encoding:", position_encoding.shape if position_encoding is not None else None)
        # 添加序列维度: (batch_size, 1, edge_feature_dim)
        x = edge_attr.unsqueeze(1)
        
        if self.use_position_encoding and position_encoding is not None:
            # 投影基础特征
            base_features = self.input_projection(x)
            
            # 投影位置编码
            pe_features = self.pe_projection(position_encoding.unsqueeze(1))
            
            # 拼接特征
            combined_features = torch.cat([base_features, pe_features], dim=-1)
            
            # Transformer编码
            encoded_features = self.transformer_encoder(combined_features)
            
            # 输出投影
            output = self.output_projection(encoded_features)
            output = output.squeeze(1)
        else:
            # 输入投影
            x = self.input_projection(x)
                
            # Transformer编码
            x = self.transformer_encoder(x)
            
            # 使用squeeze去除序列维度
            x = x.squeeze(1)
        
        return x
