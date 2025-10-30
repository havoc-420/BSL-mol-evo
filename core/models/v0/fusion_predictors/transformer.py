#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Transformer的特征融合预测器

使用Transformer架构对起始分子、目标分子和边特征进行融合，
通过自注意力机制实现全局特征交互，最后用CLS token做回归预测。
"""

import torch
import torch.nn as nn


class TransformerFusionPredictor(nn.Module):
    """
    使用Transformer的特征融合预测器
    
    将起始分子、目标分子和边特征拼接后通过Transformer进行特征交互，
    最后用CLS token做回归预测。
    """
    
    def __init__(self, node_dim: int = 512, edge_dim: int = 128,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 3,
                 output_dim: int = 15):
        """
        初始化融合预测器

        Args:
            node_dim: 节点特征维度 (来自GCN提取器的输出)
            edge_dim: 边特征维度
            d_model: Transformer模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            output_dim: 输出维度
        """
        super(TransformerFusionPredictor, self).__init__()
        
        # 总输入维度
        total_input_dim = node_dim * 2 + edge_dim
        
        # 将拼接后的特征投影到d_model维度
        self.token_proj = nn.Linear(total_input_dim, d_model)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            activation='gelu',
            batch_first=True
        )
        
        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # CLS token用于最终预测
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 回归预测器
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, from_feat: torch.Tensor, to_feat: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            from_feat: 起始分子特征 [B, 512]
            to_feat: 目标分子特征 [B, 512]
            edge_feat: 边特征 [B, 128]

        Returns:
            属性变化预测值 [B, output_dim]
        """
        # 确保所有特征都有相同的batch维度
        batch_size = from_feat.size(0)
        
        # 如果edge_feat是2D的，需要扩展为与from_feat和to_feat相同的维度
        if edge_feat.dim() == 2 and from_feat.dim() == 2 and to_feat.dim() == 2:
            # 拼接所有特征: [B, 512+512+128] = [B, 1152]
            x = torch.cat([from_feat, to_feat, edge_feat], dim=-1)
        elif edge_feat.dim() == 3 and from_feat.dim() == 3 and to_feat.dim() == 3:
            # 如果所有特征都是3D的，直接拼接
            x = torch.cat([from_feat, to_feat, edge_feat], dim=-1)
        else:
            # 处理维度不一致的情况
            # 确保所有特征都被展平到2D
            from_feat_flat = from_feat.view(batch_size, -1)
            to_feat_flat = to_feat.view(batch_size, -1)
            edge_feat_flat = edge_feat.view(batch_size, -1)
            x = torch.cat([from_feat_flat, to_feat_flat, edge_feat_flat], dim=-1)
        
        # 投影到d_model维度: [B, d_model]
        x = self.token_proj(x)
        
        # 添加序列维度: [B, 1, d_model]
        x = x.unsqueeze(1)
        
        # 扩展CLS token到批次大小: [B, 1, d_model]
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        
        # 拼接CLS token和输入特征: [B, 2, d_model]
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer编码: [B, 2, d_model]
        x = self.transformer(x)
        
        # 取出CLS token的输出: [B, d_model]
        cls_output = x[:, 0, :]
        
        # 回归预测: [B, output_dim]
        output = self.regressor(cls_output)
        
        return output