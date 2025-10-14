#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于GCN和Transformer的分子特征提取模型 v0

这是一个简化版的GCN模型，使用Transformer来处理边特征（操作信息），
用于从分子图中提取特征表示并预测分子演化过程中的属性变化。
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from molecule_feature_extractors.gcn import GCNMoleculeFeatureExtractor


class EdgeFeatureTransformer(nn.Module):
    """
    边特征Transformer编码器
    
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
        super(EdgeFeatureTransformer, self).__init__()
        
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
        x = x.mean(dim=1)
        
        # 输出投影
        x = self.output_projection(x)
        
        return x


class MoleculeEvolutionGCNTransformerPredictor(nn.Module):
    """
    基于GCN和Transformer的分子进化预测器 (v0)
    
    使用两个MoleculeFeatureExtractor分别提取起始分子和目标分子的特征，
    并结合Transformer编码器处理边特征（操作信息）来预测属性变化。
    """
    
    def __init__(self, node_feature_dim: int = 37, edge_feature_dim: int = 11,
                 hidden_dim: int = 128, output_dim: int = 15, num_layers: int = 2):
        """
        初始化预测器

        Args:
            node_feature_dim: 节点特征维度
            edge_feature_dim: 边特征维度 (操作信息)
            hidden_dim: 隐藏层维度
            output_dim: 输出维度 (属性变化)
            num_layers: GCN层数
        """
        super(MoleculeEvolutionGCNTransformerPredictor, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 起始分子和目标分子特征提取器
        # 注意：现在GCNMoleculeFeatureExtractor输出512维（256*2）
        self.from_molecule_extractor = GCNMoleculeFeatureExtractor(
            node_feature_dim)
        self.to_molecule_extractor = GCNMoleculeFeatureExtractor(
            node_feature_dim)
        
        # 边特征Transformer编码器
        # 使用Transformer替代原有的线性层序列
        self.edge_encoder = EdgeFeatureTransformer(
            input_dim=edge_feature_dim,
            d_model=hidden_dim,
            nhead=8 if hidden_dim % 8 == 0 else 4,  # 确保nhead能整除d_model
            num_layers=2,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1
        )
        
        # 特征融合和预测层
        # 输入: [from_mol_features, to_mol_features, edge_features]
        # 现在维度是: 512 + 512 + 128 = 1152
        fusion_input_dim = 512 * 2 + hidden_dim
        self.predictor = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, from_data: Data, to_data: Data, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            from_data: 起始分子图数据
            to_data: 目标分子图数据
            edge_attr: 边特征 (操作信息)

        Returns:
            属性变化预测值
        """
        # 提取起始分子特征
        from_features = self.from_molecule_extractor(from_data)
        
        # 提取目标分子特征
        to_features = self.to_molecule_extractor(to_data)
        
        # 使用Transformer编码边特征
        edge_features = self.edge_encoder(edge_attr)
        
        # 特征融合
        combined_features = torch.cat([from_features, to_features, edge_features], dim=1)
        
        # 预测属性变化
        property_changes = self.predictor(combined_features)
        
        return property_changes