#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子进化预测器模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import List

from .base import BaseModel
from .gnn import MoleculeGNN, EnhancedMoleculeGNN, MoleculeGNNWithFingerprint


class MoleculeEvolutionPredictor(BaseModel):
    """
    分子进化属性变化预测器
    
    基于源分子和边特征预测目标分子的属性变化
    """
    
    def __init__(self, node_feature_dim: int = 15, edge_feature_dim: int = 30,
                 hidden_dim: int = 128, property_dim: int = 15):
        """
        初始化预测器

        Args:
            node_feature_dim: 节点特征维度（15个QM9量子化学属性）
            edge_feature_dim: 边特征维度（30维）
            hidden_dim: 隐藏层维度
            property_dim: 属性变化维度（15个量子化学属性）
        """
        super(MoleculeEvolutionPredictor, self).__init__()
        
        self.gnn = MoleculeGNN(node_feature_dim, edge_feature_dim, hidden_dim, hidden_dim)
        
        # 属性变化预测头
        self.property_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, property_dim)
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: 图数据
            
        Returns:
            属性变化预测值
        """
        # 获取图表示
        graph_embedding = self.gnn(data)
        
        # 预测属性变化
        property_changes = self.property_predictor(graph_embedding)
        
        return property_changes


class EnhancedMoleculeEvolutionPredictor(BaseModel):
    """
    增强的分子进化属性变化预测器

    使用增强GNN模型进行属性变化预测
    """

    def __init__(self, node_feature_dim: int = 15, edge_feature_dim: int = 30,
                 hidden_dim: int = 128, property_dim: int = 15, num_layers: int = 3,
                 heads: int = 4, dropout: float = 0.1):
        """
        初始化增强预测器

        Args:
            node_feature_dim: 节点特征维度
            edge_feature_dim: 边特征维度
            hidden_dim: 隐藏层维度
            property_dim: 属性变化维度
            num_layers: GNN层数
            heads: 注意力头数
            dropout: Dropout率
        """
        super(EnhancedMoleculeEvolutionPredictor, self).__init__()

        self.gnn = EnhancedMoleculeGNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout
        )

        # 属性变化预测头
        self.property_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, property_dim)
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播

        Args:
            data: 图数据

        Returns:
            属性变化预测值
        """
        # 获取图表示
        graph_embedding = self.gnn(data)

        # 预测属性变化
        property_changes = self.property_predictor(graph_embedding)

        return property_changes


class MoleculeEvolutionPredictorWithFingerprint(BaseModel):
    """
    基于分子指纹的分子进化属性变化预测器
    """
    
    def __init__(self, node_feature_dim: int = 2048, edge_feature_dim: int = 30,
                 hidden_dim: int = 128, property_dim: int = 15):
        """
        初始化预测器

        Args:
            node_feature_dim: 节点特征维度（2048维Morgan指纹）
            edge_feature_dim: 边特征维度（30维）
            hidden_dim: 隐藏层维度
            property_dim: 属性变化维度（15个量子化学属性）
        """
        super(MoleculeEvolutionPredictorWithFingerprint, self).__init__()
        
        self.gnn = MoleculeGNNWithFingerprint(node_feature_dim, edge_feature_dim, hidden_dim, hidden_dim)
        
        # 属性变化预测头
        self.property_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, property_dim)
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: 图数据
            
        Returns:
            属性变化预测值
        """
        # 获取图表示
        graph_embedding = self.gnn(data)
        
        # 预测属性变化
        property_changes = self.property_predictor(graph_embedding)
        
        return property_changes


class MoleculeEvolutionTransformer(BaseModel):
    """
    分子进化转换器
    
    根据起始分子特征、原子和操作类型预测目标分子特征
    """
    
    def __init__(self, node_feature_dim: int = 2048, edge_feature_dim: int = 30,
                 hidden_dim: int = 128, output_dim: int = 2048):
        """
        初始化转换器

        Args:
            node_feature_dim: 节点特征维度（默认2048，对应Morgan指纹）
            edge_feature_dim: 边特征维度（30维）
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（默认2048，对应Morgan指纹）
        """
        super(MoleculeEvolutionTransformer, self).__init__()
        
        # 起始分子特征编码器
        self.source_molecule_encoder = nn.Linear(node_feature_dim, hidden_dim)
        
        # 边特征编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 转换网络
        self.transformer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 拼接起始分子和边特征
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, source_features: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            source_features: 起始分子特征 [batch_size, node_feature_dim]
            edge_features: 边特征 [batch_size, edge_feature_dim]
            
        Returns:
            目标分子特征预测值 [batch_size, output_dim]
        """
        # 编码起始分子特征
        source_encoded = self.source_molecule_encoder(source_features)
        
        # 编码边特征
        edge_encoded = self.edge_encoder(edge_features)
        
        # 拼接特征
        combined_features = torch.cat([source_encoded, edge_encoded], dim=1)
        
        # 转换
        target_features = self.transformer(combined_features)
        
        return target_features