#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础GCN分子特征提取器

这是一个通用的GCN特征提取器，可以从分子图中提取特征表示。
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .molecule_feature_extractors import GCNMoleculeFeatureExtractor


class MoleculeEvolutionGCNPredictor(nn.Module):
    """
    基于GCN的分子进化预测器 (v0)
    
    使用两个BaseGCNMoleculeFeatureExtractor分别提取起始分子和目标分子的特征，
    并结合边特征预测属性变化。
    """
    
    def __init__(self, node_feature_dim: int = 37, edge_feature_dim: int = 15,
                 hidden_dims: list = [128, 256, 256], output_dim: int = 15):
        """
        初始化预测器

        Args:
            node_feature_dim: 节点特征维度
            edge_feature_dim: 边特征维度 (操作信息)
            hidden_dims: GCN各隐藏层维度列表
            output_dim: 输出维度 (属性变化)
        """
        super(MoleculeEvolutionGCNPredictor, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # 起始分子和目标分子特征提取器
        self.from_molecule_extractor = GCNMoleculeFeatureExtractor(node_feature_dim, hidden_dims)
        self.to_molecule_extractor = GCNMoleculeFeatureExtractor(node_feature_dim, hidden_dims)
        
        # 边特征编码器
        # 从edge_feature_dim维演变到hidden_dims[-1]维
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dims[-1])
        )
        
        # 特征融合和预测层
        # 输入: [from_mol_features, to_mol_features, edge_features]
        # 现在维度是: 2*hidden_dims[-1] + 2*hidden_dims[-1] + hidden_dims[-1] = 5*hidden_dims[-1]
        fusion_input_dim = 2 * hidden_dims[-1] * 2 + hidden_dims[-1]
        self.predictor = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
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
        # import pdb; pdb.set_trace()

        # 提取起始分子特征
        from_features = self.from_molecule_extractor(from_data)
        
        # 提取目标分子特征
        to_features = self.to_molecule_extractor(to_data)
        
        # 编码边特征
        edge_features = self.edge_encoder(edge_attr)
        
        # 特征融合
        combined_features = torch.cat([from_features, to_features, edge_features], dim=1)
        
        # 预测属性变化
        property_changes = self.predictor(combined_features)
        
        return property_changes