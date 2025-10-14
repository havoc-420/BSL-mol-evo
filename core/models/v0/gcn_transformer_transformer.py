#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于GCN和Transformer的分子进化预测器 (GCN-GCN-Transformer版本)

这是一个使用混合组件的GCN模型：
- molecule: GCN特征提取器
- edge: Transformer边特征编码器
- fusion: Transformer特征融合预测器
用于从分子图中提取特征表示并预测分子演化过程中的属性变化。
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .molecule_feature_extractors.gcn import GCNMoleculeFeatureExtractor
from .edge_feature_extractors import TransformerEdgeFeatureExtractor
from .fusion_predictors import TransformerFusionPredictor


class MoleculeEvolutionGCNTransformerPredictor(nn.Module):
    """
    基于GCN和Transformer的分子进化预测器 (GCN-GCN-Transformer版本)
    
    使用混合组件组合：
    - molecule: GCNMoleculeFeatureExtractor (GCN特征提取器)
    - edge: TransformerEdgeFeatureExtractor (Transformer边特征编码器)
    - fusion: TransformerFusionPredictor (Transformer特征融合预测器)
    """
    
    def __init__(self, node_feature_dim: int = 11, edge_feature_dim: int = 11,
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
        
        # molecule组件: GCN分子特征提取器
        self.molecule_extractor_from = GCNMoleculeFeatureExtractor(node_feature_dim)
        self.molecule_extractor_to = GCNMoleculeFeatureExtractor(node_feature_dim)
        
        # edge组件: Transformer边特征编码器
        self.edge_encoder = TransformerEdgeFeatureExtractor(
            input_dim=edge_feature_dim,
            d_model=hidden_dim,
            nhead=8 if hidden_dim % 8 == 0 else 4,  # 确保nhead能整除d_model  # INFO 不过感觉这样的 nhead 可能没什么用
            num_layers=2,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1
        )
        
        # fusion组件: Transformer特征融合预测器
        self.fusion_predictor = TransformerFusionPredictor(
            node_dim=512,  # GCN提取器输出维度为512 (256*2, mean和max池化的合并结果)
            edge_dim=hidden_dim,
            d_model=256,
            nhead=8,
            num_layers=3,
            output_dim=output_dim
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
        # molecule组件: 提取起始和目标分子特征
        from_features = self.molecule_extractor_from(from_data)
        to_features = self.molecule_extractor_to(to_data)
        
        # edge组件: 使用Transformer编码边特征
        edge_features = self.edge_encoder(edge_attr)
        
        # fusion组件: 使用Transformer融合特征并预测属性变化
        property_changes = self.fusion_predictor(from_features, to_features, edge_features)
        
        return property_changes