#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于VisNet的分子进化预测器 (Linear-Linear版本)

这是一个使用线性组件的VisNet模型：
- molecule: VisNet特征提取器
- edge: 线性边特征编码器
- fusion: 线性特征融合预测器
用于从分子图中提取特征表示并预测分子演化过程中的属性变化。
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .molecule_feature_extractors import VisNetMoleculeFeatureExtractor
from .edge_feature_extractors import LinearEdgeFeatureExtractor
from .fusion_predictors import MLPFusionPredictor
from . import register_model


@register_model(display_name="VisNet Linear-Linear 模型", save_dir_name="visnet_linear_linear")
class MoleculeEvolutionVisnetLinearPredictor(nn.Module):
    """
    基于VisNet的分子进化预测器 (Linear-Linear版本)
    
    使用线性组件组合：
    - molecule: VisNetMoleculeFeatureExtractor (VisNet特征提取器)
    - edge: LinearEdgeFeatureExtractor (线性边特征编码器)
    - fusion: MLPFusionPredictor (线性特征融合预测器)
    """
    
    def __init__(self, node_feature_dim: int = 11, edge_feature_dim: int = 15,
                 hidden_dims: list = [128, 256, 256], output_dim: int = 1):
        """
        初始化预测器

        Args:
            node_feature_dim: 节点特征维度
            edge_feature_dim: 边特征维度 (操作信息)
            hidden_dims: VisNet各隐藏层维度列表
            output_dim: 输出维度 (属性变化)
        """
        super(MoleculeEvolutionVisnetLinearPredictor, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # molecule组件: VisNet分子特征提取器
        self.molecule_extractor_from = VisNetMoleculeFeatureExtractor(node_feature_dim, hidden_dims)
        self.molecule_extractor_to = VisNetMoleculeFeatureExtractor(node_feature_dim, hidden_dims)
        
        # edge组件: 线性边特征编码器
        self.edge_encoder = LinearEdgeFeatureExtractor(edge_feature_dim, hidden_dims[-1])
        
        # fusion组件: 线性特征融合预测器
        self.fusion_predictor = MLPFusionPredictor(
            node_dim=hidden_dims[-1] * 2,  # VisNet输出是mean和max拼接的结果
            edge_dim=hidden_dims[-1],
            hidden_dims=[512, 256, 128],
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
        
        # edge组件: 编码边特征
        edge_features = self.edge_encoder(edge_attr)
        
        # fusion组件: 融合特征并预测属性变化
        property_changes = self.fusion_predictor(from_features, to_features, edge_features)
        
        return property_changes