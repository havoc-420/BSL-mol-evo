#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于EquiformerV1的分子进化预测器 (Linear-Linear版本)

这是一个使用线性组件的EquiformerV1模型：
- molecule: EquiformerV1特征提取器
- edge: 线性边特征编码器
- fusion: 线性特征融合预测器
用于从分子图中提取特征表示并预测分子演化过程中的属性变化。
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .molecule_feature_extractors.equiformer_v1 import EquiformerV1MoleculeFeatureExtractor
from .fusion_predictors.mlp import MLPFusionPredictor
from .edge_feature_extractors.linear import LinearEdgeFeatureExtractor
from . import register_model


@register_model(display_name="Equiformer Linear-Linear 模型", save_dir_name="equiformer_linear_linear")
class MoleculeEvolutionEquiformerLinearPredictor(nn.Module):
    """
    基于EquiformerV1的分子进化预测器 (Linear-Linear版本)
    
    使用线性组件组合：
    - molecule: EquiformerV1MoleculeFeatureExtractor (EquiformerV1特征提取器)
    - edge: LinearEdgeFeatureExtractor (线性边特征编码器)
    - fusion: MLPFusionPredictor (线性特征融合预测器)
    """
    
    def __init__(self, 
                 irreps_in: str = '5x0e',
                 radius: float = 5.0,
                 num_basis: int = 128,
                 hidden_dims: list = [128, 256, 256],
                 edge_feature_dim: int = 11,
                 output_dim: int = 1,
                 **kwargs):
        """
        初始化预测器

        Args:
            irreps_in: 输入特征的等变表示
            radius: 原子间相互作用的最大半径
            num_basis: 基函数数量
            hidden_dims: 隐藏层维度列表
            edge_feature_dim: 边特征维度 (操作信息)
            output_dim: 输出维度 (属性变化)
            **kwargs: 传递给EquiformerV1特征提取器的其他参数
        """
        super(MoleculeEvolutionEquiformerLinearPredictor, self).__init__()
        
        self.irreps_in = irreps_in
        self.radius = radius
        self.num_basis = num_basis
        self.hidden_dims = hidden_dims
        self.edge_feature_dim = edge_feature_dim
        self.output_dim = output_dim
        
        # molecule组件: EquiformerV1分子特征提取器
        self.molecule_extractor_from = EquiformerV1MoleculeFeatureExtractor(
            irreps_in=irreps_in,
            max_radius=radius,
            number_of_basis=num_basis,
        )
        
        self.molecule_extractor_to = EquiformerV1MoleculeFeatureExtractor(
            irreps_in=irreps_in,
            max_radius=radius,
            number_of_basis=num_basis,
        )
        
        # edge组件: 线性边特征编码器
        self.edge_encoder = LinearEdgeFeatureExtractor(edge_feature_dim, hidden_dims[-1])
        
        # fusion组件: 线性特征融合预测器
        self.fusion_predictor = MLPFusionPredictor(
            node_dim=hidden_dims[-1] * 2,  # 假设Equiformer输出是mean和max拼接的结果
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