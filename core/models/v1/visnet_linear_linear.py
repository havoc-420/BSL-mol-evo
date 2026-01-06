#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于VisNet的分子进化预测器 (Linear-Linear版本) - 序列迭代式架构

这是一个使用线性组件的VisNet模型的序列迭代式版本：
- molecule: VisNet特征提取器
- edge: 线性边特征编码器
- fusion: 线性特征融合预测器

关键特性：
- 支持序列式的分子演化预测
- 输入：from_data_list [s1, s2, s3, ...], to_data_list [s2, s3, s4, ...], edge_attrs [edge1, edge2, edge3, ...]
- 逐步迭代：s1->s2使用edge1, s2->s3使用edge2, s3->s4使用edge3, ...
- 每个步骤独立预测，输出完整的属性变化序列
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from ..v0.molecule_feature_extractors import VisNetMoleculeFeatureExtractor
from ..v0.edge_feature_extractors import LinearEdgeFeatureExtractor
from ..v0.fusion_predictors import MLPFusionPredictor
from . import register_model


@register_model(display_name="VisNet Linear-Linear 序列迭代式模型", save_dir_name="visnet_linear_linear_iterative")
class MoleculeEvolutionVisnetLinearIterativePredictor(nn.Module):
    """
    基于VisNet的分子进化预测器 (Linear-Linear序列迭代式版本)
    
    使用线性组件组合，支持序列式的分子演化预测：
    - molecule: VisNetMoleculeFeatureExtractor (VisNet特征提取器)
    - edge: LinearEdgeFeatureExtractor (线性边特征编码器)
    - fusion: MLPFusionPredictor (线性特征融合预测器)
    
    序列迭代机制：
    - 输入：from_data_list [s1, s2, s3, ...], to_data_list [s2, s3, s4, ...], edge_attrs [edge1, edge2, edge3, ...]
    - 逐步迭代：s1->s2使用edge1, s2->s3使用edge2, s3->s4使用edge3, ...
    - 每个步骤独立预测，输出完整的属性变化序列
    - 适用于多步分子演化场景
    
    使用示例：
        from_data_list = [s1, s2, s3]  # 起始分子序列
        to_data_list = [s2, s3, s4]    # 目标分子序列
        edge_attrs = torch.tensor([[...], [...], [...]])  # 3个操作的特征
        predictions = model(from_data_list, to_data_list, edge_attrs)
        # predictions.shape: [3, output_dim]
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
        super(MoleculeEvolutionVisnetLinearIterativePredictor, self).__init__()
        
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
    
    def forward(self, from_data_list: list, to_data_list: list, edge_attrs: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 序列迭代式处理

        支持序列式的分子演化预测：
        - s1 -> s2 使用 edge1
        - s2 -> s3 使用 edge2
        - s3 -> s4 使用 edge3
        - ...

        Args:
            from_data_list: 起始分子图数据列表 [s1, s2, s3, ...]
            to_data_list: 目标分子图数据列表 [s2, s3, s4, ...]
            edge_attrs: 边特征数组 (操作信息序列)
                       形状: [num_steps, edge_feature_dim]

        Returns:
            属性变化预测值序列
            形状: [num_steps, output_dim]
        """
        # 验证输入长度一致
        num_steps = len(from_data_list)
        assert len(to_data_list) == num_steps, f"to_data_list长度{len(to_data_list)}与from_data_list长度{num_steps}不一致"
        assert edge_attrs.size(0) == num_steps, f"edge_attrs第一维{edge_attrs.size(0)}与步骤数{num_steps}不一致"
        
        predictions = []
        
        for step in range(num_steps):
            # 获取当前步骤的from_data和to_data
            current_from_data = from_data_list[step]
            current_to_data = to_data_list[step]
            current_edge_attr = edge_attrs[step:step+1]
            
            # molecule组件: 提取当前起始和目标分子特征
            from_features = self.molecule_extractor_from(current_from_data)
            to_features = self.molecule_extractor_to(current_to_data)
            
            # edge组件: 编码当前边特征
            edge_features = self.edge_encoder(current_edge_attr)
            
            # fusion组件: 融合特征并预测属性变化
            property_change = self.fusion_predictor(from_features, to_features, edge_features)
            predictions.append(property_change)
        
        # 堆叠所有预测结果
        predictions = torch.stack(predictions, dim=0)  # [num_steps, output_dim]
        
        return predictions

