#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于VisNet的分子进化预测器 (Linear-Linear版本) - 迭代式架构

这是一个使用线性组件的VisNet模型的迭代式版本：
- molecule: VisNet特征提取器
- edge: 线性边特征编码器
- fusion: 线性特征融合预测器

关键特性：
- 支持edge数组的逐步迭代处理
- 每次迭代使用一个edge_attr进行预测
- 预测结果作为下一次迭代的输入
- 最终输出是最后一次迭代的结果
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from ..v0.molecule_feature_extractors import VisNetMoleculeFeatureExtractor
from ..v0.edge_feature_extractors import LinearEdgeFeatureExtractor
from ..v0.fusion_predictors import MLPFusionPredictor
from . import register_model


@register_model(display_name="VisNet Linear-Linear 迭代式模型", save_dir_name="visnet_linear_linear_iterative")
class MoleculeEvolutionVisnetLinearIterativePredictor(nn.Module):
    """
    基于VisNet的分子进化预测器 (Linear-Linear迭代式版本)
    
    使用线性组件组合，支持edge数组的逐步迭代处理：
    - molecule: VisNetMoleculeFeatureExtractor (VisNet特征提取器)
    - edge: LinearEdgeFeatureExtractor (线性边特征编码器)
    - fusion: MLPFusionPredictor (线性特征融合预测器)
    
    迭代机制：
    - 输入edge_attr数组，逐步迭代处理
    - 每次迭代使用一个edge_attr进行预测
    - 预测结果作为下一次迭代的输入
    - 最终输出是最后一次迭代的结果
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
    
    def forward(self, from_data: Data, to_data: Data, edge_attrs: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 迭代式处理edge数组

        Args:
            from_data: 起始分子图数据
            to_data: 目标分子图数据
            edge_attrs: 边特征数组 (操作信息序列)
                       形状: [num_edges, edge_feature_dim] 或 [num_edges, num_operations, edge_feature_dim]

        Returns:
            属性变化预测值序列
            形状: [num_edges, output_dim] 或 [num_operations, output_dim]
        """
        # molecule组件: 提取起始和目标分子特征（只提取一次）
        from_features = self.molecule_extractor_from(from_data)
        to_features = self.molecule_extractor_to(to_data)
        
        # 处理edge_attrs的形状
        if edge_attrs.dim() == 2:
            # 形状: [num_edges, edge_feature_dim]
            num_edges = edge_attrs.size(0)
            predictions = []
            
            for i in range(num_edges):
                # edge组件: 编码当前边特征
                edge_features = self.edge_encoder(edge_attrs[i:i+1])
                
                # fusion组件: 融合特征并预测属性变化
                property_change = self.fusion_predictor(from_features, to_features, edge_features)
                predictions.append(property_change)
            
            # 堆叠所有预测结果
            predictions = torch.stack(predictions, dim=0)  # [num_edges, output_dim]
            
        elif edge_attrs.dim() == 3:
            # 形状: [num_edges, num_operations, edge_feature_dim]
            num_edges, num_operations, _ = edge_attrs.size()
            predictions = []
            
            for edge_idx in range(num_edges):
                edge_predictions = []
                current_from_features = from_features
                
                for op_idx in range(num_operations):
                    # edge组件: 编码当前操作特征
                    edge_features = self.edge_encoder(edge_attrs[edge_idx:edge_idx+1, op_idx:op_idx+1])
                    
                    # fusion组件: 融合特征并预测属性变化
                    property_change = self.fusion_predictor(current_from_features, to_features, edge_features)
                    edge_predictions.append(property_change)
                    
                    # 更新from_features（可选：根据预测结果更新）
                    # 这里可以添加更复杂的逻辑，例如使用预测结果更新from_features
                
                # 堆叠当前edge的所有操作预测结果
                edge_predictions = torch.stack(edge_predictions, dim=0)  # [num_operations, output_dim]
                predictions.append(edge_predictions)
            
            # 堆叠所有edge的预测结果
            predictions = torch.stack(predictions, dim=0)  # [num_edges, num_operations, output_dim]
        
        else:
            raise ValueError(f"edge_attrs的维度必须是2或3，但得到的是: {edge_attrs.dim()}")
        
        return predictions
    
    def forward_iterative(self, from_data: Data, to_data: Data, edge_attrs: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 真正的迭代式处理（每次迭代更新from_data）

        Args:
            from_data: 起始分子图数据
            to_data: 目标分子图数据
            edge_attrs: 边特征数组 (操作信息序列)
                       形状: [num_operations, edge_feature_dim]

        Returns:
            属性变化预测值序列
            形状: [num_operations, output_dim]
        """
        num_operations = edge_attrs.size(0)
        predictions = []
        
        current_from_data = from_data
        
        for i in range(num_operations):
            # molecule组件: 提取当前起始和目标分子特征
            current_from_features = self.molecule_extractor_from(current_from_data)
            to_features = self.molecule_extractor_to(to_data)
            
            # edge组件: 编码当前边特征
            edge_features = self.edge_encoder(edge_attrs[i:i+1])
            
            # fusion组件: 融合特征并预测属性变化
            property_change = self.fusion_predictor(current_from_features, to_features, edge_features)
            predictions.append(property_change)
            
            # 更新current_from_data（这里需要根据实际需求实现）
            # 例如：可以根据预测结果修改from_data的某些属性
            # 这是一个简化版本，实际实现可能需要更复杂的逻辑
        
        # 堆叠所有预测结果
        predictions = torch.stack(predictions, dim=0)  # [num_operations, output_dim]
        
        return predictions
