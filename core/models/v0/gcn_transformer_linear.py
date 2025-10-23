#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于GCN和Transformer的分子进化预测器 (GCN-GCN-Transformer版本)

这是一个使用混合组件的GCN模型：
- molecule: GCN特征提取器
- edge: Transformer边特征编码器
- fusion: MLP特征融合预测器
用于从分子图中提取特征表示并预测分子演化过程中的属性变化。
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .molecule_feature_extractors.gcn import GCNMoleculeFeatureExtractor
from .edge_feature_extractors import TransformerEdgeFeatureExtractor
from .fusion_predictors import MLPFusionPredictor
from . import register_model


@register_model(display_name="GCN-Transformer-Linear 模型", save_dir_name="gcn_transformer_linear")
class MoleculeEvolutionGCNTransformerLinearPredictor(nn.Module):
    """
    基于GCN和Transformer的分子进化预测器 (GCN-GCN-Transformer版本)
    
    使用混合组件组合：
    - molecule: GCNMoleculeFeatureExtractor (GCN特征提取器)
    - edge: TransformerEdgeFeatureExtractor (Transformer边特征编码器)
    - fusion: MLPFusionPredictor (MLP特征融合预测器)
    """
    
    def __init__(self, node_feature_dim: int = 11, edge_feature_dim: int = 15,
                 hidden_dims: list = [128, 256, 256], output_dim: int = 1,
                 num_heads: int = 8, num_layers: int = 2):
        """
        初始化预测器

        Args:
            node_feature_dim: 节点特征维度
            edge_feature_dim: 边特征维度 (操作信息)
            hidden_dims: GCN各隐藏层维度列表
            output_dim: 输出维度 (属性变化)
            num_heads: Transformer头数
            num_layers: Transformer层数
        """
        super(MoleculeEvolutionGCNTransformerLinearPredictor, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # molecule组件: GCN分子特征提取器
        self.molecule_extractor_from = GCNMoleculeFeatureExtractor(node_feature_dim, hidden_dims)
        self.molecule_extractor_to = GCNMoleculeFeatureExtractor(node_feature_dim, hidden_dims)
        
        # edge组件: Transformer边特征编码器
        self.edge_encoder = TransformerEdgeFeatureExtractor(
            edge_feature_dim, hidden_dims[-1], num_heads, num_layers)
        
        # fusion组件: MLP特征融合预测器
        self.fusion_predictor = MLPFusionPredictor(
            node_dim=hidden_dims[-1] * 2,  # GCN输出是mean和max拼接的结果
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