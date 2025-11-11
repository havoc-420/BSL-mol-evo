#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于GCN和Transformer的分子进化预测器 (GCN-GCN-Transformer版本，支持位置编码) - V0.2

这是一个使用混合组件的GCN模型：
- molecule: GCN特征提取器
- edge: Transformer边特征编码器（支持位置编码）
- fusion: MLP特征融合预测器
用于从分子图中提取特征表示并预测分子演化过程中的属性变化。
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from ..v0.molecule_feature_extractors.gcn import GCNMoleculeFeatureExtractor
from ..v0.edge_feature_extractors import TransformerEdgeFeatureExtractorLap
from ..v0.fusion_predictors import MLPFusionPredictor
from ..v0 import register_model


@register_model(display_name="GCN-Transformer-Linear 模型(支持位置编码) V0.2", save_dir_name="gcn_transformer_linear_v02", requires_position_encoding=True)
class MoleculeEvolutionGCNTransformerLinearPredictorV02(nn.Module):
    """
    基于GCN和Transformer的分子进化预测器 (GCN-GCN-Transformer版本，支持位置编码) - V0.2
    
    使用混合组件组合：
    - molecule: GCNMoleculeFeatureExtractor (GCN特征提取器)
    - edge: TransformerEdgeFeatureExtractorLap (Transformer边特征编码器，支持位置编码)
    - fusion: MLPFusionPredictor (MLP特征融合预测器)
    """
    
    def __init__(self, node_feature_dim: int = 11, edge_feature_dim: int = 15,
                 hidden_dims: list = [128, 256, 256], output_dim: int = 1,
                 num_heads: int = 8, num_layers: int = 2, pe_dim: int = 9):
        """
        初始化预测器

        Args:
            node_feature_dim: 节点特征维度
            edge_feature_dim: 边特征维度 (操作信息)
            hidden_dims: GCN各隐藏层维度列表
            output_dim: 输出维度 (属性变化)
            num_heads: Transformer头数
            num_layers: Transformer层数
            pe_dim: 位置编码维度
        """
        super(MoleculeEvolutionGCNTransformerLinearPredictorV02, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.pe_dim = pe_dim
        
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # molecule组件: GCN分子特征提取器
        self.molecule_extractor_from = GCNMoleculeFeatureExtractor(node_feature_dim, hidden_dims)
        self.molecule_extractor_to = GCNMoleculeFeatureExtractor(node_feature_dim, hidden_dims)
        
        # edge组件: Transformer边特征编码器（支持位置编码）
        # 当使用位置编码时，Transformer内部会将基础特征和位置编码拼接，所以d_model需要调整
        edge_d_model = 128
        if pe_dim > 0:
            # 有位置编码时，实际输出维度是2 * edge_d_model
            edge_d_model = 64  # 这样输出维度就是 64*2 = 128
            
        self.edge_encoder = TransformerEdgeFeatureExtractorLap(
            input_dim=edge_feature_dim, 
            d_model=edge_d_model,
            nhead=num_heads, 
            num_layers=num_layers,
            pe_dim=pe_dim)
        
        # fusion组件: MLP特征融合预测器
        # 根据是否有位置编码调整edge_dim参数
        edge_output_dim = 128  # 无位置编码时的输出维度
        if pe_dim > 0:
            edge_output_dim = 128  # 有位置编码时的输出维度也是128
            
        self.fusion_predictor = MLPFusionPredictor(
            node_dim=hidden_dims[-1] * 2,  # GCN输出是mean和max拼接的结果
            edge_dim=edge_output_dim,
            hidden_dims=[512, 256, 128],
            output_dim=output_dim
        )
    
    def get_model_config(self):
        """
        获取模型配置参数
        
        Returns:
            dict: 包含模型配置参数的字典
        """
        return {
            'node_feature_dim': self.node_feature_dim,
            'edge_feature_dim': self.edge_feature_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'num_heads': self.edge_encoder.nhead,
            'num_layers': self.edge_encoder.num_layers,
            'pe_dim': self.pe_dim
        }
    
    def forward(self, from_data: Data, to_data: Data, edge_attr: torch.Tensor, 
                position_encoding: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            from_data: 起始分子图数据
            to_data: 目标分子图数据
            edge_attr: 边特征 (操作信息)
            position_encoding: 位置编码特征

        Returns:
            属性变化预测值
        """
        # molecule组件: 提取起始和目标分子特征
        from_features = self.molecule_extractor_from(from_data)
        to_features = self.molecule_extractor_to(to_data)
        
        # edge组件: 编码边特征（支持位置编码）
        if position_encoding is not None:
            edge_features = self.edge_encoder(edge_attr, position_encoding)
            print('😀', edge_features.shape)
        else:
            edge_features = self.edge_encoder(edge_attr)
        
        # fusion组件: 融合特征并预测属性变化
        property_changes = self.fusion_predictor(from_features, to_features, edge_features)
        
        return property_changes