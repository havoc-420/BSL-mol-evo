#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于EquiformerV1的分子进化预测器 (Linear-Linear-Linear版本)

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


class MoleculeEvolutionEquiformerLinearPredictor(nn.Module):
    """
    基于EquiformerV1的分子进化预测器 (Linear-Linear-Linear版本)
    
    使用线性组件组合：
    - molecule: EquiformerV1MoleculeFeatureExtractor (EquiformerV1特征提取器)
    - edge: LinearEdgeFeatureExtractor (线性边特征编码器)
    - fusion: MLPFusionPredictor (线性特征融合预测器)
    """
    
    def __init__(self, 
                 # EquiformerV1参数
                 irreps_in='5x0e',
                 irreps_node_embedding='128x0e+64x1e+32x2e', 
                 num_layers=6,
                 irreps_node_attr='1x0e', 
                 irreps_sh='1x0e+1x1e+1x2e',
                 max_radius=5.0,
                 number_of_basis=128, 
                 basis_type='gaussian', 
                 fc_neurons=[64, 64], 
                 irreps_feature='512x0e',
                 irreps_head='32x0e+16x1o+8x2e', 
                 num_heads=4, 
                 irreps_pre_attn=None,
                 rescale_degree=False, 
                 nonlinear_message=False,
                 irreps_mlp_mid='128x0e+64x1e+32x2e',
                 norm_layer='layer',
                 alpha_drop=0.2, 
                 proj_drop=0.0, 
                 out_drop=0.0,
                 drop_path_rate=0.0,
                 mean=None, 
                 std=None, 
                 scale=None, 
                 atomref=None,
                 hidden_dims=None,
                 # 其他组件参数
                 edge_feature_dim: int = 11,
                 output_dim: int = 1):
        """
        初始化预测器

        Args:
            # EquiformerV1参数
            irreps_in: 输入节点的表示（irreps）
            irreps_node_embedding: 节点嵌入的表示
            num_layers: 网络层数
            irreps_node_attr: 节点属性的表示
            irreps_sh: 球谐函数的表示
            max_radius: 最大原子间距离（用于构建图）
            number_of_basis: 径向基函数的数量
            basis_type: 径向基函数类型 ('gaussian' 或 'bessel')
            fc_neurons: 全连接层神经元数
            irreps_feature: 特征表示
            irreps_head: 注意力头的表示
            num_heads: 注意力头数
            irreps_pre_attn: 注意力前的表示
            rescale_degree: 是否重新缩放度数
            nonlinear_message: 是否使用非线性消息传递
            irreps_mlp_mid: MLP中间层的表示
            norm_layer: 归一化层类型
            alpha_drop: 注意力dropout率
            proj_drop: 投影dropout率
            out_drop: 输出dropout率
            drop_path_rate: 路径dropout率
            mean: 数据集均值（用于标准化）
            std: 数据集标准差（用于标准化）
            scale: 输出缩放因子
            atomref: 原子参考值
            hidden_dims: 全连接层维度配置
            
            # 其他组件参数
            edge_feature_dim: 边特征维度 (操作信息)
            output_dim: 输出维度 (属性变化)
        """
        super(MoleculeEvolutionEquiformerLinearPredictor, self).__init__()
        
        # molecule组件: EquiformerV1分子特征提取器
        self.molecule_extractor_from = EquiformerV1MoleculeFeatureExtractor(
            irreps_in=irreps_in,
            irreps_node_embedding=irreps_node_embedding,
            num_layers=num_layers,
            irreps_node_attr=irreps_node_attr,
            irreps_sh=irreps_sh,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            basis_type=basis_type,
            fc_neurons=fc_neurons,
            irreps_feature=irreps_feature,
            irreps_head=irreps_head,
            num_heads=num_heads,
            irreps_pre_attn=irreps_pre_attn,
            rescale_degree=rescale_degree,
            nonlinear_message=nonlinear_message,
            irreps_mlp_mid=irreps_mlp_mid,
            norm_layer=norm_layer,
            alpha_drop=alpha_drop,
            proj_drop=proj_drop,
            out_drop=out_drop,
            drop_path_rate=drop_path_rate,
            mean=mean,
            std=std,
            scale=scale,
            atomref=atomref,
            hidden_dims=hidden_dims
        )
        
        self.molecule_extractor_to = EquiformerV1MoleculeFeatureExtractor(
            irreps_in=irreps_in,
            irreps_node_embedding=irreps_node_embedding,
            num_layers=num_layers,
            irreps_node_attr=irreps_node_attr,
            irreps_sh=irreps_sh,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            basis_type=basis_type,
            fc_neurons=fc_neurons,
            irreps_feature=irreps_feature,
            irreps_head=irreps_head,
            num_heads=num_heads,
            irreps_pre_attn=irreps_pre_attn,
            rescale_degree=rescale_degree,
            nonlinear_message=nonlinear_message,
            irreps_mlp_mid=irreps_mlp_mid,
            norm_layer=norm_layer,
            alpha_drop=alpha_drop,
            proj_drop=proj_drop,
            out_drop=out_drop,
            drop_path_rate=drop_path_rate,
            mean=mean,
            std=std,
            scale=scale,
            atomref=atomref,
            hidden_dims=hidden_dims
        )
        
        # edge组件: 线性边特征编码器
        self.edge_encoder = LinearEdgeFeatureExtractor(edge_feature_dim, 256)
        
        # fusion组件: 线性特征融合预测器
        self.fusion_predictor = MLPFusionPredictor(
            node_dim=512,  # EquiformerV1输出是512维特征
            edge_dim=256,
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

        # TEST
        # print('😀', from_features.shape, to_features.shape, edge_features.shape)

        # fusion组件: 融合特征并预测属性变化
        property_changes = self.fusion_predictor(from_features, to_features, edge_features)
        
        return property_changes