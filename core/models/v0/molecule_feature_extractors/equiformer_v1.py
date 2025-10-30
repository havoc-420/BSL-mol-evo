#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EquiformerV1分子特征提取器

这是一个基于GraphAttentionTransformer的分子特征提取器，用于从分子3D结构中提取特征表示。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_max_pool

from .equiformer_v1_core.graph_attention_transformer import GraphAttentionTransformer


class EquiformerV1MoleculeFeatureExtractor(nn.Module):
    """
    EquiformerV1分子特征提取器
    
    该模型将SMILES转换为3D结构后，通过GraphAttentionTransformer提取分子特征表示。
    EquiformerV1是一个基于等变图注意力的模型，适用于3D分子结构的特征提取。
    """
    
    def __init__(self, 
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
                 hidden_dims=None):
        """
        初始化EquiformerV1特征提取器

        Args:
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
            hidden_dims: 全连接层维度配置，默认为[128, 256]
        """
        super(EquiformerV1MoleculeFeatureExtractor, self).__init__()
        
        # EquiformerV1主干网络
        self.equiformer = GraphAttentionTransformer(
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
            atomref=atomref
        )
        
        # 默认隐藏层维度配置，最后一层输出512维特征
        default_hidden_dims = [128, 256, 512]
        self.hidden_dims = hidden_dims if hidden_dims is not None else default_hidden_dims
        
        # 添加线性层将拼接后的特征(1024维)映射到512维
        self.feature_projection = nn.Linear(1024, 512)
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播，提取分子特征

        Args:
            data: 包含分子3D结构信息的数据对象，需要包含:
                - f_in: 输入节点特征
                - pos: 原子3D坐标
                - batch: 原子批次信息
                - node_atom: 原子类型

        Returns:
            分子特征表示 (512维)
        """
        # 从data对象中提取所需参数；参考 mol_evo/modules/equiformer/engine.py:62
        f_in = data.x
        pos = data.pos
        batch = data.batch
        node_atom = data.z
        
        # 获取分子级别的特征表示
        node_features = self.equiformer(f_in, pos, batch, node_atom, return_features=True) # edge_d_index=edge_d_index, edge_d_attr=edge_d_attr)
        
        # 对节点特征进行池化，得到图级别的特征
        # 使用mean和max池化来捕获不同的特征信息
        x_mean = global_mean_pool(node_features, batch)  # [batch_size, feature_dim]
        x_max = global_max_pool(node_features, batch)    # [batch_size, feature_dim]
        
        # 合并mean和max池化结果
        x = torch.cat([x_mean, x_max], dim=1)  # [batch_size, 2*feature_dim]
        
        # 通过线性层将1024维特征映射到512维
        x = self.feature_projection(x)
        
        return x
