#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于GCN的分子特征提取模型 v0

这是一个简化版的GCN模型，只用于从分子图中提取特征表示。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data


class MoleculeFeatureExtractor(nn.Module):
    """
    分子特征提取器
    
    该模型将SMILES转换为图结构后，通过两层GCNConv提取分子特征表示。
    """
    
    def __init__(self, node_feature_dim: int = 1, hidden_dim: int = 128, 
                 output_dim: int = 256):
        """
        初始化特征提取器

        Args:
            node_feature_dim: 节点特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
        """
        super(MoleculeFeatureExtractor, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # GCN网络层 - 三层结构
        self.gcn_layers = nn.ModuleList()
        
        # 第一层: 从节点特征维度到128
        self.gcn_layers.append(GCNConv(node_feature_dim, 128))
        
        # 第二层: 从128到256
        self.gcn_layers.append(GCNConv(128, 256))
        
        # 第三层: 从256到256
        self.gcn_layers.append(GCNConv(256, 256))
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播，提取分子特征

        Args:
            data: 包含节点特征和边索引的图数据

        Returns:
            分子特征表示 (512维: mean和max池化的合并结果)
        """
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
        
        # 通过GCN层
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, edge_index)
            if i < len(self.gcn_layers) - 1:  # 最后一层不加激活函数
                x = F.relu(x)
        
        # 全局池化获取图表示 - 分别进行mean和max池化
        if batch is not None:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
        else:
            x_mean = torch.mean(x, dim=0, keepdim=True)
            x_max = torch.max(x, dim=0, keepdim=True)[0]
        
        # 合并mean和max池化结果
        x = torch.cat([x_mean, x_max], dim=1)
        
        return x


class MoleculeEvolutionGCNPredictor(nn.Module):
    """
    基于GCN的分子进化预测器 (v0)
    
    使用两个MoleculeFeatureExtractor分别提取起始分子和目标分子的特征，
    并结合边特征预测属性变化。
    """
    
    def __init__(self, node_feature_dim: int = 37, edge_feature_dim: int = 15,
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
        super(MoleculeEvolutionGCNPredictor, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 起始分子和目标分子特征提取器
        # 注意：现在MoleculeFeatureExtractor输出512维（256*2）
        self.from_molecule_extractor = MoleculeFeatureExtractor(
            node_feature_dim, 128, 256)
        self.to_molecule_extractor = MoleculeFeatureExtractor(
            node_feature_dim, 128, 256)
        
        # 边特征编码器
        # 从edge_feature_dim(11)维先演变到64维，再到hidden_dim(128)维
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
        
        # 特征融合和预测层
        # 输入: [from_mol_features, to_mol_features, edge_features]
        # 现在维度是: 512 + 512 + 128 = 1152
        fusion_input_dim = 512 * 2 + hidden_dim
        self.predictor = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
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