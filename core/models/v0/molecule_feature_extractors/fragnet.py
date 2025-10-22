#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FragNet分子特征提取器

这是一个基于FragNet的分子特征提取器，用于从分子图中提取特征表示。
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_max_pool

from .fragnet_core.gat2 import FragNet


class FragNetMoleculeFeatureExtractor(nn.Module):
    """
    基于FragNet的分子特征提取器
    
    该模型将SMILES转换为图结构后，通过FragNet提取分子特征表示。
    FragNet是一个专门为分子属性预测设计的图神经网络。
    """
    
    def __init__(self, 
                 atom_feature_dim: int = 167,
                 frag_feature_dim: int = 167, 
                 edge_feature_dim: int = 16,
                 num_layers: int = 4,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 dropout_ratio: float = 0.15):
        """
        初始化FragNet特征提取器

        Args:
            atom_feature_dim: 原子特征维度
            frag_feature_dim: 片段特征维度  
            edge_feature_dim: 边特征维度
            num_layers: 网络层数
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            dropout_ratio: Dropout比例
        """
        super(FragNetMoleculeFeatureExtractor, self).__init__()
        
        self.atom_feature_dim = atom_feature_dim
        self.frag_feature_dim = frag_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_ratio = dropout_ratio
        
        # FragNet主干网络
        self.fragnet = FragNet(
            num_layer=num_layers,
            drop_ratio=dropout_ratio,
            emb_dim=hidden_dim,
            atom_features=atom_feature_dim,
            frag_features=frag_feature_dim,
            edge_features=edge_feature_dim,
            num_heads=num_heads
        )
        
        # 输出层 - 将原子和片段特征合并
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        
    def forward(self, data) -> torch.Tensor:
        """
        前向传播，提取分子特征

        Args:
            data: 包含分子图信息的数据对象，可以是:
                - Data: PyG Data对象
                - dict: 字典格式的FragNet数据
                - list: 包含dict的列表（支持one-by-one处理）

        Returns:
            分子特征表示 (512维: mean和max池化的合并结果)
        """
        # 处理FragNet数据格式（列表形式）
        # 如果输入是列表，则逐个处理并返回列表
        if isinstance(data, list):
            # one-by-one处理列表中的每个元素
            results = []
            for item in data:
                result = self._forward_single(item)
                results.append(result)
            return results
        
        # 单个数据处理
        return self._forward_single(data)
    
    def _forward_single(self, data) -> torch.Tensor:
        """
        处理单个数据样本

        Args:
            data: 单个分子数据对象

        Returns:
            分子特征表示
        """
        # 如果是字典格式（FragNet数据），直接使用
        if isinstance(data, dict):
            batch_dict = data
        else:
            # 构造FragNet所需的batch字典
            batch_dict = {
                "x_atoms": data.x_atoms if hasattr(data, 'x_atoms') else data.x,
                "x_frags": data.x_frags if hasattr(data, 'x_frags') else data.x,
                "edge_index": data.edge_index,
                "edge_attr": data.edge_attr if hasattr(data, 'edge_attr') else torch.zeros(data.edge_index.size(1), self.edge_feature_dim),
                "frag_index": data.frag_index if hasattr(data, 'frag_index') else data.edge_index,
                "atom_to_frag_ids": data.atom_to_frag_ids if hasattr(data, 'atom_to_frag_ids') else torch.zeros(data.x.size(0), dtype=torch.long),
                "node_features_bonds": data.node_features_bonds if hasattr(data, 'node_features_bonds') else torch.zeros(data.edge_index.size(1), self.edge_feature_dim),
                "edge_index_bonds_graph": data.edge_index_bonds_graph if hasattr(data, 'edge_index_bonds_graph') else data.edge_index,
                "edge_attr_bonds": data.edge_attr_bonds if hasattr(data, 'edge_attr_bonds') else torch.zeros(data.edge_index.size(1), self.edge_feature_dim),
                "node_features_fbonds": data.node_features_fbonds if hasattr(data, 'node_features_fbonds') else torch.zeros(data.edge_index.size(1), self.edge_feature_dim),  # 添加缺失字段
                "edge_index_fbonds": data.edge_index_fbonds if hasattr(data, 'edge_index_fbonds') else data.edge_index,
                "edge_attr_fbonds": data.edge_attr_fbonds if hasattr(data, 'edge_attr_fbonds') else torch.zeros(data.edge_index.size(1), self.edge_feature_dim),  # 添加缺失字段
                "batch": data.batch if hasattr(data, 'batch') else torch.zeros(data.x.size(0), dtype=torch.long),
                "frag_batch": data.frag_batch if hasattr(data, 'frag_batch') else torch.zeros(data.x.size(0), dtype=torch.long) if hasattr(data, 'x_frags') else torch.zeros(data.x.size(0), dtype=torch.long),
            }
        
        # 处理字段名映射问题，确保所有必需的字段都存在
        # 特别是atom_to_frag_ids字段可能在不同处理函数中使用不同的名称
        if "atom_to_frag_ids" not in batch_dict and "atom_id_frag_id" in batch_dict:
            batch_dict["atom_to_frag_ids"] = batch_dict["atom_id_frag_id"]
        
        # 确保所有必需的字段都有默认值
        required_fields = {
            "atom_to_frag_ids": lambda: torch.zeros(batch_dict["x_atoms"].size(0), dtype=torch.long),
            "node_features_bonds": lambda: torch.zeros(batch_dict["edge_index"].size(1) if batch_dict["edge_index"].numel() > 0 else 0, self.edge_feature_dim),
            "edge_index_bonds_graph": lambda: batch_dict["edge_index"],
            "edge_attr_bonds": lambda: torch.zeros(batch_dict["edge_index"].size(1) if batch_dict["edge_index"].numel() > 0 else 0, self.edge_feature_dim),
            "node_features_fbonds": lambda: torch.zeros(0, self.edge_feature_dim),  # 空张量
            "edge_index_fbonds": lambda: torch.tensor([[], []], dtype=torch.long),
            "edge_attr_fbonds": lambda: torch.zeros(0, self.edge_feature_dim),
            "batch": lambda: torch.zeros(batch_dict["x_atoms"].size(0), dtype=torch.long),
            "frag_batch": lambda: torch.zeros(batch_dict["x_frags"].size(0), dtype=torch.long) if "x_frags" in batch_dict and batch_dict["x_frags"] is not None and batch_dict["x_frags"].numel() > 0 else torch.zeros(batch_dict["x_atoms"].size(0), dtype=torch.long),
        }
        
        for field, default_factory in required_fields.items():
            if field not in batch_dict or batch_dict[field] is None:
                batch_dict[field] = default_factory()
            # 特别处理列表类型数据
            elif isinstance(batch_dict[field], list):
                # 如果是列表，将其转换为张量
                if len(batch_dict[field]) > 0 and not isinstance(batch_dict[field][0], torch.Tensor):
                    batch_dict[field] = torch.tensor(batch_dict[field], dtype=torch.long if 'atom_to_frag_ids' in field or 'batch' in field or 'index' in field else torch.float)
        
        # 确保所有张量都在同一个设备上
        device = next(self.parameters()).device
        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor) and value.device != device:
                batch_dict[key] = value.to(device)
        
        # 通过FragNet获取原子和片段特征
        x_atoms, x_frags, _, _ = self.fragnet(batch_dict)
        
        # 池化操作获取图级别表示
        batch = batch_dict["batch"]
        frag_batch = batch_dict["frag_batch"]
        
        # 对原子特征进行池化
        if batch is not None and batch.numel() > 0:
            x_atoms_mean = global_mean_pool(x_atoms, batch)
            x_atoms_max = global_max_pool(x_atoms, batch)
        else:
            x_atoms_mean = torch.mean(x_atoms, dim=0, keepdim=True)
            x_atoms_max = torch.max(x_atoms, dim=0, keepdim=True)[0]
        
        # 对片段特征进行池化
        if frag_batch is not None and frag_batch.numel() > 0:
            x_frags_mean = global_mean_pool(x_frags, frag_batch)
            x_frags_max = global_max_pool(x_frags, frag_batch)
        else:
            x_frags_mean = torch.mean(x_frags, dim=0, keepdim=True)
            x_frags_max = torch.max(x_frags, dim=0, keepdim=True)[0]
            
        # 合并原子和片段特征
        x_mean = torch.cat([x_atoms_mean, x_frags_mean], dim=1)
        x_max = torch.cat([x_atoms_max, x_frags_max], dim=1)
        
        # 合并mean和max池化结果
        x = torch.cat([x_mean, x_max], dim=1)
        
        # 通过输出投影层
        x = self.output_projection(x)
        
        return x
