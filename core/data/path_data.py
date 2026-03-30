#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0.3 长链路路径 Dataset 与 Collate 模块

负责：
- MoleculePathDataset: 可变长度路径样本容器
- path_collate: 将多条路径 padding 到同一 max_steps, 生成批次级掩码与张量
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from typing import List, Dict, Tuple, Any, Optional


class MoleculePathDataset(Dataset):
    """
    v0.3 路径样本 Dataset。
    
    每个样本是一条路径，包含：
    - 节点图数据列表（含 None 表示非法节点）
    - 节点有效位掩码
    - 步骤操作边特征
    - 步骤有效位掩码
    - 路径级标签
    - 步骤级标签（可选）
    """
    
    def __init__(self, processed_paths: List[Dict]):
        """
        Args:
            processed_paths: build_molecule_path_dataset_v0_3 返回的路径列表
        """
        self.paths = processed_paths
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        返回单条路径样本 dict，包含所有原始字段。
        实际的 padding / batching 在 path_collate 中完成。
        """
        return self.paths[idx]


def _create_placeholder_graph() -> Data:
    """
    为非法中间节点创建占位 Data 对象。
    包含一个虚拟原子（原子序数 0）和零坐标，
    VisNet 可以处理但不包含有意义的化学信息。
    """
    return Data(
        z=torch.LongTensor([0]),     # 虚拟原子序数
        pos=torch.zeros(1, 3),       # 零坐标
    )


def path_collate(batch: List[Dict]) -> Dict[str, Any]:
    """
    将一个 batch 的路径样本 padding 到统一长度，生成批次级张量。
    
    Args:
        batch: List[Dict]，每个 dict 是 MoleculePathDataset.__getitem__ 返回的路径
        
    Returns:
        Dict 包含：
        - node_batch_list: List[Batch]，长度 = max_num_nodes
            每个位置是一个 PyG Batch，包含 batch_size 个分子图
            非法节点用占位图填充
        - node_valid_mask: Tensor (B, max_num_nodes) bool
        - edge_features: Tensor (B, max_steps, edge_dim)
        - step_valid_mask: Tensor (B, max_steps) bool
        - path_padding_mask: Tensor (B, max_steps) bool, True 表示有效位置
        - path_targets: Tensor (B, 1)
        - step_targets: Tensor (B, max_steps) 或 None
        - has_step_targets: bool
        - num_steps: List[int]，每条路径的实际步数
        - batch_size: int
    """
    batch_size = len(batch)
    
    # 找到 batch 内最大步数
    max_steps = max(p['num_steps'] for p in batch)
    max_num_nodes = max_steps + 1  # 节点数 = 步数 + 1
    
    # 找到边特征维度
    edge_dim = len(batch[0]['edge_features'][0]) if batch[0]['edge_features'] else 0
    
    # 是否所有样本都有步骤级标签
    has_step_targets = all(p['step_targets'] is not None for p in batch)
    
    placeholder = _create_placeholder_graph()
    
    # ====== 构建节点级数据 ======
    # 按"节点位置"组织：position_i -> [path_0 的第 i 个节点, path_1 的第 i 个节点, ...]
    node_batch_list = []
    node_valid_mask = torch.zeros(batch_size, max_num_nodes, dtype=torch.bool)
    
    for pos_i in range(max_num_nodes):
        graphs_at_pos = []
        for b_i, path in enumerate(batch):
            if pos_i < len(path['node_graph_data_list']):
                graph = path['node_graph_data_list'][pos_i]
                valid = path['node_valid_mask'][pos_i]
                if graph is not None and valid:
                    graphs_at_pos.append(graph)
                    node_valid_mask[b_i, pos_i] = True
                else:
                    graphs_at_pos.append(placeholder)
            else:
                # padding 位置
                graphs_at_pos.append(placeholder)
        
        # 合并为 PyG Batch
        node_batch_list.append(Batch.from_data_list(graphs_at_pos))
    
    # ====== 构建步骤级数据 ======
    edge_features = torch.zeros(batch_size, max_steps, edge_dim, dtype=torch.float32)
    step_valid_mask = torch.zeros(batch_size, max_steps, dtype=torch.bool)
    path_padding_mask = torch.zeros(batch_size, max_steps, dtype=torch.bool)
    
    for b_i, path in enumerate(batch):
        n_steps = path['num_steps']
        for s_i in range(n_steps):
            edge_features[b_i, s_i] = torch.tensor(path['edge_features'][s_i], dtype=torch.float32)
            step_valid_mask[b_i, s_i] = path['step_valid_mask'][s_i]
            path_padding_mask[b_i, s_i] = True  # 有效位置
    
    # ====== 路径级标签 ======
    path_targets = torch.tensor(
        [[p['path_target']] for p in batch],
        dtype=torch.float32
    )
    
    # ====== 步骤级标签 ======
    step_targets = None
    if has_step_targets:
        step_targets = torch.zeros(batch_size, max_steps, dtype=torch.float32)
        for b_i, path in enumerate(batch):
            n_steps = path['num_steps']
            for s_i in range(min(n_steps, len(path['step_targets']))):
                step_targets[b_i, s_i] = path['step_targets'][s_i]
    
    # ====== 步数列表 ======
    num_steps_list = [p['num_steps'] for p in batch]
    
    return {
        'node_batch_list': node_batch_list,
        'node_valid_mask': node_valid_mask,
        'edge_features': edge_features,
        'step_valid_mask': step_valid_mask,
        'path_padding_mask': path_padding_mask,
        'path_targets': path_targets,
        'step_targets': step_targets,
        'has_step_targets': has_step_targets,
        'num_steps': num_steps_list,
        'batch_size': batch_size,
    }
