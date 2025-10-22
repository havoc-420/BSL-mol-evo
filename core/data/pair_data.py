"""
分子对数据处理模块

该模块提供处理分子对数据集的通用功能，支持多种数据格式。
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class MoleculePairDataset(Dataset):
    """
    返回 (from_graph, to_graph, edge_feature, target) 的数据集
    
    支持两种数据格式：
    1. PyG Data 对象列表 (用于GCN等模型)
    2. FragNet 特征字典列表 (用于FragNet模型)
    具体的批处理逻辑由 DataLoader 的 collate_fn 参数控制
    """
    def __init__(self, from_list, to_list, edge_attrs, targets):
        assert len(from_list) == len(to_list) == len(edge_attrs) == len(targets)
        self.from_list = from_list
        self.to_list   = to_list
        self.edge_attrs = edge_attrs      # Tensor (N, edge_feature_dim)
        self.targets    = targets         # Tensor (N, 1)

    def __len__(self):
        return len(self.from_list)

    def __getitem__(self, idx):
        return self.from_list[idx], self.to_list[idx], self.edge_attrs[idx], self.targets[idx]


def pair_collate(batch):
    """把若干 (from, to, edge, target) 合并成 batch"""
    from_list, to_list, edge_list, target_list = zip(*batch)
    
    # 检查是否有None值
    if any(item is None for item in from_list) or any(item is None for item in to_list):
        raise ValueError("批次中包含None值，无法构造batch")

    # 检查数据格式
    if len(from_list) > 0 and isinstance(from_list[0], dict):
        # FragNet 字典格式数据，直接返回列表
        from_batch = list(from_list)
        to_batch = list(to_list)
    else:
        # PyG Data 对象格式数据，使用 Batch 合并
        from_batch = Batch.from_data_list(list(from_list))
        to_batch = Batch.from_data_list(list(to_list))

    # edge 特征和目标直接 stack
    edge_batch = torch.stack(edge_list, dim=0)          # (B, edge_dim)
    target_batch = torch.stack(target_list, dim=0)      # (B, 1)

    return from_batch, to_batch, edge_batch, target_batch