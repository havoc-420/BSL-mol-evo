#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础模型类定义
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data


class BaseModel(nn.Module):
    """
    所有模型的基类
    """
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("每个模型必须实现自己的forward方法")


class BaseGNNModel(BaseModel):
    """
    GNN模型的基类
    """
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, 
                 hidden_dim: int, output_dim: int):
        super(BaseGNNModel, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
    
    def forward(self, data: Data) -> torch.Tensor:
        raise NotImplementedError("每个GNN模型必须实现自己的forward方法")