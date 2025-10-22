#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorNet分子特征提取器

这是一个基于TensorNet的分子特征提取器，用于从分子3D结构中提取特征表示。
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_max_pool

from .torchmdnet_t_core.tensornet import TensorNet


class TensorNetMoleculeFeatureExtractor(nn.Module):
    """
    TensorNet分子特征提取器
    
    该模型将SMILES转换为3D结构后，通过TensorNet提取分子特征表示。
    TensorNet是一个基于张量表示的模型，适用于3D分子结构的特征提取。
    """
    
    def __init__(self,
                 hidden_channels=128,
                 num_layers=2,
                 num_rbf=32,
                 rbf_type="expnorm",
                 trainable_rbf=False,
                 activation="silu",
                 cutoff_lower=0.0,
                 cutoff_upper=4.5,
                 max_z=128,
                 max_num_neighbors=32,
                 equivariance_invariance_group="O(3)",
                 static_shapes=True,
                 check_errors=True,
                 dtype=torch.float32):
        """
        初始化TensorNet特征提取器

        Args:
            hidden_channels (int, optional): Hidden embedding size.
                (default: :obj:`128`)
            num_layers (int, optional): The number of interaction layers.
                (default: :obj:`2`)
            num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
                (default: :obj:`32`)
            rbf_type (string, optional): The type of radial basis function to use.
                (default: :obj:`"expnorm"`)
            trainable_rbf (bool, optional): Whether to train RBF parameters with
                backpropagation. (default: :obj:`False`)
            activation (string, optional): The type of activation function to use.
                (default: :obj:`"silu"`)
            cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
                (default: :obj:`0.0`)
            cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
                (default: :obj:`4.5`)
            max_z (int, optional): Maximum atomic number. Used for initializing embeddings.
                (default: :obj:`128`)
            max_num_neighbors (int, optional): Maximum number of neighbors to return for a
                given node/atom when constructing the molecular graph during forward passes.
                (default: :obj:`32`)
            equivariance_invariance_group (string, optional): Group under whose action on input
                positions internal tensor features will be equivariant and scalar predictions
                will be invariant. O(3) or SO(3).
                (default :obj:`"O(3)"`)
            static_shapes (bool, optional): Whether to enforce static shapes.
                Makes the model CUDA-graph compatible if check_errors is set to False.
                (default: :obj:`True`)
            check_errors (bool, optional): Whether to check for errors in the distance module.
                (default: :obj:`True`)
            dtype (torch.dtype, optional): Data type for the model parameters.
                (default: :obj:`torch.float32`)
        """
        super(TensorNetMoleculeFeatureExtractor, self).__init__()
        
        # TensorNet主干网络
        self.tensornet = TensorNet(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_num_neighbors=max_num_neighbors,
            max_z=max_z,
            equivariance_invariance_group=equivariance_invariance_group,
            static_shapes=static_shapes,
            check_errors=check_errors,
            dtype=dtype
        )
        
        # 用于特征投影的线性层，将拼接后的特征(2*hidden_channels维)映射到512维
        self.feature_projection = nn.Linear(2 * hidden_channels, 512)
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播，提取分子特征

        Args:
            data: 包含分子3D结构信息的数据对象，需要包含:
                - z: 原子类型
                - pos: 原子3D坐标
                - batch: 原子批次信息

        Returns:
            分子特征表示 (512维)
        """
        # 从data对象中提取所需参数
        z = data.z
        pos = data.pos
        batch = data.batch
        
        # 使用TensorNet进行前向传播
        # tensornet.forward返回 (x, None, z, pos, batch)
        x, _, _, _, _ = self.tensornet(z, pos, batch)
        
        # 对节点特征进行池化，得到图级别的特征
        # 使用mean和max池化来捕获不同的特征信息
        x_mean = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x_max = global_max_pool(x, batch)    # [batch_size, hidden_channels]
        
        # 合并mean和max池化结果
        x_pooled = torch.cat([x_mean, x_max], dim=1)  # [batch_size, 2*hidden_channels]
        
        # 通过线性层将特征映射到512维
        x_projected = self.feature_projection(x_pooled)
        
        return x_projected