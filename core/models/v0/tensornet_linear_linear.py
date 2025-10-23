#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于TensorNet的分子进化预测器 (Linear-Linear版本)

这是一个使用线性组件的TensorNet模型：
- molecule: TensorNetMoleculeFeatureExtractor
- edge: 线性边特征编码器
- fusion: 线性特征融合预测器
用于从分子图中提取特征表示并预测分子演化过程中的属性变化。
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .molecule_feature_extractors.tensornet import TensorNetMoleculeFeatureExtractor
from .fusion_predictors.mlp import MLPFusionPredictor
from .edge_feature_extractors.linear import LinearEdgeFeatureExtractor
from . import register_model


@register_model(display_name="TensorNet Linear-Linear 模型", save_dir_name="tensornet_linear_linear")
class MoleculeEvolutionTensornetLinearPredictor(nn.Module):
    """
    基于TensorNet的分子进化预测器 (Linear-Linear版本)
    
    使用线性组件组合：
    - molecule: TensorNetMoleculeFeatureExtractor (TensorNet特征提取器)
    - edge: LinearEdgeFeatureExtractor (线性边特征编码器)
    - fusion: MLPFusionPredictor (线性特征融合预测器)
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
                 dtype=torch.float32,
                 edge_feature_dim: int = 15,
                 output_dim: int = 1,
                 **kwargs):
        """
        初始化预测器

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
            edge_feature_dim: 边特征维度 (操作信息)
            output_dim: 输出维度 (属性变化)
            **kwargs: 传递给TensorNet特征提取器的其他参数
        """
        super(MoleculeEvolutionTensornetLinearPredictor, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z
        self.max_num_neighbors = max_num_neighbors
        self.equivariance_invariance_group = equivariance_invariance_group
        self.static_shapes = static_shapes
        self.check_errors = check_errors
        self.dtype = dtype
        self.edge_feature_dim = edge_feature_dim
        self.output_dim = output_dim
        
        # molecule组件: TensorNet分子特征提取器
        self.molecule_extractor_from = TensorNetMoleculeFeatureExtractor(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_z=max_z,
            max_num_neighbors=max_num_neighbors,
            equivariance_invariance_group=equivariance_invariance_group,
            static_shapes=static_shapes,
            check_errors=check_errors,
            dtype=dtype
        )
        
        self.molecule_extractor_to = TensorNetMoleculeFeatureExtractor(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_z=max_z,
            max_num_neighbors=max_num_neighbors,
            equivariance_invariance_group=equivariance_invariance_group,
            static_shapes=static_shapes,
            check_errors=check_errors,
            dtype=dtype
        )
        
        # edge组件: 线性边特征编码器
        self.edge_encoder = LinearEdgeFeatureExtractor(edge_feature_dim, 512)
        
        # fusion组件: 线性特征融合预测器
        self.fusion_predictor = MLPFusionPredictor(
            node_dim=512,  # TensorNet输出的是512维特征
            edge_dim=512,
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