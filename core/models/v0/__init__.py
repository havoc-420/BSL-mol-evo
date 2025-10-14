#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型基类和工厂实现
"""

import torch.nn as nn
from torch_geometric.data import Data
import torch


class BaseMoleculeEvolutionPredictor(nn.Module):
    """
    分子进化预测器的基类
    """
    
    def __init__(self):
        super(BaseMoleculeEvolutionPredictor, self).__init__()
    
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
        raise NotImplementedError("子类必须实现 forward 方法")


class ModelFactory:
    """
    模型工厂类，用于注册和创建不同的模型
    """
    
    _models = {}
    
    @classmethod
    def register(cls, name, model_class):
        """
        注册模型类
        
        Args:
            name: 模型名称
            model_class: 模型类
        """
        cls._models[name] = model_class
    
    @classmethod
    def create(cls, name, **kwargs):
        """
        创建模型实例
        
        Args:
            name: 模型名称
            **kwargs: 模型初始化参数
            
        Returns:
            模型实例
        """
        if name not in cls._models:
            raise ValueError(f"未知的模型类型: {name}。可用模型: {list(cls._models.keys())}")
        
        model_class = cls._models[name]
        return model_class(**kwargs)
    
    @classmethod
    def list_models(cls):
        """
        列出所有可用的模型
        
        Returns:
            模型名称列表
        """
        return list(cls._models.keys())


try: 
    from .gcn_linear_linear import MoleculeEvolutionGCNLinearPredictor
    from .gcn_transformer_transformer import MoleculeEvolutionGCNTransformerPredictor
    from .visnet_linear_linear import MoleculeEvolutionVisnetLinearPredictor
except ImportError as e:
    print(f"❌ 无法导入某些模型模块: {e}. 请确保所有依赖项已安装.")
    exit(1)

ModelFactory.register("gcn_linear_linear", MoleculeEvolutionGCNLinearPredictor)
ModelFactory.register("gcn_transformer_transformer", MoleculeEvolutionGCNTransformerPredictor)
ModelFactory.register("visnet_linear_linear", MoleculeEvolutionVisnetLinearPredictor)