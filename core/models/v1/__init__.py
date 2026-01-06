#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型基类和工厂实现 - v1版本
"""

import sys
import torch.nn as nn
from torch_geometric.data import Data
import torch
from collections import defaultdict


# 装饰器注册相关
_model_entrypoints = {}  # mapping of model names to entrypoint fns
_model_to_module = {}  # mapping of model names to module names
_module_to_models = defaultdict(set)  # dict of sets to check membership of model in module
_model_display_names = {}  # mapping of model names to display names
_model_save_dir_names = {}  # mapping of model names to save directory names
_model_requires_position_encoding = {}  # mapping of model names to position encoding requirements


def register_model(name=None, display_name=None, save_dir_name=None, requires_position_encoding=False):
    """注册模型的装饰器
    
    Args:
        name: 模型注册名称，默认为函数/类名
        display_name: 模型显示名称，用于用户界面展示
        save_dir_name: 模型保存目录名称，用于模型文件存储
        requires_position_encoding: 模型是否需要位置编码
    """
    def _register_model(fn):
        # lookup containing module
        mod = sys.modules[fn.__module__]
        module_name_split = fn.__module__.split('.')
        module_name = module_name_split[-1] if len(module_name_split) else ''

        # add model to __all__ in module
        model_name = name if name is not None else fn.__name__
        if hasattr(mod, '__all__'):
            mod.__all__.append(model_name)
        else:
            mod.__all__ = [model_name]

        # add entries to registry dict/sets
        _model_entrypoints[model_name] = fn
        _model_to_module[model_name] = module_name
        _module_to_models[module_name].add(model_name)
        
        # 设置显示名称
        if display_name is not None:
            _model_display_names[model_name] = display_name
        else:
            _model_display_names[model_name] = model_name
            
        # 设置保存目录名称
        if save_dir_name is not None:
            _model_save_dir_names[model_name] = save_dir_name
        else:
            _model_save_dir_names[model_name] = model_name
            
        # 设置位置编码需求
        _model_requires_position_encoding[model_name] = requires_position_encoding
        
        # 同时注册到ModelFactory
        from ..model_factory import ModelFactory
        ModelFactory.register(model_name, fn)
        
        return fn
    
    # 如果register_model()被直接调用（没有参数），name会是被装饰的函数
    if callable(name):
        fn = name
        name = None
        return _register_model(fn)
    
    return _register_model


def model_entrypoint(model_name):
    """获取模型入口函数"""
    return _model_entrypoints[model_name]


def get_model_display_name(model_name):
    """获取模型显示名称"""
    return _model_display_names.get(model_name, model_name)


def get_model_save_dir_name(model_name):
    """获取模型保存目录名称"""
    return _model_save_dir_names.get(model_name, model_name)


def model_requires_position_encoding(model_name):
    """检查模型是否需要位置编码"""
    return _model_requires_position_encoding.get(model_name, False)


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


# 直接导入模型类，它们已经在各自的文件中通过装饰器注册了
try: 
    from .visnet_linear_linear import MoleculeEvolutionVisnetLinearIterativePredictor
    
    # 确保触发模块导入，使装饰器得以执行
    _ = [
        MoleculeEvolutionVisnetLinearIterativePredictor,
    ]
except ImportError as e:
    print(f"❌ 无法导入某些模型模块: {e}. 请确保所有依赖项已安装.")
    exit(1)
