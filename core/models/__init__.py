# -*- coding: utf-8 -*-
"""Molecule evolution models package."""

import os
import sys
import warnings
from collections import defaultdict
from .v0 import register_model as register_model_v0

# 全局注册表
_model_entrypoints = {}
_model_to_module = {}
_module_to_models = defaultdict(set)
_model_display_names = {}
_model_save_dir_names = {}

# 注册v0版本模型
from .v0 import *  # noqa: F403, F401

# 注册v0.1版本模型
from .v0_1 import *  # noqa: F403, F401

# 注册v0.2版本模型
from .v0_2 import *  # noqa: F403, F401

# 注册v1版本模型
# from .v1 import *  # noqa: F403, F401 # INFO


def register_model(name=None, display_name=None, save_dir_name=None):
    """
    注册模型的装饰器函数
    
    Args:
        name: 模型名称
        display_name: 模型显示名称
        save_dir_name: 模型保存目录名称
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
        
        # 同时注册到ModelFactory
        # 注意：这里需要根据模型版本选择不同的注册方式
        # 当前所有模型都使用v0的注册方式
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


# 为了向后兼容，导出v0版本的BaseMoleculeEvolutionPredictor
from .v0 import BaseMoleculeEvolutionPredictor  # noqa: F401

__all__ = [
    'register_model',
    'model_entrypoint',
    'get_model_display_name',
    'get_model_save_dir_name',
    'BaseMoleculeEvolutionPredictor',
]