#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件工具函数
用于解析YAML格式的模型配置文件
"""

import os
import yaml


def load_model_config(config_name: str, config_dir: str = None) -> dict:
    """
    加载模型配置文件
    
    Args:
        config_name: 配置文件名（不包含.yaml后缀）
        config_dir: 配置文件目录，默认为项目根目录下的configs文件夹
        
    Returns:
        dict: 配置参数字典
        
    Raises:
        FileNotFoundError: 当配置文件不存在时
        yaml.YAMLError: 当YAML文件格式错误时
    """
    # 如果没有指定配置目录，则使用默认路径
    if config_dir is None:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建默认配置目录路径
        config_dir = os.path.join(current_dir, '..', 'configs')
    
    # 构建配置文件完整路径
    config_path = os.path.join(config_dir, f'{config_name}.yaml')
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 加载YAML配置文件
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config if config is not None else {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML文件格式错误: {config_path}\n{str(e)}")


def get_model_config_type(model_type: str) -> str:
    """
    根据模型类型获取对应的配置文件名
    
    Args:
        model_type: 模型类型字符串
        
    Returns:
        str: 对应的配置文件名（不包含.yaml后缀）
    """
    # 检查是否是 FragNet 模型类型
    if model_type and "frag" in model_type.lower():
        return "fragnet"
    
    # 检查是否是 Equiformer 模型类型
    if model_type and "equiformer" in model_type.lower():
        return "equiformer"
    
    # 默认返回基础配置
    return "base"


def load_config_by_model_type(model_type: str, config_dir: str = None) -> dict:
    """
    根据模型类型加载对应的配置文件
    
    Args:
        model_type: 模型类型字符串
        config_dir: 配置文件目录，默认为项目根目录下的configs文件夹
        
    Returns:
        dict: 配置参数字典
    """
    config_name = get_model_config_type(model_type)
    return load_model_config(config_name, config_dir)