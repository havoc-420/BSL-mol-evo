#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设备管理工具函数
专门用于处理张量和数据结构在不同设备(CPU/GPU)之间的移动
"""

import torch
from typing import Any, Dict, Tuple, Union


def move_dict_to_device(data_dict: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    将字典中的张量移动到指定设备
    
    Args:
        data_dict: 包含张量的字典
        device: 目标设备
        
    Returns:
        移动到指定设备后的字典
    """
    if not isinstance(data_dict, dict):
        return data_dict
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value 
            for key, value in data_dict.items()}


def move_data_to_device(
    from_batch: Union[Dict, list, torch.Tensor], 
    to_batch: Union[Dict, list, torch.Tensor],
    edge_batch: torch.Tensor, 
    target_batch: torch.Tensor,
    device: torch.device, 
    is_fragnet_model: bool
) -> Tuple[Any, Any, torch.Tensor, torch.Tensor]:
    """
    将数据移动到指定设备的统一函数
    
    Args:
        from_batch: 起始分子数据
        to_batch: 目标分子数据
        edge_batch: 边数据
        target_batch: 目标数据
        device: 目标设备
        is_fragnet_model: 是否为FragNet模型
        
    Returns:
        移动到设备后的数据元组 (from_batch, to_batch, edge_batch, target_batch)
    """
    if is_fragnet_model:
        # FragNet 模型的数据处理
        edge_batch = edge_batch.to(device)
        target_batch = target_batch.to(device)
        
        if isinstance(from_batch, list):
            from_batch = [move_dict_to_device(item, device) for item in from_batch]
        else:
            from_batch = move_dict_to_device(from_batch, device)
            
        if isinstance(to_batch, list):
            to_batch = [move_dict_to_device(item, device) for item in to_batch]
        else:
            to_batch = move_dict_to_device(to_batch, device)
    else:
        # 标准模型需要移动所有数据到设备
        from_batch = from_batch.to(device)
        to_batch = to_batch.to(device)
        edge_batch = edge_batch.to(device)
        target_batch = target_batch.to(device)
        
    return from_batch, to_batch, edge_batch, target_batch


def move_data_to_device_for_validation(
    from_batch: Union[Dict, list, torch.Tensor], 
    to_batch: Union[Dict, list, torch.Tensor],
    edge_batch: torch.Tensor, 
    target_batch: torch.Tensor,
    device: torch.device, 
    is_fragnet_model: bool
) -> Tuple[Any, Any, torch.Tensor, torch.Tensor]:
    """
    将验证数据移动到指定设备的函数
    
    Args:
        from_batch: 起始分子数据
        to_batch: 目标分子数据
        edge_batch: 边数据
        target_batch: 目标数据
        device: 目标设备
        is_fragnet_model: 是否为FragNet模型
        
    Returns:
        移动到设备后的数据元组 (from_batch, to_batch, edge_batch, target_batch)
    """
    if is_fragnet_model:
        # FragNet 模型的数据处理
        edge_batch = edge_batch.to(device)
        target_batch = target_batch.to(device)
        
        if isinstance(from_batch, list):
            from_batch = [move_dict_to_device(item, device) for item in from_batch]
        else:
            from_batch = move_dict_to_device(from_batch, device)
            
        if isinstance(to_batch, list):
            to_batch = [move_dict_to_device(item, device) for item in to_batch]
        else:
            to_batch = move_dict_to_device(to_batch, device)
    else:
        # 标准模型需要移动所有数据到设备
        from_batch = from_batch.to(device)
        to_batch = to_batch.to(device)
        edge_batch = edge_batch.to(device)
        target_batch = target_batch.to(device)
        
    return from_batch, to_batch, edge_batch, target_batch


def move_data_to_device_for_testing(
    from_batch: Union[Dict, list, torch.Tensor], 
    to_batch: Union[Dict, list, torch.Tensor],
    edge_batch: torch.Tensor, 
    device: torch.device, 
    is_fragnet_model: bool
) -> Tuple[Any, Any, torch.Tensor]:
    """
    将测试数据移动到指定设备的函数
    
    Args:
        from_batch: 起始分子数据
        to_batch: 目标分子数据
        edge_batch: 边数据
        device: 目标设备
        is_fragnet_model: 是否为FragNet模型
        
    Returns:
        移动到设备后的数据元组 (from_batch, to_batch, edge_batch)
    """
    if is_fragnet_model:
        # FragNet 模型的数据处理
        edge_batch = edge_batch.to(device)
        
        if isinstance(from_batch, list):
            from_batch = [move_dict_to_device(item, device) for item in from_batch]
        else:
            from_batch = move_dict_to_device(from_batch, device)
            
        if isinstance(to_batch, list):
            to_batch = [move_dict_to_device(item, device) for item in to_batch]
        else:
            to_batch = move_dict_to_device(to_batch, device)
    else:
        # 标准模型需要移动所有数据到设备
        from_batch = from_batch.to(device)
        to_batch = to_batch.to(device)
        edge_batch = edge_batch.to(device)
        
    return from_batch, to_batch, edge_batch