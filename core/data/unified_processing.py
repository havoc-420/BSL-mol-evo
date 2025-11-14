#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一数据处理模块
整合了各种分子数据处理和预处理函数，用于构建分子进化数据集
"""

import pandas as pd
import numpy as np
import torch
import json
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from typing import List, Tuple, Dict, Optional, Any
from tqdm import tqdm
import os

# 导入必要的本地模块
try:
    from ..utils.molecule import MoleculeCache
    from .data_v0 import smiles_to_graph_data
    from .fragnet_data import smile_to_fragnet_features
    from .processing import (
        smiles_to_fingerprint, 
        atom_type_to_onehot, 
        operation_type_to_onehot,
        prepare_edge_features as processing_prepare_edge_features,
        load_operation_config,
        get_atom_types
    )
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from mol_evo.core.utils.molecule import MoleculeCache
    from mol_evo.core.data.data_v0 import smiles_to_graph_data
    from mol_evo.core.data.fragnet_data import smile_to_fragnet_features
    from mol_evo.core.data.processing import (
        smiles_to_fingerprint, 
        atom_type_to_onehot, 
        operation_type_to_onehot,
        processing_prepare_edge_features,
        load_operation_config,
        get_atom_types
    )

def prepare_edge_features(row: pd.Series, property_stats: Dict[str, Tuple[float, float]] = None,
                         include_property_changes: bool = False) -> List[float]:
    """
    准备边特征向量（重写自processing.py中的同名函数，保持接口一致）
    
    Args:
        row: CSV文件中的一行数据
        property_stats: 属性统计信息（用于标准化）
        include_property_changes: 是否包含属性变化特征
        
    Returns:
        边特征向量
    """
    # 使用processing.py中的实现
    return processing_prepare_edge_features(row, property_stats, include_property_changes)


def load_json_data(json_file: str) -> pd.DataFrame:
    """
    从JSON文件加载数据并转换为DataFrame格式
    
    Args:
        json_file: JSON文件路径
        
    Returns:
        包含分子对数据的DataFrame
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
    # 检查是否是JSON数组格式
    if content.startswith('['):
        data = json.loads(content)
    else:
        # 处理JSON行格式文件
        lines = content.splitlines()
        data = []
        for line in lines:
            if line.strip():  # 跳过空行
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError:
                    # 跳过无效的JSON行
                    continue
    
    return pd.DataFrame(data)


def build_molecule_evolution_dataset_v0(
    data_file: str, 
    max_pairs: int = None, 
    target_property: str = 'mu_change', 
    logger=None,
    model_type: str = "gcn_linear"
) -> Tuple[List[Any], List[Any], torch.Tensor, torch.Tensor, Dict[str, Tuple[float, float]]]:
    """
    构建分子进化数据集，使用smile_to_graph_xyz函数处理分子结构
    参考文档: mol_evo/docs/model-v0/data_preprocessing_and_usage.md
    
    Args:
        data_file: 数据文件路径 (支持CSV和JSON格式)
        max_pairs: 最大对数（用于调试）
        target_property: 目标属性名称
        logger: 日志记录器
        model_type: 模型类型，用于确定数据预处理方式
        
    Returns:
        起始分子数据列表、目标分子数据列表、边特征张量、目标属性张量和属性统计信息
    """
    # 检查是否是 FragNet 模型类型
    is_fragnet_model = model_type and "frag" in model_type.lower()
    # 检查是否是 Equiformer 模型类型
    is_equiformer_model = model_type and "equiformer" in model_type.lower()
    
    # 根据文件扩展名自动选择加载方式
    df = load_json_data(data_file) if data_file.endswith('.json') else pd.read_csv(data_file)
    
    if max_pairs:
        df = df.head(max_pairs)
    
    # 计算目标属性的统计信息
    if target_property in df.columns:
        mean = df[target_property].mean()
        std = df[target_property].std()
        property_stats = {target_property: (mean, std)}
    else:
        property_stats = {target_property: (0.0, 1.0)}
    
    # 创建分子缓存实例，并传入logger
    cache = MoleculeCache(csv_file=data_file, logger=logger)    # UPDATE 避免缓存破坏
    
    # 加载操作和原子类型配置
    load_operation_config(dataset_path=data_file)
    # 定义原子类型映射（用于非FragNet模型）
    atom_types = get_atom_types()
    types = {atom: i for i, atom in enumerate(atom_types)}
    
    # 构建分子数据列表
    from_data_list = []
    to_data_list = []
    edge_attr_list = []
    target_features_list = []  # 添加用于收集目标特征的列表
    
    # 直接逐行处理数据（移除了多线程）
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="构建图数据缓存[v0]"):
        try:
            # TAG smiles data generation
            if is_fragnet_model:
                # 使用 FragNet 数据处理函数
                from_data = smile_to_fragnet_features(row['smiles_from'])
                to_data = smile_to_fragnet_features(row['smiles_to'])
            else:
                # 使用标准的 smiles_to_graph_data 函数
                from_data = smiles_to_graph_data(row['smiles_from'], cache)
                to_data = smiles_to_graph_data(row['smiles_to'], cache)
                
            # 检查数据是否有效
            if from_data is None or to_data is None:
                message = f"跳过第{idx}行分子对: {row['smiles_from']} -> {row['smiles_to']} (数据为None)"
                if logger:
                    logger.warning(message)
                continue
                
            from_data_list.append(from_data)
            to_data_list.append(to_data)
            
            # TAG 准备演化操作边特征 (操作信息特征 Hav)
            edge_feat = prepare_edge_features(row, property_stats, include_property_changes=False)
            edge_attr_list.append(edge_feat)
            
            # 准备目标属性特征
            if target_property in row and not pd.isna(row[target_property]):
                value = row[target_property]
                # 标准化目标属性值
                if target_property in property_stats:
                    mean, std = property_stats[target_property]
                    if std > 0:
                        value = (value - mean) / std
                target_features_list.append([value])
            else:
                target_features_list.append([0.0])
                
        except Exception as e:
            message = f"处理第{idx}行分子对时发生错误: {row['smiles_from']} -> {row['smiles_to']}, 错误: {str(e)}"
            if logger:
                logger.warning(message)
    
    if len(edge_attr_list) > 0:
        edge_attrs = torch.FloatTensor(np.array(edge_attr_list))
    else:
        edge_attrs = torch.FloatTensor([])
        
    # 转换目标特征为张量
    if len(target_features_list) > 0:
        target_features = torch.FloatTensor(np.array(target_features_list))
    else:
        target_features = torch.FloatTensor([])
    
    # 打印缓存统计信息
    stats = cache.get_stats()
    message = f"分子处理统计: 总数={stats['total']}, 命中={stats['hits']}, 未命中={stats['misses']}, 命中率={stats['hit_rate']:.2%}"
    if logger:
        logger.info(message)
    else:
        print(message)
    
    return from_data_list, to_data_list, edge_attrs, target_features, property_stats



def build_molecule_evolution_dataset_unified(
    data_file: str,
    model_type: str = "gcn_linear",
    max_pairs: Optional[int] = None,
    target_property: str = 'mu_change',
    include_property_changes: bool = False,
    logger=None
) -> Tuple[List[Any], List[Any], torch.Tensor, torch.Tensor, Dict[str, Tuple[float, float]]]:
    """
    统一的分子进化数据集构建函数，支持多种模型类型
    
    Args:
        data_file: 数据文件路径 (支持CSV和JSON格式)
        model_type: 模型类型 ("gcn_linear", "frag*", "equiformer*", 等)
        max_pairs: 最大对数（用于调试）
        target_property: 目标属性名称
        include_property_changes: 是否包含属性变化特征
        logger: 日志记录器
        
    Returns:
        起始分子数据列表、目标分子数据列表、边特征张量、目标属性张量和属性统计信息
    """
    # 检查模型类型
    is_fragnet_model = model_type and "frag" in model_type.lower()
    is_equiformer_model = model_type and "equiformer" in model_type.lower()
    
    # 根据文件扩展名自动选择加载方式
    if data_file.endswith('.json'):
        df = load_json_data(data_file)
    else:
        df = pd.read_csv(data_file)
    
    if max_pairs:
        df = df.head(max_pairs)
    
    # 计算属性统计信息用于标准化
    property_names = ['A_change', 'B_change', 'C_change', 'mu_change', 'alpha_change',
                      'homo_change', 'lumo_change', 'gap_change', 'r2_change', 'zpve_change',
                      'U0_change', 'U_change', 'H_change', 'G_change', 'Cv_change']
    
    property_stats = {}
    for prop in property_names:
        if prop in df.columns:
            mean = df[prop].mean()
            std = df[prop].std()
            property_stats[prop] = (mean, std)
    
    # 计算目标属性的统计信息
    if target_property in df.columns:
        mean = df[target_property].mean()
        std = df[target_property].std()
        property_stats[target_property] = (mean, std)
    else:
        property_stats[target_property] = (0.0, 1.0)
    
    # 创建分子缓存实例（仅对非FragNet模型使用）
    cache = None
    if not is_fragnet_model:
        cache = MoleculeCache(csv_file=data_file, logger=logger)
    
    # 加载操作和原子类型配置
    load_operation_config(dataset_path=data_file)
    # 定义原子类型映射（用于非FragNet和非Equiformer模型）
    atom_types = get_atom_types()
    types = {atom: i for i, atom in enumerate(atom_types)}
    
    # 构建分子数据列表
    from_data_list = []
    to_data_list = []
    edge_attr_list = []
    target_features_list = []
    
    # 直接逐行处理数据（移除了多线程）
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="构建图数据缓存[unified]"):
        try:
            # 根据模型类型选择适当的数据处理函数
            if is_fragnet_model:
                # 使用 FragNet 数据处理函数
                from_data = smile_to_fragnet_features(row['smiles_from'])
                to_data = smile_to_fragnet_features(row['smiles_to'])
            elif is_equiformer_model:
                # 对于Equiformer模型，暂时使用指纹方法
                from_fp = smiles_to_fingerprint(row['smiles_from'])
                to_fp = smiles_to_fingerprint(row['smiles_to'])
                from_data = Data(x=torch.FloatTensor(from_fp).unsqueeze(0))
                to_data = Data(x=torch.FloatTensor(to_fp).unsqueeze(0))
            else:
                # 使用标准的 smiles_to_graph_data 函数
                from_data = smiles_to_graph_data(row['smiles_from'], cache)
                to_data = smiles_to_graph_data(row['smiles_to'], cache)
                
            # 检查数据是否有效
            if from_data is None or to_data is None:
                message = f"跳过第{idx}行分子对: {row['smiles_from']} -> {row['smiles_to']} (数据为None)"
                if logger:
                    logger.warning(message)
                continue
                
            from_data_list.append(from_data)
            to_data_list.append(to_data)
            
            # 准备边特征
            edge_feat = prepare_edge_features(row, property_stats, include_property_changes=include_property_changes)
            edge_attr_list.append(edge_feat)
            
            # 准备目标属性特征
            if target_property in row and not pd.isna(row[target_property]):
                value = row[target_property]
                # 标准化目标属性值
                if target_property in property_stats:
                    mean, std = property_stats[target_property]
                    if std > 0:
                        value = (value - mean) / std
                target_features_list.append([value])
            else:
                target_features_list.append([0.0])
                
        except Exception as e:
            message = f"处理第{idx}行分子对时发生错误: {row['smiles_from']} -> {row['smiles_to']}, 错误: {str(e)}"
            if logger:
                logger.warning(message)
    
    if len(edge_attr_list) > 0:
        edge_attrs = torch.FloatTensor(np.array(edge_attr_list))
    else:
        edge_attrs = torch.FloatTensor([])
        
    # 转换目标特征为张量
    if len(target_features_list) > 0:
        target_features = torch.FloatTensor(np.array(target_features_list))
    else:
        target_features = torch.FloatTensor([])
    
    # 打印缓存统计信息（如果使用了缓存）
    if cache is not None:
        stats = cache.get_stats()
        message = f"分子处理统计: 总数={stats['total']}, 命中={stats['hits']}, 未命中={stats['misses']}, 命中率={stats['hit_rate']:.2%}"
        if logger:
            logger.info(message)
        else:
            print(message)
    
    return from_data_list, to_data_list, edge_attrs, target_features, property_stats


# 保持processing.py中的一些有用函数的可用性
__all__ = [
    'smiles_to_fingerprint',
    'atom_type_to_onehot',
    'operation_type_to_onehot',
    'calculate_molecular_similarity',
    'prepare_edge_features',
    'build_molecule_evolution_dataset_v0',
    'build_molecule_evolution_dataset_unified',
]