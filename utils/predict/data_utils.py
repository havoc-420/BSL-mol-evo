"""
数据工具模块
包含与数据准备和处理相关的工具函数
"""

import sys
import os
import torch
import pandas as pd
import numpy as np

# 设置项目根目录路径
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

# 导入自定义模块
try:
    from mol_evo.core.data.processing import prepare_edge_features
    from mol_evo.core.utils.molecule import MoleculeCache
    from mol_evo.core.data.data_v0 import smiles_to_graph_data
except ImportError as e:
    print("无法导入自定义模块", e)
    raise


def prepare_single_prediction_data(smiles_from, smiles_to, to_atom_symbol, operation_type, model_dir):
    """
    准备单个预测的数据
    
    Args:
        smiles_from: 起始分子的SMILES
        smiles_to: 目标分子的SMILES
        to_atom_symbol: 变化涉及的原子类型
        operation_type: 操作类型
        model_dir: 模型目录路径
        
    Returns:
        图数据对象和属性统计信息
    """
    # 创建虚拟数据框以复用现有函数
    data_dict = {
        'smiles_from': [smiles_from],
        'smiles_to': [smiles_to],
        'to_atom_symbol': [to_atom_symbol],
        'operation_type': [operation_type]
    }
    df = pd.DataFrame(data_dict)
    
    # 获取属性统计信息
    property_stats = load_property_stats(model_dir)
    
    # 创建分子缓存实例
    cache = MoleculeCache("prediction_dataset")
    
    # 构建起始和目标分子图数据
    from_data = smiles_to_graph_data(smiles_from, cache)
    to_data = smiles_to_graph_data(smiles_to, cache)
    
    if from_data is None or to_data is None:
        raise ValueError("无法将SMILES转换为图数据")
    
    # 构建边特征（不含属性变化）
    edge_feat = prepare_edge_features(df.iloc[0], property_stats, include_property_changes=False)
    edge_attr = torch.FloatTensor(np.array([edge_feat]))
    
    return from_data, to_data, edge_attr, property_stats


# 为避免循环依赖，这里复制load_property_stats函数
def load_property_stats(model_dir):
    """
    从模型目录加载属性统计信息
    
    Args:
        model_dir: 模型目录路径
        
    Returns:
        属性统计信息字典
    """
    from .model_utils import load_property_stats as _load_property_stats
    return _load_property_stats(model_dir)