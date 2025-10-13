"""
分子演化模型v0版本的数据处理模块
"""

import pandas as pd
import numpy as np
from torch_geometric.data import Data
from typing import List, Tuple, Dict


def smiles_to_graph_data(smiles, cache):
    """
    将SMILES字符串转换为图数据（使用项目中的实际函数）
    
    Args:
        smiles (str): SMILES字符串
        cache (MoleculeCache): 分子缓存实例
    
    Returns:
        Data: PyTorch Geometric Data对象
    """
    # 定义原子类型映射
    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    
    # 使用项目中的函数将SMILES转换为图结构
    x, z, pos, edge_index, edge_attr = cache.process_smiles(smiles, types)
    
    # 检查转换是否成功
    if x is None:
        print(f"无法处理SMILES: {smiles}")
        return None
    
    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def atom_type_to_onehot(atom_symbol: str) -> List[int]:
    """
    将原子类型转换为独热编码
    
    Args:
        atom_symbol: 原子符号
        
    Returns:
        独热编码向量
    """
    atom_types = ['C', 'N', 'O', 'F', 'P']  # 常见原子类型
    onehot = [0] * len(atom_types)
    
    if atom_symbol in atom_types:
        idx = atom_types.index(atom_symbol)
        onehot[idx] = 1
        
    return onehot


def operation_type_to_onehot(operation_type: str) -> List[int]:
    """
    将操作类型转换为独热编码
    
    Args:
        operation_type: 操作类型
        
    Returns:
        独热编码向量
    """
    op_types = ['add', 'replace', 'del', 'add_multi', 'del_multi', 'complex']
    onehot = [0] * len(op_types)
    
    if operation_type in op_types:
        idx = op_types.index(operation_type)
        onehot[idx] = 1
        
    return onehot


def prepare_edge_features(row: pd.Series, property_stats: Dict[str, Tuple[float, float]] = None, 
                         include_property_changes: bool = False) -> List[float]:
    """
    准备边特征向量
    
    Args:
        row: CSV文件中的一行数据
        property_stats: 属性统计信息（用于标准化）
        include_property_changes: 是否包含属性变化特征
        
    Returns:
        边特征向量
    """
    # 原子类型特征（5维）
    atom_features = atom_type_to_onehot(row['to_atom_symbol'] if 'to_atom_symbol' in row else '')
    
    # 操作类型特征（6维）
    op_features = operation_type_to_onehot(row['operation_type'] if 'operation_type' in row else 'unknown')

    # TODO 还有一个 op-position 这个特征可以加上
    
    # 属性变化特征（15维，可选）
    property_changes = []
    if include_property_changes:
        property_names = ['A_change', 'B_change', 'C_change', 'mu_change', 'alpha_change',
                          'homo_change', 'lumo_change', 'gap_change', 'r2_change', 'zpve_change',
                          'U0_change', 'U_change', 'H_change', 'G_change', 'Cv_change']

        for prop in property_names:
            if prop in row and not pd.isna(row[prop]):
                value = row[prop]
                # 标准化属性变化值
                if property_stats and prop in property_stats:
                    mean, std = property_stats[prop]
                    if std > 0:
                        value = (value - mean) / std
                property_changes.append(value)
            else:
                property_changes.append(0.0)
    
    # 组合所有特征
    if include_property_changes:
        edge_features = atom_features + op_features + property_changes
    else:
        edge_features = atom_features + op_features
    
    return edge_features