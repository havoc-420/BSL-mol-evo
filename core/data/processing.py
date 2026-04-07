#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子数据处理工具
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from typing import List, Tuple, Dict
import yaml
import os

# 添加networkx导入以支持图操作
import networkx as nx

# 全局配置变量
_OPERATION_TYPES = None
_ATOM_TYPES = None


def load_operation_config(config_path: str = None, dataset_path: str = None):
    """
    从YAML配置文件加载操作类型和原子类型
    
    Args:
        config_path: 配置文件路径
        dataset_path: 数据集路径，用于生成同步的配置文件名
    """
    global _OPERATION_TYPES, _ATOM_TYPES
    
    # 如果没有指定配置文件路径，则根据数据集路径生成
    if config_path is None and dataset_path is not None:
        # 获取数据集文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(dataset_path))[0]
        # 生成对应的配置文件路径
        config_path = os.path.join(os.path.dirname(dataset_path), f"{base_name}-config.yaml")

    # 如果没有指定配置文件路径，尝试使用默认配置文件
    if config_path is None:
        # 尝试在项目中查找默认配置文件
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_config_path = os.path.join(script_dir, '..', '..', 'dataset', 'configs', 'default_config.yaml')
        if os.path.exists(default_config_path):
            config_path = default_config_path
        else:
            # 如果还找不到配置文件，则创建并使用一个基本配置
            _OPERATION_TYPES = ['add_atom', 'delete_atom', 'change_atom', 'add_bond', 'delete_bond', 'change_bond']
            _ATOM_TYPES = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
            print("使用默认操作类型和原子类型配置")
            return
    
    if not os.path.exists(config_path):
        # 如果配置文件不存在，则创建并使用基本配置
        _OPERATION_TYPES = ['add_atom', 'delete_atom', 'change_atom', 'add_bond', 'delete_bond', 'change_bond']
        _ATOM_TYPES = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        print("配置文件未找到，使用默认操作类型和原子类型配置")
        # 创建默认配置文件
        default_config = {
            'operation_types': _OPERATION_TYPES,
            'atom_types': _ATOM_TYPES
        }
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
        print(f"已创建默认配置文件: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    _OPERATION_TYPES = config['operation_types']
    _ATOM_TYPES = config['atom_types']
    
    print(f"😀 已加载 {len(_OPERATION_TYPES)} 种操作类型和 {len(_ATOM_TYPES)} 种原子类型")


def get_operation_types() -> List[str]:
    """
    获取操作类型列表
    
    Returns:
        操作类型列表
    """
    global _OPERATION_TYPES
    if _OPERATION_TYPES is None:
        raise RuntimeError("操作类型未初始化，请先调用 load_operation_config()")
    return _OPERATION_TYPES


def get_atom_types() -> List[str]:
    """
    获取原子类型列表
    
    Returns:
        原子类型列表
    """
    global _ATOM_TYPES
    if _ATOM_TYPES is None:
        raise RuntimeError("原子类型未初始化，请先调用 load_operation_config()")
    return _ATOM_TYPES


def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    将SMILES转换为Morgan指纹
    
    Args:
        smiles: SMILES字符串
        radius: Morgan指纹半径
        n_bits: 指纹位数
        
    Returns:
        指纹数组
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        
        # 生成Morgan指纹，使用新的API避免弃用警告
        try:
            # 尝试使用新的MorganGenerator API
            generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
            fingerprint = generator.GetFingerprint(mol)
            return np.array(fingerprint)
        except AttributeError:
            # 如果新API不可用，回退到旧API
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return np.array(fingerprint)
    except Exception:
        return np.zeros(n_bits)


def atom_type_to_onehot(atom_symbol: str) -> List[int]:
    """
    将原子类型转换为独热编码
    
    Args:
        atom_symbol: 原子符号
        
    Returns:
        独热编码向量
    """
    atom_types = get_atom_types()
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
    op_types = get_operation_types()
    onehot = [0] * len(op_types)
    
    if operation_type in op_types:
        idx = op_types.index(operation_type)
        onehot[idx] = 1
        
    return onehot


def calculate_molecular_similarity(smiles1: str, smiles2: str) -> float:
    """
    计算两个分子的Tanimoto相似度

    Args:
        smiles1: 第一个分子的SMILES
        smiles2: 第二个分子的SMILES

    Returns:
        Tanimoto相似度 (0-1)
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return 0.0

        # 计算Morgan指纹相似度，使用新的API避免弃用警告
        try:
            # 尝试使用新的MorganGenerator API (RDKit 2023.09+)
            generator = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
            fp1 = generator.GetFingerprint(mol1)
            fp2 = generator.GetFingerprint(mol2)
        except AttributeError:
            # 如果新API不可用，回退到旧API
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)

        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


def get_laplacian_pe_from_smiles(smiles: str, k: int = 8, target_position: int = None) -> np.ndarray:
    """
    从SMILES获取拉普拉斯位置编码，并特别标记目标位置
    
    Args:
        smiles: SMILES字符串
        k: 位置编码维度
        target_position: 要特别标记的目标原子位置（从0开始）
        
    Returns:
        增强的位置编码数组
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    # 构建分子图
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx())
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        G.add_edge(i, j)
    
    # INFO 计算归一化拉普拉斯矩阵
    L = nx.normalized_laplacian_matrix(G).astype(float)
    
    # 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(L.toarray())
    
    # 选择最小的k个非零特征值对应的特征向量
    valid_indices = np.where(eigenvalues > 1e-8)[0]
    if len(valid_indices) == 0:
        # 如果所有特征值都很小，使用随机编码
        pe_vectors = np.random.normal(0, 0.1, (len(G.nodes()), k))
    else:
        k_actual = min(k, len(valid_indices))
        selected_indices = valid_indices[:k_actual]
        pe_vectors = eigenvectors[:, selected_indices]
        
        # 如果维度不够，用零填充
        if pe_vectors.shape[1] < k:
            padding = np.zeros((pe_vectors.shape[0], k - pe_vectors.shape[1]))
            pe_vectors = np.hstack([pe_vectors, padding])
    
    # 添加目标位置标记
    if target_position is not None and target_position < len(G.nodes()):
        # 二进制标记法
        position_marker = np.zeros((len(G.nodes()), 1))
        position_marker[target_position] = 1.0
        pe_vectors = np.hstack([pe_vectors, position_marker])
    
    return pe_vectors


def get_laplacian_pe_for_multiple_positions(smiles: str, k: int = 8, target_positions: List[int] = None) -> np.ndarray:
    """
    从SMILES获取拉普拉斯位置编码，并标记多个目标位置
    
    Args:
        smiles: SMILES字符串
        k: 位置编码维度
        target_positions: 要特别标记的目标原子位置列表（从0开始）
        
    Returns:
        增强的位置编码数组
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    # 构建分子图
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx())
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        G.add_edge(i, j)
    
    # 计算归一化拉普拉斯矩阵
    L = nx.normalized_laplacian_matrix(G).astype(float)
    
    # 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(L.toarray())
    
    # 选择最小的k个非零特征值对应的特征向量
    valid_indices = np.where(eigenvalues > 1e-8)[0]
    if len(valid_indices) == 0:
        # 如果所有特征值都很小，使用随机编码
        pe_vectors = np.random.normal(0, 0.1, (len(G.nodes()), k))
    else:
        k_actual = min(k, len(valid_indices))
        selected_indices = valid_indices[:k_actual]
        pe_vectors = eigenvectors[:, selected_indices]
        
        # 如果维度不够，用零填充
        if pe_vectors.shape[1] < k:
            padding = np.zeros((pe_vectors.shape[0], k - pe_vectors.shape[1]))
            pe_vectors = np.hstack([pe_vectors, padding])
    
    # 添加目标位置标记
    position_marker = np.zeros((len(G.nodes()), 1))
    if target_positions:
        for pos in target_positions:
            if 0 <= pos < len(G.nodes()):
                position_marker[pos] = 1.0
    
    pe_vectors = np.hstack([pe_vectors, position_marker])
    
    return pe_vectors


def parse_position_string(position_str: str) -> List[int]:
    """
    解析位置字符串，支持多种格式：
    - "0" -> [0] (单个位置)
    - "0-1" -> [0, 1] (两个特定位置)
    - "0,1,2" -> [0, 1, 2] (逗号分隔的多个位置)
    - "0-2" -> [0, 2] (两个特定位置，不是范围)
    
    Args:
        position_str: 位置字符串
        
    Returns:
        位置整数列表
    """
    if not position_str or pd.isna(position_str):
        return []
    
    position_str = str(position_str).strip()
    
    # 处理连字符分隔格式 "0-1" 或 "0-2" (表示两个特定位置)
    if '-' in position_str and ',' not in position_str:
        try:
            # 直接分割，不处理范围
            positions = [int(x.strip()) for x in position_str.split('-')]
            return positions
        except ValueError:
            return []
    
    # 处理逗号分隔格式 "0,1,2"
    elif ',' in position_str:
        try:
            return [int(x.strip()) for x in position_str.split(',')]
        except ValueError:
            return []
    
    # 处理单个位置 "0"
    else:
        try:
            return [int(position_str)]
        except ValueError:
            return []


def parse_complex_position_string(position_str: str) -> List[int]:
    """
    解析复杂的位置字符串格式，支持混合分隔符
    
    Args:
        position_str: 位置字符串，如 "0-1,2-3"
        
    Returns:
        位置整数列表
    """
    if not position_str or pd.isna(position_str):
        return []
    
    position_str = str(position_str).strip()
    all_positions = []
    
    # 先按逗号分割
    parts = position_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # 处理连字符分隔的部分
            sub_parts = part.split('-')
            try:
                positions = [int(x.strip()) for x in sub_parts]
                all_positions.extend(positions)
            except ValueError:
                continue
        else:
            # 处理单个位置
            try:
                all_positions.append(int(part))
            except ValueError:
                continue
    
    # 去重并排序
    return sorted(set(all_positions))


def prepare_position_encoding_features(row: pd.Series, pe_dim: int = 8) -> List[float]:
    """
    准备位置编码特征向量（分离的位置编码）
    
    Args:
        row: CSV文件中的一行数据
        pe_dim: 位置编码维度
        
    Returns:
        位置编码特征向量
    """
    # 从JSON数据中提取操作信息
    if 'operations' in row and isinstance(row['operations'], list) and len(row['operations']) > 0:
        operation = row['operations'][0]  # 取第一个操作
        position_str = operation.get('position', '')  # 获取操作位置字符串
    else:
        # 使用CSV数据中的操作信息
        position_str = ''
    
    # 解析位置字符串 - 使用增强的解析函数
    if ',' in position_str and '-' in position_str:
        # 复杂格式如 "0-1,2-3"
        target_positions = parse_complex_position_string(position_str)
    else:
        # 简单格式
        target_positions = parse_position_string(position_str)
    
    # 位置编码特征
    position_features = []
    if len(target_positions) > 0:
        try:
            smiles_from = row['smiles_from']
            
            # 使用多位置版本的位置编码
            pe = get_laplacian_pe_for_multiple_positions(smiles_from, k=pe_dim, target_positions=target_positions)
            
            if pe is not None and len(pe) > 0:
                # 对于多位置操作，我们取所有目标位置编码的平均值
                target_pe_list = []
                for pos in target_positions:
                    if 0 <= pos < len(pe):
                        target_pe_list.append(pe[pos])
                    else:
                        target_pe_list.append(np.zeros(pe.shape[1]))
                
                # 计算平均位置编码
                if target_pe_list:
                    avg_target_pe = np.mean(target_pe_list, axis=0)
                    position_features = avg_target_pe.tolist()
                else:
                    position_features = [0.0] * (pe_dim + 1)
            else:
                position_features = [0.0] * (pe_dim + 1)
        except Exception as e:
            print(f"生成位置编码时出错: {e}")
            position_features = [0.0] * (pe_dim + 1)
    else:
        position_features = [0.0] * (pe_dim + 1)
    
    return position_features


def _row_get(row: Any, key: str, default=None):
    """兼容 `dict` / `pd.Series` 两种输入的轻量读取函数。"""
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[key] if key in row else default
    except Exception:
        return default


def _small_vocab_onehot(value: Any, vocab: List[str]) -> List[float]:
    onehot = [0.0] * len(vocab)
    if value in vocab:
        onehot[vocab.index(value)] = 1.0
    return onehot


def prepare_semantic_step_features(row: Any,
                                   property_stats: Dict[str, Tuple[float, float]] = None,
                                   include_property_changes: bool = False) -> List[float]:
    """
    准备 `semantic_step` 的最小可用特征向量。

    第一版只编码语义层级、annotation 状态、fragment-op 核心类型、
    connection / topology 以及若干 trace summary 标量，先为 `04` 提供
    一个稳定的入口，不改变现有训练主链。
    """
    semantic_step = _row_get(row, 'semantic_step', {}) or {}
    fragment_op = semantic_step.get('fragment_op') or _row_get(row, 'fragment_op', {}) or {}

    semantic_level = semantic_step.get('semantic_level', _row_get(row, 'semantic_level', 'atomic_fallback'))
    annotation_status = semantic_step.get('annotation_status', _row_get(row, 'annotation_status', 'unresolved'))
    annotation_confidence = semantic_step.get('annotation_confidence', _row_get(row, 'annotation_confidence', 0.0))
    annotation_confidence = float(annotation_confidence or 0.0)

    primitive_ops = semantic_step.get('primitive_ops') or _row_get(row, 'primitive_ops', []) or []
    primitive_span = semantic_step.get('primitive_span')
    if primitive_span is None and primitive_ops:
        primitive_span = [0, max(0, len(primitive_ops) - 1)]

    connection = fragment_op.get('connection') or {}
    constraints = fragment_op.get('constraints') or {}
    anchor = fragment_op.get('anchor') or {}
    fragment = fragment_op.get('fragment') or {}
    leaving_group = fragment_op.get('leaving_group') or {}

    span_length = 0.0
    if isinstance(primitive_span, list) and len(primitive_span) == 2:
        try:
            span_length = float(max(0, int(primitive_span[1]) - int(primitive_span[0]) + 1))
        except Exception:
            span_length = 0.0

    semantic_level_vocab = ['fragment', 'atomic_fallback']
    annotation_status_vocab = ['resolved', 'approximate', 'unresolved']
    fragment_op_vocab = [
        'attach_fragment',
        'replace_substituent',
        'grow_r_group',
        'bioisostere_swap',
        'delete_fragment',
    ]
    bond_type_vocab = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'none']
    topology_vocab = ['adds_branch', 'replaces_branch', 'extends_chain', 'forms_ring', 'breaks_ring', 'removes_branch']

    semantic_level_features = _small_vocab_onehot(semantic_level, semantic_level_vocab)
    annotation_status_features = _small_vocab_onehot(annotation_status, annotation_status_vocab)
    fragment_op_features = _small_vocab_onehot(fragment_op.get('op_type'), fragment_op_vocab)
    bond_type_features = _small_vocab_onehot(connection.get('bond_type'), bond_type_vocab)
    topology_features = _small_vocab_onehot(connection.get('topology_change'), topology_vocab)

    scalar_features = [
        1.0 if fragment_op else 0.0,
        float(len(primitive_ops)),
        span_length,
        float(len(anchor.get('anchor_atom_indices', []) or [])),
        float(fragment.get('fragment_size') or 0.0),
        float(leaving_group.get('leaving_group_size') or 0.0),
        1.0 if constraints.get('scaffold_preserving') else 0.0,
        1.0 if constraints.get('rgroup_only') else 0.0,
        1.0 if constraints.get('ring_change') else 0.0,
        1.0 if constraints.get('charge_change') else 0.0,
        1.0 if constraints.get('valence_safe') else 0.0,
        annotation_confidence,
    ]

    # 暂不在 edge 特征里拼接 property change，避免把标签信息直接泄漏到输入。
    _ = property_stats, include_property_changes

    return (
        semantic_level_features
        + annotation_status_features
        + fragment_op_features
        + bond_type_features
        + topology_features
        + scalar_features
    )


# INFO 准备边特征向量
def prepare_edge_features_with_position(row: pd.Series, property_stats: Dict[str, Tuple[float, float]] = None, 
                                       include_property_changes: bool = False, 
                                       include_position_encoding: bool = True,
                                       pe_dim: int = 8) -> List[float]:
    """
    准备边特征向量（包含位置编码）
    
    Args:
        row: CSV文件中的一行数据
        property_stats: 属性统计信息（用于标准化）
        include_property_changes: 是否包含属性变化特征
        include_position_encoding: 是否包含位置编码
        pe_dim: 位置编码维度
        
    Returns:
        边特征向量
    """
    # 从JSON数据中提取操作信息
    if 'operations' in row and isinstance(row['operations'], list) and len(row['operations']) > 0:
        operation = row['operations'][0]  # 取第一个操作    # UPDATE 如何支持多步操作；
        atom_symbol = operation.get('atom', '')
        operation_type = operation.get('operation', 'unknown')
        position_str = operation.get('position', '')  # 获取操作位置字符串
    else:
        # 使用CSV数据中的操作信息
        atom_symbol = row['to_atom_symbol'] if 'to_atom_symbol' in row else ''
        operation_type = row['operation_type'] if 'operation_type' in row else 'unknown'
        position_str = ''
    
    # 解析位置字符串 - 使用增强的解析函数
    if ',' in position_str and '-' in position_str:
        # 复杂格式如 "0-1,2-3"
        target_positions = parse_complex_position_string(position_str)
    else:
        # 简单格式
        target_positions = parse_position_string(position_str)
    
    # 原子类型特征
    atom_features = atom_type_to_onehot(atom_symbol)
    
    # 操作类型特征
    op_features = operation_type_to_onehot(operation_type)
    
    # 位置编码特征
    position_features = []
    if include_position_encoding and len(target_positions) > 0:
        try:
            smiles_from = row['smiles_from']
            
            # 使用多位置版本的位置编码
            pe = get_laplacian_pe_for_multiple_positions(smiles_from, k=pe_dim, target_positions=target_positions)
            
            if pe is not None and len(pe) > 0:
                # 对于多位置操作，我们取所有目标位置编码的平均值
                target_pe_list = []
                for pos in target_positions:
                    if 0 <= pos < len(pe):
                        target_pe_list.append(pe[pos])
                    else:
                        target_pe_list.append(np.zeros(pe.shape[1]))
                
                # 计算平均位置编码
                if target_pe_list:
                    avg_target_pe = np.mean(target_pe_list, axis=0)
                    position_features = avg_target_pe.tolist()
                else:
                    position_features = [0.0] * (pe_dim + 1)
            else:
                position_features = [0.0] * (pe_dim + 1)
        except Exception as e:
            print(f"生成位置编码时出错: {e}")
            position_features = [0.0] * (pe_dim + 1)
    elif include_position_encoding:
        # include_position_encoding=True但没有有效位置时，仍然添加占位符
        position_features = [0.0] * (pe_dim + 1)
    # 当include_position_encoding=False时不添加任何位置特征
    
    # 属性变化特征（可选）【无用】
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
        if include_position_encoding:
            edge_features = atom_features + op_features + position_features + property_changes
        else:
            edge_features = atom_features + op_features + property_changes
    else:
        if include_position_encoding:
            edge_features = atom_features + op_features + position_features
        else:
            edge_features = atom_features + op_features
    
    return edge_features


def prepare_edge_features(row: pd.Series, property_stats: Dict[str, Tuple[float, float]] = None, 
                         include_property_changes: bool = False,
                         include_position_encoding: bool = True,
                         pe_dim: int = 8) -> List[float]:
    """
    准备边特征向量
    
    Args:
        row: CSV文件中的一行数据
        property_stats: 属性统计信息（用于标准化）
        include_property_changes: 是否包含属性变化特征
        include_position_encoding: 是否包含位置编码
        pe_dim: 位置编码维度
        
    Returns:
        边特征向量
    """
    # 调用带位置编码的新函数，并默认启用位置编码
    return prepare_edge_features_with_position(
        row, 
        property_stats, 
        include_property_changes, 
        include_position_encoding,
        pe_dim=pe_dim
    )


def load_qm9_properties() -> Dict[str, np.ndarray]:
    """
    加载所有QM9分子的原始属性

    Returns:
        SMILES到属性向量的映射
    """
    smiles_to_properties = {}

    # QM9属性列名
    qm9_property_columns = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
                           'U0', 'U', 'H', 'G', 'Cv', 'A', 'B', 'C']

    # 加载所有重原子数的QM9数据文件
    for heavy_atoms in range(1, 10):
        file_path = f"mol_evo/dataset/data/qm9_smiles_heavy_{heavy_atoms}_atoms.csv"
        try:
            df = pd.read_csv(file_path)

            for _, row in df.iterrows():
                smiles = row['smiles']
                properties = []

                # 提取15个量子化学属性
                for prop in qm9_property_columns:
                    if prop in row and not pd.isna(row[prop]):
                        properties.append(row[prop])
                    else:
                        properties.append(0.0)

                smiles_to_properties[smiles] = np.array(properties, dtype=np.float32)

        except FileNotFoundError:
            print(f"警告: 找不到文件 {file_path}")
            continue

    return smiles_to_properties


def build_molecule_graph_with_properties(csv_file: str, max_molecules: int = None) -> Tuple[Data, Dict[str, int]]:
    """
    从CSV文件构建分子进化图（使用QM9原始属性作为节点特征）

    Args:
        csv_file: CSV文件路径
        max_molecules: 最大分子数（用于调试）

    Returns:
        图数据和SMILES到索引的映射
    """
    # 读取数据
    df = pd.read_csv(csv_file)

    if max_molecules:
        df = df.head(max_molecules)

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

    # 收集所有唯一的SMILES
    all_smiles = set(df['smiles_from'].tolist() + df['smiles_to'].tolist())
    smiles_to_idx = {smiles: idx for idx, smiles in enumerate(all_smiles)}

    # 加载QM9原始属性
    print("正在加载QM9原始属性...")
    qm9_properties = load_qm9_properties()
    print(f"已加载 {len(qm9_properties)} 个分子的属性")

    # 构建节点特征 (使用QM9原始属性)
    num_nodes = len(all_smiles)
    node_features = []
    missing_smiles = []

    for smiles in all_smiles:
        if smiles in qm9_properties:
            properties = qm9_properties[smiles]
        else:
            # 如果找不到原始属性，使用默认值
            properties = np.zeros(15, dtype=np.float32)
            missing_smiles.append(smiles)
        node_features.append(properties)

    if missing_smiles:
        print(f"警告: 找不到 {len(missing_smiles)} 个分子的QM9属性")

    node_features = torch.FloatTensor(np.array(node_features))

    # 构建边索引和边特征
    edge_indices = []
    edge_features = []

    for _, row in df.iterrows():
        src_idx = smiles_to_idx[row['smiles_from']]
        dst_idx = smiles_to_idx[row['smiles_to']]

        edge_indices.append([src_idx, dst_idx])
        edge_feat = prepare_edge_features(row, property_stats)
        edge_features.append(edge_feat)

    edge_index = torch.LongTensor(edge_indices).t().contiguous()
    edge_attr = torch.FloatTensor(np.array(edge_features))

    # 创建图数据对象
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    print(f"图构建完成: {num_nodes} 个节点, {len(edge_indices)} 条边")

    return data, smiles_to_idx


def build_molecule_graph_with_fingerprints(csv_file: str, max_molecules: int = None) -> Tuple[Data, Dict[str, int], Dict[str, Tuple[float, float]]]:
    """
    从CSV文件构建分子进化图（使用Morgan指纹作为节点特征）

    Args:
        csv_file: CSV文件路径
        max_molecules: 最大分子数（用于调试）

    Returns:
        图数据、SMILES到索引的映射和属性统计信息
    """
    # 读取数据
    df = pd.read_csv(csv_file)

    if max_molecules:
        df = df.head(max_molecules)

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

    # 收集所有唯一的SMILES
    all_smiles = set(df['smiles_from'].tolist() + df['smiles_to'].tolist())
    smiles_to_idx = {smiles: idx for idx, smiles in enumerate(all_smiles)}

    # 构建节点特征 (使用Morgan指纹)
    # num_nodes = len(all_smiles)
    node_features = []

    for smiles in all_smiles:
        fingerprint = smiles_to_fingerprint(smiles)
        node_features.append(fingerprint)

    node_features = torch.FloatTensor(np.array(node_features))

    # 构建边索引和边特征
    edge_index = []
    edge_attr = []

    for _, row in df.iterrows():
        from_idx = smiles_to_idx[row['smiles_from']]
        to_idx = smiles_to_idx[row['smiles_to']]
        
        # 添加有向边
        edge_index.append([from_idx, to_idx])
        
        # 准备边特征（不含属性变化）
        edge_feat = prepare_edge_features(row, property_stats, include_property_changes=False)
        edge_attr.append(edge_feat)

    edge_index = torch.LongTensor(edge_index).t().contiguous()
    edge_attr = torch.FloatTensor(np.array(edge_attr))

    # 创建图数据对象
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    return data, smiles_to_idx, property_stats


def build_molecule_evolution_dataset(csv_file: str, max_pairs: int = None, target_property: str = 'mu_change', 
                                   include_position_encoding: bool = False, pe_dim: int = 8) -> Tuple[List[Data], List[Data], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    构建分子进化数据集，用于v0模型训练
    
    Args:
        csv_file: CSV文件路径
        max_pairs: 最大对数（用于调试）
        target_property: 目标属性名称
        include_position_encoding: 是否包含位置编码
        pe_dim: 位置编码维度
        
    Returns:
        起始分子数据列表、目标分子数据列表、边特征张量、目标属性张量和位置编码张量（如果启用）
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    
    if max_pairs:
        df = df.head(max_pairs)
    
    # 计算目标属性的统计信息
    if target_property in df.columns:
        mean = df[target_property].mean()
        std = df[target_property].std()
        property_stats = {target_property: (mean, std)}
    else:
        property_stats = {target_property: (0.0, 1.0)}
    
    # 准备目标属性值
    target_features = []
    for _, row in df.iterrows():
        if target_property in row and not pd.isna(row[target_property]):
            value = row[target_property]
            # 标准化目标属性值
            if target_property in property_stats:
                mean, std = property_stats[target_property]
                if std > 0:
                    value = (value - mean) / std
            target_features.append([value])
        else:
            target_features.append([0.0])
    
    target_features = torch.FloatTensor(np.array(target_features))
    
    # 构建分子数据列表
    from_data_list = []
    to_data_list = []
    edge_attr_list = []
    position_encoding_list = []
    
    for _, row in df.iterrows():
        # 生成起始分子和目标分子的指纹
        from_fp = smiles_to_fingerprint(row['smiles_from'])
        to_fp = smiles_to_fingerprint(row['smiles_to'])
        
        # 创建Data对象
        from_data = Data(x=torch.FloatTensor(from_fp).unsqueeze(0))
        to_data = Data(x=torch.FloatTensor(to_fp).unsqueeze(0))
        
        from_data_list.append(from_data)
        to_data_list.append(to_data)
        
        # 准备边特征（不含位置编码）
        edge_feat = prepare_edge_features(row, property_stats, include_property_changes=False, 
                                        include_position_encoding=False)
        edge_attr_list.append(edge_feat)
        
        # 如果启用位置编码，则准备位置编码特征
        if include_position_encoding:
            position_encoding = prepare_position_encoding_features(row, pe_dim)
            position_encoding_list.append(position_encoding)
    
    edge_attrs = torch.FloatTensor(np.array(edge_attr_list))
    
    if include_position_encoding:
        position_encodings = torch.FloatTensor(np.array(position_encoding_list))
        return from_data_list, to_data_list, edge_attrs, target_features, position_encodings
    else:
        return from_data_list, to_data_list, edge_attrs, target_features, None


def prepare_property_change_targets(csv_file: str, property_stats: Dict[str, Tuple[float, float]], 
                                  max_pairs: int = None) -> torch.Tensor:
    """
    准备属性变化目标值
    
    Args:
        csv_file: CSV文件路径
        property_stats: 属性统计信息（均值和标准差）
        max_pairs: 最大对数（用于调试）
        
    Returns:
        标准化后的属性变化目标值
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    
    if max_pairs:
        df = df.head(max_pairs)
    
    # 准备目标特征（15个属性变化值）
    target_features = []
    
    property_names = ['A_change', 'B_change', 'C_change', 'mu_change', 'alpha_change',
                      'homo_change', 'lumo_change', 'gap_change', 'r2_change', 'zpve_change',
                      'U0_change', 'U_change', 'H_change', 'G_change', 'Cv_change']
    
    for _, row in df.iterrows():
        properties = []
        
        for prop in property_names:
            if prop in row and not pd.isna(row[prop]):
                value = row[prop]
                # 标准化属性变化值
                if prop in property_stats:
                    mean, std = property_stats[prop]
                    if std > 0:
                        value = (value - mean) / std
                properties.append(value)
            else:
                properties.append(0.0)
                
        target_features.append(properties)
    
    target_features = torch.FloatTensor(np.array(target_features))
    
    return target_features