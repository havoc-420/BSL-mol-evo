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
    except:
        return np.zeros(n_bits)


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
    except:
        return 0.0


def prepare_edge_features(row: pd.Series, property_stats: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """
    准备边特征

    Args:
        row: 数据行
        property_stats: 属性统计信息（均值和标准差）

    Returns:
        边特征向量
    """
    # 原子类型特征
    atom_features = atom_type_to_onehot(row['to_atom_symbol'])

    # 操作类型特征
    op_features = operation_type_to_onehot(row['operation_type'])

    # 属性变化特征
    property_changes = []
    property_names = ['A_change', 'B_change', 'C_change', 'mu_change', 'alpha_change',
                      'homo_change', 'lumo_change', 'gap_change', 'r2_change', 'zpve_change',
                      'U0_change', 'U_change', 'H_change', 'G_change', 'Cv_change']

    for prop in property_names:
        if prop in row and not pd.isna(row[prop]):
            value = row[prop]
            property_changes.append(value)
        else:
            property_changes.append(0.0)

    # 位置敏感特征
    position_features = []
    if 'from_heavy_atoms' in row and 'to_heavy_atoms' in row:
        position_features = [
            row['from_heavy_atoms'],
            row['to_heavy_atoms'],
            row['to_heavy_atoms'] - row['from_heavy_atoms']  # 原子数变化
        ]
    else:
        position_features = [0, 0, 0]

    # 分子结构相似性特征
    similarity_features = []
    if 'smiles_from' in row and 'smiles_to' in row:
        similarity = calculate_molecular_similarity(row['smiles_from'], row['smiles_to'])
        similarity_features = [similarity]
    else:
        similarity_features = [0.0]

    # 组合所有特征
    edge_features = atom_features + op_features + property_changes + position_features + similarity_features

    return np.array(edge_features, dtype=np.float32)


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


def build_molecule_graph_with_fingerprints(csv_file: str, max_molecules: int = None) -> Tuple[Data, Dict[str, int]]:
    """
    从CSV文件构建分子进化图（使用Morgan指纹作为节点特征）
    
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
    
    # 构建节点特征 (使用Morgan指纹)
    num_nodes = len(all_smiles)
    node_features = []
    
    # 为每个SMILES构建特征
    for smiles in all_smiles:
        fp = smiles_to_fingerprint(smiles)
        node_features.append(fp)
    
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
    
    return data, smiles_to_idx


def prepare_evolution_data(csv_file: str, max_pairs: int = None):
    """
    准备分子进化数据用于转换器模型训练
    
    Args:
        csv_file: CSV文件路径
        max_pairs: 最大对数（用于调试）
        
    Returns:
        起始分子特征、边特征、目标分子特征和属性统计信息
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    
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
    
    # 准备特征
    source_features = []
    edge_features = []
    target_features = []
    
    for _, row in df.iterrows():
        # 起始分子特征
        source_fp = smiles_to_fingerprint(row['smiles_from'])
        source_features.append(source_fp)
        
        # 边特征
        edge_feat = prepare_edge_features(row, property_stats)
        edge_features.append(edge_feat)
        
        # 目标分子特征
        target_fp = smiles_to_fingerprint(row['smiles_to'])
        target_features.append(target_fp)
    
    source_features = torch.FloatTensor(np.array(source_features))
    edge_features = torch.FloatTensor(np.array(edge_features))
    target_features = torch.FloatTensor(np.array(target_features))
    
    return source_features, edge_features, target_features, property_stats


def split_data_by_molecules(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[List[int], List[int], List[int]]:
    """
    按分子分割数据，避免数据泄露

    Args:
        df: 数据DataFrame
        test_size: 测试集比例
        val_size: 验证集比例

    Returns:
        训练集、验证集、测试集索引
    """
    # 获取所有唯一的SMILES分子
    all_smiles = set(df['smiles_from'].tolist() + df['smiles_to'].tolist())
    all_smiles = list(all_smiles)

    # 随机打乱分子
    np.random.shuffle(all_smiles)

    # 计算分割点
    n_total = len(all_smiles)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    n_train = n_total - n_test - n_val

    # 分割分子
    train_molecules = set(all_smiles[:n_train])
    val_molecules = set(all_smiles[n_train:n_train + n_val])
    test_molecules = set(all_smiles[n_train + n_val:])

    # 根据分子分配数据索引
    train_idx, val_idx, test_idx = [], [], []

    for idx, row in df.iterrows():
        from_smiles = row['smiles_from']
        to_smiles = row['smiles_to']

        # 如果起始分子和目标分子都在训练集，则分配到训练集
        if from_smiles in train_molecules and to_smiles in train_molecules:
            train_idx.append(idx)
        # 如果起始分子和目标分子都在验证集，则分配到验证集
        elif from_smiles in val_molecules and to_smiles in val_molecules:
            val_idx.append(idx)
        # 如果起始分子和目标分子都在测试集，则分配到测试集
        elif from_smiles in test_molecules and to_smiles in test_molecules:
            test_idx.append(idx)
        # 否则分配到训练集（避免数据泄露）
        else:
            train_idx.append(idx)

    return train_idx, val_idx, test_idx