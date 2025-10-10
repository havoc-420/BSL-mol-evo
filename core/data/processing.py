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
    except Exception:
        return 0.0


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
    num_nodes = len(all_smiles)
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


def build_molecule_evolution_dataset(csv_file: str, max_pairs: int = None, target_property: str = 'mu_change') -> Tuple[List[Data], List[Data], torch.Tensor, torch.Tensor]:
    """
    构建分子进化数据集，用于v0模型训练
    
    Args:
        csv_file: CSV文件路径
        max_pairs: 最大对数（用于调试）
        target_property: 目标属性名称
        
    Returns:
        起始分子数据列表、目标分子数据列表、边特征张量和目标属性张量
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
    
    for _, row in df.iterrows():
        # 生成起始分子和目标分子的指纹
        from_fp = smiles_to_fingerprint(row['smiles_from'])
        to_fp = smiles_to_fingerprint(row['smiles_to'])
        
        # 创建Data对象
        from_data = Data(x=torch.FloatTensor(from_fp).unsqueeze(0))
        to_data = Data(x=torch.FloatTensor(to_fp).unsqueeze(0))
        
        from_data_list.append(from_data)
        to_data_list.append(to_data)
        
        # 准备边特征
        edge_feat = prepare_edge_features(row, property_stats, include_property_changes=False)
        edge_attr_list.append(edge_feat)
    
    edge_attrs = torch.FloatTensor(np.array(edge_attr_list))
    
    return from_data_list, to_data_list, edge_attrs, target_features


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