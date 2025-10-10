#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子处理工具函数
"""

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType, BondType

# 定义需要的全局变量和辅助函数
try:
    ETKDG_PARAMS = AllChem.ETKDGv3()
    ETKDG_VERSION_USED = "ETKDGv3"
except AttributeError:
    ETKDG_PARAMS = AllChem.ETKDGv2()
    ETKDG_VERSION_USED = "ETKDGv2 (fallback)"

bonds = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3}


def one_hot(tensor, num_classes):
    """
    将整数张量转换为 one-hot 编码。
    
    Args:
        tensor: 输入张量。
        num_classes: 类别数量。
        
    Returns:
        one-hot 编码张量。
    """
    return torch.eye(num_classes, dtype=torch.float)[tensor]


def smile_to_graph_xyz(smile, types):
    """
    将SMILES字符串转换为图结构表示，包括3D坐标信息。
    
    Args:
        smile (str): SMILES字符串
        types (dict): 原子类型映射字典
        
    Returns:
        tuple: (x, z, pos, edge_index, edge_attr) 其中:
            - x: 节点特征矩阵
            - z: 原子序数
            - pos: 3D坐标
            - edge_index: 边索引
            - edge_attr: 边属性: 边类型 one-hot 编码
    """
    mol = Chem.MolFromSmiles(smile)
    
    if mol is None:
        print(f"无法解析SMILES: {smile}")
        return None, None, None, None, None
    
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, ETKDG_PARAMS)
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        pos = torch.tensor(pos, dtype=torch.float)
    except Exception as e:
        print(f'无法生成3D坐标，尝试使用{ETKDG_VERSION_USED}生成坐标...', repr(e))
        # 去除立体构型
        Chem.RemoveStereochemistry(mol)
        try:
            # 第二次尝试生成坐标
            AllChem.EmbedMolecule(mol, ETKDG_PARAMS)
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)
        except Exception as e:
            print(f'Cannot generate 3D coordinates for {smile}')
            pos = None  # 明确设置为None
    
    # 检查是否成功生成坐标
    if pos is None:
        # 返回空的结果而不是未定义的变量
        return None, None, None, None, None
        
    # 获取原子特征
    type_idx = []
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in types:
            continue
        type_idx.append(types[symbol])
        atomic_number.append(atom.GetAtomicNum())                       # 原子序数
        aromatic.append(1 if atom.GetIsAromatic() else 0)               # 芳香性
        hybridization = atom.GetHybridization()                         # 杂化类型 * 3
        sp.append(1 if hybridization == HybridizationType.SP else 0)    
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    # 获取键信息
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = one_hot(edge_type, num_classes=len(bonds))

    # 排序边
    N = mol.GetNumAtoms()
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    edge_attr = edge_attr[perm]

    # 计算氢原子数量
    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = torch.zeros(N, dtype=torch.float).scatter_add_(0, col, hs[row]).tolist()

    # 构建节点特征
    x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))        # 原子类型 one-hot 编码
    x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],  # 直接将各种属性信息作为节点特征
                        dtype=torch.float).t().contiguous()
    x = torch.cat([x1, x2], dim=-1)  
    
    return x, z, pos, edge_index, edge_attr