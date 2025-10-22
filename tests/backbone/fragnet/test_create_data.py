#!/usr/bin/env python3
"""
测试 CreateData 类及其子类的功能
"""

import sys
import os

# 添加项目路径，确保可以导入项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'mol_evo', 'modules', 'FragNet'))

from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from fragnet.dataset.data import CreateData, CreateDataDTA, CreateDataCDRP


def test_create_data():
    """
    测试基础的 CreateData 类
    """
    print("测试 CreateData 类...")
    
    # 创建一个简单的分子 (乙醇)
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    
    # 创建 CreateData 实例
    creator = CreateData(data_type="exp")
    
    # 准备参数 (smiles, y, mol, conf, frag_type)
    args = (smiles, [1.0], mol, conf, "brics")  # 将y改为列表形式
    
    # 创建数据点
    data = creator.create_data_point(args)
    
    if data is not None:
        print(f"  成功创建数据点:")
        print(f"    SMILES: {data.smiles}")
        print(f"    原子特征形状: {data.x_atoms.shape}")
        print(f"    边索引形状: {data.edge_index.shape}")
        print(f"    边属性形状: {data.edge_attr.shape}")
        print(f"    片段索引形状: {data.frag_index.shape}")
        print(f"    片段数量: {data.n_frags}")
        print(f"    键图节点特征形状: {data.node_features_bonds.shape}")
        print(f"    键图边索引形状: {data.edge_index_bonds.shape}")
        print(f"    键图边属性形状: {data.edge_attr_bonds.shape}")
    else:
        print("  创建数据点失败")


def test_create_data_dta():
    """
    测试 CreateDataDTA 类 (药物-靶标亲和力)
    """
    print("\n测试 CreateDataDTA 类...")
    
    # 创建一个简单的分子 (乙醇)
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    
    # 简单的蛋白质序列
    protein_seq = "ACDEFGHIKLMNPQRSTVWY"  # 简短的氨基酸序列
    
    # 创建 CreateDataDTA 实例
    creator = CreateDataDTA(data_type="exp")
    
    # 准备参数 (smiles, y, mol, conf, protein)
    args = (smiles, [1.0], mol, conf, protein_seq)  # 将y改为列表形式
    
    # 创建数据点
    data = creator.create_data_point(args)
    
    if data is not None:
        print(f"  成功创建DTA数据点:")
        print(f"    SMILES: {data.smiles}")
        print(f"    原子特征形状: {data.x_atoms.shape}")
        print(f"    蛋白质特征形状: {data.protein.shape}")
        print(f"    片段数量: {data.n_frags}")
        print(f"    键图节点特征形状: {data.node_features_bonds.shape}")
    else:
        print("  创建DTA数据点失败")


def test_create_data_cdrp():
    """
    测试 CreateDataCDRP 类 (化学-基因响应预测)
    """
    print("\n测试 CreateDataCDRP 类...")
    
    # 创建一个简单的分子 (乙醇)
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    
    # 模拟基因表达数据 (通常为向量)
    gene_expression = [0.1, 0.2, 0.3, 0.4, 0.5]  # 简化的基因表达向量
    
    # 创建 CreateDataCDRP 实例
    creator = CreateDataCDRP(data_type="exp")
    
    # 准备参数 (smiles, y, mol, conf, gene_expr)
    args = (smiles, [1.0], mol, conf, gene_expression)  # 将y改为列表形式
    
    # 创建数据点
    data = creator.create_data_point(args)
    
    if data is not None:
        print(f"  成功创建CDRP数据点:")
        print(f"    SMILES: {data.smiles}")
        print(f"    原子特征形状: {data.x_atoms.shape}")
        print(f"    基因表达特征形状: {data.gene_expr.shape}")
        print(f"    片段数量: {data.n_frags}")
        print(f"    键图节点特征形状: {data.node_features_bonds.shape}")
    else:
        print("  创建CDRP数据点失败")


def test_collate_functions():
    """
    测试批处理函数
    """
    print("\n测试批处理函数...")
    
    from fragnet.dataset.data import collate_fn
    
    # 创建多个数据点进行批处理测试
    smiles_list = ["CCO", "CCN", "CCC"]
    data_list = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)
            conf = mol.GetConformer()
            
            creator = CreateData(data_type="exp")
            args = (smiles, [1.0], mol, conf, "brics")  # 将y改为列表形式
            data = creator.create_data_point(args)
            if data is not None:
                data_list.append(data)
    
    if data_list:
        batch = collate_fn(data_list)
        print(f"  成功创建批处理数据:")
        print(f"    原子特征形状: {batch['x_atoms'].shape}")
        print(f"    边索引形状: {batch['edge_index'].shape}")
        print(f"    目标值形状: {batch['y'].shape}")
        print(f"    批处理大小: {len(data_list)}")
    else:
        print("  批处理测试失败")


if __name__ == "__main__":
    print("开始测试 CreateData 类及其子类")
    print("=" * 50)
    
    test_create_data()
    test_create_data_dta()
    test_create_data_cdrp()
    test_collate_functions()
    
    print("\n" + "=" * 50)
    print("测试完成")