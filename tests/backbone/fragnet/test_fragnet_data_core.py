#!/usr/bin/env python3
"""
测试修改后的 mol_evo/core/data/fragnet_data.py 文件
"""

import sys
import os

# 添加项目路径，确保可以导入项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'mol_evo', 'modules', 'FragNet'))

from rdkit import Chem
import torch
from mol_evo.core.data.fragnet_data import smile_to_fragnet_features, smile_to_fragnet_batch


def test_smile_to_fragnet_features():
    """
    测试 smile_to_fragnet_features 函数
    """
    print("测试 smile_to_fragnet_features 函数...")
    
    # 测试单个SMILES
    smiles = "CCO"  # 乙醇
    
    features = smile_to_fragnet_features(smiles)
    
    if features is not None:
        print(f"  成功生成特征:")
        print(f"    SMILES: {features['smiles']}")
        print(f"    原子特征形状: {features['x_atoms'].shape}")
        print(f"    边索引形状: {features['edge_index'].shape}")
        print(f"    边属性形状: {features['edge_attr'].shape}")
        print(f"    片段索引形状: {features['frag_index'].shape}")
        print(f"    片段数量: {features['n_frags']}")
        print(f"    键图节点特征形状: {features['node_features_bonds'].shape}")
        print(f"    键图边索引形状: {features['edge_index_bonds'].shape}")
        print(f"    键图边属性形状: {features['edge_attr_bonds'].shape}")
        print(f"    原子到片段映射形状: {features['atom_to_frag_ids'].shape}")
    else:
        print("  特征生成失败")


def test_smile_to_fragnet_batch():
    """
    测试 smile_to_fragnet_batch 函数
    """
    print("\n测试 smile_to_fragnet_batch 函数...")
    
    # 测试多个SMILES
    smiles_list = ["CCO", "CCN", "CCC"]
    
    batch_data = smile_to_fragnet_batch(smiles_list)
    
    if batch_data is not None:
        print(f"  成功生成批处理数据:")
        print(f"    原子特征形状: {batch_data['x_atoms'].shape}")
        print(f"    边索引形状: {batch_data['edge_index'].shape}")
        print(f"    边属性形状: {batch_data['edge_attr'].shape}")
        print(f"    片段索引形状: {batch_data['frag_index'].shape}")
        print(f"    批处理大小: {len(smiles_list)}")
        print(f"    原子到片段映射形状: {batch_data['atom_to_frag_ids'].shape}")
        print(f"    键图节点特征形状: {batch_data['node_features_bonds'].shape}")
        print(f"    键图边索引形状: {batch_data['edge_index_bonds_graph'].shape}")
        print(f"    键图边属性形状: {batch_data['edge_attr_bonds'].shape}")
    else:
        print("  批处理数据生成失败")


def compare_with_direct_fragnet():
    """
    比较通过新接口和直接使用 FragNet 生成的数据是否一致
    """
    print("\n比较新接口和直接使用 FragNet 的结果...")
    
    from fragnet.dataset.data import CreateData, collate_fn
    from rdkit.Chem import AllChem
    
    smiles = "CCO"
    
    # 使用新接口
    features = smile_to_fragnet_features(smiles)
    
    # 直接使用 FragNet
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    
    creator = CreateData(data_type="exp")
    args = (smiles, [0.0], mol, conf, "brics")
    data = creator.create_data_point(args)
    
    if features is not None and data is not None:
        # 比较原子特征
        atoms_equal = torch.allclose(features['x_atoms'], data.x_atoms, equal_nan=True)
        print(f"    原子特征是否一致: {atoms_equal}")
        
        # 比较边索引
        edge_index_equal = torch.equal(features['edge_index'], data.edge_index)
        print(f"    边索引是否一致: {edge_index_equal}")
        
        # 比较片段索引
        frag_index_equal = torch.equal(features['frag_index'], data.frag_index)
        print(f"    片段索引是否一致: {frag_index_equal}")
        
    else:
        print("  比较失败，至少有一个结果为空")


if __name__ == "__main__":
    print("开始测试修改后的 fragnet_data.py")
    print("=" * 50)
    
    test_smile_to_fragnet_features()
    test_smile_to_fragnet_batch()
    compare_with_direct_fragnet()
    
    print("\n" + "=" * 50)
    print("测试完成")