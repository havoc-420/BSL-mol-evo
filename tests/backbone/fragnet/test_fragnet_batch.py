#!/usr/bin/env python3
"""
测试 Frag 项目的 data batch 化函数
"""

import sys
import os

# 添加项目路径，确保可以导入项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'mol_evo', 'modules', 'FragNet'))

from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from fragnet.dataset.data import CreateData, collate_fn


def test_fragnet_batch_function():
    """
    测试 Frag 项目的 batch 化函数
    """
    print("测试 Frag 项目的 batch 化函数...")
    
    # 创建多个分子数据点
    smiles_list = ["CCO", "CCN", "CCC", "C1CC1", "CCOCC"]
    data_list = []
    
    for smiles in smiles_list:
        # 创建分子对象和3D构象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
            
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        
        # 使用 FragNet 的 CreateData 类
        creator = CreateData(data_type="exp")
        
        # 创建数据点 (smiles, y, mol, conf, frag_type)
        args = (smiles, [0.0], mol, conf, "brics")
        data = creator.create_data_point(args)
        
        if data is not None:
            data_list.append(data)
            print(f"  成功创建分子 {smiles} 的数据点")
    
    print(f"\n总共创建了 {len(data_list)} 个数据点")
    
    if len(data_list) > 0:
        # 使用 FragNet 的 collate_fn 进行批处理
        try:
            batch_data = collate_fn(data_list)
            print(f"\n成功进行批处理:")
            print(f"  原子特征形状: {batch_data['x_atoms'].shape}")
            print(f"  边索引形状: {batch_data['edge_index'].shape}")
            print(f"  边属性形状: {batch_data['edge_attr'].shape}")
            print(f"  片段索引形状: {batch_data['frag_index'].shape}")
            print(f"  片段连接属性形状: {batch_data['cnx_attr'].shape}")
            print(f"  片段特征形状: {batch_data['x_frags'].shape}")
            print(f"  原子到片段映射形状: {batch_data['atom_to_frag_ids'].shape}")
            print(f"  批处理向量形状: {batch_data['batch'].shape}")
            print(f"  片段批处理向量形状: {batch_data['frag_batch'].shape}")
            print(f"  键图节点特征形状: {batch_data['node_features_bonds'].shape}")
            print(f"  键图边索引形状: {batch_data['edge_index_bonds_graph'].shape}")
            print(f"  键图边属性形状: {batch_data['edge_attr_bonds'].shape}")
            print(f"  片段键图节点特征形状: {batch_data['node_features_fbonds'].shape}")
            print(f"  片段键图边索引形状: {batch_data['edge_index_fbonds'].shape}")
            print(f"  片段键图边属性形状: {batch_data['edge_attr_fbonds'].shape}")
            print(f"  目标值形状: {batch_data['y'].shape}")
            
            return batch_data
        except Exception as e:
            print(f"批处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print("没有成功创建任何数据点")
        return None


def test_batch_with_different_molecules():
    """
    测试包含不同复杂度分子的批处理
    """
    print("\n\n测试包含不同复杂度分子的批处理...")
    
    # 创建不同复杂度的分子
    smiles_list = [
        "C",           # 甲烷 - 简单分子
        "CCO",         # 乙醇 - 中等复杂度
        "c1ccccc1",    # 苯 - 芳香环
        "CC(C)CC1=CC=C(C=C1)C(C)C(C(=O)O)NC(=O)C",  # 复杂分子
    ]
    
    data_list = []
    
    for smiles in smiles_list:
        # 创建分子对象和3D构象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  无法解析 SMILES: {smiles}")
            continue
            
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        
        # 使用 FragNet 的 CreateData 类
        creator = CreateData(data_type="exp")
        
        # 创建数据点 (smiles, y, mol, conf, frag_type)
        args = (smiles, [0.0], mol, conf, "brics")
        data = creator.create_data_point(args)
        
        if data is not None:
            data_list.append(data)
            print(f"  成功创建分子 {smiles} 的数据点")
            print(f"    原子数: {data.x_atoms.shape[0]}")
            print(f"    边数: {data.edge_index.shape[1]}")
            print(f"    片段数: {data.n_frags.item()}")
        else:
            print(f"  无法创建分子 {smiles} 的数据点")
    
    if len(data_list) > 0:
        # 使用 FragNet 的 collate_fn 进行批处理
        try:
            batch_data = collate_fn(data_list)
            print(f"\n成功进行批处理:")
            print(f"  总原子数: {batch_data['x_atoms'].shape[0]}")
            print(f"  总边数: {batch_data['edge_index'].shape[1]}")
            print(f"  总片段数: {batch_data['x_frags'].shape[0]}")
            print(f"  批处理大小: {len(data_list)}")
            print(f"  目标值形状: {batch_data['y'].shape}")
            
            # 验证批处理向量的正确性
            unique_batches = torch.unique(batch_data['batch'])
            unique_frag_batches = torch.unique(batch_data['frag_batch'])
            print(f"  原子批处理向量唯一值: {unique_batches}")
            print(f"  片段批处理向量唯一值: {unique_frag_batches}")
            
            return batch_data
        except Exception as e:
            print(f"批处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print("没有成功创建任何数据点")
        return None


def verify_batch_consistency():
    """
    验证批处理结果的一致性
    """
    print("\n\n验证批处理结果的一致性...")
    
    # 创建相同的分子两次，检查批处理结果是否一致
    smiles = "CCO"
    
    data_list1 = []
    data_list2 = []
    
    # 创建两组相同的数据
    for i in range(2):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        
        creator = CreateData(data_type="exp")
        args = (smiles, [0.0], mol, conf, "brics")
        data = creator.create_data_point(args)
        
        if data is not None:
            data_list1.append(data)
            data_list2.append(data)
    
    if len(data_list1) == 2 and len(data_list2) == 2:
        try:
            batch1 = collate_fn(data_list1)
            batch2 = collate_fn(data_list2)
            
            # 检查关键属性是否一致
            is_consistent = (
                batch1['x_atoms'].shape == batch2['x_atoms'].shape and
                batch1['edge_index'].shape == batch2['edge_index'].shape and
                batch1['x_frags'].shape == batch2['x_frags'].shape and
                torch.allclose(batch1['x_atoms'], batch2['x_atoms'], equal_nan=True)
            )
            
            print(f"  批处理一致性检查: {'通过' if is_consistent else '失败'}")
            return is_consistent
        except Exception as e:
            print(f"一致性检查失败: {e}")
            return False
    else:
        print("无法创建足够的数据点进行一致性检查")
        return False


if __name__ == "__main__":
    print("开始测试 Frag 项目的 data batch 化函数")
    print("=" * 60)
    
    batch_data1 = test_fragnet_batch_function()
    batch_data2 = test_batch_with_different_molecules()
    consistency_result = verify_batch_consistency()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print(f"基本批处理测试: {'通过' if batch_data1 is not None else '失败'}")
    print(f"复杂分子批处理测试: {'通过' if batch_data2 is not None else '失败'}")
    print(f"一致性测试: {'通过' if consistency_result else '失败'}")