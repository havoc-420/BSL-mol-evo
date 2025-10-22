#!/usr/bin/env python3
"""
测试与 train_v0.py 中类似的 batch 数据化功能
"""

import sys
import os

# 添加项目路径，确保可以导入项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'mol_evo', 'modules', 'FragNet'))

from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch.utils.data import Dataset, DataLoader
# 修正导入路径
from fragnet.dataset.data import CreateData, collate_fn


class MoleculeDataset(Dataset):
    """分子数据集类"""
    
    def __init__(self, smiles_list, targets=None):
        self.smiles_list = smiles_list
        self.targets = targets if targets is not None else [0.0] * len(smiles_list)
        
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        target = self.targets[idx]
        
        # 创建分子对象和3D构象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无法解析 SMILES: {smiles}")
            
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        
        # 使用 FragNet 的 CreateData 类
        creator = CreateData(data_type="exp")
        
        # 创建数据点 (smiles, y, mol, conf, frag_type)
        args = (smiles, [target], mol, conf, "brics")
        data = creator.create_data_point(args)
        
        if data is None:
            raise ValueError(f"无法创建分子数据: {smiles}")
            
        return data


def test_data_loader_batching():
    """
    测试 DataLoader 的批处理功能
    """
    print("测试 DataLoader 的批处理功能...")
    
    # 创建测试数据
    smiles_list = ["CCO", "CCN", "CCC", "C1CC1", "CC(=O)O"]
    targets = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # 创建数据集
    dataset = MoleculeDataset(smiles_list, targets)
    print(f"  创建了包含 {len(dataset)} 个样本的数据集")
    
    # 创建 DataLoader
    batch_size = 3
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 为了测试结果一致性，不打乱顺序
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print(f"  创建了 batch_size={batch_size} 的 DataLoader")
    
    # 遍历 DataLoader
    batch_count = 0
    total_samples = 0
    
    for batch_idx, batch_data in enumerate(data_loader):
        batch_count += 1
        batch_size_actual = batch_data['y'].shape[0]
        total_samples += batch_size_actual
        
        print(f"\n  批次 {batch_idx + 1}:")
        print(f"    实际批次大小: {batch_size_actual}")
        print(f"    原子特征形状: {batch_data['x_atoms'].shape}")
        print(f"    边索引形状: {batch_data['edge_index'].shape}")
        print(f"    目标值形状: {batch_data['y'].shape}")
        print(f"    片段特征形状: {batch_data['x_frags'].shape}")
        print(f"    键图节点特征形状: {batch_data['node_features_bonds'].shape}")
        
        # 检查批处理向量
        unique_batches = torch.unique(batch_data['batch'])
        print(f"    原子批处理向量唯一值: {unique_batches}")
        
        # 检查目标值
        print(f"    目标值: {batch_data['y'].tolist()}")
    
    print(f"\n  总共处理了 {batch_count} 个批次，{total_samples} 个样本")
    print(f"  数据集大小: {len(dataset)}")
    
    return batch_count, total_samples


def test_different_batch_sizes():
    """
    测试不同批次大小的处理
    """
    print("\n\n测试不同批次大小的处理...")
    
    # 创建测试数据
    smiles_list = ["CCO", "CCN", "CCC", "C1CC1", "CC(=O)O", "c1ccccc1"]
    targets = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    
    # 创建数据集
    dataset = MoleculeDataset(smiles_list, targets)
    
    # 测试不同的批次大小
    batch_sizes = [1, 2, 3, 4, 6, 8]  # 包括能整除和不能整除的情况
    
    for batch_size in batch_sizes:
        print(f"\n  测试 batch_size={batch_size}:")
        
        try:
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn
            )
            
            batch_count = 0
            total_samples = 0
            
            for batch_idx, batch_data in enumerate(data_loader):
                batch_count += 1
                batch_size_actual = batch_data['y'].shape[0]
                total_samples += batch_size_actual
                
            print(f"    成功处理 {batch_count} 个批次，{total_samples} 个样本")
            
        except Exception as e:
            print(f"    处理失败: {e}")


def test_edge_cases():
    """
    测试边界情况
    """
    print("\n\n测试边界情况...")
    
    # 测试单个样本
    print("  测试单个样本:")
    single_smiles = ["CCO"]
    single_target = [1.0]
    
    try:
        dataset = MoleculeDataset(single_smiles, single_target)
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        for batch_data in data_loader:
            print(f"    成功处理单个样本:")
            print(f"      原子特征形状: {batch_data['x_atoms'].shape}")
            print(f"      目标值: {batch_data['y'].item()}")
            
    except Exception as e:
        print(f"    处理失败: {e}")
    
    # 测试复杂分子
    print("\n  测试复杂分子:")
    complex_smiles = ["CC(C)CC1=CC=C(C=C1)C(C)C(C(=O)O)NC(=O)C"]
    complex_target = [42.0]
    
    try:
        dataset = MoleculeDataset(complex_smiles, complex_target)
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        for batch_data in data_loader:
            print(f"    成功处理复杂分子:")
            print(f"      原子数: {batch_data['x_atoms'].shape[0]}")
            print(f"      边数: {batch_data['edge_index'].shape[1]}")
            print(f"      片段数: {batch_data['x_frags'].shape[0]}")
            print(f"      目标值: {batch_data['y'].item()}")
            
    except Exception as e:
        print(f"    处理失败: {e}")


def verify_batch_consistency():
    """
    验证批次数据的一致性
    """
    print("\n\n验证批次数据的一致性...")
    
    # 创建相同的数据集
    smiles_list = ["CCO", "CCN", "CCC"]
    targets = [1.0, 2.0, 3.0]
    
    dataset = MoleculeDataset(smiles_list, targets)
    
    # 创建两个 DataLoader
    data_loader1 = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    data_loader2 = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # 获取两个批次的数据
    batches1 = list(data_loader1)
    batches2 = list(data_loader2)
    
    # 检查批次数量是否一致
    if len(batches1) == len(batches2):
        print(f"  批次数量一致: {len(batches1)}")
        
        # 检查每个批次的数据是否一致
        is_consistent = True
        for i, (batch1, batch2) in enumerate(zip(batches1, batches2)):
            # 检查关键属性
            if not (batch1['x_atoms'].shape == batch2['x_atoms'].shape and
                    batch1['edge_index'].shape == batch2['edge_index'].shape and
                    torch.allclose(batch1['y'], batch2['y'], equal_nan=True)):
                is_consistent = False
                break
                
        print(f"  批次数据一致性: {'通过' if is_consistent else '失败'}")
        return is_consistent
    else:
        print(f"  批次数量不一致: {len(batches1)} vs {len(batches2)}")
        return False


if __name__ == "__main__":
    print("开始测试与 train_v0.py 中类似的 batch 数据化功能")
    print("=" * 60)
    
    batch_count, total_samples = test_data_loader_batching()
    test_different_batch_sizes()
    test_edge_cases()
    consistency_result = verify_batch_consistency()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print(f"基本批处理测试: {'通过' if batch_count > 0 and total_samples > 0 else '失败'}")
    print(f"一致性测试: {'通过' if consistency_result else '失败'}")