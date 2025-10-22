#!/usr/bin/env python3
"""
测试 TorchMD-Net 数据集功能
参考 test_create_data.py 和 test_batch_data_loader.py 脚本
"""

import sys
import os
import torch
import tempfile
import numpy as np

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'modules', 'torchmd-net'))

from torch_geometric.data import Data, DataLoader
from torchmdnet.datasets.memdataset import MemmappedDataset


def test_data_structure():
    """
    测试 TorchMD-Net 数据结构
    """
    print("测试 TorchMD-Net 数据结构...")
    
    try:
        # 创建简单的分子数据 (水分子 H2O)
        # 原子序数: H=1, O=8
        z = torch.tensor([8, 1, 1], dtype=torch.long)  # O, H, H
        
        # 简单的3D坐标 (以埃为单位)
        pos = torch.tensor([
            [0.0, 0.0, 0.0],    # O
            [0.0, 0.0, 1.0],    # H
            [0.0, 1.0, 0.0],    # H
        ], dtype=torch.float32)
        
        # 能量值
        y = torch.tensor([[-10.5]], dtype=torch.float64)  # 虚拟能量值
        
        # 创建数据对象
        data = Data(z=z, pos=pos, y=y)
        
        print(f"  成功创建数据对象")
        print(f"    原子序数: {data.z}")
        print(f"    原子坐标形状: {data.pos.shape}")
        print(f"    能量值: {data.y}")
        print(f"    数据对象类型: {type(data)}")
        
        return data
    except Exception as e:
        print(f"  数据结构测试失败: {e}")
        return None


def test_custom_dataset():
    """
    测试自定义数据集
    """
    print("\n测试自定义数据集...")
    
    class SimpleMoleculeDataset:
        """简单的分子数据集类"""
        
        def __init__(self):
            # 创建一些虚拟分子数据
            self.data_list = []
            
            # 水分子 (H2O)
            data1 = Data(
                z=torch.tensor([8, 1, 1], dtype=torch.long),
                pos=torch.tensor([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ], dtype=torch.float32),
                y=torch.tensor([[-10.5]], dtype=torch.float64)
            )
            self.data_list.append(data1)
            
            # 甲烷 (CH4)
            data2 = Data(
                z=torch.tensor([6, 1, 1, 1, 1], dtype=torch.long),
                pos=torch.tensor([
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                ], dtype=torch.float32),
                y=torch.tensor([[-15.2]], dtype=torch.float64)
            )
            self.data_list.append(data2)
            
            # 氨 (NH3)
            data3 = Data(
                z=torch.tensor([7, 1, 1, 1], dtype=torch.long),
                pos=torch.tensor([
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ], dtype=torch.float32),
                y=torch.tensor([[-8.7]], dtype=torch.float64)
            )
            self.data_list.append(data3)
        
        def __len__(self):
            return len(self.data_list)
        
        def __getitem__(self, idx):
            return self.data_list[idx]
    
    try:
        # 创建数据集
        dataset = SimpleMoleculeDataset()
        print(f"  成功创建自定义数据集，包含 {len(dataset)} 个分子")
        
        # 测试数据访问
        for i in range(len(dataset)):
            data = dataset[i]
            print(f"    分子 {i+1}: {len(data.z)} 个原子, 能量 = {data.y.item():.2f}")
        
        return dataset
    except Exception as e:
        print(f"  自定义数据集测试失败: {e}")
        return None


def test_dataloader_batching():
    """
    测试 DataLoader 批处理功能
    """
    print("\n测试 DataLoader 批处理功能...")
    
    # 使用自定义数据集
    dataset = test_custom_dataset()
    if dataset is None:
        print("  跳过 DataLoader 测试，因为数据集创建失败")
        return
    
    try:
        # 创建 DataLoader
        from torch_geometric.loader import DataLoader as PyGDataLoader
        data_loader = PyGDataLoader(dataset, batch_size=2, shuffle=False)
        
        print(f"  成功创建 DataLoader，批次大小: 2")
        
        # 遍历批次
        for batch_idx, batch in enumerate(data_loader):
            print(f"    批次 {batch_idx+1}:")
            print(f"      总原子数: {batch.z.shape[0]}")
            print(f"      原子坐标形状: {batch.pos.shape}")
            print(f"      能量值形状: {batch.y.shape}")
            print(f"      批次向量形状: {batch.batch.shape}")
            
            # 显示原子类型
            print(f"      原子类型: {batch.z.tolist()}")
            
        return True
    except Exception as e:
        print(f"  DataLoader 测试失败: {e}")
        return False


def test_memmapped_dataset():
    """
    测试内存映射数据集
    """
    print("\n测试内存映射数据集...")
    
    try:
        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"  使用临时目录: {tmpdir}")
            
            # 创建简单的内存映射数据集
            class TestMemDataset(MemmappedDataset):
                def __init__(self, root):
                    super().__init__(
                        root=root,
                        transform=None,
                        pre_transform=None,
                        pre_filter=None,
                        properties=("y",)
                    )
                
                def sample_iter(self, mol_ids=False):
                    # 水分子
                    z = torch.tensor([8, 1, 1], dtype=torch.long)
                    pos = torch.tensor([
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0],
                    ], dtype=torch.float32)
                    y = torch.tensor([[-10.5]], dtype=torch.float64)
                    
                    data = Data(z=z, pos=pos, y=y)
                    if self.filter_and_pre_transform is not None:
                        data = self.filter_and_pre_transform(data)
                    if data is not None:
                        yield data
            
            # 创建数据集实例
            dataset = TestMemDataset(root=tmpdir)
            print(f"  成功创建内存映射数据集")
            print(f"  数据集大小: {len(dataset)}")
            
            # 测试数据访问
            if len(dataset) > 0:
                data = dataset[0]
                print(f"    数据样本:")
                print(f"      原子数: {len(data.z)}")
                print(f"      原子类型: {data.z.tolist()}")
                print(f"      能量值: {data.y.item()}")
            
            return dataset
    except Exception as e:
        print(f"  内存映射数据集测试失败: {e}")
        return None


def test_different_properties():
    """
    测试不同的分子属性
    """
    print("\n测试不同的分子属性...")
    
    try:
        # 创建包含多种属性的分子数据
        z = torch.tensor([8, 1, 1], dtype=torch.long)  # O, H, H
        pos = torch.tensor([
            [0.0, 0.0, 0.0],    # O
            [0.0, 0.0, 1.0],    # H
            [0.0, 1.0, 0.0],    # H
        ], dtype=torch.float32)
        
        # 各种属性
        y = torch.tensor([[-10.5]], dtype=torch.float64)  # 能量
        neg_dy = torch.tensor([  # 负梯度（力）
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ], dtype=torch.float32)
        
        # 创建包含多种属性的数据对象
        data = Data(z=z, pos=pos, y=y, neg_dy=neg_dy)
        
        print(f"  成功创建多属性数据对象")
        print(f"    原子序数形状: {data.z.shape}")
        print(f"    坐标形状: {data.pos.shape}")
        print(f"    能量形状: {data.y.shape}")
        print(f"    力形状: {data.neg_dy.shape}")
        
        return data
    except Exception as e:
        print(f"  多属性数据测试失败: {e}")
        return None


if __name__ == "__main__":
    print("开始测试 TorchMD-Net 数据集功能")
    print("=" * 50)
    
    test_data_structure()
    test_custom_dataset()
    test_dataloader_batching()
    test_memmapped_dataset()
    test_different_properties()
    
    print("\n" + "=" * 50)
    print("TorchMD-Net 数据集测试完成")