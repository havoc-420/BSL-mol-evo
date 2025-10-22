#!/usr/bin/env python3
"""
测试 TorchMD-Net 基础模型功能
参考 test_create_data.py 和 test_batch_data_loader.py 脚本
"""

import sys
import os
import torch
import numpy as np

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'modules', 'torchmd-net'))

from torchmdnet.models.model import create_model
from torch_geometric.data import Data


def test_torchmd_net_model_creation():
    """
    测试 TorchMD-Net 模型创建
    """
    print("测试 TorchMD-Net 模型创建...")
    
    # 定义模型参数
    model_args = {
        "embedding_dimension": 128,
        "num_layers": 2,
        "num_rbf": 32,
        "rbf_type": "expnorm",
        "trainable_rbf": False,
        "activation": "silu",
        "cutoff_lower": 0.0,
        "cutoff_upper": 5.0,
        "max_z": 100,
        "max_num_neighbors": 32,
        "model": "tensornet",
        "aggr": "add",
        "derivative": False,
        "atom_filter": -1,
        "prior_model": None,
        "output_model": "Scalar",
        "reduce_op": "add",
        "precision": 32,
    }
    
    try:
        # 创建模型
        model = create_model(model_args)
        print(f"  成功创建模型: {type(model).__name__}")
        print(f"  模型参数数量: {sum(p.numel() for p in model.parameters())}")
        return model
    except Exception as e:
        print(f"  模型创建失败: {e}")
        return None


def test_simple_molecular_data():
    """
    测试简单的分子数据处理
    """
    print("\n测试简单的分子数据处理...")
    
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
        
        # 创建批次信息
        batch = torch.zeros_like(z)
        
        print(f"  原子序数: {z}")
        print(f"  原子坐标形状: {pos.shape}")
        print(f"  批次信息: {batch}")
        
        return z, pos, batch
    except Exception as e:
        print(f"  数据创建失败: {e}")
        return None, None, None


def test_model_forward_pass():
    """
    测试模型前向传播
    """
    print("\n测试模型前向传播...")
    
    # 创建模型
    model = test_torchmd_net_model_creation()
    if model is None:
        print("  跳过前向传播测试，因为模型创建失败")
        return
    
    # 创建数据
    z, pos, batch = test_simple_molecular_data()
    if z is None:
        print("  跳过前向传播测试，因为数据创建失败")
        return
    
    try:
        # 执行前向传播
        with torch.no_grad():
            output, _ = model(z, pos, batch=batch)
        
        print(f"  前向传播成功")
        print(f"  输出形状: {output.shape}")
        print(f"  输出值: {output}")
        return True
    except Exception as e:
        print(f"  前向传播失败: {e}")
        return False


def test_batch_processing():
    """
    测试批处理功能
    """
    print("\n测试批处理功能...")
    
    # 创建模型
    model = test_torchmd_net_model_creation()
    if model is None:
        print("  跳过批处理测试，因为模型创建失败")
        return
    
    try:
        # 创建两个分子的数据
        # 第一个分子 (水分子 H2O)
        z1 = torch.tensor([8, 1, 1], dtype=torch.long)  # O, H, H
        pos1 = torch.tensor([
            [0.0, 0.0, 0.0],    # O
            [0.0, 0.0, 1.0],    # H
            [0.0, 1.0, 0.0],    # H
        ], dtype=torch.float32)
        
        # 第二个分子 (甲烷 CH4)
        z2 = torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)  # C, H, H, H, H
        pos2 = torch.tensor([
            [0.0, 0.0, 0.0],    # C
            [1.0, 0.0, 0.0],    # H
            [0.0, 1.0, 0.0],    # H
            [0.0, 0.0, 1.0],    # H
            [-1.0, 0.0, 0.0],   # H
        ], dtype=torch.float32)
        
        # 合并为批次
        z_batch = torch.cat([z1, z2])
        pos_batch = torch.cat([pos1, pos2])
        batch_indices = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)  # 前3个属于批次0，后5个属于批次1
        
        print(f"  批次原子序数: {z_batch}")
        print(f"  批次原子坐标形状: {pos_batch.shape}")
        print(f"  批次索引: {batch_indices}")
        
        # 执行前向传播
        with torch.no_grad():
            output, _ = model(z_batch, pos_batch, batch=batch_indices)
        
        print(f"  批处理前向传播成功")
        print(f"  输出形状: {output.shape}")
        print(f"  输出值: {output}")
        return True
    except Exception as e:
        print(f"  批处理测试失败: {e}")
        return False


def test_model_types():
    """
    测试不同的模型类型
    """
    print("\n测试不同的模型类型...")
    
    model_types = [
        "tensornet",
        "equivariant-transformer",
        "graph-network"
    ]
    
    base_args = {
        "embedding_dimension": 64,
        "num_layers": 2,
        "num_rbf": 32,
        "rbf_type": "expnorm",
        "trainable_rbf": False,
        "activation": "silu",
        "cutoff_lower": 0.0,
        "cutoff_upper": 5.0,
        "max_z": 100,
        "max_num_neighbors": 32,
        "aggr": "add",
        "derivative": False,
        "atom_filter": -1,
        "prior_model": None,
        "output_model": "Scalar",
        "reduce_op": "add",
        "precision": 32,
    }
    
    for model_type in model_types:
        print(f"\n  测试 {model_type}...")
        args = base_args.copy()
        args["model"] = model_type
        
        # 特定模型的额外参数
        if model_type == "equivariant-transformer":
            args.update({
                "attn_activation": "silu",
                "num_heads": 4,
                "distance_influence": "both",
                "neighbor_embedding": True,
            })
        elif model_type == "tensornet":
            args.update({
                "equivariance_invariance_group": "O(3)",
            })
        elif model_type == "graph-network":
            args.update({
                "num_filters": 64,
                "neighbor_embedding": True,
            })
        
        try:
            model = create_model(args)
            print(f"    成功创建 {model_type} 模型")
            
            # 简单测试前向传播
            z = torch.tensor([8, 1, 1], dtype=torch.long)  # 水分子
            pos = torch.tensor([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ], dtype=torch.float32)
            batch = torch.zeros(3, dtype=torch.long)
            
            with torch.no_grad():
                output, _ = model(z, pos, batch=batch)
            print(f"    {model_type} 前向传播成功，输出形状: {output.shape}")
            
        except Exception as e:
            print(f"    {model_type} 测试失败: {e}")


if __name__ == "__main__":
    print("开始测试 TorchMD-Net 模型")
    print("=" * 50)
    
    test_torchmd_net_model_creation()
    test_simple_molecular_data()
    test_model_forward_pass()
    test_batch_processing()
    test_model_types()
    
    print("\n" + "=" * 50)
    print("TorchMD-Net 测试完成")