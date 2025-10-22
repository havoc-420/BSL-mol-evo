#!/usr/bin/env python3
"""
测试 TorchMD-Net 模型性能
参考 test_create_data.py 和 test_batch_data_loader.py 脚本
"""

import sys
import os
import torch
import time
import numpy as np

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'modules', 'torchmd-net'))

from torchmdnet.models.model import create_model
from torch_geometric.data import Data, DataLoader as PyGDataLoader


def create_large_dataset(num_molecules=100):
    """
    创建大型数据集用于性能测试
    """
    print(f"创建包含 {num_molecules} 个分子的大型数据集...")
    
    data_list = []
    
    # 创建不同大小的分子混合数据集
    for i in range(num_molecules):
        # 随机选择分子类型
        mol_type = i % 4  # 0=H2O, 1=CH4, 2=NH3, 3=C6H6
        
        if mol_type == 0:  # 水分子 (3 原子)
            z = torch.tensor([8, 1, 1], dtype=torch.long)
            pos = torch.randn(3, 3, dtype=torch.float32) * 0.5
            y = torch.tensor([[-10.0]], dtype=torch.float64)
        elif mol_type == 1:  # 甲烷 (5 原子)
            z = torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)
            pos = torch.randn(5, 3, dtype=torch.float32) * 0.7
            y = torch.tensor([[-15.0]], dtype=torch.float64)
        elif mol_type == 2:  # 氨 (4 原子)
            z = torch.tensor([7, 1, 1, 1], dtype=torch.long)
            pos = torch.randn(4, 3, dtype=torch.float32) * 0.6
            y = torch.tensor([[-8.0]], dtype=torch.float64)
        else:  # 苯 (12 原子)
            z = torch.tensor([6] * 6 + [1] * 6, dtype=torch.long)
            pos = torch.randn(12, 3, dtype=torch.float32) * 1.0
            y = torch.tensor([[-20.0]], dtype=torch.float64)
        
        data = Data(z=z, pos=pos, y=y)
        data_list.append(data)
    
    print(f"  成功创建大型数据集")
    return data_list


def benchmark_model_forward(model, data_loader, num_iterations=10):
    """
    基准测试模型前向传播性能
    """
    print(f"基准测试模型前向传播性能 ({num_iterations} 次迭代)...")
    
    model.eval()
    times = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            iteration_start = time.time()
            
            for batch in data_loader:
                _, _ = model(batch.z, batch.pos, batch=batch.batch)
            
            iteration_time = time.time() - iteration_start
            times.append(iteration_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"  平均每次迭代时间: {avg_time:.4f} ± {std_time:.4f} 秒")
    print(f"  最快迭代时间: {np.min(times):.4f} 秒")
    print(f"  最慢迭代时间: {np.max(times):.4f} 秒")
    
    return avg_time, std_time


def test_memory_usage(model, data_loader):
    """
    测试内存使用情况
    """
    print("\n测试内存使用情况...")
    
    try:
        import psutil
        import gc
        
        # 获取初始内存使用
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"  初始内存使用: {initial_memory:.2f} MB")
        
        # 执行几次前向传播
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= 3:  # 只测试前3个批次
                    break
                _, _ = model(batch.z, batch.pos, batch=batch.batch)
        
        # 强制垃圾回收
        gc.collect()
        
        # 获取最终内存使用
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"  最终内存使用: {final_memory:.2f} MB")
        print(f"  内存增长: {memory_increase:.2f} MB")
        
        return memory_increase
    except ImportError:
        print("  无法导入 psutil，跳过内存测试")
        return None


def compare_model_types():
    """
    比较不同模型类型的性能
    """
    print("\n比较不同模型类型的性能...")
    
    # 创建测试数据
    dataset = create_large_dataset(50)  # 使用较小的数据集进行比较
    data_loader = PyGDataLoader(dataset, batch_size=10, shuffle=False)
    
    model_configs = [
        {
            "name": "TensorNet",
            "args": {
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
                "model": "tensornet",
                "aggr": "add",
                "derivative": False,
                "atom_filter": -1,
                "prior_model": None,
                "output_model": "Scalar",
                "reduce_op": "add",
                "precision": 32,
            }
        },
        {
            "name": "EquivariantTransformer",
            "args": {
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
                "model": "equivariant-transformer",
                "aggr": "add",
                "derivative": False,
                "atom_filter": -1,
                "prior_model": None,
                "output_model": "Scalar",
                "reduce_op": "add",
                "precision": 32,
                "attn_activation": "silu",
                "num_heads": 4,
                "distance_influence": "both",
                "neighbor_embedding": True,
            }
        }
    ]
    
    results = []
    
    for config in model_configs:
        print(f"\n  测试 {config['name']}...")
        try:
            # 创建模型
            model = create_model(config['args'])
            print(f"    模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
            
            # 测试前向传播时间
            start_time = time.time()
            model.eval()
            with torch.no_grad():
                for batch in data_loader:
                    _, _ = model(batch.z, batch.pos, batch=batch.batch)
                    break  # 只测试第一个批次
            
            elapsed_time = time.time() - start_time
            print(f"    前向传播时间: {elapsed_time:.6f} 秒")
            
            results.append({
                "model": config['name'],
                "params": sum(p.numel() for p in model.parameters()),
                "time": elapsed_time
            })
            
        except Exception as e:
            print(f"    {config['name']} 测试失败: {e}")
    
    # 打印比较结果
    print("\n  性能比较结果:")
    print("  {:<20} {:<15} {:<15}".format("模型", "参数数量", "前向传播时间"))
    print("  " + "-" * 50)
    for result in results:
        print("  {:<20} {:<15,} {:<15.6f}".format(
            result['model'], 
            result['params'], 
            result['time']
        ))


def test_scalability_with_batch_size():
    """
    测试不同批次大小的可扩展性
    """
    print("\n测试不同批次大小的可扩展性...")
    
    # 创建中等大小的数据集
    dataset = create_large_dataset(100)
    
    batch_sizes = [1, 5, 10, 20, 50]
    results = []
    
    # 创建模型
    model_args = {
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
        "model": "tensornet",
        "aggr": "add",
        "derivative": False,
        "atom_filter": -1,
        "prior_model": None,
        "output_model": "Scalar",
        "reduce_op": "add",
        "precision": 32,
    }
    
    model = create_model(model_args)
    model.eval()
    
    for batch_size in batch_sizes:
        try:
            data_loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            # 测试时间
            start_time = time.time()
            with torch.no_grad():
                for batch in data_loader:
                    _, _ = model(batch.z, batch.pos, batch=batch.batch)
                    break  # 只测试第一个批次
            
            elapsed_time = time.time() - start_time
            
            print(f"  批次大小 {batch_size:2d}: {elapsed_time:.6f} 秒")
            
            results.append({
                "batch_size": batch_size,
                "time": elapsed_time
            })
            
        except Exception as e:
            print(f"  批次大小 {batch_size:2d}: 测试失败 - {e}")
    
    return results


def test_gpu_performance():
    """
    测试GPU性能（如果可用）
    """
    print("\n测试GPU性能...")
    
    if not torch.cuda.is_available():
        print("  CUDA不可用，跳过GPU测试")
        return
    
    device = torch.device('cuda')
    print(f"  使用设备: {device} ({torch.cuda.get_device_name(0)})")
    
    # 创建模型并移动到GPU
    model_args = {
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
        "model": "tensornet",
        "aggr": "add",
        "derivative": False,
        "atom_filter": -1,
        "prior_model": None,
        "output_model": "Scalar",
        "reduce_op": "add",
        "precision": 32,
    }
    
    model = create_model(model_args).to(device)
    
    # 创建数据并移动到GPU
    z = torch.tensor([8, 1, 1] * 20, dtype=torch.long).to(device)  # 60个原子
    pos = torch.randn(60, 3, dtype=torch.float32).to(device)
    batch = torch.tensor([i//20 for i in range(60)], dtype=torch.long).to(device)
    
    # GPU预热
    with torch.no_grad():
        for _ in range(5):
            _, _ = model(z, pos, batch=batch)
    
    # 测试GPU性能
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            _, _ = model(z, pos, batch=batch)
    
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    
    print(f"  GPU前向传播时间 (10次迭代): {gpu_time:.6f} 秒")
    print(f"  平均每次迭代: {gpu_time/10:.6f} 秒")


if __name__ == "__main__":
    print("开始测试 TorchMD-Net 模型性能")
    print("=" * 50)
    
    # 创建数据集
    large_dataset = create_large_dataset(100)
    data_loader = PyGDataLoader(large_dataset, batch_size=10, shuffle=False)
    
    # 测试模型创建
    model_args = {
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
        "model": "tensornet",
        "aggr": "add",
        "derivative": False,
        "atom_filter": -1,
        "prior_model": None,
        "output_model": "Scalar",
        "reduce_op": "add",
        "precision": 32,
    }
    
    model = create_model(model_args)
    
    # 运行各种性能测试
    benchmark_model_forward(model, data_loader)
    test_memory_usage(model, data_loader)
    compare_model_types()
    test_scalability_with_batch_size()
    test_gpu_performance()
    
    print("\n" + "=" * 50)
    print("TorchMD-Net 性能测试完成")