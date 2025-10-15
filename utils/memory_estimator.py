#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型显存估算工具
用于在训练前根据网络架构和batch大小估算需要的显存大小
"""

import torch
import numpy as np
from torch_geometric.data import Data
import gc


def estimate_memory_usage(model, batch_size, avg_nodes_per_graph=20, avg_edges_per_graph=30, 
                         node_feature_dim=11, edge_feature_dim=11, verbose=False):
    """
    估算模型训练时的显存使用量
    
    Args:
        model: 要估算的模型实例
        batch_size: 批处理大小
        avg_nodes_per_graph: 每个图的平均节点数
        avg_edges_per_graph: 每个图的平均边数
        node_feature_dim: 节点特征维度
        edge_feature_dim: 边特征维度
        verbose: 是否输出详细信息
    
    Returns:
        dict: 包含各种内存使用情况的字典
    """
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # 获取初始显存使用量
    initial_memory = 0
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated()
    
    # 创建模拟数据
    # 假设每个分子图有大约 avg_nodes_per_graph 个节点
    num_graphs = batch_size * 2  # from和to两个图
    
    # 构建模拟图数据
    from_data_list = []
    to_data_list = []
    
    for i in range(batch_size):
        # 创建起始分子图
        num_nodes = max(3, int(np.random.normal(avg_nodes_per_graph, avg_nodes_per_graph * 0.3)))
        num_edges = max(num_nodes, int(np.random.normal(avg_edges_per_graph, avg_edges_per_graph * 0.3)))
        
        # 确保边索引有效
        edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
        
        from_data = Data(
            x=torch.randn(num_nodes, node_feature_dim, requires_grad=True),
            edge_index=edge_index
        )
        
        # 创建目标分子图
        num_nodes_to = max(3, int(np.random.normal(avg_nodes_per_graph, avg_nodes_per_graph * 0.3)))
        num_edges_to = max(num_nodes_to, int(np.random.normal(avg_edges_per_graph, avg_edges_per_graph * 0.3)))
        
        edge_index_to = torch.randint(0, num_nodes_to, (2, num_edges_to), dtype=torch.long)
        
        to_data = Data(
            x=torch.randn(num_nodes_to, node_feature_dim, requires_grad=True),
            edge_index=edge_index_to
        )
        
        from_data_list.append(from_data)
        to_data_list.append(to_data)
    
    # 创建边特征 (操作信息)
    edge_attrs = torch.randn(batch_size, edge_feature_dim, requires_grad=True)
    targets = torch.randn(batch_size, 1, requires_grad=True)
    
    # 使用PyG的Batch来合并图
    from torch_geometric.data import Batch
    from_batch = Batch.from_data_list(from_data_list)
    to_batch = Batch.from_data_list(to_data_list)
    
    if verbose:
        print(f"模拟数据信息:")
        print(f"  Batch大小: {batch_size}")
        print(f"  起始图Batch节点数: {from_batch.x.shape[0]}")
        print(f"  目标图Batch节点数: {to_batch.x.shape[0]}")
        print(f"  边特征维度: {edge_attrs.shape}")
    
    # 将数据移到GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    from_batch = from_batch.to(device)
    to_batch = to_batch.to(device)
    edge_attrs = edge_attrs.to(device)
    targets = targets.to(device)
    
    # 获取模型参数内存使用量
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    if verbose:
        print(f"模型参数内存使用量: {param_memory / 1024 / 1024:.2f} MB")
    
    # 前向传播估算
    try:
        model.eval()  # 使用eval模式避免一些训练时特有的操作
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        
        # 前向传播
        with torch.no_grad():
            outputs = model(from_batch, to_batch, edge_attrs)
        
        forward_memory = 0
        if torch.cuda.is_available():
            forward_memory = torch.cuda.max_memory_allocated() - initial_memory
        
        if verbose:
            print(f"前向传播后显存使用量: {forward_memory / 1024 / 1024:.2f} MB")
        
        # 反向传播估算（模拟）
        loss = torch.mean((outputs - targets) ** 2)
        # 创建新的带有梯度的变量用于反向传播测试
        loss_for_backward = loss + 0.0  # 创建一个有梯度的新变量
        loss_for_backward.backward()
        
        backward_memory = 0
        if torch.cuda.is_available():
            backward_memory = torch.cuda.max_memory_allocated() - initial_memory
        
        if verbose:
            print(f"反向传播后显存使用量: {backward_memory / 1024 / 1024:.2f} MB")
        
        # 清理
        del from_batch, to_batch, edge_attrs, targets, outputs, loss, loss_for_backward
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        # 返回估算结果
        result = {
            'param_memory_mb': param_memory / 1024 / 1024,
            'forward_memory_mb': forward_memory / 1024 / 1024 if forward_memory > 0 else param_memory / 1024 / 1024 * 2,
            'backward_memory_mb': backward_memory / 1024 / 1024 if backward_memory > 0 else param_memory / 1024 / 1024 * 4,
            'estimated_total_mb': (param_memory * 4) / 1024 / 1024,  # 粗略估算
            'batch_size': batch_size,
            'device': str(device)
        }
        
        return result
        
    except Exception as e:
        # 清理
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        raise e


def print_memory_estimate(model, batch_size, model_name=""):
    """
    打印模型显存估算结果
    
    Args:
        model: 模型实例
        batch_size: 批处理大小
        model_name: 模型名称
    """
    try:
        result = estimate_memory_usage(model, batch_size)
        
        print("=" * 50)
        if model_name:
            print(f"模型显存使用量估算: {model_name}")
        else:
            print("模型显存使用量估算")
        print("=" * 50)
        print(f"模型参数内存: {result['param_memory_mb']:.2f} MB")
        print(f"前向传播内存: {result['forward_memory_mb']:.2f} MB")
        print(f"反向传播内存: {result['backward_memory_mb']:.2f} MB")
        print(f"预估总内存: {result['estimated_total_mb']:.2f} MB")
        print(f"批处理大小: {result['batch_size']}")
        print(f"设备: {result['device']}")
        print("=" * 50)
        
        return result
    except Exception as e:
        print(f"显存估算失败: {str(e)}")
        return None