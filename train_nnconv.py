#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练基于NNConv的分子进化预测器模型
"""

import sys
import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.optim as optim
import random
from datetime import datetime

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建项目根目录路径
project_root = os.path.join(script_dir, '..')
# 添加项目根目录到Python路径
sys.path.insert(0, project_root)

# 导入自定义模块
from mol_evo.core.models.nnconv_predictor import MoleculeEvolutionNNConvPredictor
from mol_evo.core.data.processing import build_molecule_graph_with_fingerprints, prepare_property_change_targets
from mol_evo.core.utils.training import train_gnn_model


def split_data_indices(total_count: int, train_ratio: float = 0.7, val_ratio: float = 0.2, 
                      test_ratio: float = 0.1, seed: int = 42) -> tuple:
    """
    划分数据集索引
    
    Args:
        total_count: 总数据量
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        
    Returns:
        训练集、验证集和测试集的索引
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "数据集划分比例之和必须为1"
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 计算各数据集大小
    train_size = int(total_count * train_ratio)
    val_size = int(total_count * val_ratio)
    test_size = total_count - train_size - val_size
    
    # 创建索引列表并打乱
    indices = list(range(total_count))
    np.random.shuffle(indices)
    
    # 划分索引
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    print(f"数据集划分完成:")
    print(f"  - 训练集: {len(train_idx)} ({len(train_idx)/total_count*100:.1f}%)")
    print(f"  - 验证集: {len(val_idx)} ({len(val_idx)/total_count*100:.1f}%)")
    print(f"  - 测试集: {len(test_idx)} ({len(test_idx)/total_count*100:.1f}%)")
    
    return train_idx, val_idx, test_idx


def train_nnconv_model(data_file: str, max_pairs: int = None, epochs: int = 100):
    """
    训练基于NNConv的分子进化预测器模型
    
    Args:
        data_file: 数据文件路径
        max_pairs: 最大对数（用于调试）
        epochs: 训练轮数
    """
    print("=" * 60)
    print("基于NNConv的分子进化预测器模型训练")
    print("=" * 60)
    
    # 构建图数据
    print(f"正在构建图数据: {data_file}")
    data, smiles_to_idx, property_stats = build_molecule_graph_with_fingerprints(data_file, max_pairs)
    
    # 准备属性变化目标
    target_features = prepare_property_change_targets(data_file, property_stats, max_pairs)
    
    print(f"数据构建完成:")
    print(f"  - 节点数: {data.num_nodes}")
    print(f"  - 边数: {data.num_edges}")
    print(f"  - 节点特征维度: {data.x.shape[1]}")
    print(f"  - 边特征维度: {data.edge_attr.shape[1]}")
    print(f"  - 目标属性变化维度: {target_features.shape[1]}")
    
    # 划分数据集
    train_idx, val_idx, test_idx = split_data_indices(data.num_edges, 0.7, 0.2, 0.1)
    
    # 创建模型
    print("\n正在创建模型...")
    model = MoleculeEvolutionNNConvPredictor(
        node_feature_dim=2048,  # Morgan指纹维度
        edge_feature_dim=11,    # 边特征维度（5原子类型 + 6操作类型）
        hidden_dim=128,
        output_dim=15,          # 属性变化维度
        num_layers=3
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    print(f"\n开始训练 ({epochs} 轮)...")
    train_losses, val_losses = train_gnn_model(
        model, data, target_features,
        epochs=epochs, lr=0.001, train_idx=train_idx, val_idx=val_idx
    )
    
    # 测试阶段
    model.eval()
    with torch.no_grad():
        predictions = model(data)
        criterion = nn.MSELoss()
        test_loss = criterion(predictions[test_idx], target_features[test_idx])
    
    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(project_root, 'mol_evo', 'model-data', f"training_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "molecule_evolution_nnconv_predictor.pth")
    torch.save(model.state_dict(), model_path)
    
    # 保存训练日志
    log_path = os.path.join(model_dir, "nnconv_training_log.txt")
    with open(log_path, 'w') as f:
        f.write(f"训练完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"训练轮数: {epochs}\n")
        f.write(f"最终训练损失: {train_losses[-1]:.6f}\n")
        f.write(f"最佳验证损失: {min(val_losses):.6f}\n")
        f.write(f"测试损失: {test_loss.item():.6f}\n")
        f.write(f"训练集大小: {len(train_idx)}\n")
        f.write(f"验证集大小: {len(val_idx)}\n")
        f.write(f"测试集大小: {len(test_idx)}\n")
    
    print(f"\n训练完成:")
    print(f"  - 最终训练损失: {train_losses[-1]:.6f}")
    print(f"  - 最佳验证损失: {min(val_losses):.6f}")
    print(f"  - 测试损失: {test_loss.item():.6f}")
    print(f"  - 模型已保存到: {model_path}")
    print(f"  - 训练日志已保存到: {log_path}")
    
    return model, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description='训练基于NNConv的分子进化预测器模型')
    parser.add_argument('--data-file', type=str, 
                       default='mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv',
                       help='数据文件路径')
    parser.add_argument('--max-pairs', type=int, help='最大分子对数（用于调试）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    
    args = parser.parse_args()
    
    try:
        model, train_losses, val_losses = train_nnconv_model(
            args.data_file, args.max_pairs, args.epochs
        )
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()