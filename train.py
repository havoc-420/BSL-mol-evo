#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练基于分子进化的转换器模型

该脚本训练转换器模型，用于根据起始分子特征、原子类型和操作类型预测目标分子特征。
"""

import sys
import os
import argparse
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
from datetime import datetime

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建项目根目录路径
project_root = os.path.join(script_dir, '..')
# 构建model-data目录路径
model_data_dir = os.path.join(project_root, 'mol_evo', 'model-data')
# 添加项目根目录到Python路径
sys.path.insert(0, project_root)

# 更新导入语句以使用新的模块结构
from core.models.predictors import MoleculeEvolutionTransformer
from core.data.processing import prepare_evolution_data
from core.utils.training import train_transformer_model


def split_data_indices(total_count, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    按照指定比例划分数据集索引
    
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


def train_transformer(data_file: str, max_pairs: int = None, epochs: int = 100):
    """
    训练分子进化转换器模型
    
    Args:
        data_file: 数据文件路径
        max_pairs: 最大对数（用于调试）
        epochs: 训练轮数
    """
    print("=" * 60)
    print("分子进化转换器模型训练")
    print("=" * 60)
    
    # 加载数据
    print(f"正在加载数据: {data_file}")
    source_features, edge_features, target_features, property_stats = prepare_evolution_data(data_file, max_pairs)
    
    print(f"数据加载完成:")
    print(f"  - 起始分子特征维度: {source_features.shape[1]}")
    print(f"  - 边特征维度: {edge_features.shape[1]}")
    print(f"  - 目标分子特征维度: {target_features.shape[1]}")
    print(f"  - 分子对数: {source_features.shape[0]}")
    
    # 划分数据集
    train_idx, val_idx, test_idx = split_data_indices(source_features.shape[0], 0.7, 0.2, 0.1)
    
    # 创建模型
    print("\n正在创建模型...")
    model = MoleculeEvolutionTransformer(
        node_feature_dim=2048,  # Morgan指纹维度
        edge_feature_dim=30,    # 边特征维度（保持为30，与数据处理一致）
        hidden_dim=128,
        output_dim=2048         # 目标分子指纹维度
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    print(f"\n开始训练 ({epochs} 轮)...")
    train_losses, val_losses = train_transformer_model(
        model, source_features, edge_features, target_features,
        epochs=epochs, lr=0.001, train_idx=train_idx, val_idx=val_idx
    )
    
    # 测试阶段
    model.eval()
    with torch.no_grad():
        predictions = model(source_features[test_idx], edge_features[test_idx])
        criterion = nn.MSELoss()
        test_loss = criterion(predictions, target_features[test_idx])
    
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(model_data_dir, f"training_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出最终损失
    print(f"\n训练完成:")
    print(f"  - 最终训练损失: {train_losses[-1]:.6f}")
    print(f"  - 最佳验证损失: {min(val_losses) if val_losses else 'N/A':.6f}")
    print(f"  - 测试损失: {test_loss.item():.6f}")
    
    # 保存模型到带时间戳的目录
    model_filename = "molecule_evolution_transformer.pth"
    model_path = os.path.join(output_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"  - 模型已保存到: {model_path}")
    
    # 保存训练日志到带时间戳的目录
    log_filename = "transformer_training_log.txt"
    log_path = os.path.join(output_dir, log_filename)
    with open(log_path, 'w') as f:
        f.write(f"训练完成报告\n")
        f.write(f"==================\n")
        f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"训练集大小: {len(train_idx)}\n")
        f.write(f"验证集大小: {len(val_idx)}\n")
        f.write(f"测试集大小: {len(test_idx)}\n")
        f.write(f"最终训练损失: {train_losses[-1]:.6f}\n")
        f.write(f"最佳验证损失: {min(val_losses) if val_losses else 'N/A'}\n")
        f.write(f"测试损失: {test_loss.item():.6f}\n")
        f.write(f"训练轮数: {epochs}\n")
    print(f"  - 训练日志已保存到: {log_path}")
    
    return model, train_losses, val_losses


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练分子进化转换器模型')
    parser.add_argument(
        '--data', 
        type=str, 
        default=os.path.join(project_root, 'mol_evo', 'dataset', 'data', 'qm9-evo-pairs-step-1-with-properties.csv'),
        help='数据文件路径'
    )
    parser.add_argument(
        '--max-pairs', 
        type=int, 
        default=None,
        help='最大分子对数（用于调试）'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=100,
        help='训练轮数'
    )
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data):
        print(f"错误: 数据文件不存在: {args.data}")
        # 尝试在项目根目录下查找
        alternative_path = os.path.join(project_root, 'dataset', 'data', 'qm9-evo-pairs-step-1-with-properties.csv')
        if os.path.exists(alternative_path):
            print(f"在替代路径找到数据文件: {alternative_path}")
            args.data = alternative_path
        else:
            sys.exit(1)
    
    # 训练模型
    try:
        model, train_losses, val_losses = train_transformer(
            args.data, 
            max_pairs=args.max_pairs, 
            epochs=args.epochs
        )
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()