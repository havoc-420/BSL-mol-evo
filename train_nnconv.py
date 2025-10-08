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
import json
import matplotlib.pyplot as plt
import logging

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


def setup_logger(model_dir):
    """
    设置日志记录器
    
    Args:
        model_dir: 模型目录路径
        
    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger('nnconv_training')
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 创建文件处理器
        log_file = os.path.join(model_dir, "training.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式器并添加到处理器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


def inverse_standardize(predictions, property_stats):
    """
    反标准化预测结果
    
    Args:
        predictions: 标准化的预测结果
        property_stats: 属性统计信息（均值和标准差）
        
    Returns:
        反标准化后的预测结果
    """
    # 属性名称顺序必须与处理时一致
    property_names = ['A_change', 'B_change', 'C_change', 'mu_change', 'alpha_change',
                      'homo_change', 'lumo_change', 'gap_change', 'r2_change', 'zpve_change',
                      'U0_change', 'U_change', 'H_change', 'G_change', 'Cv_change']
    
    # 创建副本避免修改原始数据
    inv_predictions = predictions.clone()
    
    # 对每个属性进行反标准化
    for i, prop in enumerate(property_names):
        if prop in property_stats:
            mean, std = property_stats[prop]
            if std > 0:
                inv_predictions[:, i] = predictions[:, i] * std + mean
    
    return inv_predictions


def split_data_indices(total_count: int, train_ratio: float = 0.7, 
                      test_ratio: float = 0.1, val_ratio: float = 0.2, 
                      seed: int = 42) -> tuple:
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
    
    return train_idx, val_idx, test_idx


def save_training_data_as_json(train_losses, val_losses, test_metrics, model_dir):
    """
    将训练数据保存为JSON格式
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        test_metrics: 测试指标字典
        model_dir: 模型目录路径
    """
    # 构建训练数据字典
    training_data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_metrics": test_metrics
    }
    
    # 保存为JSON文件
    json_path = os.path.join(model_dir, "training_data.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)


def plot_training_trends(train_losses, val_losses, model_dir):
    """
    绘制训练趋势图
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        model_dir: 模型目录路径
    """
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 绘制训练和验证损失
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'o-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 's-', label='Validation Loss', linewidth=2)
    
    # 设置图表属性
    plt.title('Training and Validation Loss Trends')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plot_path = os.path.join(model_dir, "training_trends.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图表以释放内存


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
    
    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(project_root, 'mol_evo', 'model-data', f"training_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # 设置日志记录器
    logger = setup_logger(model_dir)
    logger.info("=" * 60)
    logger.info("基于NNConv的分子进化预测器模型训练开始")
    logger.info("=" * 60)
    logger.info(f"数据文件: {data_file}")
    logger.info(f"最大对数: {max_pairs}")
    logger.info(f"训练轮数: {epochs}")
    
    # 构建图数据
    logger.info(f"正在构建图数据: {data_file}")
    data, smiles_to_idx, property_stats = build_molecule_graph_with_fingerprints(data_file, max_pairs)
    
    # 准备属性变化目标
    target_features = prepare_property_change_targets(data_file, property_stats, max_pairs)
    
    logger.info(f"数据构建完成:")
    logger.info(f"  - 节点数: {data.num_nodes}")
    logger.info(f"  - 边数: {data.num_edges}")
    logger.info(f"  - 节点特征维度: {data.x.shape[1]}")
    logger.info(f"  - 边特征维度: {data.edge_attr.shape[1]}")
    logger.info(f"  - 目标属性变化维度: {target_features.shape[1]}")
    
    # 划分数据集
    train_idx, val_idx, test_idx = split_data_indices(data.num_edges, 0.7, 0.2, 0.1)
    
    logger.info(f"数据集划分完成:")
    logger.info(f"  - 训练集: {len(train_idx)} ({len(train_idx)/data.num_edges*100:.1f}%)")
    logger.info(f"  - 验证集: {len(val_idx)} ({len(val_idx)/data.num_edges*100:.1f}%)")
    logger.info(f"  - 测试集: {len(test_idx)} ({len(test_idx)/data.num_edges*100:.1f}%)")
    
    # 创建模型
    logger.info("正在创建模型...")
    model = MoleculeEvolutionNNConvPredictor(
        node_feature_dim=2048,  # Morgan指纹维度
        edge_feature_dim=11,    # 边特征维度（5原子类型 + 6操作类型）
        hidden_dim=128,
        output_dim=15,          # 属性变化维度
        num_layers=3
    )
    
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    logger.info(f"开始训练 ({epochs} 轮)...")
    train_losses, val_losses = train_gnn_model(
        model, data, target_features,
        epochs=epochs, lr=0.001, train_idx=train_idx, val_idx=val_idx, logger=logger
    )
    
    # 定义属性名称列表
    property_names = ['A_change', 'B_change', 'C_change', 'mu_change', 'alpha_change',
                      'homo_change', 'lumo_change', 'gap_change', 'r2_change', 'zpve_change',
                      'U0_change', 'U_change', 'H_change', 'G_change', 'Cv_change']

    def get_accuracy_color_code(accuracy):
        """
        根据准确率值返回相应的ANSI颜色代码
        
        Args:
            accuracy: 准确率值 (0-1)
            
        Returns:
            ANSI颜色代码字符串
        """
        if accuracy >= 0.9:
            return '\033[32m'  # 绿色 - 优秀
        elif accuracy >= 0.7:
            return '\033[36m'  # 青色 - 良好
        elif accuracy >= 0.5:
            return '\033[33m'  # 黄色 - 一般
        elif accuracy >= 0.3:
            return '\033[35m'  # 紫色 - 较差
        else:
            return '\033[31m'  # 红色 - 很差
    
    
    def print_table_accuracy(accuracies, thresholds, property_names):
        """
        以表格形式打印各维度阈值准确率，并添加颜色分区效果
        先构建不带颜色的字符串确保对齐，再添加颜色
        表格按横向展示形式，第一行为属性名称，后续每行显示一个阈值下所有属性的准确率

        Args:
            accuracies: 不同阈值下的准确率字典
            thresholds: 阈值列表
            property_names: 属性名称列表
        """
        print("各维度阈值准确率:")

        # 定义列宽
        col_width = 9
        threshold_col_width = 12

        # 构建表头
        header = "Threshold".ljust(threshold_col_width)
        for prop_name in property_names:
            # 去除"_change"后缀并截取或填充属性名到固定宽度
            clean_name = prop_name.replace("_change", "")
            short_name = (clean_name[:col_width-1] if len(clean_name) >= col_width else clean_name).ljust(col_width)
            header += short_name
        print(header)

        # 打印分隔线
        separator_length = threshold_col_width + len(property_names) * col_width
        print("-" * separator_length)

        # 逐行打印每个阈值下的准确率（带颜色）
        for threshold in thresholds:
            row = f"@{threshold:.3f}".ljust(threshold_col_width)
            for i in range(len(property_names)):
                acc = accuracies[threshold][i]
                color_code = get_accuracy_color_code(acc)
                formatted_acc = f"{acc:.4f}"
                # 构建带颜色的单元格，确保固定宽度
                colored_cell = f"{color_code}{formatted_acc}\033[0m"
                # 手动计算并添加空格以保持对齐
                padding = col_width - len(formatted_acc)
                row += colored_cell + " " * padding
            print(row)

    # 测试阶段
    model.eval()
    with torch.no_grad():
        predictions = model(data)
        criterion = nn.MSELoss()
        test_loss = criterion(predictions[test_idx], target_features[test_idx])
        
        # 计算额外的评估指标
        test_predictions = predictions[test_idx]
        test_targets = target_features[test_idx]
        
        # RMSE (Root Mean Square Error)
        mse = torch.mean((test_predictions - test_targets) ** 2)
        rmse = torch.sqrt(mse)
        
        # MAE (Mean Absolute Error)
        mae = torch.mean(torch.abs(test_predictions - test_targets))
        
        # R² (Coefficient of Determination)
        ss_res = torch.sum((test_targets - test_predictions) ** 2)
        ss_tot = torch.sum((test_targets - torch.mean(test_targets)) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        # 阈值准确率评估
        thresholds = [0.1, 0.05]
        threshold_accs = {}
        
        # 计算各维度阈值准确率
        dimension_accuracies = {}
        for th in thresholds:
            # 计算所有维度同时满足阈值的样本比例
            all_dims_correct = (torch.abs(test_predictions - test_targets) < th).all(dim=1).float()
            threshold_accs[f'all_{th}'] = all_dims_correct.mean().item()
            
            # 计算整体平均准确率（每个维度单独计算然后平均）
            dim_correct = (torch.abs(test_predictions - test_targets) < th).float()
            threshold_accs[f'mean_{th}'] = dim_correct.mean().item()
            
            # 保存各维度的准确率
            dimension_accuracies[th] = dim_correct.mean(dim=0).cpu().numpy()
        
        # 反标准化预测结果和目标值以获得原始尺度的评估指标
        original_predictions = inverse_standardize(test_predictions, property_stats)
        original_targets = inverse_standardize(test_targets, property_stats)
        
        # 在原始尺度上计算评估指标
        orig_mse = torch.mean((original_predictions - original_targets) ** 2)
        orig_rmse = torch.sqrt(orig_mse)
        orig_mae = torch.mean(torch.abs(original_predictions - original_targets))
        
        logger.info(f"原始尺度评估指标:")
        logger.info(f"  - MSE: {orig_mse.item():.6f}")
        logger.info(f"  - RMSE: {orig_rmse.item():.6f}")
        logger.info(f"  - MAE: {orig_mae.item():.6f}")

        # 使用表格形式展示各维度阈值准确率
        print_table_accuracy(dimension_accuracies, thresholds, property_names)

    # 保存模型
    model_path = os.path.join(model_dir, "molecule_evolution_nnconv_predictor.pth")
    torch.save(model.state_dict(), model_path)
    
    # 准备测试指标数据
    test_metrics = {
        "test_loss": test_loss.item(),
        "rmse": rmse.item(),
        "mae": mae.item(),
        "r2": r2.item(),
        "threshold_accs": threshold_accs,
        "dimension_accuracies": {str(th): acc.tolist() for th, acc in dimension_accuracies.items()},
        "original_scale_metrics": {
            "mse": orig_mse.item(),
            "rmse": orig_rmse.item(),
            "mae": orig_mae.item()
        },
        "dataset_info": {
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx)
        }
    }
    
    # 保存训练数据为JSON格式
    save_training_data_as_json(train_losses, val_losses, test_metrics, model_dir)
    
    # 生成训练趋势图
    plot_training_trends(train_losses, val_losses, model_dir)
    
    # 记录完整评估结果到日志
    logger.info(f"训练完成:")
    logger.info(f"  - 最终训练损失: {train_losses[-1]:.6f}")
    logger.info(f"  - 最佳验证损失: {min(val_losses):.6f}")
    logger.info(f"  - 测试损失 (MSE): {test_loss.item():.6f}")
    logger.info(f"  - 测试RMSE: {rmse.item():.6f}")
    logger.info(f"  - 测试MAE: {mae.item():.6f}")
    logger.info(f"  - 测试R²: {r2.item():.6f}")
    logger.info(f"  - 测试阈값准确率 (0.1, 所有维度): {threshold_accs['all_0.1']:.4f}")
    logger.info(f"  - 测试阈값准确率 (0.1, 平均): {threshold_accs['mean_0.1']:.4f}")
    logger.info(f"  - 测试阈값准确率 (0.05, 所有维度): {threshold_accs['all_0.05']:.4f}")
    logger.info(f"  - 测试阈값准确率 (0.05, 平均): {threshold_accs['mean_0.05']:.4f}")
    
    logger.info(f"  - 原始尺度MSE: {orig_mse.item():.6f}")
    logger.info(f"  - 原始尺度RMSE: {orig_rmse.item():.6f}")
    logger.info(f"  - 原始尺度MAE: {orig_mae.item():.6f}")
    logger.info(f"  - 模型已保存到: {model_path}")
    
    # 记录各维度准确率到日志
    logger.info("各维度阈值准确率详情:")
    for th in thresholds:
        logger.info(f"  阈值 @{th}:")
        for i, prop_name in enumerate(property_names):
            logger.info(f"    {prop_name}: {dimension_accuracies[th][i]:.4f}")
    
    print(f"\n训练完成:")
    print(f"  - 最终训练损失: {train_losses[-1]:.6f}")
    print(f"  - 最佳验证损失: {min(val_losses):.6f}")
    print(f"  - 测试损失 (MSE): {test_loss.item():.6f}")
    print(f"  - 测试RMSE: {rmse.item():.6f}")
    print(f"  - 测试MAE: {mae.item():.6f}")
    print(f"  - 测试R²: {r2.item():.6f}")
    print(f"  - 测试阈값准确率 (0.1, 所有维度): {threshold_accs['all_0.1']:.4f}")
    print(f"  - 测试阈값准确率 (0.1, 平均): {threshold_accs['mean_0.1']:.4f}")
    print(f"  - 测试阈값准确率 (0.05, 所有维度): {threshold_accs['all_0.05']:.4f}")
    print(f"  - 测试阈값准确率 (0.05, 平均): {threshold_accs['mean_0.05']:.4f}")
    
    print(f"  - 原始尺度MSE: {orig_mse.item():.6f}")
    print(f"  - 原始尺度RMSE: {orig_rmse.item():.6f}")
    print(f"  - 原始尺度MAE: {orig_mae.item():.6f}")
    print(f"  - 模型已保存到: {model_path}")
    print(f"  - 完整日志已保存到: {os.path.join(model_dir, 'training.log')}")
    
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