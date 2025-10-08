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
import random
from datetime import datetime
import json
import matplotlib.pyplot as plt
import logging

""" BASE SETTINGS """

BASE_DIR_NAME = "nnconv"

# 定义模型参数字典
model_params = {
    "node_feature_dim": 2048,
    "edge_feature_dim": 11,
    "hidden_dim": 128,
    "output_dim": 15,
    "num_layers": 3
}

""" 设置项目根目录路径 """

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
from mol_evo.utils.training_utils import (
    setup_logger, 
    inverse_standardize, 
    split_data_indices, 
    save_training_data_as_json, 
    plot_training_trends,
    print_table_accuracy,
    log_training_completion,
    log_training_start,
    log_data_construction,
    log_dataset_split,
    log_model_creation,
    log_training_start_message,
    log_original_scale_metrics
)


def plot_training_trends(train_losses, val_losses, val_r2s, val_maes, model_dir):
    """
    绘制训练趋势图
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        val_r2s: 验证R²值列表
        val_maes: 验证MAE值列表
        model_dir: 模型目录路径
    """
    # 创建损失图表
    plt.figure(figsize=(10, 6))
    
    # 绘制训练和验证损失
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'o-', label='Training Loss', linewidth=1, markersize=1.5, alpha=0.7)
    plt.plot(epochs, val_losses, 'o-', label='Validation Loss', linewidth=1, markersize=1.5, alpha=0.7)
    plt.title('Training and Validation Loss Trends')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 调整布局并保存损失图表
    plt.tight_layout()
    loss_plot_path = os.path.join(model_dir, "loss_trends.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图表以释放内存
    print(f"损失趋势图已保存: {loss_plot_path}")
    
    # 创建其他指标图表
    plt.figure(figsize=(10, 5))
    
    # 绘制验证R2趋势
    plt.subplot(1, 2, 1)
    val_epochs = range(10, len(train_losses) + 1, 10)  # R2和MAE每10个epoch记录一次
    plt.plot(val_epochs, val_r2s, 'o-', label='Validation R²', linewidth=2, markersize=1.5, color='green', alpha=0.7)
    plt.title('Validation R² Trend')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制验证MAE趋势
    plt.subplot(1, 2, 2)
    plt.plot(val_epochs, val_maes, 's-', label='Validation MAE', linewidth=2, markersize=1.5, color='red', alpha=0.7)
    plt.title('Validation MAE Trend')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 调整子图间距并保存其他指标图表
    plt.tight_layout()
    metrics_plot_path = os.path.join(model_dir, "metrics_trends.png")
    plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图表以释放内存
    print(f"指标趋势图已保存: {metrics_plot_path}")


def train_nnconv_model(data_file: str, max_pairs: int = None, epochs: int = 100, seed: int = 42):
    """
    训练基于NNConv的分子进化预测器模型
    
    Args:
        data_file: 数据文件路径
        max_pairs: 最大对数（用于调试）
        epochs: 训练轮数
        seed: 随机种子
    """
    
    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(project_root, 'mol_evo', 'model-data', BASE_DIR_NAME, f"training_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # 设置日志记录器
    logger = setup_logger(model_dir)
    log_training_start(logger, data_file, max_pairs, epochs)
    
    # 构建图数据
    data, smiles_to_idx, property_stats = build_molecule_graph_with_fingerprints(data_file, max_pairs)
    
    # 准备属性变化目标
    target_features = prepare_property_change_targets(data_file, property_stats, max_pairs)
    
    log_data_construction(logger, data_file, data, target_features)
    
    # 划分数据集
    train_idx, val_idx, test_idx = split_data_indices(data.num_edges, 0.7, 0.2, 0.1, seed)
    
    log_dataset_split(logger, data, train_idx, val_idx, test_idx)
    
    # INFO 创建模型
    model = MoleculeEvolutionNNConvPredictor(
        node_feature_dim=model_params["node_feature_dim"],  # Morgan指纹维度
        edge_feature_dim=model_params["edge_feature_dim"],    # 边特征维度（5原子类型 + 6操作类型）
        hidden_dim=model_params["hidden_dim"],
        output_dim=model_params["output_dim"],          # 属性变化维度
        num_layers=model_params["num_layers"]
    )
    
    log_model_creation(logger, model)
    
    # 训练模型
    log_training_start_message(logger, epochs)
    train_losses, val_losses, val_r2s, val_maes = train_gnn_model(
        model, data, target_features,
        epochs=epochs, lr=0.001, train_idx=train_idx, val_idx=val_idx, logger=logger
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 将模型和数据移动到设备
    model = model.to(device)
    data = data.to(device)
    target_features = target_features.to(device)
    
    # 定义属性名称列表
    property_names = ['A_change', 'B_change', 'C_change', 'mu_change', 'alpha_change',
                      'homo_change', 'lumo_change', 'gap_change', 'r2_change', 'zpve_change',
                      'U0_change', 'U_change', 'H_change', 'G_change', 'Cv_change']

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
        thresholds = [0.4, 0.3, 0.2, 0.1, 0.05]
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
        
        log_original_scale_metrics(logger, orig_mse, orig_rmse, orig_mae)

        # 使用表格形式展示各维度阈值准确率
        print_table_accuracy(dimension_accuracies, thresholds, property_names, logger)
        
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
        
        # 保存训练数据为JSON格式，包含训练参数
        training_params = {
            "data_file": data_file,
            "max_pairs": max_pairs,
            "epochs": epochs,
            "seed": seed
        }
        save_training_data_as_json(train_losses, val_losses, test_metrics, model_dir, model_params, training_params)
        
        # 生成训练趋势图
        plot_training_trends(train_losses, val_losses, val_r2s, val_maes, model_dir)
        
        # 记录完整评估结果到日志
        log_training_completion(logger, train_losses, val_losses, test_loss, rmse, mae, r2,
                               threshold_accs, orig_mse, orig_rmse, orig_mae, model_path)
        return model, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description='训练基于NNConv的分子进化预测器模型')
    parser.add_argument('--data-file', type=str, 
                       default='mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv',
                       help='数据文件路径')
    parser.add_argument('--max-pairs', type=int, help='最大分子对数（用于调试）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    try:
        model, train_losses, val_losses = train_nnconv_model(
            args.data_file, args.max_pairs, args.epochs, args.seed
        )
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()