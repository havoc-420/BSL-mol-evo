#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0版本训练工具函数
专门用于训练基于双GCN网络的分子进化预测模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import matplotlib.pyplot as plt
import logging
import os
from typing import List, Tuple, Dict
from torch_geometric.data import Data


def setup_logger(model_dir: str) -> logging.Logger:
    """
    设置训练日志记录器

    Args:
        model_dir: 模型保存目录

    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 创建文件处理器
    log_file = os.path.join(model_dir, 'training.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def split_data_indices(num_samples: int, train_ratio: float = 0.7, 
                      val_ratio: float = 0.2, test_ratio: float = 0.1, 
                      seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """
    按比例划分数据索引

    Args:
        num_samples: 样本总数
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        训练集、验证集、测试集索引列表
    """
    # 设置随机种子
    np.random.seed(seed)
    
    # 验证比例之和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    # 生成随机索引
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    # 计算分割点
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))
    
    # 划分索引
    train_idx = indices[:train_end].tolist()
    val_idx = indices[train_end:val_end].tolist()
    test_idx = indices[val_end:].tolist()
    
    return train_idx, val_idx, test_idx


def save_training_data_as_json(train_losses: List[float], val_losses: List[float], 
                              test_metrics: Dict, model_dir: str, 
                              model_params: Dict, training_params: Dict):
    """
    将训练数据保存为JSON格式

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        test_metrics: 测试指标字典
        model_dir: 模型保存目录
        model_params: 模型参数
        training_params: 训练参数
    """
    training_data = {
        "model_params": model_params,
        "training_params": training_params,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_metrics": test_metrics
    }
    
    json_file = os.path.join(model_dir, 'training_data.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)


def plot_training_trends(train_losses: List[float], val_losses: List[float],
                        val_r2s: List[float], val_maes: List[float], model_dir: str):
    """
    绘制训练趋势图

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        val_r2s: 验证R2列表
        val_maes: 验证MAE列表
        model_dir: 模型保存目录
    """
    epochs = range(1, len(train_losses) + 1)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('训练趋势图')
    
    # 训练和验证损失
    axes[0, 0].plot(epochs, train_losses, label='训练损失')
    axes[0, 0].plot(epochs, val_losses, label='验证损失')
    axes[0, 0].set_title('损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 验证R2
    if val_r2s:
        eval_epochs = range(10, len(train_losses) + 1, 10)
        axes[0, 1].plot(eval_epochs, val_r2s, 'g-', label='验证R²')
        axes[0, 1].set_title('R²曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # 验证MAE
    if val_maes:
        eval_epochs = range(10, len(train_losses) + 1, 10)
        axes[1, 0].plot(eval_epochs, val_maes, 'r-', label='验证MAE')
        axes[1, 0].set_title('MAE曲线')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # 损失对比（对数尺度）
    axes[1, 1].semilogy(epochs, train_losses, label='训练损失')
    axes[1, 1].semilogy(epochs, val_losses, label='验证损失')
    axes[1, 1].set_title('损失曲线（对数尺度）')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('损失 (log scale)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_trends.png'), dpi=300, bbox_inches='tight')
    plt.close()


def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    计算多种评估指标
    
    Args:
        predictions: 预测值
        targets: 真实值
        
    Returns:
        包含各种评估指标的字典
    """
    # MSE (Mean Squared Error)
    mse = torch.mean((predictions - targets) ** 2)
    
    # RMSE (Root Mean Squared Error)
    rmse = torch.sqrt(mse)
    
    # MAE (Mean Absolute Error)
    mae = torch.mean(torch.abs(predictions - targets))
    
    # R² (Coefficient of Determination) - 增强数值稳定性
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    
    # 添加数值稳定性检查
    if ss_tot.item() == 0:
        r2 = 0.0
    else:
        r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
    
    return {
        'mse': mse.item(),
        'rmse': rmse.item(),
        'mae': mae.item(),
        'r2': r2
    }


def log_training_start(logger, data_file: str, max_pairs: int, epochs: int):
    """记录训练开始日志"""
    logger.info("=" * 60)
    logger.info("基于双GCN的分子进化预测器模型训练开始")
    logger.info("=" * 60)
    logger.info(f"数据文件: {data_file}")
    logger.info(f"最大对数: {max_pairs}")
    logger.info(f"训练轮数: {epochs}")


def log_data_construction(logger, data_file: str, data: Data, target_features: torch.Tensor):
    """记录数据构建完成日志"""
    logger.info(f"正在构建图数据: {data_file}")
    logger.info("数据构建完成:")
    logger.info(f"  - 节点数: {data.num_nodes}")
    logger.info(f"  - 边数: {data.num_edges}")
    logger.info(f"  - 节点特征维度: {data.x.shape[1]}")
    logger.info(f"  - 边特征维度: {data.edge_attr.shape[1] if data.edge_attr is not None else 0}")
    logger.info(f"  - 目标属性变化维度: {target_features.shape[1] if len(target_features.shape) > 1 else 1}")


def log_dataset_split(logger, data: Data, train_idx: List[int], val_idx: List[int], test_idx: List[int]):
    """记录数据集划分日志"""
    total_samples = len(train_idx) + len(val_idx) + len(test_idx)
    logger.info("数据集划分完成:")
    logger.info(f"  - 训练集: {len(train_idx)} ({len(train_idx)/total_samples*100:.1f}%)")
    logger.info(f"  - 验证集: {len(val_idx)} ({len(val_idx)/total_samples*100:.1f}%)")
    logger.info(f"  - 测试集: {len(test_idx)} ({len(test_idx)/total_samples*100:.1f}%)")


def log_model_creation(logger, model):
    """记录模型创建日志"""
    logger.info("模型创建完成:")
    logger.info(f"  - 模型类型: {type(model).__name__}")
    logger.info(f"  - 模型参数数量: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"  - 可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


def log_training_start_message(logger, epochs: int):
    """记录训练开始消息"""
    logger.info(f"开始训练，共 {epochs} 轮")


def log_training_completion(logger, train_losses: List[float], val_losses: List[float],
                           test_loss: float, rmse: float, mae: float, r2: float,
                           threshold_accs: Dict, val_r2s: List[float], val_maes: List[float],
                           val_threshold_accs: Dict, model_path: str):
    """记录训练完成日志"""
    logger.info("=" * 60)
    logger.info("模型训练完成")
    logger.info("=" * 60)
    logger.info("训练结果:")
    logger.info(f"  - 最终训练损失: {train_losses[-1]:.6f}")
    if val_losses:
        logger.info(f"  - 最终验证损失: {val_losses[-1]:.6f}")
    logger.info(f"  - 测试损失: {test_loss:.6f}")
    logger.info(f"  - 测试RMSE: {rmse:.6f}")
    logger.info(f"  - 测试MAE: {mae:.6f}")
    logger.info(f"  - 测试R²: {r2:.6f}")
    
    if threshold_accs:
        logger.info("  - 阈值准确率:")
        for th, acc in threshold_accs.items():
            logger.info(f"    - 阈值 {th}: {acc*100:.2f}%")
    
    logger.info(f"  - 模型保存路径: {model_path}")