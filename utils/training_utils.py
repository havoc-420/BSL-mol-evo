#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练相关的通用工具函数
"""

import os
import random
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import logging


def setup_logger(model_dir, logger_name='training'):
    """
    设置日志记录器
    
    Args:
        model_dir: 模型目录路径
        logger_name: 日志记录器名称
        
    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger(logger_name)
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


def inverse_standardize(data, stats_dict):
    """
    反标准化数据
    
    Args:
        data: 标准化后的数据
        stats_dict: 包含均值和标准差的字典
        
    Returns:
        反标准化后的数据
    """
    # 确保数据在正确的设备上
    device = data.device
    
    # 将统计数据移动到相同设备
    # 修复：stats_dict[prop]是一个元组(mean, std)，而不是一个字典
    means = torch.tensor([stats_dict[prop][0] for prop in stats_dict.keys() 
                         if prop.endswith('_change')], device=device)
    stds = torch.tensor([stats_dict[prop][1] for prop in stats_dict.keys() 
                        if prop.endswith('_change')], device=device)
    
    # 反标准化
    return data * stds + means


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


def save_training_data_as_json(train_losses, val_losses, test_metrics, model_dir, model_params, training_params=None):
    """
    将训练数据保存为JSON格式
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        test_metrics: 测试指标字典
        model_dir: 模型目录路径
        model_params: 模型参数字典
        training_params: 训练参数字典（可选）
    """
    # 构建训练数据字典
    training_data = {
        "model_params": model_params,
        "test_metrics": test_metrics,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    
    # 如果提供了训练参数，则添加到数据中
    if training_params is not None:
        training_data["training_params"] = training_params
    
    # 保存为JSON文件
    json_path = os.path.join(model_dir, "training_data.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)


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
    plt.plot(epochs, train_losses, 'o-', label='Training Loss', linewidth=2, markersize=3)
    plt.plot(epochs, val_losses, 's-', label='Validation Loss', linewidth=2, markersize=3)
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
    plt.plot(val_epochs, val_r2s, 'o-', label='Validation R²', linewidth=2, markersize=3, color='green')
    plt.title('Validation R² Trend')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制验证MAE趋势
    plt.subplot(1, 2, 2)
    plt.plot(val_epochs, val_maes, 's-', label='Validation MAE', linewidth=2, markersize=3, color='red')
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


def print_table_accuracy(accuracies, thresholds, property_names, logger=None):
    """
    以表格形式打印各维度阈值准确率，并添加颜色分区效果
    先构建不带颜色的字符串确保对齐，再添加颜色
    表格按横向展示形式，第一行为属性名称，后续每行显示一个阈值下所有属性的准确率

    Args:
        accuracies: 不同阈值下的准确率字典
        thresholds: 阈值列表
        property_names: 属性名称列表
        logger: 日志记录器实例（可选）
    """
    # 定义列宽
    col_width = 9
    threshold_col_width = 12
    
    # 打印到控制台（带颜色）
    print("各维度阈值准确率:")
    
    # 表头
    header_line = "Threshold".ljust(threshold_col_width)
    for prop_name in property_names:
        clean_name = prop_name.replace("_change", "")
        short_name = (clean_name[:col_width-1] if len(clean_name) >= col_width else clean_name).ljust(col_width)
        header_line += short_name
    print(header_line)
    
    # 分隔线
    separator_length = threshold_col_width + len(property_names) * col_width
    print("-" * separator_length)
    
    # 数据行（带颜色）
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
    
    # 记录到日志（无颜色，逐行记录）
    if logger:
        logger.info("各维度阈值准确率:")
        # 表头
        header_line = "Threshold".ljust(threshold_col_width)
        for prop_name in property_names:
            clean_name = prop_name.replace("_change", "")
            short_name = (clean_name[:col_width-1] if len(clean_name) >= col_width else clean_name).ljust(col_width)
            header_line += short_name
        logger.info(header_line)
        
        # 分隔线
        logger.info("-" * separator_length)
        
        # 数据行
        for threshold in thresholds:
            row = f"@{threshold:.3f}".ljust(threshold_col_width)
            for i in range(len(property_names)):
                acc = accuracies[threshold][i]
                formatted_acc = f"{acc:.4f}"
                # 构建单元格，确保固定宽度
                cell = formatted_acc.ljust(col_width)
                row += cell
            logger.info(row)


def log_training_completion(logger, train_losses, val_losses, test_loss, rmse, mae, r2, 
                           threshold_accs, orig_mse, orig_rmse, orig_mae, model_path):
    """
    记录训练完成信息
    
    Args:
        logger: 日志记录器
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        test_loss: 测试损失
        rmse: 均方根误差
        mae: 平均绝对误差
        r2: 决定系数
        threshold_accs: 阈值准确率字典
        orig_mse: 原始尺度MSE
        orig_rmse: 原始尺度RMSE
        orig_mae: 原始尺度MAE
        model_path: 模型保存路径
    """
    logger.info("训练完成:")
    logger.info(f"  - 最终训练损失: {train_losses[-1]:.6f}")
    logger.info(f"  - 最佳验证损失: {min(val_losses):.6f}")
    logger.info(f"  - 测试损失 (MSE): {test_loss.item():.6f}")
    logger.info(f"  - 测试RMSE: {rmse.item():.6f}")
    logger.info(f"  - 测试MAE: {mae.item():.6f}")
    logger.info(f"  - 测试R²: {r2.item():.6f}")
    logger.info(f"  - 测试阈值准确率 (0.4, 所有维度): {threshold_accs['all_0.4']:.4f}")
    logger.info(f"  - 测试阈值准确率 (0.4, 平均): {threshold_accs['mean_0.4']:.4f}")
    logger.info(f"  - 测试阈值准确率 (0.3, 所有维度): {threshold_accs['all_0.3']:.4f}")
    logger.info(f"  - 测试阈值准确率 (0.3, 平均): {threshold_accs['mean_0.3']:.4f}")
    logger.info(f"  - 测试阈值准确率 (0.2, 所有维度): {threshold_accs['all_0.2']:.4f}")
    logger.info(f"  - 测试阈值准确率 (0.2, 平均): {threshold_accs['mean_0.2']:.4f}")
    logger.info(f"  - 测试阈值准确率 (0.1, 所有维度): {threshold_accs['all_0.1']:.4f}")
    logger.info(f"  - 测试阈值准确率 (0.1, 平均): {threshold_accs['mean_0.1']:.4f}")
    logger.info(f"  - 测试阈值准确率 (0.05, 所有维度): {threshold_accs['all_0.05']:.4f}")
    logger.info(f"  - 测试阈值准确率 (0.05, 平均): {threshold_accs['mean_0.05']:.4f}")
    logger.info(f"  - 原始尺度MSE: {orig_mse.item():.6f}")
    logger.info(f"  - 原始尺度RMSE: {orig_rmse.item():.6f}")
    logger.info(f"  - 原始尺度MAE: {orig_mae.item():.6f}")
    logger.info(f"  - 模型已保存到: {model_path}")


def log_training_start(logger, data_file, max_pairs, epochs):
    """
    记录训练开始信息
    
    Args:
        logger: 日志记录器
        data_file: 数据文件路径
        max_pairs: 最大对数
        epochs: 训练轮数
    """
    logger.info("=" * 60)
    logger.info("基于NNConv的分子进化预测器模型训练开始")
    logger.info("=" * 60)
    logger.info(f"数据文件: {data_file}")
    logger.info(f"最大对数: {max_pairs}")
    logger.info(f"训练轮数: {epochs}")


def log_data_construction(logger, data_file, data, target_features):
    """
    记录数据构建信息
    
    Args:
        logger: 日志记录器
        data_file: 数据文件路径
        data: 图数据
        target_features: 目标特征
    """
    logger.info(f"正在构建图数据: {data_file}")
    logger.info(f"数据构建完成:")
    logger.info(f"  - 节点数: {data.num_nodes}")
    logger.info(f"  - 边数: {data.num_edges}")
    logger.info(f"  - 节点特征维度: {data.x.shape[1]}")
    logger.info(f"  - 边特征维度: {data.edge_attr.shape[1]}")
    logger.info(f"  - 目标属性变化维度: {target_features.shape[1]}")


def log_dataset_split(logger, data, train_idx, val_idx, test_idx):
    """
    记录数据集划分信息
    
    Args:
        logger: 日志记录器
        data: 图数据
        train_idx: 训练集索引
        val_idx: 验证集索引
        test_idx: 测试集索引
    """
    logger.info(f"数据集划分完成:")
    logger.info(f"  - 训练集: {len(train_idx)} ({len(train_idx)/data.num_edges*100:.1f}%)")
    logger.info(f"  - 验证集: {len(val_idx)} ({len(val_idx)/data.num_edges*100:.1f}%)")
    logger.info(f"  - 测试集: {len(test_idx)} ({len(test_idx)/data.num_edges*100:.1f}%)")


def log_model_creation(logger, model):
    """
    记录模型创建信息
    
    Args:
        logger: 日志记录器
        model: 模型实例
    """
    logger.info("正在创建模型...")
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")


def log_training_start_message(logger, epochs):
    """
    记录训练开始消息
    
    Args:
        logger: 日志记录器
        epochs: 训练轮数
    """
    logger.info(f"开始训练 ({epochs} 轮)...")


def log_original_scale_metrics(logger, orig_mse, orig_rmse, orig_mae):
    """
    记录原始尺度评估指标
    
    Args:
        logger: 日志记录器
        orig_mse: 原始尺度MSE
        orig_rmse: 原始尺度RMSE
        orig_mae: 原始尺度MAE
    """
    logger.info(f"原始尺度评估指标:")
    logger.info(f"  - MSE: {orig_mse.item():.6f}")
    logger.info(f"  - RMSE: {orig_rmse.item():.6f}")
    logger.info(f"  - MAE: {orig_mae.item():.6f}")