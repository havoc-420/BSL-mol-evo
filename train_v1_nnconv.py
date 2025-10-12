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
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt

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

# 定义属性名称列表
ALL_PROPERTY_NAMES = ['A_change', 'B_change', 'C_change', 'mu_change', 'alpha_change',
                      'homo_change', 'lumo_change', 'gap_change', 'r2_change', 'zpve_change',
                      'U0_change', 'U_change', 'H_change', 'G_change', 'Cv_change']

""" 设置项目根目录路径 """

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建项目根目录路径
project_root = os.path.join(script_dir, '..')
# 添加项目根目录到Python路径
sys.path.insert(0, project_root)

# 导入自定义模块
try: 
    from mol_evo.core.models.v1.nnconv import MoleculeEvolutionNNConvPredictor
    from mol_evo.core.data.processing import build_molecule_graph_with_fingerprints
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
        log_original_scale_metrics,
        display_sample_data
    )
except ImportError:
    print("请确保已安装所需的依赖项。")


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
    
    # 只有当val_losses不为空且长度与epochs相同时才绘制验证损失
    if len(val_losses) > 0:
        if len(val_losses) == len(epochs):
            plt.plot(epochs, val_losses, 'o-', label='Validation Loss', linewidth=1, markersize=1.5, alpha=0.7)
        else:
            # 如果长度不匹配，则只截取前面部分进行绘制
            val_epochs = range(1, len(val_losses) + 1)
            plt.plot(val_epochs, val_losses, 'o-', label='Validation Loss', linewidth=1, markersize=1.5, alpha=0.7)
    
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
    
    # 只有当val_r2s不为空时才绘制
    if len(val_r2s) > 0:
        if len(val_r2s) == len(val_epochs):
            plt.plot(val_epochs, val_r2s, 'o-', label='Validation R²', linewidth=2, markersize=1.5, color='green', alpha=0.7)
        else:
            # 处理长度不匹配的情况
            actual_val_epochs = range(10, len(val_r2s) * 10 + 1, 10)
            plt.plot(actual_val_epochs, val_r2s, 'o-', label='Validation R²', linewidth=2, markersize=1.5, color='green', alpha=0.7)
    
    plt.title('Validation R² Trend')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制验证MAE趋势
    plt.subplot(1, 2, 2)
    
    # 只有当val_maes不为空时才绘制
    if len(val_maes) > 0:
        if len(val_maes) == len(val_epochs):
            plt.plot(val_epochs, val_maes, 's-', label='Validation MAE', linewidth=2, markersize=1.5, color='red', alpha=0.7)
        else:
            # 处理长度不匹配的情况
            actual_val_epochs = range(10, len(val_maes) * 10 + 1, 10)
            plt.plot(actual_val_epochs, val_maes, 's-', label='Validation MAE', linewidth=2, markersize=1.5, color='red', alpha=0.7)
    
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


def prepare_property_change_targets_no_standardization(csv_file: str, max_pairs: int = None, selected_properties=None) -> torch.Tensor:
    """
    准备属性变化目标值（不进行标准化）
    
    Args:
        csv_file: CSV文件路径
        max_pairs: 最大对数（用于调试）
        selected_properties: 选定的属性列表，默认为None表示使用全部属性
        
    Returns:
        属性变化目标值（未标准化）
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    
    if max_pairs:
        df = df.head(max_pairs)
    
    # 准备目标特征
    target_features = []
    
    property_names = selected_properties if selected_properties is not None else ALL_PROPERTY_NAMES
    
    for _, row in df.iterrows():
        properties = []
        
        for prop in property_names:
            if prop in row and not pd.isna(row[prop]):
                value = row[prop]
                properties.append(value)
            else:
                properties.append(0.0)
                
        target_features.append(properties)
    
    target_features = torch.FloatTensor(np.array(target_features))
    
    return target_features


def select_properties_interactive():
    """
    交互式选择属性
    """
    print("请选择要训练的属性（输入序号，多个序号用空格分隔）:")
    for i, prop in enumerate(ALL_PROPERTY_NAMES, 1):
        print(f"{i:2d}. {prop}")
    
    while True:
        try:
            user_input = input("请输入序号（例如: 1 3 5-7 10）: ").strip()
            if not user_input:
                print("使用全部属性进行训练")
                return ALL_PROPERTY_NAMES
            
            # 解析用户输入
            selected_indices = []
            for part in user_input.split():
                if '-' in part:
                    # 处理范围输入如 5-7
                    start, end = map(int, part.split('-'))
                    selected_indices.extend(range(start, end + 1))
                else:
                    # 处理单个数字
                    selected_indices.append(int(part))
            
            # 转换为0基索引并验证
            selected_indices = [i - 1 for i in selected_indices]
            for idx in selected_indices:
                if idx < 0 or idx >= len(ALL_PROPERTY_NAMES):
                    raise ValueError(f"索引 {idx + 1} 超出范围")
            
            # 获取选中的属性名称
            selected_properties = [ALL_PROPERTY_NAMES[i] for i in selected_indices]
            
            print(f"选中的属性: {selected_properties}")
            return selected_properties
            
        except ValueError as e:
            print(f"输入错误: {e}，请重新输入")
        except Exception as e:
            print(f"输入错误: {e}，请重新输入")


def filter_property_stats(property_stats, selected_properties):
    """
    根据选定的属性过滤属性统计信息
    
    Args:
        property_stats: 完整的属性统计信息字典
        selected_properties: 选定的属性列表
        
    Returns:
        过滤后的属性统计信息字典
    """
    if selected_properties is None:
        return property_stats
    
    filtered_stats = {}
    for prop in selected_properties:
        if prop in property_stats:
            filtered_stats[prop] = property_stats[prop]
    
    return filtered_stats


def train_nnconv_model(data_file: str, max_pairs: int = None, epochs: int = 100, seed: int = 42, 
                      normalize: bool = True, selected_properties=None):
    """
    训练基于NNConv的分子进化预测器模型
    
    Args:
        data_file: 数据文件路径
        max_pairs: 最大对数（用于调试）
        epochs: 训练轮数
        seed: 随机种子
        normalize: 是否对属性进行标准化
        selected_properties: 选定的属性列表，默认为None表示使用全部属性
    """
    
    # 确定使用的属性列表
    if selected_properties is not None:
        property_names = selected_properties
        output_dim = len(selected_properties)
        print(f"使用选定的 {output_dim} 个属性进行训练: {property_names}")
    else:
        property_names = ALL_PROPERTY_NAMES
        output_dim = len(ALL_PROPERTY_NAMES)
        print(f"使用全部 {output_dim} 个属性进行训练")
    
    # 更新模型参数
    model_params["output_dim"] = output_dim
    
    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(project_root, 'mol_evo', 'output', BASE_DIR_NAME, f"training_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # 设置日志记录器
    logger = setup_logger(model_dir)
    log_training_start(logger, data_file, max_pairs, epochs)
    
    # 构建图数据
    data, smiles_to_idx, property_stats = build_molecule_graph_with_fingerprints(data_file, max_pairs)
    
    # 过滤属性统计信息，只保留选定属性的统计信息
    filtered_property_stats = filter_property_stats(property_stats, property_names)
    
    # 准备属性变化目标
    if normalize:
        # 修改prepare_property_change_targets以支持选定属性
        target_features = prepare_property_change_targets_selected(data_file, filtered_property_stats, max_pairs, property_names)
    else:
        target_features = prepare_property_change_targets_no_standardization(data_file, max_pairs, property_names)
        # 创建空的属性统计信息，表示未进行标准化
        filtered_property_stats = {}
    
    log_data_construction(logger, data_file, data, target_features)
    
    # 划分数据集
    train_idx, val_idx, test_idx = split_data_indices(data.num_edges, 0.7, 0.2, 0.1, seed)
    
    log_dataset_split(logger, data, train_idx, val_idx, test_idx)
    
    # 显示部分数据样本用于查验
    display_sample_data(data_file, data, target_features, train_idx, val_idx, property_names)
    
    # INFO 创建模型
    model = MoleculeEvolutionNNConvPredictor(
        node_feature_dim=model_params["node_feature_dim"],  # Morgan指纹维度
        edge_feature_dim=model_params["edge_feature_dim"],  # 边特征维度（5原子类型 + 6操作类型）
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
        
        # R² (Coefficient of Determination) - 增强数值稳定性
        ss_res = torch.sum((test_targets - test_predictions) ** 2, dim=0)  # 按维度计算
        ss_tot = torch.sum((test_targets - torch.mean(test_targets, dim=0)) ** 2, dim=0)  # 按维度计算
        
        # 添加数值稳定性检查
        # 对于每个维度，如果ss_tot为0，则R²为0；否则计算1 - ss_res/ss_tot
        # 添加小的epsilon值提高数值稳定性
        r2 = torch.ones_like(ss_res)  # 默认为1
        non_zero_mask = ss_tot != 0
        r2[non_zero_mask] = 1 - ss_res[non_zero_mask] / (ss_tot[non_zero_mask] + 1e-8)
        
        # 如果只有一个属性，则取标量值
        if r2.numel() == 1:
            r2 = r2.item()
        else:
            r2 = r2.mean().item()  # 多个属性时取平均
        
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
        
        # 如果进行了标准化，则反标准化预测结果和目标值以获得原始尺度的评估指标
        if normalize and filtered_property_stats:
            original_predictions = inverse_standardize(test_predictions, filtered_property_stats)
            original_targets = inverse_standardize(test_targets, filtered_property_stats)
            
            # 在原始尺度上计算评估指标
            orig_mse = torch.mean((original_predictions - original_targets) ** 2)
            orig_rmse = torch.sqrt(orig_mse)
            orig_mae = torch.mean(torch.abs(original_predictions - original_targets))
            
            # 在原始尺度上计算R²
            orig_ss_res = torch.sum((original_targets - original_predictions) ** 2)
            orig_ss_tot = torch.sum((original_targets - torch.mean(original_targets)) ** 2)
            if orig_ss_tot.item() == 0:
                orig_r2 = 0.0
            else:
                orig_r2 = (1 - orig_ss_res / (orig_ss_tot + 1e-8)).item()
            
            log_original_scale_metrics(logger, orig_mse, orig_rmse, orig_mae)
        else:
            # 如果没有标准化，则原始尺度的指标就是标准化后的指标
            orig_mse = mse
            orig_rmse = rmse
            orig_mae = mae
            # 对于未标准化的数据，我们不计算原始尺度的R²，因为这没有意义

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
            "r2": r2,
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
        
        # 保存训练数据为JSON格式，包含属性统计信息，包含训练参数
        training_params = {
            "data_file": data_file,
            "max_pairs": max_pairs,
            "epochs": epochs,
            "seed": seed,
            "normalize": normalize,
            "selected_properties": selected_properties
        }
        save_training_data_as_json(train_losses, val_losses, test_metrics, model_dir, model_params, training_params, filtered_property_stats)
        
        # 生成训练趋势图
        plot_training_trends(train_losses, val_losses, val_r2s, val_maes, model_dir)
        
        # 记录完整评估结果到日志
        log_training_completion(logger, train_losses, val_losses, test_loss, rmse, mae, r2,
                               threshold_accs, orig_mse, orig_rmse, orig_mae, model_path)
        return model, train_losses, val_losses


def prepare_property_change_targets_selected(csv_file: str, property_stats: dict, 
                                           max_pairs: int = None, selected_properties=None) -> torch.Tensor:
    """
    准备选定属性的变化目标值
    
    Args:
        csv_file: CSV文件路径
        property_stats: 属性统计信息（均值和标准差）
        max_pairs: 最大对数（用于调试）
        selected_properties: 选定的属性列表，默认为None表示使用全部属性
        
    Returns:
        标准化后的属性变化目标值
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    
    if max_pairs:
        df = df.head(max_pairs)
    
    # 准备目标特征
    target_features = []
    
    property_names = selected_properties if selected_properties is not None else ALL_PROPERTY_NAMES
    
    for _, row in df.iterrows():
        properties = []
        
        for prop in property_names:
            if prop in row and not pd.isna(row[prop]):
                value = row[prop]
                # 标准化属性变化值
                if prop in property_stats:
                    mean, std = property_stats[prop]
                    if std > 0:
                        value = (value - mean) / std
                properties.append(value)
            else:
                properties.append(0.0)
                
        target_features.append(properties)
    
    target_features = torch.FloatTensor(np.array(target_features))
    
    return target_features


def main():
    parser = argparse.ArgumentParser(description='训练基于NNConv的分子进化预测器模型')
    parser.add_argument('--data-file', type=str, 
                       default='mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv',
                       help='数据文件路径')
    parser.add_argument('--max-pairs', type=int, help='最大分子对数（用于调试）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no-normalize', action='store_true', 
                       help='不进行属性标准化')
    parser.add_argument('--prop', action='store_true',
                       help='交互式选择属性')
    
    args = parser.parse_args()
    
    # 如果指定了--prop参数，则进行交互式属性选择
    selected_properties = None
    if args.prop:
        selected_properties = select_properties_interactive()
    
    try:
        model, train_losses, val_losses = train_nnconv_model(
            args.data_file, args.max_pairs, args.epochs, args.seed, not args.no_normalize, selected_properties
        )
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()