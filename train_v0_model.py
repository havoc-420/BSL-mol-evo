#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0版本模型训练脚本
专门用于训练针对单属性（mu_change）预测的GCN模型
"""

import sys
import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm

""" BASE SETTINGS """

# 定义模型参数字典
model_params = {
    "node_feature_dim": 11,   # 修改为smile_to_graph_xyz生成的特征维度
    "edge_feature_dim": 11,   # 保持不变
    "hidden_dim": 128,
    "output_dim": 1,  # 单属性预测
    "num_layers": 2  # INFO 现阶段默认都是 2 层
}

# 目标属性名称
TARGET_PROPERTY = 'mu_change'

# 设置项目根目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

# 导入自定义模块
try:
    from mol_evo.core.models.v0.gcn import MoleculeEvolutionGCNPredictor
    from mol_evo.core.utils.molecule import MoleculeCache
    from mol_evo.core.data.processing import prepare_edge_features
    from mol_evo.utils.training_utils import (
        split_data_indices, 
        save_training_data_as_json, 
        plot_training_trends,
        log_training_completion,
        log_training_start,
        log_model_creation,
        log_training_start_message
    )
    from mol_evo.utils.logger_utils import DualLogger
except ImportError as e:
    print(f"无法导入所需的模块: {e}")
    exit(1)


def smiles_to_graph_data(smiles, cache):
    """
    将SMILES字符串转换为图数据（使用项目中的实际函数）
    参考文档: mol_evo/docs/model-v0/data_preprocessing_and_usage.md
    
    Args:
        smiles (str): SMILES字符串
        cache (MoleculeCache): 分子缓存实例
    
    Returns:
        Data: PyTorch Geometric Data对象
    """
    # 定义原子类型映射
    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    
    # 使用项目中的函数将SMILES转换为图结构
    x, z, pos, edge_index, edge_attr = cache.process_smiles(smiles, types)
    
    # 检查转换是否成功
    if x is None:
        print(f"无法处理SMILES: {smiles}")
        return None
    
    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def build_molecule_evolution_dataset_v0(csv_file: str, max_pairs: int = None, 
                                      target_property: str = 'mu_change', logger=None):
    """
    构建分子进化数据集，使用smile_to_graph_xyz函数处理分子结构
    参考文档: mol_evo/docs/model-v0/data_preprocessing_and_usage.md
    
    Args:
        csv_file: CSV文件路径
        max_pairs: 最大对数（用于调试）
        target_property: 目标属性名称
        logger: 日志记录器
        
    Returns:
        起始分子数据列表、目标分子数据列表、边特征张量和目标属性张量
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    
    if max_pairs:
        df = df.head(max_pairs)
    
    # 计算目标属性的统计信息
    if target_property in df.columns:
        mean = df[target_property].mean()
        std = df[target_property].std()
        property_stats = {target_property: (mean, std)}
    else:
        property_stats = {target_property: (0.0, 1.0)}
    
    # 准备目标属性值
    target_features = []
    for _, row in df.iterrows():
        if target_property in row and not pd.isna(row[target_property]):
            value = row[target_property]
            # 标准化目标属性值
            if target_property in property_stats:
                mean, std = property_stats[target_property]
                if std > 0:
                    value = (value - mean) / std
            target_features.append([value])
        else:
            target_features.append([0.0])
    
    target_features = torch.FloatTensor(np.array(target_features))
    
    # 创建分子缓存实例，并传入logger
    cache = MoleculeCache("training_dataset", logger=logger)
    
    # 构建分子数据列表
    from_data_list = []
    to_data_list = []
    edge_attr_list = []
    
    for _, row in df.iterrows():
        # 使用smile_to_graph_xyz函数生成起始分子和目标分子的图结构
        from_data = smiles_to_graph_data(row['smiles_from'], cache)
        to_data = smiles_to_graph_data(row['smiles_to'], cache)
        
        # 检查转换是否成功
        if from_data is None or to_data is None:
            message = f"跳过无法处理的分子对: {row['smiles_from']} -> {row['smiles_to']}"
            if logger:
                logger.info(message)
            else:
                print(message)
            continue
            
        from_data_list.append(from_data)
        to_data_list.append(to_data)
        
        # 准备边特征 (操作信息特征)
        edge_feat = prepare_edge_features(row, property_stats, include_property_changes=False)
        edge_attr_list.append(edge_feat)
    
    if len(edge_attr_list) > 0:
        edge_attrs = torch.FloatTensor(np.array(edge_attr_list))
    else:
        edge_attrs = torch.FloatTensor([])
    
    # 打印缓存统计信息
    stats = cache.get_stats()
    message = f"分子处理统计: 总数={stats['total']}, 命中={stats['hits']}, 未命中={stats['misses']}, 命中率={stats['hit_rate']:.2%}"
    if logger:
        logger.info(message)
    else:
        print(message)
    
    return from_data_list, to_data_list, edge_attrs, target_features, property_stats


def prepare_single_property_targets(csv_file: str, property_name: str, 
                                  max_pairs: int = None) -> torch.Tensor:
    """
    准备单个属性的变化目标值
    
    Args:
        csv_file: CSV文件路径
        property_name: 属性名称
        max_pairs: 最大对数（用于调试）
        
    Returns:
        属性变化目标值
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    
    if max_pairs:
        df = df.head(max_pairs)
    
    # 准备目标特征
    target_features = []
    
    for _, row in df.iterrows():
        if property_name in row and not pd.isna(row[property_name]):
            value = row[property_name]
            target_features.append([value])
        else:
            target_features.append([0.0])
    
    target_features = torch.FloatTensor(np.array(target_features))
    
    return target_features


def create_model(model_params: dict):
    """
    创建v0版本的模型实例
    
    Args:
        model_params: 模型参数字典
        
    Returns:
        模型实例
    """
    model = MoleculeEvolutionGCNPredictor(
        node_feature_dim=model_params["node_feature_dim"],
        edge_feature_dim=model_params["edge_feature_dim"],
        hidden_dim=model_params["hidden_dim"],
        output_dim=model_params["output_dim"],
        num_layers=model_params["num_layers"]
    )
    
    return model



def train_model(data_file: str, max_pairs: int = None, epochs: int = 100, 
                seed: int = 42):
    """
    训练v0版本的分子进化预测器模型（单属性预测）
    
    Args:
        data_file: 数据文件路径
        max_pairs: 最大对数（用于调试）
        epochs: 训练轮数
        seed: 随机种子
    """
    
    # train-data 存储位置
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(project_root, 'mol_evo', 'model-data', 'v0', f"training_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建自定义的DualLogger实例
    logger = DualLogger(model_dir)
    log_training_start(logger, data_file, max_pairs, epochs)
    
    # 构建图数据 - 使用新的数据处理方法
    logger.info(f"正在构建图数据: {data_file}")  # 默认输出到控制台和文件
    from_data_list, to_data_list, edge_attrs, target_features, property_stats = build_molecule_evolution_dataset_v0(
        data_file, max_pairs, TARGET_PROPERTY, logger)
    
    # 输出训练集的头部信息（前几个样本示例）
    logger.info("训练集样本示例 (头部数据):")
    num_examples = min(3, len(from_data_list))  # 显示前3个样本或全部样本（如果少于3个）
    for i in range(num_examples):
        logger.info(f"  样本 {i+1}:")
        logger.info(f"    起始分子节点数: {from_data_list[i].x.size(0)}")
        logger.info(f"    起始分子边数: {from_data_list[i].edge_index.size(1)}")
        logger.info(f"    目标分子节点数: {to_data_list[i].x.size(0)}")
        logger.info(f"    目标分子边数: {to_data_list[i].edge_index.size(1)}")
        logger.info(f"    边特征: {edge_attrs[i].tolist()}")
        logger.info(f"    目标属性值: {target_features[i].item()}")

    # 直接记录数据构建信息，避免创建不必要的dummy_data对象
    logger.info("数据构建完成:")  # 默认输出到控制台和文件
    logger.info(f"  - 样本数: {len(from_data_list)}")  # 默认输出到控制台和文件
    logger.info(f"  - 节点特征维度: {model_params['node_feature_dim']}")  # 默认输出到控制台和文件
    logger.info(f"  - 边特征维度: {edge_attrs.shape[1] if len(edge_attrs.shape) > 1 else 1}")  # 默认输出到控制台和文件
    logger.info(f"  - 目标属性变化维度: {target_features.shape[1] if len(target_features.shape) > 1 else 1}")  # 默认输出到控制台和文件
    
    # 划分数据集
    train_idx, val_idx, test_idx = split_data_indices(len(from_data_list), 0.8, 0.1, 0.1, seed)
    
    # 直接记录数据集划分信息
    total_samples = len(train_idx) + len(val_idx) + len(test_idx)
    logger.info("数据集划分完成:")  # 默认输出到控制台和文件
    logger.info(f"  - 训练集: {len(train_idx)} ({len(train_idx)/total_samples*100:.1f}%)")  # 默认输出到控制台和文件
    logger.info(f"  - 验证集: {len(val_idx)} ({len(val_idx)/total_samples*100:.1f}%)")  # 默认输出到控制台和文件
    logger.info(f"  - 测试集: {len(test_idx)} ({len(test_idx)/total_samples*100:.1f}%)")  # 默认输出到控制台和文件
    
    # 创建模型
    model = create_model(model_params)
    log_model_creation(logger, model)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    message = f"使用设备: {device}"
    logger.info(message)  # 默认输出到控制台和文件
    
    # 将模型移到设备上
    model = model.to(device)
    edge_attrs = edge_attrs.to(device)
    target_features = target_features.to(device)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss(reduction='mean')
    
    # 初始化损失记录
    train_losses = []
    val_losses = []
    val_mses = []
    val_rmses = []
    val_r2s = []
    val_maes = []
    
    # 早停机制参数
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 50
    
    # 训练循环
    model.train()
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        optimizer.zero_grad()
        
        # 前向传播
        all_predictions = []
        for i in range(len(from_data_list)):
            from_data = from_data_list[i].to(device)
            to_data = to_data_list[i].to(device)
            edge_attr = edge_attrs[i].unsqueeze(0)
            
            prediction = model(from_data, to_data, edge_attr)
            all_predictions.append(prediction)
        
        # 合并所有预测结果
        predictions = torch.cat(all_predictions, dim=0)
        
        # 确保预测值和目标值维度匹配
        if predictions.shape != target_features.shape:
            raise ValueError(f"预测值维度 {predictions.shape} 与目标值维度 {target_features.shape} 不匹配")
        
        # 计算训练损失
        train_loss = criterion(predictions[train_idx], target_features[train_idx])
        
        # 检查是否有NaN或inf值
        if torch.isnan(train_loss) or torch.isinf(train_loss):
            message = f"警告: 在第 {epoch+1} 轮检测到NaN或inf损失值，停止训练"
            logger.info(message)  # 默认输出到控制台和文件
            break
        
        # 反向传播
        train_loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 每100轮保存一次checkpoint
        if (epoch + 1) % 100 == 0:
            checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss.item(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
            }, checkpoint_path)
            logger.info(f"已保存checkpoint: {checkpoint_path}")
        
        # 记录训练损失
        train_losses.append(train_loss.item())
        
        # 验证阶段
        if val_idx is not None and len(val_idx) > 0:
            model.eval()
            with torch.no_grad():
                # 为验证集创建预测
                val_predictions_list = []
                for i in val_idx:
                    from_data = from_data_list[i].to(device)
                    to_data = to_data_list[i].to(device)
                    edge_attr = edge_attrs[i].unsqueeze(0)
                    
                    prediction = model(from_data, to_data, edge_attr)
                    val_predictions_list.append(prediction)
                
                # 合并验证集预测结果
                val_predictions = torch.cat(val_predictions_list, dim=0)
                val_loss = criterion(val_predictions, target_features[val_idx])
                val_losses.append(val_loss.item())
                
                # 检查验证损失是否有NaN或inf
                if torch.isnan(val_loss) or torch.isinf(val_loss):
                    message = f"警告: 在第 {epoch+1} 轮验证时检测到NaN或inf损失值"
                    logger.info(message)  # 默认输出到控制台和文件
                
                # 更新学习率调度器
                scheduler.step(val_loss)
                
                # 计算验证集的额外评估指标 (每10个epoch计算一次)
                if (epoch + 1) % 10 == 0:
                    # RMSE
                    val_mse = torch.mean((val_predictions - target_features[val_idx]) ** 2)
                    val_rmse = torch.sqrt(val_mse)
                    
                    # 添加MSE和RMSE到对应的列表中
                    val_mses.append(val_mse.item())
                    val_rmses.append(val_rmse.item())
                    
                    # MAE
                    val_mae = torch.mean(torch.abs(val_predictions - target_features[val_idx]))
                    val_maes.append(val_mae.item())
                    
                    # R²
                    val_ss_res = torch.sum((target_features[val_idx] - val_predictions) ** 2)
                    val_ss_tot = torch.sum((target_features[val_idx] - torch.mean(target_features[val_idx])) ** 2)
                    if val_ss_tot.item() == 0:
                        val_r2 = 0.0
                    else:
                        val_r2 = (1 - val_ss_res / (val_ss_tot + 1e-8)).item()
                    val_r2s.append(val_r2)
                    
            model.train()
        else:
            val_loss = None
        
        # 早停机制检查
        if val_loss is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                
            if patience_counter >= patience_limit:
                message = f"早停机制触发，在第 {epoch+1} 轮停止训练"
                logger.info(message)  # 默认输出到控制台和文件
                # 恢复最佳模型状态
                model.load_state_dict(best_model_state)
                break
        
        # 每10个epoch输出一次信息
        if (epoch + 1) % 10 == 0:
            message = f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.6f}"
            if val_loss is not None:
                message += f", Val Loss: {val_loss:.6f}"
                if len(val_r2s) > 0:
                    message += f", R²: {val_r2s[-1]:.4f}, MAE: {val_maes[-1]:.4f}"
            
            # 详细训练信息只记录到文件，不在控制台显示
            logger.info(message, to_console=False)
            # 控制台只显示基本进度信息
            tqdm.write(message)
    
    log_training_start_message(logger, epochs)

    # 测试阶段
    model.eval()
    with torch.no_grad():
        # 为测试集创建预测
        test_predictions_list = []
        for i in test_idx:
            from_data = from_data_list[i].to(device)
            to_data = to_data_list[i].to(device)
            edge_attr = edge_attrs[i].unsqueeze(0)
            
            prediction = model(from_data, to_data, edge_attr)
            test_predictions_list.append(prediction)
        
        # 合并测试集预测结果
        test_predictions = torch.cat(test_predictions_list, dim=0)
        test_targets = target_features[test_idx]
        
        criterion = nn.MSELoss()
        test_loss = criterion(test_predictions, test_targets)
        
        # 计算额外的评估指标
        # RMSE (Root Mean Square Error)
        mse = torch.mean((test_predictions - test_targets) ** 2)
        rmse = torch.sqrt(mse)
        
        # MAE (Mean Absolute Error)
        mae = torch.mean(torch.abs(test_predictions - test_targets))
        
        # R² (Coefficient of Determination) - 增强数值稳定性
        ss_res = torch.sum((test_targets - test_predictions) ** 2)
        ss_tot = torch.sum((test_targets - torch.mean(test_targets)) ** 2)
        
        # 添加数值稳定性检查
        if ss_tot.item() == 0:
            r2 = 0.0
        else:
            r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
        
        # 阈值准确率评估
        thresholds = [0.4, 0.3, 0.2, 0.1, 0.05]
        threshold_accs = {}
        
        for th in thresholds:
            # 计算满足阈值的样本比例
            correct = (torch.abs(test_predictions - test_targets) < th).float()
            threshold_accs[th] = correct.mean().item()
        
        # 保存模型
        model_filename = "molecule_evolution_gcn_v0_mu_predictor.pth"
        model_path = os.path.join(model_dir, model_filename)
        torch.save(model.state_dict(), model_path)
        
        # 准备测试指标数据
        test_metrics = {
            "test_loss": test_loss.item(),
            "rmse": rmse.item(),
            "mae": mae.item(),
            "r2": r2,
            "threshold_accs": threshold_accs,
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
            "seed": seed,
            "target_property": TARGET_PROPERTY
        }
        save_training_data_as_json(train_losses, val_losses, test_metrics, model_dir, model_params, training_params, property_stats)
        
        # 生成训练趋势图
        plot_training_trends(train_losses, val_losses, val_r2s, val_maes, model_dir)
        
        # 记录完整评估结果到日志
        log_training_completion(logger, train_losses, val_losses, test_loss, rmse, mae, r2,
                               threshold_accs, None, None, None, model_path)
        return model, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description='v0版本分子进化预测器模型训练脚本（单属性预测）')
    parser.add_argument('--data-file', type=str, 
                       default=os.path.join('mol_evo', 'dataset', 'data', 'qm9-evo-pairs-step-1-with-properties.csv'),
                       help='数据文件路径')
    parser.add_argument('--max-pairs', type=int, help='最大分子对数（用于调试）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    try:
        model, train_losses, val_losses = train_model(
            args.data_file, args.max_pairs, args.epochs, args.seed
        )
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()