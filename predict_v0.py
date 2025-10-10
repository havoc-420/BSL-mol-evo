#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的v0模型进行分子进化属性变化预测
"""

import sys
import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import argparse
import json
import glob
from datetime import datetime
import random
import logging

# 设置项目根目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

# 导入自定义模块
from mol_evo.core.models.v0.gcn import MoleculeEvolutionGCNPredictor
from mol_evo.core.data.processing import prepare_edge_features
from mol_evo.core.utils.molecule import MoleculeCache
from mol_evo.utils.training_utils import inverse_standardize


def setup_logger(log_level=logging.INFO):
    """
    设置日志记录器
    
    Args:
        log_level: 日志级别
        
    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger('prediction')
    logger.setLevel(log_level)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # 创建格式器并添加到处理器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # 添加处理器到logger
        logger.addHandler(console_handler)
    
    return logger


def find_model_files():
    """
    自动查找模型文件
    
    Returns:
        模型文件列表
    """
    model_pattern = os.path.join(project_root, 'mol_evo', 'model-data', 'v0', 'training_*', 'molecule_evolution_gcn_v0_mu_predictor.pth')
    model_files = glob.glob(model_pattern)
    return sorted(model_files)


def select_models_interactively(model_files):
    """
    交互式选择模型（支持多选）
    
    Args:
        model_files: 模型文件列表
        
    Returns:
        选择的模型文件路径列表
    """
    if not model_files:
        print("未找到任何模型文件")
        return []
    
    if len(model_files) == 1:
        print(f"找到一个模型文件: {os.path.basename(os.path.dirname(model_files[0]))}")
        return [model_files[0]]
    
    print("找到多个模型，请选择一个或多个:")
    for i, model_file in enumerate(model_files):
        model_dir = os.path.dirname(model_file)
        model_name = os.path.basename(model_dir)
        print(f"{i+1}. {model_name}")
    
    while True:
        try:
            choice = input(f"请选择模型 (1-{len(model_files)})，多个选择用逗号分隔，或按回车选择最新模型: ").strip()
            if not choice:
                # 选择最新的模型
                latest_model = max(model_files, key=os.path.getctime)
                print(f"选择最新模型: {os.path.basename(os.path.dirname(latest_model))}")
                return [latest_model]
            
            # 解析选择
            choices = [int(c.strip()) - 1 for c in choice.split(',')]
            selected_models = []
            for choice_idx in choices:
                if 0 <= choice_idx < len(model_files):
                    selected_models.append(model_files[choice_idx])
                else:
                    print(f"请输入 1 到 {len(model_files)} 之间的数字")
                    raise ValueError("无效选择")
            
            return selected_models
        except ValueError:
            print("请输入有效的数字，多个选择用逗号分隔")


def load_property_stats(model_dir):
    """
    从模型目录加载属性统计信息
    
    Args:
        model_dir: 模型目录路径
        
    Returns:
        属性统计信息字典
    """
    stats_file = os.path.join(model_dir, "training_data.json")
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            data = json.load(f)
            # 从训练数据中提取属性统计信息
            if 'property_stats' in data:
                return data['property_stats']
            else:
                print("警告: 训练数据中未找到属性统计信息，将无法进行反标准化")
                return None
    else:
        print(f"警告: 未找到训练数据文件 {stats_file}，将无法进行反标准化")
        return None


def load_training_params(model_dir):
    """
    从模型目录加载训练参数
    
    Args:
        model_dir: 模型目录路径
        
    Returns:
        训练参数字典
    """
    stats_file = os.path.join(model_dir, "training_data.json")
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            data = json.load(f)
            # 从训练数据中提取训练参数
            if 'training_params' in data:
                return data['training_params']
            else:
                print("警告: 训练数据中未找到训练参数")
                return None
    else:
        print(f"警告: 未找到训练数据文件 {stats_file}")
        return None


def smiles_to_graph_data(smiles, cache):
    """
    将SMILES字符串转换为图数据（使用项目中的实际函数）
    
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


def prepare_single_prediction_data(smiles_from, smiles_to, to_atom_symbol, operation_type, model_dir):
    """
    准备单个预测的数据
    
    Args:
        smiles_from: 起始分子的SMILES
        smiles_to: 目标分子的SMILES
        to_atom_symbol: 变化涉及的原子类型
        operation_type: 操作类型
        model_dir: 模型目录路径
        
    Returns:
        图数据对象和属性统计信息
    """
    # 创建虚拟数据框以复用现有函数
    data_dict = {
        'smiles_from': [smiles_from],
        'smiles_to': [smiles_to],
        'to_atom_symbol': [to_atom_symbol],
        'operation_type': [operation_type]
    }
    df = pd.DataFrame(data_dict)
    
    # 获取属性统计信息
    property_stats = load_property_stats(model_dir)
    
    # 创建分子缓存实例
    cache = MoleculeCache("prediction_dataset")
    
    # 构建起始和目标分子图数据
    from_data = smiles_to_graph_data(smiles_from, cache)
    to_data = smiles_to_graph_data(smiles_to, cache)
    
    if from_data is None or to_data is None:
        raise ValueError("无法将SMILES转换为图数据")
    
    # 构建边特征（不含属性变化）
    edge_feat = prepare_edge_features(df.iloc[0], property_stats, include_property_changes=False)
    edge_attr = torch.FloatTensor(np.array([edge_feat]))
    
    return from_data, to_data, edge_attr, property_stats


def is_model_normalized(model_dir):
    """
    检查模型是否使用了标准化数据进行训练
    
    Args:
        model_dir: 模型目录路径
        
    Returns:
        bool: 如果模型使用了标准化数据则返回True，否则返回False
    """
    stats_file = os.path.join(model_dir, "training_data.json")
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            data = json.load(f)
            # 检查训练参数中是否启用了标准化
            if 'training_params' in data and 'normalize' in data['training_params']:
                return data['training_params']['normalize']
            # 默认情况下，检查是否有属性统计信息
            elif 'property_stats' in data and data['property_stats']:
                return True
            else:
                return False
    return False


def predict_property_changes(model_path, model_dir, smiles_from, smiles_to, to_atom_symbol, operation_type):
    """
    使用训练好的模型预测属性变化
    
    Args:
        model_path: 模型文件路径
        model_dir: 模型目录路径
        smiles_from: 起始分子的SMILES
        smiles_to: 目标分子的SMILES
        to_atom_symbol: 变化涉及的原子类型
        operation_type: 操作类型
        
    Returns:
        预测的属性变化值
    """
    # 准备数据
    from_data, to_data, edge_attr, property_stats = prepare_single_prediction_data(
        smiles_from, smiles_to, to_atom_symbol, operation_type, model_dir)
    
    # 创建模型
    model = MoleculeEvolutionGCNPredictor(
        node_feature_dim=11,      # v0模型节点特征维度
        edge_feature_dim=11,      # 边特征维度（5原子类型 + 6操作类型）
        hidden_dim=128,
        output_dim=1,             # v0模型只预测单个属性 (mu_change)
        num_layers=2
    )
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # 进行预测
    with torch.no_grad():
        predictions = model(from_data, to_data, edge_attr)
    
    # 获取预测结果
    predicted_changes = predictions[0].numpy()
    
    # 检查模型是否使用了标准化
    is_normalized = is_model_normalized(model_dir)
    
    if is_normalized and property_stats:
        # 如果模型使用了标准化，则需要反标准化
        standardized_changes = {}
        original_changes = {}
        
        # v0模型只预测mu_change属性
        prop_name = 'mu_change'
        standardized_changes[prop_name] = predicted_changes[0]
        # 反标准化得到原始尺度的预测值
        if prop_name in property_stats:
            mean, std = property_stats[prop_name]
            original_changes[prop_name] = predicted_changes[0] * std + mean
        else:
            original_changes[prop_name] = predicted_changes[0]
        
        return standardized_changes, original_changes
    else:
        # 如果模型没有使用标准化，则直接返回预测值
        original_changes = {'mu_change': predicted_changes[0]}
        return None, original_changes  # 第一个返回值为None表示没有标准化值


def compare_with_ground_truth(model_paths, model_dirs, csv_file, row_index):
    """
    与CSV文件中指定行的真实值进行对比（支持多模型）
    
    Args:
        model_paths: 模型文件路径列表
        model_dirs: 模型目录路径列表
        csv_file: CSV文件路径
        row_index: 行索引
        
    Returns:
        预测值和真实值的对比
    """
    # 读取指定行的数据
    df = pd.read_csv(csv_file)
    if row_index >= len(df):
        raise ValueError(f"行索引 {row_index} 超出范围，CSV文件共有 {len(df)} 行")
    
    row = df.iloc[row_index]
    
    # 获取真实值
    true_values = {'mu_change': row['mu_change']}
    
    # 对每个模型进行预测
    predictions_list = []
    for model_path, model_dir in zip(model_paths, model_dirs):
        standardized_pred, original_pred = predict_property_changes(
            model_path, model_dir,
            row['smiles_from'], row['smiles_to'],
            row['to_atom_symbol'], row['operation_type']
        )
        
        # 使用原始尺度预测值
        predictions_list.append(original_pred)
    
    return predictions_list, true_values, row


def print_multi_model_comparison_results(predictions_list, true_values, row, model_dirs):
    """
    打印多模型预测值与真实值的对比结果
    
    Args:
        predictions_list: 多个模型的预测值列表
        true_values: 真实值字典
        row: 数据行
        model_dirs: 模型目录路径列表
    """
    print("\n多模型预测结果与真实值对比:")
    print("=" * 80)
    print(f"起始分子 SMILES: {row['smiles_from']}")
    print(f"目标分子 SMILES: {row['smiles_to']}")
    print(f"变化原子类型: {row['to_atom_symbol']}")
    print(f"操作类型: {row['operation_type']}")
    print(f"数据集行号: {row.name}")
    print("=" * 80)
    
    # 表头
    header = f"{'属性名称':<15}"
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        header += f"{model_name:<15}"
    header += f"{'真实值':<15}"
    print(header)
    print("-" * 80)
    
    # 数据行
    prop = 'mu_change'
    prop_name = prop.replace("_change", "") if prop.endswith("_change") else prop
    row_str = f"{prop_name:<15}"
    for predictions in predictions_list:
        pred_value = predictions[prop]
        row_str += f"{pred_value:<15.4f}"
    true_value = true_values[prop]
    row_str += f"{true_value:<15.4f}"
    print(row_str)
    
    # 计算每个模型的总体指标
    print("-" * 80)
    print(f"{'模型指标':<15}", end="")
    for i in range(len(model_dirs)):
        print(f"{'MSE':<5} {'RMSE':<5} {'MAE':<5}", end="")
    print(f"{'':<15}")  # 真实值列为空
    print("-" * 80)
    
    true_array = np.array([true_values['mu_change']])
    for i, predictions in enumerate(predictions_list):
        pred_array = np.array([predictions['mu_change']])
        mse = np.mean((pred_array - true_array) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_array - true_array))
        print(f"{'':<15}{mse:<5.2f} {rmse:<5.2f} {mae:<5.2f}", end="")
    print(f"{'':<15}")  # 真实值列为空


def print_comparison_results(predicted_values, true_values, row, model_dir):
    """
    打印预测值与真实值的对比结果
    
    Args:
        predicted_values: 预测值字典
        true_values: 真实值字典
        row: 数据行
        model_dir: 模型目录路径
    """
    print("\n预测结果与真实值对比:")
    print("=" * 60)
    print(f"起始分子 SMILES: {row['smiles_from']}")
    print(f"目标分子 SMILES: {row['smiles_to']}")
    print(f"变化原子类型: {row['to_atom_symbol']}")
    print(f"操作类型: {row['operation_type']}")
    print(f"使用模型: {os.path.basename(model_dir)}")
    print(f"数据集行号: {row.name}")
    print("=" * 60)
    
    # 表头
    print(f"{'属性名称':<15} {'预测值':<15} {'真实值':<15} {'差值':<15}")
    print("-" * 60)
    
    # 数据行
    prop = 'mu_change'
    pred_value = predicted_values[prop]
    true_value = true_values[prop]
    diff = pred_value - true_value
    prop_name = prop.replace("_change", "") if prop.endswith("_change") else prop
    print(f"{prop_name:<15} {pred_value:<15.4f} {true_value:<15.4f} {diff:<15.4f}")
    
    # 计算总体指标
    pred_array = np.array([predicted_values['mu_change']])
    true_array = np.array([true_values['mu_change']])
    
    mse = np.mean((pred_array - true_array) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_array - true_array))
    
    print("-" * 60)
    print(f"{'总体指标':<15} {'MSE':<15} {'RMSE':<15} {'MAE':<15}")
    print(f"{'':<15} {mse:<15.4f} {rmse:<15.4f} {mae:<15.4f}")


def batch_predict(model_path, model_dir, csv_file, num_samples=10, random_seed=42, logger=None):
    """
    批量预测并计算误差
    
    Args:
        model_path: 模型文件路径
        model_dir: 模型目录路径
        csv_file: CSV文件路径
        num_samples: 采样数量
        random_seed: 随机种子
        logger: 日志记录器
        
    Returns:
        预测结果和误差统计
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    
    # 随机采样
    random.seed(random_seed)
    np.random.seed(random_seed)
    sampled_indices = np.random.choice(len(df), min(num_samples, len(df)), replace=False)
    sampled_df = df.iloc[sampled_indices].reset_index(drop=True)
    
    # 获取属性统计信息用于反标准化
    property_stats = load_property_stats(model_dir)
    
    # 创建分子缓存实例
    cache = MoleculeCache("batch_prediction_dataset")
    
    # 存储预测结果和真实值
    predictions_list = []
    targets_list = []
    
    logger.info(f"开始批量预测 {len(sampled_df)} 个样本...")
    
    for idx, row in sampled_df.iterrows():
        try:
            # 获取真实值
            true_changes = {'mu_change': row['mu_change']}
            targets_list.append(true_changes)
            
            # 准备数据
            from_data = smiles_to_graph_data(row['smiles_from'], cache)
            to_data = smiles_to_graph_data(row['smiles_to'], cache)
            
            if from_data is None or to_data is None:
                logger.warning(f"跳过无法处理的分子对: {row['smiles_from']} -> {row['smiles_to']}")
                continue
            
            # 构建边特征（不含属性变化）
            edge_feat = prepare_edge_features(row, property_stats, include_property_changes=False)
            edge_attr = torch.FloatTensor(np.array([edge_feat]))
            
            # 创建模型
            model = MoleculeEvolutionGCNPredictor(
                node_feature_dim=11,      # v0模型节点特征维度
                edge_feature_dim=11,      # 边特征维度（5原子类型 + 6操作类型）
                hidden_dim=128,
                output_dim=1,             # v0模型只预测单个属性 (mu_change)
                num_layers=2
            )
            
            # 加载模型权重
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            
            # 进行预测
            with torch.no_grad():
                predictions = model(from_data, to_data, edge_attr)
            
            # 获取预测结果
            predicted_changes = predictions[0].numpy()
            
            # 检查模型是否使用了标准化
            is_normalized = is_model_normalized(model_dir)
            
            if is_normalized and property_stats:
                # 反标准化得到原始尺度的预测值
                prop_name = 'mu_change'
                if prop_name in property_stats:
                    mean, std = property_stats[prop_name]
                    original_pred = predicted_changes[0] * std + mean
                else:
                    original_pred = predicted_changes[0]
            else:
                original_pred = predicted_changes[0]
            
            predictions_list.append({'mu_change': original_pred})
            
            if (idx + 1) % 10 == 0:
                logger.info(f"已完成 {idx + 1}/{len(sampled_df)} 个样本的预测")
                
        except Exception as e:
            logger.error(f"处理第 {idx} 个样本时出错: {e}")
            continue
    
    # 计算误差统计
    if not predictions_list or not targets_list:
        logger.error("没有有效的预测结果用于计算误差")
        return None, None
    
    # 转换为数组便于计算
    pred_array = np.array([pred['mu_change'] for pred in predictions_list])
    target_array = np.array([target['mu_change'] for target in targets_list])
    
    # 计算各种误差指标
    mse = np.mean((pred_array - target_array) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_array - target_array))
    
    # 计算R2分数
    ss_res = np.sum((target_array - pred_array) ** 2)
    ss_tot = np.sum((target_array - np.mean(target_array)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)  # 添加小值避免除零
    
    # 计算平均相对误差，处理接近零的情况
    # 使用一个阈值来避免除以接近零的数
    epsilon = 1e-8
    relative_error = np.mean(
        np.abs((pred_array - target_array) / 
               np.where(np.abs(target_array) > epsilon, target_array, epsilon))
    ) * 100
    
    # 计算预测值和真实值的统计信息
    pred_mean = np.mean(pred_array)
    target_mean = np.mean(target_array)
    pred_std = np.std(pred_array)
    target_std = np.std(target_array)
    
    # 创建误差统计字典
    error_stats = {
        'property_names': ['mu_change'],
        'mse': np.array([mse]),
        'rmse': np.array([rmse]),
        'mae': np.array([mae]),
        'r2': np.array([r2]),
        'relative_error_percent': np.array([relative_error]),
        'num_samples': len(predictions_list),
        'pred_mean': np.array([pred_mean]),
        'target_mean': np.array([target_mean]),
        'pred_std': np.array([pred_std]),
        'target_std': np.array([target_std])
    }
    
    return error_stats, sampled_df


def print_error_statistics(error_stats, logger=None):
    """
    打印误差统计结果
    
    Args:
        error_stats: 误差统计字典
        logger: 日志记录器
    """
    if not error_stats:
        print("没有误差统计数据可显示")
        if logger:
            logger.warning("没有误差统计数据可显示")
        return
    
    property_names = error_stats['property_names']
    mse = error_stats['mse']
    rmse = error_stats['rmse']
    mae = error_stats['mae']
    r2 = error_stats['r2']
    relative_error = error_stats['relative_error_percent']
    pred_mean = error_stats['pred_mean']
    target_mean = error_stats['target_mean']
    pred_std = error_stats['pred_std']
    target_std = error_stats['target_std']
    
    print("\n" * 2 + "="*80)
    print("批量预测误差统计结果")
    print("="*80)
    print(f"样本数量: {error_stats['num_samples']}")
    print("-"*80)
    
    # 表头
    header = f"{'属性':<12} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R2':<12} {'相对误差(%)':<12} {'预测均值':<12} {'真实均值':<12} {'预测std':<12} {'真实std':<12}"
    print(header)
    print("-"*80)
    
    # 数据行
    for i, prop in enumerate(property_names):
        prop_name = prop.replace('_change', '')
        row = (f"{prop_name:<12} {mse[i]:<12.4f} {rmse[i]:<12.4f} {mae[i]:<12.4f} {r2[i]:<12.4f} "
               f"{relative_error[i]:<12.2f} {pred_mean[i]:<12.4f} {target_mean[i]:<12.4f} "
               f"{pred_std[i]:<12.4f} {target_std[i]:<12.4f}")
        print(row)
    
    if logger:
        logger.info("批量预测完成，误差统计结果已显示")


def main():
    parser = argparse.ArgumentParser(description='使用训练好的v0模型进行分子进化属性变化预测')
    parser.add_argument('--model-path', type=str,
                       help='模型文件路径 (.pth文件)')
    parser.add_argument('--model-dir', type=str,
                       help='模型目录路径 (包含training_data.json)')
    parser.add_argument('--smiles-from', type=str,
                       help='起始分子的SMILES (单次预测)')
    parser.add_argument('--smiles-to', type=str,
                       help='目标分子的SMILES (单次预测)')
    parser.add_argument('--atom-symbol', type=str,
                       help='变化涉及的原子类型 (如: C, N, O, F, P) (单次预测)')
    parser.add_argument('--operation-type', type=str,
                       choices=['add', 'replace', 'del', 'add_multi', 'del_multi', 'complex'],
                       help='操作类型 (单次预测)')
    parser.add_argument('--csv-file', type=str, default='mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv',
                       help='CSV文件路径 (批量预测)')
    parser.add_argument('--row-index', type=int,
                       help='CSV文件中的行索引，用于对比预测值和真实值')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='批量预测的样本数量 (默认: 100)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='日志级别 (默认: INFO)')
    
    args = parser.parse_args()
    
    # 设置日志记录器
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logger(log_level)
    
    # 检查是单次预测还是批量预测
    single_prediction = args.smiles_from and args.smiles_to and args.atom_symbol and args.operation_type
    batch_prediction = args.csv_file and args.row_index is None
    comparison_prediction = args.csv_file and args.row_index is not None
    
    if not single_prediction and not batch_prediction and not comparison_prediction:
        logger.error("请指定单次预测参数或批量预测参数")
        parser.print_help()
        return
    
    try:
        # 如果没有指定模型路径，则自动查找并选择
        if not args.model_path:
            model_files = find_model_files()
            selected_models = select_models_interactively(model_files)
            
            if not selected_models:
                logger.warning("未选择模型，程序退出")
                return
            
            # 如果选择了多个模型且是对比预测，则进行多模型对比
            if len(selected_models) > 1 and comparison_prediction:
                model_paths = selected_models
                model_dirs = [os.path.dirname(model_path) for model_path in model_paths]
            else:
                # 默认使用第一个选择的模型
                args.model_path = selected_models[0]
                args.model_dir = os.path.dirname(selected_models[0])
        elif not args.model_dir:
            # 如果指定了模型路径但没有指定模型目录
            args.model_dir = os.path.dirname(args.model_path)
        
        # 检查模型目录中是否包含训练数据
        if hasattr(args, 'model_dir') and args.model_dir:
            stats_file = os.path.join(args.model_dir, "training_data.json")
            if not os.path.exists(stats_file):
                logger.warning(f"模型目录中未找到训练数据文件 {stats_file}")
                logger.warning("将无法进行反标准化以获得原始尺度的预测值")
        
        # 执行单次预测
        if single_prediction:
            standardized_changes, original_changes = predict_property_changes(
                args.model_path, args.model_dir,
                args.smiles_from, args.smiles_to,
                args.atom_symbol, args.operation_type
            )
            
            print("\n预测结果:")
            print("=" * 50)
            print(f"起始分子 SMILES: {args.smiles_from}")
            print(f"目标分子 SMILES: {args.smiles_to}")
            print(f"变化原子类型: {args.atom_symbol}")
            print(f"操作类型: {args.operation_type}")
            print(f"使用模型: {os.path.basename(args.model_dir)}")
            print("=" * 50)
            
            # 创建表格展示预测结果
            print("\n预测结果汇总:")
            # 表头
            print(f"{'属性名称':<15} {'标准化值':<12} {'原始值':<12}")
            print("-" * 45)
            
            # 表格数据行
            prop = 'mu_change'
            orig_value = original_changes[prop]
            std_value = standardized_changes.get(prop, "N/A") if standardized_changes else "N/A"
            prop_name = prop.replace("_change", "") if prop.endswith("_change") else prop
            if standardized_changes:
                print(f"{prop_name:<15} {std_value:<12.4f} {orig_value:<12.4f}" if isinstance(std_value, (int, float)) else f"{prop_name:<15} {std_value:<12} {orig_value:<12.4f}")
            else:
                print(f"{prop_name:<15} {'N/A':<12} {orig_value:<12.4f}")
                
            if not standardized_changes:
                print("\n注意: 模型未使用标准化数据进行训练")
        
        # 执行预测值与真实值对比（支持多模型）
        elif comparison_prediction:
            if 'model_paths' in locals():
                # 多模型对比
                predictions_list, true_values, row = compare_with_ground_truth(
                    model_paths, model_dirs, args.csv_file, args.row_index
                )
                print_multi_model_comparison_results(predictions_list, true_values, row, model_dirs)
            else:
                # 单模型对比
                predictions_list, true_values, row = compare_with_ground_truth(
                    [args.model_path], [args.model_dir], args.csv_file, args.row_index
                )
                # 对于单模型，predictions_list应该只包含一个元素
                predicted_values = predictions_list[0] if predictions_list else {}
                print_comparison_results(predicted_values, true_values, row, args.model_dir)
        
        # 执行批量预测
        elif batch_prediction:
            error_stats, sampled_df = batch_predict(
                args.model_path, args.model_dir,
                args.csv_file, args.num_samples, args.random_seed, logger
            )
            
            if error_stats:
                print_error_statistics(error_stats, logger)
                
                # 询问是否保存结果
                save_results = input("\n是否保存预测结果到CSV文件? (y/N): ").strip().lower()
                if save_results == 'y':
                    output_file = f"batch_prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    sampled_df.to_csv(output_file, index=False)
                    logger.info(f"预测结果已保存到: {output_file}")
            
    except Exception as e:
        logger.error(f"预测过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()