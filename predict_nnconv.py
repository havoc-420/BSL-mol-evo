#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的NNConv模型进行分子进化属性变化预测
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
from mol_evo.core.models.nnconv_predictor import MoleculeEvolutionNNConvPredictor
from mol_evo.core.data.processing import (
    smiles_to_fingerprint, 
    prepare_edge_features
)
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
    model_pattern = os.path.join(project_root, 'mol_evo', 'model-data', 'nnconv', 'training_*', 'molecule_evolution_nnconv_predictor.pth')
    model_files = glob.glob(model_pattern)
    return sorted(model_files)


def select_model_interactively(model_files):
    """
    交互式选择模型
    
    Args:
        model_files: 模型文件列表
        
    Returns:
        选择的模型文件路径
    """
    if not model_files:
        print("未找到任何模型文件")
        return None
    
    if len(model_files) == 1:
        print(f"找到一个模型文件: {os.path.basename(os.path.dirname(model_files[0]))}")
        return model_files[0]
    
    print("找到多个模型，请选择一个:")
    for i, model_file in enumerate(model_files):
        model_dir = os.path.dirname(model_file)
        model_name = os.path.basename(model_dir)
        print(f"{i+1}. {model_name}")
    
    while True:
        try:
            choice = input(f"请选择模型 (1-{len(model_files)}) 或按回车选择最新模型: ").strip()
            if not choice:
                # 选择最新的模型
                latest_model = max(model_files, key=os.path.getctime)
                print(f"选择最新模型: {os.path.basename(os.path.dirname(latest_model))}")
                return latest_model
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(model_files):
                return model_files[choice_idx]
            else:
                print(f"请输入 1 到 {len(model_files)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")


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
    
    # 构建节点特征 (使用Morgan指纹)
    fingerprint_from = smiles_to_fingerprint(smiles_from)
    fingerprint_to = smiles_to_fingerprint(smiles_to)
    node_features = torch.FloatTensor(np.array([fingerprint_from, fingerprint_to]))
    
    # 构建边索引和边特征
    edge_index = torch.LongTensor([[0], [1]])  # 从节点0到节点1的边
    
    # 准备边特征（不含属性变化）
    edge_feat = prepare_edge_features(df.iloc[0], property_stats, include_property_changes=False)
    edge_attr = torch.FloatTensor(np.array([edge_feat]))
    
    # 创建图数据对象
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    
    return data, property_stats


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
    data, property_stats = prepare_single_prediction_data(
        smiles_from, smiles_to, to_atom_symbol, operation_type, model_dir)
    
    # 创建模型
    model = MoleculeEvolutionNNConvPredictor(
        node_feature_dim=2048,    # Morgan指纹维度
        edge_feature_dim=11,      # 边特征维度（5原子类型 + 6操作类型）
        hidden_dim=128,
        output_dim=15,            # 属性变化维度
        num_layers=3
    )
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # 进行预测
    with torch.no_grad():
        predictions = model(data)
    
    # 获取预测结果
    predicted_changes = predictions[0].numpy()
    
    # 定义属性名称
    property_names = ['A_change', 'B_change', 'C_change', 'mu_change', 'alpha_change',
                      'homo_change', 'lumo_change', 'gap_change', 'r2_change', 'zpve_change',
                      'U0_change', 'U_change', 'H_change', 'G_change', 'Cv_change']
    
    # 如果有属性统计信息，进行反标准化
    if property_stats:
        standardized_changes = {}
        original_changes = {}
        
        for i, prop_name in enumerate(property_names):
            standardized_changes[prop_name] = predicted_changes[i]
            # 反标准化得到原始尺度的预测值
            if prop_name in property_stats:
                mean, std = property_stats[prop_name]
                original_changes[prop_name] = predicted_changes[i] * std + mean
            else:
                original_changes[prop_name] = predicted_changes[i]
        
        return standardized_changes, original_changes
    
    # 如果没有统计信息，只返回标准化的预测值
    standardized_changes = {prop_name: predicted_changes[i] 
                           for i, prop_name in enumerate(property_names)}
    return standardized_changes, None


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
    
    # 定义属性名称
    property_names = ['A_change', 'B_change', 'C_change', 'mu_change', 'alpha_change',
                      'homo_change', 'lumo_change', 'gap_change', 'r2_change', 'zpve_change',
                      'U0_change', 'U_change', 'H_change', 'G_change', 'Cv_change']
    
    # 存储预测结果和真实值
    predictions_list = []
    targets_list = []
    
    logger.info(f"开始批量预测 {len(sampled_df)} 个样本...")
    
    for idx, row in sampled_df.iterrows():
        try:
            # 获取真实值
            true_changes = {prop: row[prop] for prop in property_names}
            targets_list.append(true_changes)
            
            # 进行预测
            standardized_pred, original_pred = predict_property_changes(
                model_path, model_dir,
                row['smiles_from'], row['smiles_to'],
                row['to_atom_symbol'], row['operation_type']
            )
            
            # 使用原始尺度预测值或标准化预测值
            if original_pred:
                predictions_list.append(original_pred)
            else:
                predictions_list.append(standardized_pred)
            
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
    pred_array = np.array([[pred[prop] for prop in property_names] for pred in predictions_list])
    target_array = np.array([[target[prop] for prop in property_names] for target in targets_list])
    
    # 计算各种误差指标
    mse = np.mean((pred_array - target_array) ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_array - target_array), axis=0)
    
    # 计算R2分数
    ss_res = np.sum((target_array - pred_array) ** 2, axis=0)
    ss_tot = np.sum((target_array - np.mean(target_array, axis=0)) ** 2, axis=0)
    r2 = 1 - ss_res / (ss_tot + 1e-8)  # 添加小值避免除零
    
    # 计算平均相对误差，处理接近零的情况
    # 使用一个阈值来避免除以接近零的数
    epsilon = 1e-8
    relative_error = np.mean(
        np.abs((pred_array - target_array) / 
               np.where(np.abs(target_array) > epsilon, target_array, epsilon)), 
        axis=0
    ) * 100
    
    # 计算预测值和真实值的统计信息
    pred_mean = np.mean(pred_array, axis=0)
    target_mean = np.mean(target_array, axis=0)
    pred_std = np.std(pred_array, axis=0)
    target_std = np.std(target_array, axis=0)
    
    # 创建误差统计字典
    error_stats = {
        'property_names': property_names,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'relative_error_percent': relative_error,
        'num_samples': len(predictions_list),
        'pred_mean': pred_mean,
        'target_mean': target_mean,
        'pred_std': pred_std,
        'target_std': target_std
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
    
    print("\n" + "="*120)
    print("批量预测误差统计结果")
    print("="*120)
    print(f"样本数量: {error_stats['num_samples']}")
    print("-"*120)
    
    # 表头
    header = f"{'属性':<12} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R2':<12} {'相对误差(%)':<12} {'预测均值':<12} {'真实均值':<12} {'预测std':<12} {'真实std':<12}"
    print(header)
    print("-"*120)
    
    # 数据行
    for i, prop in enumerate(property_names):
        prop_name = prop.replace('_change', '')
        row = (f"{prop_name:<12} {mse[i]:<12.4f} {rmse[i]:<12.4f} {mae[i]:<12.4f} {r2[i]:<12.4f} "
               f"{relative_error[i]:<12.2f} {pred_mean[i]:<12.4f} {target_mean[i]:<12.4f} "
               f"{pred_std[i]:<12.4f} {target_std[i]:<12.4f}")
        print(row)
    
    # 平均值
    print("-"*120)
    avg_row = (f"{'平均':<12} {np.mean(mse):<12.4f} {np.mean(rmse):<12.4f} {np.mean(mae):<12.4f} "
               f"{np.mean(r2):<12.4f} {np.mean(relative_error):<12.2f} {np.mean(pred_mean):<12.4f} "
               f"{np.mean(target_mean):<12.4f} {np.mean(pred_std):<12.4f} {np.mean(target_std):<12.4f}")
    print(avg_row)
    
    if logger:
        logger.info("批量预测完成，误差统计结果已显示")


def main():
    parser = argparse.ArgumentParser(description='使用训练好的NNConv模型进行分子进化属性变化预测')
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
    parser.add_argument('--csv-file', type=str,
                       help='CSV文件路径 (批量预测)')
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
    batch_prediction = args.csv_file
    
    if not single_prediction and not batch_prediction:
        logger.error("请指定单次预测参数或批量预测参数")
        parser.print_help()
        return
    
    try:
        # 如果没有指定模型路径，则自动查找并选择
        if not args.model_path:
            model_files = find_model_files()
            selected_model = select_model_interactively(model_files)
            
            if not selected_model:
                logger.warning("未选择模型，程序退出")
                return
            
            args.model_path = selected_model
            args.model_dir = os.path.dirname(selected_model)
        elif not args.model_dir:
            # 如果指定了模型路径但没有指定模型目录
            args.model_dir = os.path.dirname(args.model_path)
        
        # 检查模型目录中是否包含训练数据
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
            
            print("\n标准化预测值:")
            for prop, value in standardized_changes.items():
                print(f"{prop:15s}: {value:8.4f}")
                
            if original_changes:
                print("\n原始尺度预测值:")
                for prop, value in original_changes.items():
                    print(f"{prop:15s}: {value:8.4f}")
            else:
                print("\n注意: 未找到属性统计信息，无法进行反标准化")
                print("要获得原始尺度的预测值，请确保模型目录中包含完整的训练数据文件")
        
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