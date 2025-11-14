#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0版本模型训练脚本
专门用于训练针对单属性（mu_change）预测的GCN模型
"""

import sys
import os
import argparse
from scipy import optimize
import torch
import numpy as np
import math
from datetime import datetime
from tqdm import tqdm
import shutil
import random
import json

import torch.nn as nn
from torch_geometric.loader import DataLoader


""" BASE SETTINGS """
# 设置项目根目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

# 导入自定义模块
try:
    from mol_evo.core.models.v0 import ModelFactory
    from mol_evo.core.data import (
        build_molecule_evolution_dataset_v0,  # 从新的统一模块导入
        load_operation_config  # 导入配置加载函数
    )
    from mol_evo.core.data.pair_data import MoleculePairDataset, pair_collate  # 使用新的数据处理模块
    from mol_evo.utils.training_utils import (
        split_data_indices, 
        save_training_data_as_json, 
        plot_training_trends,
        log_training_completion,
        log_training_start,
        log_model_creation,
        log_training_start_message
    )
    from mol_evo.utils.device_utils import (
        move_data_to_device,
        move_data_to_device_for_validation,
        move_data_to_device_for_testing
    )
    from mol_evo.utils.logger_utils import DualLogger
    from mol_evo.utils.logger_v0 import (
        log_dataset_examples,
        log_data_construction_info,
        log_dataset_split_info,
        log_device_info,
        log_checkpoint_saved,
        log_training_interrupted,
        log_early_stopping,
        log_epoch_progress,
        log_training_metrics,
        log_model_saved,
    )
    from mol_evo.utils.training_metrics import TrainingMetricsRecorder
    from mol_evo.utils.config_utils import load_config_by_model_type
except ImportError as e:
    import traceback
    print(f"无法导入所需的模块 train-v0.py: {e}")
    print("详细错误堆栈信息:")
    traceback.print_exc()
    exit(1)

# 目标属性名称 - 默认值，将被命令行参数覆盖
TARGET_PROPERTY = 'mu_change'


def train_model(data_file: str, max_pairs: int = None, epochs: int = 100, 
                seed: int = 42, batch_size: int = 64, learning_rate: float = 0.01,
                model_type: str = "gcn_linear", model_config: dict = None):
    """
    训练v0版本的分子进化预测器模型（单属性预测）
    
    Args:
        data_file: 数据文件路径
        max_pairs: 最大对数（用于调试）
        epochs: 训练轮数
        seed: 随机种子
        batch_size: 批处理大小
        learning_rate: 学习率
        model_type: 模型类型
        model_config: 模型配置参数字典
    """
    # 加载操作配置
    try:
        load_operation_config(dataset_path=data_file)
    except Exception as e:
        print(f"警告: 无法加载操作配置: {e}")
        try:
            load_operation_config()
        except Exception as e2:
            print(f"警告: 无法加载默认操作配置: {e2}")
    
    # 设置所有随机种子以确保可重复性
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # INFO 检查是否是 FragNet 模型类型
    is_fragnet_model = model_type and "frag" in model_type.lower()
        
    # 初始化最终使用的模型配置
    final_model_config = {}
    
    # 如果提供了 model_config（来自配置文件），则优先使用它
    if model_config:
        # 从配置中获取训练参数，并覆盖函数的默认参数
        train_config = model_config.get('train', {})
        data_file = train_config.get('data_file', data_file)
        epochs = train_config.get('epochs', epochs)
        seed = train_config.get('seed', seed)
        batch_size = train_config.get('batch_size', batch_size)
        learning_rate = train_config.get('learning_rate', learning_rate)
        max_pairs = train_config.get('max_pairs', max_pairs)

        # 提取模型本身的参数，用于创建模型
        final_model_config = model_config.get('model', {}).copy()  # 使用 copy() 避免原地修改
    else:
        # 只有在没有提供外部配置时，才从默认路径加载模型配置
        final_model_config = load_config_by_model_type(model_type)
    
    # train-data 存储位置
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 模型目录命名规则: train_{TIMESTAMP}_{TARGET-ATTR}_{max-pairs}_{epoches}
    dir_name = f"train-{timestamp}-{TARGET_PROPERTY}-{max_pairs}-{epochs}"
    model_dir = os.path.join(project_root, 'mol_evo', 'output', 'v0', model_type, dir_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # 早停机制参数
    patience_limit = 50
    # 初始化property_stats为空字典
    property_stats = {}
    
    # 创建自定义的DualLogger实例
    logger = DualLogger(model_dir)
    log_training_start(logger, data_file, max_pairs, epochs)
    
    try:
        # 保存模型配置参数到JSON文件（在训练开始时）
        model_config_data = {
            "timestamp": datetime.now().isoformat(),
            "model_params": final_model_config,
            "model-train-config": {
                "model_type": model_type,
                "data_file": data_file,
                "max_pairs": max_pairs,
                "epochs": epochs,
                "seed": seed,
                "target_property": TARGET_PROPERTY,
                "batch_size": batch_size,
                "device": str(device) if 'device' in locals() else "cpu",
                "optimizer": "Adam",
                "learning_rate": learning_rate,
                "weight_decay": 1e-5,
                "scheduler": "ReduceLROnPlateau",
                "loss_function": "L1Loss",
                "patience_limit": patience_limit
            }
        }
        
        model_config_path = os.path.join(model_dir, "model_config.json")
        with open(model_config_path, 'w', encoding='utf-8') as f:
            json.dump(model_config_data, f, ensure_ascii=False, indent=2)
        logger.info(f"模型配置已保存到: {model_config_path}")
        
        # STAGE 构建图数据 - 使用新的数据处理方法
        logger.info(f"正在构建图数据: {data_file}")  # 默认输出到控制台和文件
        from_data_list, to_data_list, edge_attrs, target_features, property_stats = build_molecule_evolution_dataset_v0(
            data_file, max_pairs, TARGET_PROPERTY, logger, model_type)
        
        # 输出训练集的头部信息（前几个样本示例）
        log_dataset_examples(logger, from_data_list, to_data_list, edge_attrs, target_features)

        # 记录数据构建信息
        log_data_construction_info(logger, from_data_list, model_config, edge_attrs, target_features)
        
        # 检查是否有有效数据
        if len(from_data_list) == 0:
            logger.error("没有有效的训练数据，请检查数据预处理步骤")
            return
            
        # 检查数据是否匹配
        if not (len(from_data_list) == len(to_data_list) == len(edge_attrs) == len(target_features)):
            logger.error(f"数据长度不匹配: from_data_list={len(from_data_list)}, to_data_list={len(to_data_list)}, edge_attrs={len(edge_attrs)}, target_features={len(target_features)}")
            return
        
        # 划分数据集
        train_idx, val_idx, test_idx = split_data_indices(len(from_data_list), 0.8, 0.1, 0.1, seed)
        log_dataset_split_info(logger, train_idx, val_idx, test_idx)
        
        # 检查训练集是否为空
        if len(train_idx) == 0:
            logger.error("训练集为空，请检查数据划分")
            return
        
        # 创建数据集 - 统一使用 MoleculePairDataset
        train_dataset = MoleculePairDataset(
            [from_data_list[i] for i in train_idx],
            [to_data_list[i] for i in train_idx],
            edge_attrs[train_idx],
            target_features[train_idx])
        
        # 检查训练数据集中是否有有效数据
        if len(train_dataset) == 0:
            logger.error("训练数据集为空，请检查数据处理过程")
            return
            
        val_dataset = MoleculePairDataset(
            [from_data_list[i] for i in val_idx],
            [to_data_list[i] for i in val_idx],
            edge_attrs[val_idx],
            target_features[val_idx]) if len(val_idx) > 0 else None
            
        test_dataset = MoleculePairDataset(
            [from_data_list[i] for i in test_idx],
            [to_data_list[i] for i in test_idx],
            edge_attrs[test_idx],
            target_features[test_idx]) if len(test_idx) > 0 else None
        
        # 创建 DataLoader
        if is_fragnet_model:
            # 导入FragNet的collate_fn
            from mol_evo.modules.FragNet.fragnet.dataset.data import collate_fn as fragnet_collate_fn
            from torch.utils.data import DataLoader as torchDataLoader

            train_loader = torchDataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=fragnet_collate_fn,
                pin_memory=True)
            
            val_loader = torchDataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=fragnet_collate_fn) if val_dataset is not None else None
                
            test_loader = torchDataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=fragnet_collate_fn) if test_dataset is not None else None
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=pair_collate,
                pin_memory=True)
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=pair_collate) if val_dataset is not None else None
                
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=pair_collate) if test_dataset is not None else None
        
        # 创建模型
        # 根据模型类型传递不同的参数
        # 创建模型，直接使用最终的模型配置
        # 动态设置边特征维度
        if edge_attrs.size(1) != final_model_config.get('edge_feature_dim', 11):
            logger.info(f"更新模型边特征维度: {final_model_config.get('edge_feature_dim', 11)} -> {edge_attrs.size(1)}")
            final_model_config['edge_feature_dim'] = edge_attrs.size(1)
            
        model = ModelFactory.create(model_type, **final_model_config)
        log_model_creation(logger, model)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_device_info(logger, device)
        
        # 将模型移到设备上
        model = model.to(device)
        
        # TAG 定义优化器和损失函数
        # INFO -1 AdamW
        # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        # INFO -2 Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        # 从配置中获取min_lr参数，如果不存在则默认为1e-8
        min_lr = model_config.get('min_lr', 1e-8) if model_config and 'min_lr' in model_config else \
                (model_config or {}).get('train', {}).get('min_lr', 1e-6)
        # 确保min_lr是浮点数类型，避免字符串和数字比较的错误
        min_lr = float(min_lr)
        # 使用更积极的学习率衰减策略
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.3, min_lr=min_lr)
        criterion = nn.L1Loss()
        
        # 创建训练指标记录器
        metrics_recorder = TrainingMetricsRecorder()
        
        # 记录模型和训练参数
        # 获取模型的实际参数
        model_actual_params = final_model_config.copy()
        model_actual_params["model_type"] = model_type
        
        # 如果模型有特定的参数，也可以添加进来
        if hasattr(model, 'get_model_config'):
            model_actual_params.update(model.get_model_config())
            
        metrics_recorder.set_model_params(model_actual_params)
        training_params = {
            "model_type": model_type,
            "data_file": data_file,
            "max_pairs": max_pairs,
            "epochs": epochs,
            "seed": seed,
            "target_property": TARGET_PROPERTY,
            "batch_size": batch_size,
            "device": str(device),
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "weight_decay": 1e-5,
            "scheduler": "ReduceLROnPlateau",
            "loss_function": "L1Loss",
            "patience_limit": 50
        }
        metrics_recorder.set_training_params(training_params)
        metrics_recorder.set_property_stats(property_stats)
        
        # 早停机制参数
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        should_stop = False  # 初始化 should_stop 变量
        
        # INFO 开始训练
        logger.info(f"开始训练 ({epochs} 轮)...")
        
        # STAGE 训练循环
        model.train()
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            epoch_loss = 0.0
            for from_batch, to_batch, edge_batch, target_batch in train_loader:
                optimizer.zero_grad()
                
                # 移动到设备
                from_batch, to_batch, edge_batch, target_batch = move_data_to_device(
                    from_batch, to_batch, edge_batch, target_batch, device, is_fragnet_model)
                
                # 前向传播
                predictions = model(from_batch, to_batch, edge_batch)
                
                # 计算训练损失
                train_loss = criterion(predictions, target_batch)
                
                # 检查是否有NaN或inf值
                if math.isnan(train_loss.item()) or math.isinf(train_loss.item()):
                    log_training_interrupted(logger, epoch)
                    # 确保训练真正停止
                    should_stop = True
                    break
                
                # 反向传播
                train_loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += train_loss.item() * target_batch.size(0)
            
            # 如果需要停止训练，则跳出训练循环
            if should_stop:
                break
            
            # 计算平均epoch损失
            epoch_loss /= len(train_dataset)
            metrics_recorder.record_train_loss(epoch_loss)
            
            # TEST 每100轮保存一次checkpoint
            # if (epoch + 1) % 100 == 0:
            #     checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pth")
            #     torch.save({
            #         'epoch': epoch + 1,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'train_loss': epoch_loss,
            #         'best_val_loss': best_val_loss,
            #         'patience_counter': patience_counter,
            #     }, checkpoint_path)
            #     log_checkpoint_saved(logger, checkpoint_path)
            
            # STAGE 验证阶段
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    all_val_predictions = []
                    all_val_targets = []
                    
                    for from_batch, to_batch, edge_batch, target_batch in val_loader:
                        # 移动到设备
                        from_batch, to_batch, edge_batch, target_batch = move_data_to_device_for_validation(
                            from_batch, to_batch, edge_batch, target_batch, device, is_fragnet_model)
                        
                        # 前向传播
                        val_predictions = model(from_batch, to_batch, edge_batch)
                        vloss = criterion(val_predictions, target_batch)
                        val_loss += vloss.item() * target_batch.size(0)
                        
                        all_val_predictions.append(val_predictions)
                        all_val_targets.append(target_batch)
                    
                    # 计算平均验证损失
                    val_loss /= len(val_dataset)
                    metrics_recorder.record_val_loss(val_loss)
                    
                    # 检查验证损失是否有NaN或inf
                    if math.isnan(val_loss) or math.isinf(val_loss):
                        log_training_interrupted(logger, epoch, "NaN或inf验证损失值")
                        # 确保训练真正停止
                        should_stop = True
                        break  # 立即跳出验证数据加载器的循环
                    
                    # 如果需要停止训练，则跳出训练循环
                    if should_stop:
                        # 恢复最佳模型状态
                        if best_model_state is not None:
                            model.load_state_dict(best_model_state)
                        break
                    
                    # 更新学习率调度器
                    scheduler.step(val_loss)
                    
                    # 计算验证集的额外评估指标 (每10个epoch计算一次)
                    if (epoch + 1) % 10 == 0:
                        # 合并所有验证预测和目标
                        all_val_predictions = torch.cat(all_val_predictions, dim=0)
                        all_val_targets = torch.cat(all_val_targets, dim=0)
                        
                        # MARK RMSE
                        val_mse = torch.mean((all_val_targets - all_val_predictions) ** 2)
                        val_rmse = torch.sqrt(val_mse)
                        
                        # MAE
                        val_mae = torch.mean(torch.abs(all_val_predictions - all_val_targets))
                        
                        # R²
                        val_ss_res = torch.sum((all_val_targets - all_val_predictions) ** 2)
                        val_ss_tot = torch.sum((all_val_targets - torch.mean(all_val_targets)) ** 2)
                        if val_ss_tot.item() == 0:
                            val_r2 = 0.0
                        else:
                            val_r2 = (1 - val_ss_res / (val_ss_tot + 1e-8)).item()
                        
                        # PCC (Pearson Correlation Coefficient)
                        pred_mean = torch.mean(all_val_predictions)
                        target_mean = torch.mean(all_val_targets)
                        pred_centered = all_val_predictions - pred_mean
                        target_centered = all_val_targets - target_mean
                        numerator = torch.sum(pred_centered * target_centered)
                        pred_sq_sum = torch.sum(pred_centered ** 2)
                        target_sq_sum = torch.sum(target_centered ** 2)
                        denominator = torch.sqrt(pred_sq_sum * target_sq_sum)
                        
                        if denominator.item() == 0:
                            val_pcc = 0.0
                        else:
                            val_pcc = (numerator / denominator).item()
                        
                        # 记录验证指标
                        metrics_recorder.record_val_metrics(epoch+1, val_loss, val_mse.item(), val_rmse.item(), val_mae.item(), val_r2, val_pcc)
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
                    # 记录早停信息
                    metrics_recorder.record_early_stopping(epoch+1, best_val_loss, patience_counter)
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience_limit:
                    log_early_stopping(logger, epoch)
                    # 恢复最佳模型状态
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
            
            # 每10个epoch输出一次信息
            if (epoch + 1) % 10 == 0:
                metrics_recorder.log_epoch_progress(logger, epoch, epochs, epoch_loss, val_loss, log_epoch_progress=log_epoch_progress)
        
        # 如果训练因NaN/inf而停止，也要记录早停信息
        if should_stop:
            log_early_stopping(logger, epoch)
        
        log_training_start_message(logger, epochs)

        # STAGE 测试阶段
        if test_loader is not None:
            model.eval()
            with torch.no_grad():
                all_test_predictions = []
                all_test_targets = []
                
                for from_batch, to_batch, edge_batch, target_batch in test_loader:
                    # 移动到设备
                    from_batch, to_batch, edge_batch = move_data_to_device_for_testing(
                        from_batch, to_batch, edge_batch, device, is_fragnet_model)
                    
                    # 前向传播
                    test_predictions = model(from_batch, to_batch, edge_batch)
                    all_test_predictions.append(test_predictions.cpu())
                    all_test_targets.append(target_batch.cpu())
                
                # 合并所有测试预测和目标
                test_predictions = torch.cat(all_test_predictions, dim=0).to(device)
                test_targets = torch.cat(all_test_targets, dim=0).to(device)
                
                # criterion = nn.MSELoss()
                criterion = nn.L1Loss()
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
                
                # PCC (Pearson Correlation Coefficient)
                pred_mean = torch.mean(test_predictions)
                target_mean = torch.mean(test_targets)
                pred_centered = test_predictions - pred_mean
                target_centered = test_targets - target_mean
                numerator = torch.sum(pred_centered * target_centered)
                pred_sq_sum = torch.sum(pred_centered ** 2)
                target_sq_sum = torch.sum(target_centered ** 2)
                denominator = torch.sqrt(pred_sq_sum * target_sq_sum)
                
                if denominator.item() == 0:
                    pcc = 0.0
                else:
                    pcc = (numerator / denominator).item()
                
                # Rank Loss计算
                def compute_rank_loss(preds, targets):
                    """
                    计算Rank Loss，衡量预测值和真实值之间的排序一致性
                    """
                    # 获取所有样本对
                    n = preds.shape[0]
                    if n < 2:
                        return 0.0
                    
                    # 计算所有可能的样本对
                    pred_diffs = preds.unsqueeze(1) - preds.unsqueeze(0)
                    target_diffs = targets.unsqueeze(1) - targets.unsqueeze(0)
                    
                    # 只考虑目标值不同的样本对
                    mask = target_diffs != 0
                    sign_diffs = torch.sign(target_diffs[mask])
                    
                    # 计算hinge loss
                    loss = torch.relu(1.0 - sign_diffs * pred_diffs[mask])
                    return loss.mean().item()
                
                rank_loss = compute_rank_loss(test_predictions, test_targets)
                
                # TAG 阈值准确率评估：在标准化情况下，阈值表示标准差的倍数
                float_threshold = 0.5           #初始阈值
                threshold_scale = 0.1           #阈值数量级记录

                accuracy = 1.0                  #上一轮准确率
                target_accuracy = 0.5           #目标准确率
                threshold_accuracies = {}       #阈值-准确率映射

                #动态调整阈值以计算不同阈值下的准确率
                while accuracy > target_accuracy:
                    #计算预测值与真实值差异小于当前阈值的样本比例
                    correct_predictions = (torch.abs(test_predictions - test_targets) < float_threshold).float()
                    accuracy = correct_predictions.mean().item()
                    threshold_accuracies[float_threshold] = accuracy

                    #动态调整阈值：在同一数量级内递减，达到边界后进入下一更小数量级（1.1 防 float-th 的浮点数溢出）
                    if float_threshold <= threshold_scale * 1.1:
                        threshold_scale /= 10
                        float_threshold /= 2
                    else:
                        float_threshold -= threshold_scale
                
                # 保存模型
                model_filename = "last.pth"
                model_path = os.path.join(model_dir, model_filename)
                
                # # TEST 添加调试日志，输出模型参数的前一小部分
                # logger.info("Debug: 模型参数前10个值:")
                # state_dict = model.state_dict()
                # for name, param in list(state_dict.items())[:3]:  # 取前几个参数
                #     values = param.flatten()[:10]  # 取前10个值
                #     logger.info(f"Debug: 参数 {name} (shape: {param.shape}) => 前10个值: {values.tolist()}")
                
                torch.save(model.state_dict(), model_path)
                
                # 记录测试指标
                dataset_info = {
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                    "test_size": len(test_idx)
                }
                
                metrics_recorder.record_test_metrics(
                    test_loss=test_loss.item(), rmse=rmse.item(), mae=mae.item(), 
                    r2=r2, threshold_accs=threshold_accuracies, dataset_info=dataset_info,
                    pcc=pcc, rank_loss=rank_loss)
                
                # 准备测试指标数据
                test_metrics = {
                    "test_loss": test_loss.item(),
                    "rmse": rmse.item(),
                    "mae": mae.item(),
                    "r2": r2,
                    "pcc": pcc,
                    "rank_loss": rank_loss,
                    "threshold_accs": threshold_accuracies,
                    "dataset_info": {
                        "train_size": len(train_idx),
                        "val_size": len(val_idx),
                        "test_size": len(test_idx)
                    }
                }
                
                # 保存训练数据为JSON格式，包含训练参数
                training_params = {
                    "model_type": model_type,
                    "data_file": data_file,
                    "max_pairs": max_pairs,
                    "epochs": epochs,
                    "seed": seed,
                    "target_property": TARGET_PROPERTY,
                    "batch_size": batch_size,
                    "device": str(device),
                    "optimizer": "Adam",
                    "learning_rate": learning_rate,
                    "weight_decay": 1e-5,
                    "scheduler": "ReduceLROnPlateau",
                    "loss_function": "L1Loss",
                    "patience_limit": patience_limit
                }
                # 保存训练过程数据（不包括模型配置）
                training_process_data = {
                    "timestamp": datetime.now().isoformat(),
                    "test_metrics": test_metrics,
                    "losses": {
                        "train_losses": metrics_recorder.train_losses,
                        "val_losses": metrics_recorder.val_losses
                    },
                    "val_metrics_history": metrics_recorder.val_metrics_history,
                    "property_stats": property_stats
                }
                
                # 保存训练过程数据到JSON文件
                training_process_path = os.path.join(model_dir, "training_process.json")
                with open(training_process_path, 'w', encoding='utf-8') as f:
                    json.dump(training_process_data, f, ensure_ascii=False, indent=2)
                
                
                # 生成训练趋势图
                plot_training_trends(
                    metrics_recorder.train_losses, metrics_recorder.val_losses,
                    metrics_recorder.val_r2s, metrics_recorder.val_maes, model_dir)
                
                # 记录完整评估结果到日志
                log_model_saved(logger, model_path)
                log_training_metrics(
                    logger, metrics_recorder.train_losses, metrics_recorder.val_losses,
                    test_loss, rmse, mae, r2, threshold_accuracies, pcc, rank_loss)
                log_training_completion(
                    logger, metrics_recorder.train_losses, metrics_recorder.val_losses, 
                    test_loss, rmse, mae, r2, threshold_accuracies, model_path, pcc, rank_loss)
                
                # 输出训练结果文件路径，方便查看
                result_file = os.path.join(model_dir, "training_results.json")
                logger.info(f"训练结果已保存到: {result_file}")
                logger.info(f"模型目录: {model_dir}")
    except KeyboardInterrupt:
        logger.info("\n训练被用户中断 (Ctrl+C)")
        choice = input("是否要清理模型目录 {}? (Y/n): ".format(model_dir)).strip().lower()
        if choice in ['Y', 'yes']:
            logger.info("正在清理模型目录...")
            shutil.rmtree(model_dir)
            logger.info("模型目录已清理")
        else:
            logger.info("保留模型目录和文件")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='v0版本分子进化预测器模型训练脚本（单属性预测）')
    
    # 添加自定义类型函数用于限制最大值
    def max_pairs_type(x):
        x = int(x)
        # if x > 30000:
        #     raise argparse.ArgumentTypeError("最大分子对数不能超过30000")
        return x
    
    parser.add_argument('-d', '-data', '--data-file', type=str, 
                       default=os.path.join('mol_evo', 'dataset', 'data', 'qm9-evo-pairs-step-1-with-properties-pct.csv'),  # INFO 这里使用附带 pct 的数据集了
                       help='数据文件路径')
    parser.add_argument('-m', '-max', '--max-pairs', type=max_pairs_type, default=30000, help='最大分子对数（用于调试，最大不超过30000）')
    parser.add_argument('-e', '-ep', '--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('-b', '-batch', '--batch-size', type=int, default=512, help='批处理大小')
    parser.add_argument('-p', '--target-property', type=str, default='mu_change', 
                       help='目标属性')
    parser.add_argument('-s', '--seed', type=int, default=42, help='随机种子')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01, help='学习率')
    parser.add_argument('-mt', '--model-type', type=str, default=None, 
                       help='模型类型: 可通过交互式方式选择')
    parser.add_argument('-c', '--config-file', type=str, default=None,
                       help='YAML配置文件路径: 可直接指定模型配置文件')
    parser.add_argument('--min-lr', type=float, default=None,
                       help='学习率调度器的最小学习率')
    
    args = parser.parse_args()
    
    # 如果提供了配置文件，则使用配置文件中的参数
    model_config = {}
    if args.config_file:
        import yaml
        try:
            with open(args.config_file, 'r', encoding='utf-8') as f:
                model_config = yaml.safe_load(f)
            print(f"已加载配置文件: {args.config_file}")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return
    
    # 如果通过CLI指定了min_lr，则优先使用CLI的值
    if args.min_lr is not None:
        if 'train' not in model_config:
            model_config['train'] = {}
        model_config['train']['min_lr'] = args.min_lr
    
    # 如果没有提供模型类型，则交互式选择
    model_type = args.model_type
    if model_type is None and not args.config_file:
        try:
            from mol_evo.utils.train.model_utils import select_model_type_interactively
            model_type = select_model_type_interactively()
            if model_type is None:
                print("未选择模型类型，退出训练")
                return
        except ImportError as e:
            print(f"无法导入交互式选择工具: {e}")
            print("请提供 --model-type 参数")
            return
    elif args.config_file and not model_type:
        # 如果提供了配置文件但没有指定模型类型，可以根据配置文件名推断模型类型
        config_name = os.path.splitext(os.path.basename(args.config_file))[0]
        model_type = config_name
        # 验证推断的模型类型是否有效
        if model_type not in ModelFactory.list_models():
            # 如果推断的模型类型无效，则使用交互式选择
            try:
                from mol_evo.utils.train.model_utils import select_model_type_interactively
                model_type = select_model_type_interactively()
                if model_type is None:
                    print("未选择模型类型，退出训练")
                    return
            except ImportError as e:
                print(f"无法导入交互式选择工具: {e}")
                print("请提供 --model-type 参数")
                return
    
    # 验证模型类型是否有效
    if model_type not in ModelFactory.list_models():
        print(f"错误: 无效的模型类型 '{model_type}'")
        print(f"支持的模型类型: {', '.join(ModelFactory.list_models())}")
        return
    
    # 使用命令行参数设置目标属性
    global TARGET_PROPERTY
    if model_config and 'train' in model_config and 'target_property' in model_config['train']:
        TARGET_PROPERTY = model_config['train']['target_property']
    elif model_config and 'target_property' in model_config:
        TARGET_PROPERTY = model_config['target_property']
    else:
        TARGET_PROPERTY = args.target_property
    
    try:
        train_model(
            args.data_file, args.max_pairs, args.epochs, args.seed, args.batch_size, args.learning_rate,
            model_type, model_config
        )
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()