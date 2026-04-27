#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0.3 长链路路径训练脚本

路径级主监督训练，步骤级损失默认关闭。
支持课程学习（short-to-long）、长度分桶、弱幅度正则、辅助合法性损失。
输入为路径 JSON 文件，每条样本包含完整演化路径。
兼容非法中间节点容错、可变长度路径 padding。

使用方式:
    python mol_evo/train_v0_3_path.py \
        -d mol_evo/dataset/data/paths.json \
        -p lumo_change \
        -e 200 \
        --step-loss-weight 0.0
"""

import sys
import os
import argparse
import torch
import numpy as np
import math
from datetime import datetime
from tqdm import tqdm
import shutil
import random
import json

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 设置项目根目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

try:
    from mol_evo.core.models.v0 import ModelFactory
    from mol_evo.core.data.path_processing import build_molecule_path_dataset_v0_3
    from mol_evo.core.data.path_data import MoleculePathDataset, path_collate
    from mol_evo.core.data import load_operation_config
    from mol_evo.utils.training_utils import split_data_indices, plot_training_trends
    from mol_evo.utils.logger_utils import DualLogger
    from mol_evo.utils.training_metrics import TrainingMetricsRecorder
    from mol_evo.utils.config_utils import load_config_by_model_type
except ImportError as e:
    import traceback
    print(f"无法导入所需的模块 train_v0_3_path.py: {e}")
    traceback.print_exc()
    exit(1)


# 默认目标属性
TARGET_PROPERTY = 'lumo_change'


def move_path_batch_to_device(batch_dict: dict, device: torch.device) -> dict:
    """
    将 path_collate 返回的 batch dict 移到指定设备。
    """
    batch_dict['edge_features'] = batch_dict['edge_features'].to(device)
    batch_dict['node_valid_mask'] = batch_dict['node_valid_mask'].to(device)
    batch_dict['step_valid_mask'] = batch_dict['step_valid_mask'].to(device)
    batch_dict['path_padding_mask'] = batch_dict['path_padding_mask'].to(device)
    batch_dict['path_targets'] = batch_dict['path_targets'].to(device)
    
    if batch_dict['step_targets'] is not None:
        batch_dict['step_targets'] = batch_dict['step_targets'].to(device)
    
    # node_batch_list 中的 Batch 对象在模型前向时 lazy 移动
    # 这里不预先移动，避免大量小张量搬运
    
    return batch_dict


def compute_path_loss(
    output: dict,
    batch_dict: dict,
    criterion: nn.Module,
    step_loss_weight: float = 0.0,
    validity_loss_weight: float = 0.0,
    magnitude_reg_weight: float = 0.0,
) -> tuple:
    """
    计算路径级主损失，可选步骤辅助损失、合法性辅助损失和弱幅度正则。
    
    Args:
        output: 模型输出 dict {
            'path_pred': (B,1),
            'step_pred': (B,S,1) or None,
            'validity_pred': (B,S,1) or None,
        }
        batch_dict: path_collate 返回的 batch dict
        criterion: 损失函数 (L1Loss)
        step_loss_weight: 步骤辅助损失权重（默认 0，即仅路径主监督）
        validity_loss_weight: 合法性辅助损失权重
        magnitude_reg_weight: 弱幅度 L2 正则权重
        
    Returns:
        (total_loss, loss_details_dict)
        loss_details_dict 包含各分项的 float 值，供日志使用
    """
    path_pred = output['path_pred']      # (B, output_dim)
    path_targets = batch_dict['path_targets']  # (B, 1)
    
    # 路径级主损失
    path_loss = criterion(path_pred, path_targets)
    total_loss = path_loss
    
    loss_details = {
        'path_loss': path_loss.item(),
        'step_loss': 0.0,
        'validity_loss': 0.0,
        'magnitude_reg': 0.0,
    }
    
    # ====== 步骤级辅助损失（可选，默认关闭）======
    step_pred = output.get('step_pred')
    if (step_loss_weight > 0
            and step_pred is not None
            and batch_dict.get('has_step_targets')
            and batch_dict['step_targets'] is not None):
        step_targets = batch_dict['step_targets']  # (B, S)
        step_valid_mask = batch_dict['step_valid_mask']  # (B, S)
        
        if step_pred.dim() == 3 and step_pred.size(-1) == 1:
            step_pred_flat = step_pred.squeeze(-1)
        else:
            step_pred_flat = step_pred
        
        if step_valid_mask.any():
            valid_pred = step_pred_flat[step_valid_mask]
            valid_target = step_targets[step_valid_mask]
            step_loss = criterion(valid_pred, valid_target)
            loss_details['step_loss'] = step_loss.item()
            total_loss = total_loss + step_loss_weight * step_loss
    
    # ====== 合法性辅助损失（可选）======
    validity_pred = output.get('validity_pred')
    if (validity_loss_weight > 0
            and validity_pred is not None
            and 'step_valid_mask' in batch_dict):
        step_valid_mask = batch_dict['step_valid_mask']  # (B, S) bool — True 表示该步合法
        path_padding_mask = batch_dict['path_padding_mask']  # (B, S) bool — True 表示有效位置
        
        if validity_pred.dim() == 3 and validity_pred.size(-1) == 1:
            validity_pred_flat = validity_pred.squeeze(-1)  # (B, S)
        else:
            validity_pred_flat = validity_pred
        
        # 仅在非 padding 位置计算合法性损失
        if path_padding_mask.any():
            valid_positions = path_padding_mask
            pred_at_valid = validity_pred_flat[valid_positions]  # logits
            target_at_valid = step_valid_mask[valid_positions].float()  # 0/1
            validity_loss = F.binary_cross_entropy_with_logits(pred_at_valid, target_at_valid)
            loss_details['validity_loss'] = validity_loss.item()
            total_loss = total_loss + validity_loss_weight * validity_loss
    
    # ====== 弱幅度 L2 正则（可选）======
    if magnitude_reg_weight > 0 and step_pred is not None:
        if step_pred.dim() == 3 and step_pred.size(-1) == 1:
            step_vals = step_pred.squeeze(-1)  # (B, S)
        else:
            step_vals = step_pred
        
        path_padding_mask = batch_dict['path_padding_mask']
        mask_float = path_padding_mask.float()
        # L2 正则: mean of squared step predictions at valid positions
        mag_reg = (step_vals ** 2 * mask_float).sum() / mask_float.sum().clamp(min=1.0)
        loss_details['magnitude_reg'] = mag_reg.item()
        total_loss = total_loss + magnitude_reg_weight * mag_reg
    
    return total_loss, loss_details


def train_path_model(
    data_file: str,
    max_paths: int = None,
    epochs: int = 200,
    seed: int = 42,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    model_type: str = "visnet_path_v0_3",
    model_config: dict = None,
    step_loss_weight: float = 0.0,
    validity_loss_weight: float = 0.0,
    magnitude_reg_weight: float = 0.0,
    max_path_length: int = 20,
    keep_invalid_middle: bool = True,
    enable_curriculum: bool = False,
    curriculum_schedule: list = None,
    enable_length_bucket: bool = False,
):
    """
    v0.3 路径训练主函数。
    """
    global TARGET_PROPERTY
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 初始化模型配置
    final_model_config = {}
    if model_config:
        train_config = model_config.get('train', {})
        data_file = train_config.get('data_file', data_file)
        epochs = train_config.get('epochs', epochs)
        seed = train_config.get('seed', seed)
        batch_size = train_config.get('batch_size', batch_size)
        learning_rate = train_config.get('learning_rate', learning_rate)
        max_paths = train_config.get('max_paths', max_paths)
        step_loss_weight = train_config.get('step_loss_weight', step_loss_weight)
        validity_loss_weight = train_config.get('validity_loss_weight', validity_loss_weight)
        magnitude_reg_weight = train_config.get('magnitude_reg_weight', magnitude_reg_weight)
        max_path_length = train_config.get('max_path_length', max_path_length)
        keep_invalid_middle = train_config.get('keep_invalid_middle', keep_invalid_middle)
        enable_curriculum = train_config.get('enable_curriculum', enable_curriculum)
        curriculum_schedule = train_config.get('curriculum_schedule', curriculum_schedule)
        enable_length_bucket = train_config.get('enable_length_bucket', enable_length_bucket)
        final_model_config = model_config.get('model', {}).copy()
    else:
        try:
            final_model_config = load_config_by_model_type(model_type)
        except Exception:
            final_model_config = {}
    
    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"train-{timestamp}-{TARGET_PROPERTY}-path-{max_paths}-{epochs}"
    model_dir = os.path.join(project_root, 'mol_evo', 'output', 'v0_3', model_type, dir_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # 日志
    logger = DualLogger(model_dir)
    logger.info(f"v0.3 长链路路径训练开始")
    logger.info(f"数据文件: {data_file}")
    logger.info(f"目标属性: {TARGET_PROPERTY}")
    logger.info(f"模型类型: {model_type}")
    logger.info(f"步骤损失权重: {step_loss_weight}")
    logger.info(f"合法性损失权重: {validity_loss_weight}")
    logger.info(f"幅度正则权重: {magnitude_reg_weight}")
    logger.info(f"最大路径长度: {max_path_length}")
    logger.info(f"保留非法中间节点: {keep_invalid_middle}")
    logger.info(f"课程学习: {enable_curriculum}")
    if enable_curriculum and curriculum_schedule:
        logger.info(f"课程学习计划: {curriculum_schedule}")
    logger.info(f"长度分桶: {enable_length_bucket}")
    
    # 早停参数
    patience_limit = 50
    
    try:
        # ====== 数据构建 ======
        logger.info("正在构建路径数据集...")
        processed_paths, property_stats, build_stats = build_molecule_path_dataset_v0_3(
            data_file=data_file,
            max_paths=max_paths,
            target_property=TARGET_PROPERTY,
            logger=logger,
            max_path_length=max_path_length,
            keep_invalid_middle=keep_invalid_middle,
        )
        
        logger.info(f"数据构建统计: {json.dumps(build_stats, indent=2, ensure_ascii=False)}")
        
        if len(processed_paths) == 0:
            logger.error("没有有效的路径数据，请检查数据预处理")
            return
        
        # ====== 数据划分 ======
        train_idx, val_idx, test_idx = split_data_indices(len(processed_paths), 0.8, 0.1, 0.1, seed)
        logger.info(f"数据划分: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        
        train_paths = [processed_paths[i] for i in train_idx]
        val_paths = [processed_paths[i] for i in val_idx]
        test_paths = [processed_paths[i] for i in test_idx]
        
        train_dataset = MoleculePathDataset(train_paths)
        val_dataset = MoleculePathDataset(val_paths) if val_paths else None
        test_dataset = MoleculePathDataset(test_paths) if test_paths else None
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=path_collate, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=path_collate
        ) if val_dataset else None
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=path_collate
        ) if test_dataset else None
        
        # ====== 创建模型 ======
        # 动态确定边特征维度
        sample_edge_dim = len(processed_paths[0]['edge_features'][0])
        if sample_edge_dim != final_model_config.get('edge_feature_dim', 15):
            logger.info(f"更新边特征维度: {final_model_config.get('edge_feature_dim', 15)} -> {sample_edge_dim}")
            final_model_config['edge_feature_dim'] = sample_edge_dim
        
        # 设置步骤头开关（默认关闭，除非数据有步骤标签且用户配置了 step_loss_weight > 0）
        has_any_step_targets = any(p['step_targets'] is not None for p in processed_paths)
        if step_loss_weight > 0 and has_any_step_targets:
            final_model_config.setdefault('enable_step_head', True)
        else:
            final_model_config.setdefault('enable_step_head', False)
        
        # 设置合法性头开关
        if validity_loss_weight > 0:
            final_model_config.setdefault('enable_validity_head', True)
        else:
            final_model_config.setdefault('enable_validity_head', False)
        
        model = ModelFactory.create(model_type, **final_model_config)
        logger.info(f"模型创建成功: {model.__class__.__name__}")
        logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")
        model = model.to(device)
        
        # ====== 优化器 & 损失 ======
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        min_lr = float(model_config.get('train', {}).get('min_lr', 1e-6)) if model_config else 1e-6
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.3, min_lr=min_lr
        )
        criterion = nn.L1Loss()
        
        # 指标记录
        metrics_recorder = TrainingMetricsRecorder()
        metrics_recorder.set_training_params({
            "model_type": model_type,
            "data_file": data_file,
            "max_paths": max_paths,
            "epochs": epochs,
            "seed": seed,
            "target_property": TARGET_PROPERTY,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "step_loss_weight": step_loss_weight,
            "validity_loss_weight": validity_loss_weight,
            "magnitude_reg_weight": magnitude_reg_weight,
            "max_path_length": max_path_length,
            "keep_invalid_middle": keep_invalid_middle,
            "enable_curriculum": enable_curriculum,
            "curriculum_schedule": curriculum_schedule,
            "enable_length_bucket": enable_length_bucket,
        })
        metrics_recorder.set_property_stats(property_stats)
        
        # 保存模型配置
        config_save = {
            "timestamp": datetime.now().isoformat(),
            "model_params": final_model_config,
            "train_config": {
                "model_type": model_type,
                "data_file": data_file,
                "target_property": TARGET_PROPERTY,
                "max_paths": max_paths,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "step_loss_weight": step_loss_weight,
                "validity_loss_weight": validity_loss_weight,
                "magnitude_reg_weight": magnitude_reg_weight,
                "max_path_length": max_path_length,
                "keep_invalid_middle": keep_invalid_middle,
                "enable_curriculum": enable_curriculum,
                "curriculum_schedule": curriculum_schedule,
                "enable_length_bucket": enable_length_bucket,
            },
            "build_stats": build_stats,
            "property_stats": {k: list(v) for k, v in property_stats.items()},
        }
        with open(os.path.join(model_dir, "model_config.json"), 'w', encoding='utf-8') as f:
            json.dump(config_save, f, ensure_ascii=False, indent=2)
        
        # ====== 课程学习计划 ======
        # 默认 schedule: [(0, 2), (epoch//4, 5), (epoch//2, 10), (epoch*3//4, max_path_length)]
        if enable_curriculum:
            if curriculum_schedule is None:
                curriculum_schedule = [
                    [0, 2],
                    [epochs // 4, 5],
                    [epochs // 2, 10],
                    [epochs * 3 // 4, max_path_length],
                ]
            logger.info(f"课程学习计划（自动生成）: {curriculum_schedule}")
        
        # ====== 路径长度分布日志 ======
        length_dist = {}
        for p in train_paths:
            l = p['num_steps']
            length_dist[l] = length_dist.get(l, 0) + 1
        logger.info(f"训练集路径长度分布: {dict(sorted(length_dist.items()))}")
        
        # 非法步骤比例
        total_steps_train = sum(p['num_steps'] for p in train_paths)
        valid_steps_train = sum(sum(p['step_valid_mask']) for p in train_paths)
        invalid_ratio = 1.0 - valid_steps_train / max(total_steps_train, 1)
        logger.info(f"训练集非法步骤比例: {invalid_ratio:.4f}")
        
        # ====== 训练循环 ======
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        should_stop = False
        
        logger.info(f"开始训练 ({epochs} 轮)...")
        
        model.train()
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            # ---- 课程学习：动态决定当前 epoch 的最大允许路径长度 ----
            curr_max_len = max_path_length
            if enable_curriculum and curriculum_schedule:
                for stage_epoch, stage_max_len in curriculum_schedule:
                    if epoch >= stage_epoch:
                        curr_max_len = stage_max_len
            
            # 如果课程学习限制了长度，需要过滤训练数据并重建 loader
            if enable_curriculum and curr_max_len < max_path_length:
                filtered_train = [p for p in train_paths if p['num_steps'] <= curr_max_len]
                if len(filtered_train) == 0:
                    filtered_train = train_paths  # fallback
                curr_train_dataset = MoleculePathDataset(filtered_train)
                curr_train_loader = DataLoader(
                    curr_train_dataset, batch_size=batch_size, shuffle=True,
                    num_workers=0, collate_fn=path_collate, pin_memory=True
                )
            else:
                curr_train_loader = train_loader
            
            epoch_total_loss = 0.0
            epoch_path_loss = 0.0
            epoch_step_loss = 0.0
            epoch_validity_loss = 0.0
            epoch_mag_reg = 0.0
            epoch_samples = 0
            
            for batch_dict in curr_train_loader:
                optimizer.zero_grad()
                
                batch_dict = move_path_batch_to_device(batch_dict, device)
                
                # 前向传播
                output = model(batch_dict)
                
                # 计算损失
                total_loss, loss_details = compute_path_loss(
                    output, batch_dict, criterion,
                    step_loss_weight=step_loss_weight,
                    validity_loss_weight=validity_loss_weight,
                    magnitude_reg_weight=magnitude_reg_weight,
                )
                
                if math.isnan(total_loss.item()) or math.isinf(total_loss.item()):
                    logger.warning(f"Epoch {epoch}: 检测到 NaN/Inf 损失，跳过该 batch")
                    should_stop = True
                    break
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                bs = batch_dict['batch_size']
                epoch_total_loss += total_loss.item() * bs
                epoch_path_loss += loss_details['path_loss'] * bs
                epoch_step_loss += loss_details['step_loss'] * bs
                epoch_validity_loss += loss_details['validity_loss'] * bs
                epoch_mag_reg += loss_details['magnitude_reg'] * bs
                epoch_samples += bs
            
            if should_stop:
                break
            
            epoch_total_loss /= max(epoch_samples, 1)
            epoch_path_loss /= max(epoch_samples, 1)
            epoch_step_loss /= max(epoch_samples, 1)
            epoch_validity_loss /= max(epoch_samples, 1)
            epoch_mag_reg /= max(epoch_samples, 1)
            metrics_recorder.record_train_loss(epoch_total_loss)
            
            # ====== 验证 ======
            val_loss = None
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    val_total = 0.0
                    val_samples = 0
                    
                    for batch_dict in val_loader:
                        batch_dict = move_path_batch_to_device(batch_dict, device)
                        output = model(batch_dict)
                        total_loss_val, _ = compute_path_loss(
                            output, batch_dict, criterion,
                            step_loss_weight=step_loss_weight,
                            validity_loss_weight=validity_loss_weight,
                            magnitude_reg_weight=magnitude_reg_weight,
                        )
                        bs = batch_dict['batch_size']
                        val_total += total_loss_val.item() * bs
                        val_samples += bs
                    
                    val_loss = val_total / max(val_samples, 1)
                    metrics_recorder.record_val_loss(val_loss)
                    
                    if math.isnan(val_loss) or math.isinf(val_loss):
                        logger.warning(f"Epoch {epoch}: NaN/Inf 验证损失")
                        should_stop = True
                        break
                    
                    scheduler.step(val_loss)
                model.train()
            
            # 早停
            if val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience_limit:
                    logger.info(f"早停触发 (epoch {epoch+1})")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
            
            # 日志
            if (epoch + 1) % 10 == 0:
                msg = (f"Epoch {epoch+1}/{epochs} | Total: {epoch_total_loss:.6f} "
                       f"| Path: {epoch_path_loss:.6f}")
                if step_loss_weight > 0:
                    msg += f" | Step: {epoch_step_loss:.6f}"
                if validity_loss_weight > 0:
                    msg += f" | Valid: {epoch_validity_loss:.6f}"
                if magnitude_reg_weight > 0:
                    msg += f" | MagReg: {epoch_mag_reg:.6f}"
                if val_loss is not None:
                    msg += f" | Val: {val_loss:.6f}"
                if enable_curriculum:
                    msg += f" | CurrMaxLen: {curr_max_len}"
                logger.info(msg)
        
        if should_stop:
            logger.info("训练因异常值而提前停止")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
        
        # ====== 测试 ======
        if test_loader is not None:
            model.eval()
            with torch.no_grad():
                all_path_preds = []
                all_path_targets = []
                
                for batch_dict in test_loader:
                    batch_dict = move_path_batch_to_device(batch_dict, device)
                    output = model(batch_dict)
                    all_path_preds.append(output['path_pred'].cpu())
                    all_path_targets.append(batch_dict['path_targets'].cpu())
                
                preds = torch.cat(all_path_preds, dim=0)
                targets = torch.cat(all_path_targets, dim=0)
                
                test_loss = criterion(preds, targets).item()
                mse = torch.mean((preds - targets) ** 2).item()
                rmse = math.sqrt(mse)
                mae = torch.mean(torch.abs(preds - targets)).item()
                
                ss_res = torch.sum((targets - preds) ** 2).item()
                ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
                r2 = 1 - ss_res / (ss_tot + 1e-8) if ss_tot > 0 else 0.0
                
                # PCC
                pm = preds.mean()
                tm = targets.mean()
                num = torch.sum((preds - pm) * (targets - tm)).item()
                den = math.sqrt(torch.sum((preds - pm)**2).item() * torch.sum((targets - tm)**2).item())
                pcc = num / den if den > 0 else 0.0
                
                logger.info(f"测试结果: Loss={test_loss:.6f} RMSE={rmse:.6f} MAE={mae:.6f} R2={r2:.6f} PCC={pcc:.6f}")
        
        # 保存模型
        model_path = os.path.join(model_dir, "last.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"模型已保存到: {model_path}")
        
        # 保存训练过程
        training_process = {
            "timestamp": datetime.now().isoformat(),
            "test_metrics": {
                "test_loss": test_loss if test_loader else None,
                "rmse": rmse if test_loader else None,
                "mae": mae if test_loader else None,
                "r2": r2 if test_loader else None,
                "pcc": pcc if test_loader else None,
            },
            "losses": {
                "train_losses": metrics_recorder.train_losses,
                "val_losses": metrics_recorder.val_losses,
            },
            "property_stats": {k: list(v) for k, v in property_stats.items()},
            "build_stats": build_stats,
        }
        with open(os.path.join(model_dir, "training_process.json"), 'w', encoding='utf-8') as f:
            json.dump(training_process, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练完成，模型目录: {model_dir}")
        
    except KeyboardInterrupt:
        logger.info("\n训练被用户中断 (Ctrl+C)")
        choice = input(f"是否要清理模型目录 {model_dir}? (Y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            shutil.rmtree(model_dir)
            logger.info("模型目录已清理")
        else:
            logger.info("保留模型目录")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='v0.3 长链路路径训练脚本')
    
    parser.add_argument('-d', '--data-file', type=str, required=True,
                       help='路径 JSON 数据文件')
    parser.add_argument('-m', '--max-paths', type=int, default=None,
                       help='最大路径数（调试用）')
    parser.add_argument('-e', '--epochs', type=int, default=200,
                       help='训练轮数')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                       help='批处理大小')
    parser.add_argument('-p', '--target-property', type=str, default='lumo_change',
                       help='目标属性')
    parser.add_argument('-s', '--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('-mt', '--model-type', type=str, default='visnet_path_v0_3',
                       help='模型类型')
    parser.add_argument('-c', '--config-file', type=str, default=None,
                       help='YAML 配置文件路径')
    parser.add_argument('--step-loss-weight', type=float, default=0.0,
                       help='步骤辅助损失权重（默认 0，即仅路径主监督）')
    parser.add_argument('--validity-loss-weight', type=float, default=0.0,
                       help='合法性辅助损失权重')
    parser.add_argument('--magnitude-reg-weight', type=float, default=0.0,
                       help='弱幅度 L2 正则权重')
    parser.add_argument('--max-path-length', type=int, default=20,
                       help='最大路径步数')
    parser.add_argument('--keep-invalid-middle', action='store_true', default=True,
                       help='保留含非法中间节点的路径')
    parser.add_argument('--drop-invalid-middle', action='store_true', default=False,
                       help='丢弃含非法中间节点的路径')
    parser.add_argument('--enable-curriculum', action='store_true', default=False,
                       help='启用课程学习（short-to-long）')
    parser.add_argument('--enable-length-bucket', action='store_true', default=False,
                       help='启用按路径长度分桶')
    
    args = parser.parse_args()
    
    # 配置文件
    model_config = {}
    if args.config_file:
        import yaml
        with open(args.config_file, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)
        print(f"已加载配置文件: {args.config_file}")
    
    # 设置目标属性
    global TARGET_PROPERTY
    if model_config and 'train' in model_config and 'target_property' in model_config['train']:
        TARGET_PROPERTY = model_config['train']['target_property']
    else:
        TARGET_PROPERTY = args.target_property
    
    keep_invalid = args.keep_invalid_middle and not args.drop_invalid_middle
    
    train_path_model(
        data_file=args.data_file,
        max_paths=args.max_paths,
        epochs=args.epochs,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_type=args.model_type,
        model_config=model_config if model_config else None,
        step_loss_weight=args.step_loss_weight,
        validity_loss_weight=args.validity_loss_weight,
        magnitude_reg_weight=args.magnitude_reg_weight,
        max_path_length=args.max_path_length,
        keep_invalid_middle=keep_invalid,
        enable_curriculum=args.enable_curriculum,
        enable_length_bucket=args.enable_length_bucket,
    )


if __name__ == "__main__":
    main()
