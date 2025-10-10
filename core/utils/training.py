#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练工具函数
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
from torch_geometric.data import Data


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
    # 按维度计算
    ss_res = torch.sum((targets - predictions) ** 2, dim=0)
    ss_tot = torch.sum((targets - torch.mean(targets, dim=0)) ** 2, dim=0)
    
    # 添加数值稳定性检查
    r2 = torch.ones_like(ss_res)  # 默认为1
    non_zero_mask = ss_tot != 0
    r2[non_zero_mask] = 1 - ss_res[non_zero_mask] / (ss_tot[non_zero_mask] + 1e-8)
    
    # 如果只有一个维度，则取标量值；否则取平均值
    if r2.numel() == 1:
        r2 = r2.item()
    else:
        r2 = r2.mean().item()
    
    # 阈值准确率 (Threshold Accuracy)
    thresholds = [0.1, 0.05]
    threshold_accs = {}
    
    for th in thresholds:
        # 计算所有维度同时满足阈值的样本比例
        all_dims_correct = (torch.abs(predictions - targets) < th).all(dim=1).float()
        threshold_accs[f'all_{th}'] = all_dims_correct.mean().item()
        
        # 计算整体平均准确率（每个维度单独计算然后平均）
        dim_correct = (torch.abs(predictions - targets) < th).float()
        threshold_accs[f'mean_{th}'] = dim_correct.mean().item()
    
    return {
        'mse': mse.item(),
        'rmse': rmse.item(),
        'mae': mae.item(),
        'r2': r2,
        'threshold_accs': threshold_accs
    }


def train_gnn_model(model: nn.Module, data: Data, target_features: torch.Tensor,
                   epochs: int = 100, lr: float = 0.001, 
                   train_idx: List[int] = None, val_idx: List[int] = None, logger=None) -> Tuple[List[float], List[float]]:
    """
    训练GNN模型
    
    Args:
        model: GNN模型
        data: 图数据
        target_features: 目标特征
        epochs: 训练轮数
        lr: 学习率
        train_idx: 训练集索引
        val_idx: 验证集索引
        logger: 日志记录器
        
    Returns:
        训练损失和验证损失列表
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 将模型和数据移到设备上
    model = model.to(device)
    data = data.to(device)
    target_features = target_features.to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # 添加权重衰减作为正则化
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss(reduction='mean')  # 明确指定reduction方式
    
    # 初始化损失记录
    train_losses = []
    val_losses = []
    val_r2s = []
    val_maes = []
    
    # 早停机制参数
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 50  # 早停耐心次数
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向传播
        predictions = model(data)
        
        # 确保预测值和目标值维度匹配
        if predictions.shape != target_features.shape:
            raise ValueError(f"预测值维度 {predictions.shape} 与目标值维度 {target_features.shape} 不匹配")
        
        # 计算训练损失
        train_loss = criterion(predictions[train_idx], target_features[train_idx])
        
        # 检查是否有NaN或inf值
        if torch.isnan(train_loss) or torch.isinf(train_loss):
            print(f"警告: 在第 {epoch+1} 轮检测到NaN或inf损失值，停止训练")
            break
        
        # 反向传播
        train_loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 记录训练损失
        train_losses.append(train_loss.item())
        
        # 验证阶段
        if val_idx and len(val_idx) > 0:
            model.eval()
            with torch.no_grad():
                val_predictions = model(data)
                val_loss = criterion(val_predictions[val_idx], target_features[val_idx])
                val_losses.append(val_loss.item())
                
                # 检查验证损失是否有NaN或inf
                if torch.isnan(val_loss) or torch.isinf(val_loss):
                    print(f"警告: 在第 {epoch+1} 轮验证时检测到NaN或inf损失值")
                
                # 更新学习率调度器
                scheduler.step(val_loss)
                
                # 计算验证集的额外评估指标
                if (epoch + 1) % 10 == 0:
                    val_metrics = calculate_metrics(val_predictions[val_idx], target_features[val_idx])
                    val_r2s.append(val_metrics['r2'])
                    val_maes.append(val_metrics['mae'])
            model.train()
        else:
            val_loss = None
            val_metrics = None
        
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
                if logger:
                    logger.info(message)
                else:
                    print(message)
                # 恢复最佳模型状态
                model.load_state_dict(best_model_state)
                break
        
        # 每10个epoch输出一次信息
        if (epoch + 1) % 10 == 0:
            message = f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.6f}"
            if val_loss is not None:
                message += f", Val Loss: {val_loss:.6f}"
                if val_metrics:
                    message += f", R²: {val_metrics['r2']:.4f}, MAE: {val_metrics['mae']:.4f}"
            
            if logger:
                logger.info(message)
            else:
                print(message)
    
    return train_losses, val_losses, val_r2s, val_maes