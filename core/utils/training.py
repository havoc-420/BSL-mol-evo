#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练工具
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from typing import List, Dict, Optional, Tuple


def train_model_enhanced(model: nn.Module, data: Data, target_changes: torch.Tensor,
                        train_idx: List[int], val_idx: List[int], test_idx: List[int] = None,
                        epochs: int = 200, lr: float = 0.001, weight_decay: float = 1e-5,
                        patience: int = 20, min_delta: float = 1e-4) -> Dict[str, List[float]]:
    """
    增强的训练函数

    Args:
        model: 模型
        data: 图数据
        target_changes: 目标属性变化
        train_idx: 训练集索引
        val_idx: 验证集索引
        test_idx: 测试集索引
        epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        patience: 早停耐心值
        min_delta: 最小改进阈值

    Returns:
        训练历史
    """
    # 优化器（带权重衰减）
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    # 损失函数（Huber损失，对异常值更鲁棒）
    criterion = nn.HuberLoss()

    train_idx_tensor = torch.LongTensor(train_idx)
    val_idx_tensor = torch.LongTensor(val_idx)
    test_idx_tensor = torch.LongTensor(test_idx) if test_idx is not None else None

    model.train()
    train_losses = []
    val_losses = []
    test_losses = [] if test_idx is not None else None
    learning_rates = []

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 前向传播
        predictions = model(data)

        # 对于当前模型结构，我们重复预测结果以匹配目标数量
        if predictions.shape[0] == 1:
            predictions = predictions.repeat(target_changes.shape[0], 1)

        # 计算训练损失
        train_loss = criterion(predictions[train_idx_tensor], target_changes[train_idx_tensor])

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 反向传播
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.item())

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_predictions = model(data)
            if val_predictions.shape[0] == 1:
                val_predictions = val_predictions.repeat(target_changes.shape[0], 1)

            val_loss = criterion(val_predictions[val_idx_tensor], target_changes[val_idx_tensor])
            val_losses.append(val_loss.item())

            # 测试阶段（如果提供）
            if test_idx is not None:
                test_predictions = model(data)
                if test_predictions.shape[0] == 1:
                    test_predictions = test_predictions.repeat(target_changes.shape[0], 1)
                test_loss = criterion(test_predictions[test_idx_tensor], target_changes[test_idx_tensor])
                test_losses.append(test_loss.item())

        model.train()

        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # 早停机制
        if val_loss.item() < best_val_loss - min_delta:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停: 验证损失在 {patience} 轮内未改善")
            break

        # 打印进度
        if (epoch + 1) % 10 == 0:
            test_info = f", Test Loss: {test_loss.item():.6f}" if test_idx is not None else ""
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.6f}, '
                  f'Val Loss: {val_loss.item():.6f}{test_info}, LR: {current_lr:.2e}')

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 返回训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates
    }

    if test_losses is not None:
        history['test_losses'] = test_losses

    return history


def train_transformer_model(model: nn.Module, source_features: torch.Tensor, 
                           edge_features: torch.Tensor, target_features: torch.Tensor,
                           epochs: int = 100, lr: float = 0.001, train_idx=None, val_idx=None):
    """
    训练转换器模型
    
    Args:
        model: 转换器模型
        source_features: 起始分子特征
        edge_features: 边特征
        target_features: 目标分子特征
        epochs: 训练轮数
        lr: 学习率
        train_idx: 训练集索引
        val_idx: 验证集索引
        
    Returns:
        损失历史
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 如果没有提供索引，则使用全部数据
    if train_idx is None:
        train_idx = list(range(source_features.shape[0]))
    
    train_idx_tensor = torch.LongTensor(train_idx)
    val_idx_tensor = torch.LongTensor(val_idx) if val_idx is not None else None
    
    model.train()
    train_losses = []
    val_losses = [] if val_idx is not None else None
    
    best_val_loss = float('inf') if val_idx is not None else None
    best_model_state = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向传播
        predictions = model(source_features[train_idx_tensor], edge_features[train_idx_tensor])
        
        # 计算训练损失
        train_loss = criterion(predictions, target_features[train_idx_tensor])
        
        # 反向传播
        train_loss.backward()
        optimizer.step()
        
        train_losses.append(train_loss.item())
        
        # 验证阶段
        if val_idx is not None:
            model.eval()
            with torch.no_grad():
                val_predictions = model(source_features[val_idx_tensor], edge_features[val_idx_tensor])
                val_loss = criterion(val_predictions, target_features[val_idx_tensor])
                val_losses.append(val_loss.item())
                
                # 保存最佳模型
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_model_state = model.state_dict().copy()
            
            model.train()
        
        if (epoch + 1) % 10 == 0:
            if val_idx is not None:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}')
            else:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss.item():.6f}')
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses