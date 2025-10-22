#!/usr/bin/env python3
"""
测试 TorchMD-Net 模型训练功能
参考 test_create_data.py 和 test_batch_data_loader.py 脚本
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'modules', 'torchmd-net'))

from torchmdnet.models.model import create_model
from torchmdnet.module import LNNP
from torch_geometric.data import Data, DataLoader as PyGDataLoader


def create_sample_dataset():
    """
    创建示例数据集用于训练测试
    """
    print("创建示例数据集...")
    
    data_list = []
    
    # 创建几个具有不同能量的水分子构象
    for i in range(10):
        # 随机扰动坐标
        pos = torch.tensor([
            [0.0 + 0.1*i, 0.0, 0.0],     # O
            [0.0, 0.0, 1.0 + 0.05*i],    # H
            [0.0, 1.0 - 0.05*i, 0.0],    # H
        ], dtype=torch.float32) + torch.randn(3, 3) * 0.1
        
        # 原子序数保持不变
        z = torch.tensor([8, 1, 1], dtype=torch.long)  # O, H, H
        
        # 能量值与构象相关
        y = torch.tensor([[-10.0 - 0.1*i]], dtype=torch.float64)
        
        data = Data(z=z, pos=pos, y=y)
        data_list.append(data)
    
    # 创建一些甲烷分子构象
    for i in range(5):
        # 随机扰动坐标
        pos = torch.tensor([
            [0.0 + 0.05*i, 0.0, 0.0],    # C
            [1.0, 0.0 + 0.02*i, 0.0],    # H
            [0.0, 1.0 - 0.03*i, 0.0],    # H
            [0.0, 0.0, 1.0 + 0.01*i],    # H
            [-1.0 + 0.02*i, 0.0, 0.0],   # H
        ], dtype=torch.float32) + torch.randn(5, 3) * 0.1
        
        z = torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)  # C, H, H, H, H
        
        # 能量值与构象相关
        y = torch.tensor([[-15.0 - 0.05*i]], dtype=torch.float64)
        
        data = Data(z=z, pos=pos, y=y)
        data_list.append(data)
    
    print(f"  成功创建包含 {len(data_list)} 个样本的数据集")
    return data_list


def test_model_initialization():
    """
    测试模型初始化
    """
    print("\n测试模型初始化...")
    
    model_args = {
        "embedding_dimension": 64,
        "num_layers": 2,
        "num_rbf": 32,
        "rbf_type": "expnorm",
        "trainable_rbf": False,
        "activation": "silu",
        "cutoff_lower": 0.0,
        "cutoff_upper": 5.0,
        "max_z": 100,
        "max_num_neighbors": 32,
        "model": "tensornet",
        "aggr": "add",
        "derivative": False,
        "atom_filter": -1,
        "prior_model": None,
        "output_model": "Scalar",
        "reduce_op": "add",
        "precision": 32,
    }
    
    try:
        # 创建模型
        model = create_model(model_args)
        print(f"  成功创建模型: {type(model).__name__}")
        
        # 创建 Lightning 模块
        lightning_model = LNNP(
            model,
            optimizer_cls=torch.optim.Adam,
            optimizer_args={"lr": 1e-3},
            scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_args={"mode": "min", "factor": 0.8, "patience": 10},
        )
        
        print(f"  成功创建 Lightning 模块")
        return lightning_model
    except Exception as e:
        print(f"  模型初始化失败: {e}")
        return None


def test_single_training_step():
    """
    测试单步训练
    """
    print("\n测试单步训练...")
    
    # 创建数据集和数据加载器
    dataset = create_sample_dataset()
    if not dataset:
        print("  跳过训练测试，因为数据集创建失败")
        return
    
    try:
        data_loader = PyGDataLoader(dataset, batch_size=4, shuffle=True)
        print(f"  成功创建数据加载器，批次大小: 4")
        
        # 创建模型
        model = test_model_initialization()
        if model is None:
            print("  跳过训练测试，因为模型初始化失败")
            return
        
        # 获取一个批次的数据
        batch = next(iter(data_loader))
        print(f"  获取训练批次，包含 {len(torch.unique(batch.batch))} 个分子")
        
        # 执行单步训练
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 前向传播
        pred, _ = model(batch.z, batch.pos, batch=batch.batch)
        
        # 计算损失
        loss_fn = nn.MSELoss()
        loss = loss_fn(pred, batch.y)
        
        print(f"  前向传播成功")
        print(f"    预测形状: {pred.shape}")
        print(f"    目标形状: {batch.y.shape}")
        print(f"    损失值: {loss.item():.6f}")
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  反向传播和优化步骤成功")
        return True
    except Exception as e:
        print(f"  单步训练测试失败: {e}")
        return False


def test_multiple_training_steps():
    """
    测试多步训练
    """
    print("\n测试多步训练...")
    
    # 创建数据集和数据加载器
    dataset = create_sample_dataset()
    if not dataset:
        print("  跳过训练测试，因为数据集创建失败")
        return
    
    try:
        data_loader = PyGDataLoader(dataset, batch_size=4, shuffle=True)
        print(f"  成功创建数据加载器")
        
        # 创建模型
        model = test_model_initialization()
        if model is None:
            print("  跳过训练测试，因为模型初始化失败")
            return
        
        # 设置优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss()
        
        # 执行多个训练步骤
        model.train()
        losses = []
        
        for epoch in range(3):
            epoch_losses = []
            for batch_idx, batch in enumerate(data_loader):
                # 前向传播
                pred, _ = model(batch.z, batch.pos, batch=batch.batch)
                
                # 计算损失
                loss = loss_fn(pred, batch.y)
                epoch_losses.append(loss.item())
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f"    Epoch {epoch+1}, 平均损失: {avg_loss:.6f}")
        
        print(f"  多步训练成功完成")
        print(f"    初始损失: {losses[0]:.6f}")
        print(f"    最终损失: {losses[-1]:.6f}")
        print(f"    损失下降: {losses[0] - losses[-1]:.6f}")
        
        return True
    except Exception as e:
        print(f"  多步训练测试失败: {e}")
        return False


def test_validation_step():
    """
    测试验证步骤
    """
    print("\n测试验证步骤...")
    
    # 创建数据集和数据加载器
    dataset = create_sample_dataset()
    if not dataset:
        print("  跳过验证测试，因为数据集创建失败")
        return
    
    try:
        # 分割训练和验证集
        train_dataset = dataset[:12]  # 12个样本用于训练
        val_dataset = dataset[12:]    # 剩余样本用于验证
        
        train_loader = PyGDataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = PyGDataLoader(val_dataset, batch_size=4, shuffle=False)
        
        print(f"  数据集分割: {len(train_dataset)} 训练, {len(val_dataset)} 验证")
        
        # 创建模型
        model = test_model_initialization()
        if model is None:
            print("  跳过验证测试，因为模型初始化失败")
            return
        
        # 设置优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss()
        
        # 训练几步
        model.train()
        for batch in train_loader:
            pred, _ = model(batch.z, batch.pos, batch=batch.batch)
            loss = loss_fn(pred, batch.y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 验证步骤
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                pred, _ = model(batch.z, batch.pos, batch=batch.batch)
                loss = loss_fn(pred, batch.y)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        print(f"  验证步骤成功")
        print(f"    验证损失: {avg_val_loss:.6f}")
        
        return True
    except Exception as e:
        print(f"  验证步骤测试失败: {e}")
        return False


def test_model_saving_loading():
    """
    测试模型保存和加载
    """
    print("\n测试模型保存和加载...")
    
    try:
        # 创建模型
        model = test_model_initialization()
        if model is None:
            print("  跳过保存加载测试，因为模型初始化失败")
            return
        
        # 保存模型
        save_path = "test_model.ckpt"
        torch.save(model.state_dict(), save_path)
        print(f"  模型保存成功: {save_path}")
        
        # 加载模型
        loaded_model = test_model_initialization()
        if loaded_model is not None:
            loaded_model.load_state_dict(torch.load(save_path))
            print(f"  模型加载成功")
        
        # 清理文件
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"  临时文件清理成功")
        
        return True
    except Exception as e:
        print(f"  模型保存加载测试失败: {e}")
        return False


if __name__ == "__main__":
    print("开始测试 TorchMD-Net 模型训练功能")
    print("=" * 50)
    
    create_sample_dataset()
    test_model_initialization()
    test_single_training_step()
    test_multiple_training_steps()
    test_validation_step()
    test_model_saving_loading()
    
    print("\n" + "=" * 50)
    print("TorchMD-Net 模型训练测试完成")