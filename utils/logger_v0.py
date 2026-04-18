#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0版本模型训练日志工具函数
"""
import torch
from typing import Optional

def log_training_start(logger, data_file, max_pairs, epochs):
    """
    记录训练开始信息
    
    Args:
        logger: V0TrainingLogger实例
        data_file: 数据文件路径
        max_pairs: 最大对数
        epochs: 训练轮数
    """
    logger.info("=" * 60)
    logger.info("开始训练 v0 版本分子进化预测器模型")
    logger.info("=" * 60)
    logger.info(f"数据文件: {data_file}")
    if max_pairs:
        logger.info(f"最大分子对数: {max_pairs}")
    logger.info(f"训练轮数: {epochs}")
    logger.info("")


def log_dataset_examples(logger, from_data_list, to_data_list, edge_attrs, target_features, num_examples=3):
    """
    输出训练集样本示例信息（头部数据）
    
    Args:
        logger: V0TrainingLogger实例
        from_data_list: 起始分子数据列表
        to_data_list: 目标分子数据列表
        edge_attrs: 边特征张量
        target_features: 目标属性特征张量
        num_examples: 示例数量
    """
    logger.info("--- 训练集样本示例 (头部数据) ---")
    
    # 检查数据是否为空
    if len(from_data_list) == 0:
        logger.info("  没有可用的训练样本")
        return
        
    num_examples = min(num_examples, len(from_data_list))
    
    # 表格头部
    logger.info("  {:<8} {:<12} {:<12} {:<12} {:<12} {:<15} {:<15}".format(
        "样本", "起始节点数", "起始边数", "目标节点数", "目标边数", "边特征", "目标属性值"))
    logger.info("  " + "-" * 80)
    
    # 表格内容
    for i in range(num_examples):
        from_item = from_data_list[i]
        to_item = to_data_list[i]

        # 跳过 None 数据
        if from_item is None or to_item is None:
            logger.warning(f"  样本 {i+1}: 数据为 None，跳过")
            continue

        # 检查数据格式（PyG Data对象还是字典）
        if hasattr(from_item, 'x') and from_item.x is not None:
            from_nodes = from_item.x.size(0)
            from_edges = from_item.edge_index.size(1) if hasattr(from_item, 'edge_index') and from_item.edge_index is not None else 0
        elif hasattr(from_item, 'z') and from_item.z is not None:
            # Visnet 等模型使用 z (原子序数) 和 pos
            from_nodes = from_item.z.size(0)
            from_edges = from_item.edge_index.size(1) if hasattr(from_item, 'edge_index') and from_item.edge_index is not None else 0
        elif isinstance(from_item, dict):
            from_nodes = from_item.get('x_atoms', torch.tensor([])).size(0)
            from_edges = from_item.get('edge_index', torch.tensor([[]])).size(1)
        else:
            from_nodes = '?'
            from_edges = '?'

        if hasattr(to_item, 'x') and to_item.x is not None:
            to_nodes = to_item.x.size(0)
            to_edges = to_item.edge_index.size(1) if hasattr(to_item, 'edge_index') and to_item.edge_index is not None else 0
        elif hasattr(to_item, 'z') and to_item.z is not None:
            to_nodes = to_item.z.size(0)
            to_edges = to_item.edge_index.size(1) if hasattr(to_item, 'edge_index') and to_item.edge_index is not None else 0
        elif isinstance(to_item, dict):
            to_nodes = to_item.get('x_atoms', torch.tensor([])).size(0)
            to_edges = to_item.get('edge_index', torch.tensor([[]])).size(1)
        else:
            to_nodes = '?'
            to_edges = '?'
            
        # 检查edge_attrs是否为空
        if edge_attrs.size(0) > 0:
            edge_attr = str(edge_attrs[i].tolist())
            # DEBUG 输出边特征维度信息
            if i == 0:  # 只输出第一个样本的详细信息
                logger.info(f"  DEBUG: 边特征维度 = {edge_attrs.size(1)}")
                logger.info(f"  DEBUG: 第一个样本的边特征值 = {edge_attr}")
                # 分析边特征组成部分
                try:
                    # 导入处理函数以获取实际的原子类型和操作类型数量
                    from mol_evo.core.data.processing import get_atom_types, get_operation_types
                    atom_types = get_atom_types()
                    op_types = get_operation_types()
                    logger.info(f"  DEBUG: 原子类型数量 = {len(atom_types)}, 操作类型数量 = {len(op_types)}")
                    
                    edge_feat = edge_attrs[i].tolist()
                    logger.info(f"  DEBUG: 边特征总长度 = {len(edge_feat)}")
                    # 根据实际的原子类型和操作类型数量分割特征
                    if len(edge_feat) >= len(atom_types) + len(op_types):
                        atom_onehot = edge_feat[:len(atom_types)]
                        op_onehot = edge_feat[len(atom_types):len(atom_types)+len(op_types)]
                        logger.info(f"  DEBUG: 原子类型one-hot编码 = {atom_onehot}")
                        logger.info(f"  DEBUG: 操作类型one-hot编码 = {op_onehot}")
                        logger.info(f"  DEBUG: 原子类型列表 = {atom_types}")
                        logger.info(f"  DEBUG: 操作类型列表 = {op_types}")
                except Exception as e:
                    logger.info(f"  DEBUG: 无法获取原子类型和操作类型信息: {e}")
        else:
            edge_attr = "[]"
            
        # 检查target_features是否为空
        if target_features.size(0) > 0:
            target_val = f"{target_features[i].item():.6f}"
        else:
            target_val = "0.0"
        
        logger.info("  {:<8} {:<12} {:<12} {:<12} {:<12} {:<15} {:<15}".format(
            i+1, from_nodes, from_edges, to_nodes, to_edges, edge_attr[:13], target_val))
        
    logger.info("  " + "-" * 80)


def log_data_construction_info(logger, from_data_list, model_params, edge_attrs, target_features):
    """
    记录数据构建完成信息
    
    Args:
        logger: V0TrainingLogger实例
        from_data_list: 起始分子数据列表
        model_params: 模型参数字典
        edge_attrs: 边特征张量
        target_features: 目标属性特征张量
    """
    logger.info("\n数据构建完成:")
    
    # 检查数据格式确定节点特征维度
    if len(from_data_list) > 0:
        if hasattr(from_data_list[0], 'x'):
            # PyG Data对象格式
            node_feature_dim = from_data_list[0].x.size(1)
        else:
            # 字典格式（如FragNet）
            node_feature_dim = from_data_list[0]['x_atoms'].size(1)
    else:
        node_feature_dim = model_params.get("node_feature_dim", 0)
    
    table_data = [
        ["项目", "值"],
        ["-" * 40, "-" * 20],
        ["样本数", str(len(from_data_list))],
        ["节点特征维度", str(node_feature_dim)],
        ["边特征维度", str(edge_attrs.size(1) if edge_attrs.numel() > 0 else 0)],
        ["目标属性变化维度", str(target_features.size(1) if target_features.numel() > 0 else 0)]
    ]
    
    for row in table_data:
        logger.info(f"  {row[0]:<20} {row[1]:<20}")


def log_dataset_split_info(logger, train_idx, val_idx, test_idx):
    """
    记录数据集划分信息
    
    Args:
        logger: V0TrainingLogger实例
        train_idx: 训练集索引列表
        val_idx: 验证集索引列表
        test_idx: 测试集索引列表
    """
    total_samples = len(train_idx) + len(val_idx) + len(test_idx)
    logger.info("数据集划分完成:")
    
    if total_samples > 0:
        logger.info(f"  - 训练集: {len(train_idx)} ({len(train_idx)/total_samples*100:.1f}%)")
        logger.info(f"  - 验证集: {len(val_idx)} ({len(val_idx)/total_samples*100:.1f}%)")
        logger.info(f"  - 测试集: {len(test_idx)} ({len(test_idx)/total_samples*100:.1f}%)")
    else:
        logger.info("  - 数据集为空，无法划分")


def log_device_info(logger, device):
    """
    记录设备信息
    
    Args:
        logger: V0TrainingLogger实例
        device: 使用的设备
    """
    message = f"使用设备: {device}"
    logger.info(message)


def log_checkpoint_saved(logger, checkpoint_path):
    """
    记录检查点保存信息
    
    Args:
        logger: V0TrainingLogger实例
        checkpoint_path: 检查点文件路径
    """
    logger.info(f"已保存checkpoint: {checkpoint_path}")


def log_training_interrupted(logger, epoch, reason="NaN或inf损失值"):
    """
    记录训练中断信息
    
    Args:
        logger: V0TrainingLogger实例
        epoch: 当前轮数
        reason: 中断原因
    """
    message = f"警告: 在第 {epoch+1} 轮检测到{reason}，停止训练"
    logger.info(message)


def log_early_stopping(logger, epoch):
    """
    记录早停机制触发信息
    
    Args:
        logger: V0TrainingLogger实例
        epoch: 当前轮数
    """
    message = f"早停机制触发，在第 {epoch+1} 轮停止训练"
    logger.info(message)


def log_epoch_progress(logger, epoch: int, total_epochs: int, 
                      epoch_loss: torch.Tensor, val_loss: Optional[float] = None,
                      val_r2: Optional[float] = None, val_mae: Optional[float] = None,
                      val_pcc: Optional[float] = None):
    """
    记录epoch进度信息到日志
    
    Args:
        logger: 日志记录器
        epoch: 当前epoch
        total_epochs: 总epochs数
        epoch_loss: 训练损失
        val_loss: 验证损失
        val_r2: 验证R²
        val_mae: 验证MAE
        val_pcc: 验证PCC
    """
    message = f"Epoch [{epoch+1}/{total_epochs}], Train Loss: {epoch_loss.item():.6f}"
    if val_loss is not None:
        message += f", Val Loss: {val_loss:.6f}"
        if val_r2 is not None:
            message += f", R²: {val_r2:.4f}"
        if val_mae is not None:
            message += f", MAE: {val_mae:.4f}"
        if val_pcc is not None:
            message += f", PCC: {val_pcc:.4f}"
    logger.info(message)


def log_training_metrics(logger, train_losses, val_losses, test_loss, rmse, mae, r2, threshold_accs, pcc=None, rank_loss=None):
    """
    记录训练完成后的评估指标
    
    Args:
        logger: V0TrainingLogger实例
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        test_loss: 测试损失
        rmse: 均方根误差
        mae: 平均绝对误差
        r2: 决定系数
        threshold_accs: 阈值准确率字典
        pcc: Pearson相关系数
        rank_loss: 排序损失
    """
    logger.info("=" * 60)
    logger.info("训练完成 - 最终评估结果")
    logger.info("=" * 60)
    logger.info(f"测试集损失: {test_loss:.6f}")
    logger.info(f"RMSE: {rmse:.6f}")
    logger.info(f"MAE: {mae:.6f}")
    logger.info(f"R²: {r2:.6f}")
    if pcc is not None:
        logger.info(f"PCC: {pcc:.6f}")
    if rank_loss is not None:
        logger.info(f"Rank Loss: {rank_loss:.6f}")
    logger.info("")
    logger.info("阈值准确率:")
    for threshold, acc in threshold_accs.items():
        logger.info(f"  阈值 {threshold}: {acc*100:.2f}%")
    logger.info("")


def log_model_saved(logger, model_path):
    """
    记录模型保存信息
    
    Args:
        logger: V0TrainingLogger实例
        model_path: 模型文件路径
    """
    logger.info(f"模型已保存至: {model_path}")
    logger.info("")
