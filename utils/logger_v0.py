#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0版本模型训练日志工具函数
"""

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
    记录数据集样本示例
    
    Args:
        logger: V0TrainingLogger实例
        from_data_list: 起始分子数据列表
        to_data_list: 目标分子数据列表
        edge_attrs: 边特征张量
        target_features: 目标属性张量
        num_examples: 示例数量
    """
    logger.info("--- 训练集样本示例 (头部数据) ---")
    num_examples = min(num_examples, len(from_data_list))
    
    # 表格头部
    logger.info("  {:<8} {:<12} {:<12} {:<12} {:<12} {:<15} {:<15}".format(
        "样本", "起始节点数", "起始边数", "目标节点数", "目标边数", "边特征", "目标属性值"))
    logger.info("  " + "-" * 80)
    
    # 表格内容
    for i in range(num_examples):
        from_nodes = from_data_list[i].x.size(0)
        from_edges = from_data_list[i].edge_index.size(1)
        to_nodes = to_data_list[i].x.size(0)
        to_edges = to_data_list[i].edge_index.size(1)
        edge_attr = str(edge_attrs[i].tolist())
        target_val = f"{target_features[i].item():.6f}"
        
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
        target_features: 目标属性张量
    """
    logger.info("\n数据构建完成:")
    logger.info("  {:<20} {:<20}".format("项目", "值"))
    logger.info("  " + "-" * 40)
    logger.info("  {:<20} {:<20}".format("样本数", str(len(from_data_list))))
    logger.info("  {:<20} {:<20}".format("节点特征维度", str(model_params['node_feature_dim'])))
    logger.info("  {:<20} {:<20}".format("边特征维度", str(edge_attrs.shape[1] if len(edge_attrs.shape) > 1 else 1)))
    logger.info("  {:<20} {:<20}".format("目标属性变化维度", str(target_features.shape[1] if len(target_features.shape) > 1 else 1)))


def log_dataset_split_info(logger, train_idx, val_idx, test_idx):
    """
    记录数据集划分信息
    
    Args:
        logger: V0TrainingLogger实例
        train_idx: 训练集索引
        val_idx: 验证集索引
        test_idx: 测试集索引
    """
    total_samples = len(train_idx) + len(val_idx) + len(test_idx)
    logger.info("数据集划分完成:")
    logger.info(f"  - 训练集: {len(train_idx)} ({len(train_idx)/total_samples*100:.1f}%)")
    logger.info(f"  - 验证集: {len(val_idx)} ({len(val_idx)/total_samples*100:.1f}%)")
    logger.info(f"  - 测试集: {len(test_idx)} ({len(test_idx)/total_samples*100:.1f}%)")


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


def log_epoch_progress(logger, epoch, epochs, train_loss, val_loss=None, val_r2=None, val_mae=None):
    """
    记录每轮训练进度信息
    
    Args:
        logger: V0TrainingLogger实例
        epoch: 当前轮数
        epochs: 总轮数
        train_loss: 训练损失
        val_loss: 验证损失
        val_r2: 验证R²
        val_mae: 验证MAE
    """
    message = f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.6f}"
    if val_loss is not None:
        message += f", Val Loss: {val_loss:.6f}"
        if val_r2 is not None and val_mae is not None:
            message += f", R²: {val_r2:.4f}, MAE: {val_mae:.4f}"
    
    # 详细训练信息只记录到文件，不在控制台显示
    logger.info(message, to_console=False)


def log_training_metrics(logger, train_losses, val_losses, test_loss, rmse, mae, r2, threshold_accs):
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
    """
    logger.info("=" * 60)
    logger.info("训练完成 - 最终评估结果")
    logger.info("=" * 60)
    logger.info(f"测试集损失: {test_loss:.6f}")
    logger.info(f"RMSE: {rmse:.6f}")
    logger.info(f"MAE: {mae:.6f}")
    logger.info(f"R²: {r2:.6f}")
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
