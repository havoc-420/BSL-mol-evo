#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的v0模型对整个数据集进行预测
"""

import sys
import os
import argparse
import torch
import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import torch.nn as nn
from datetime import datetime

# 设置项目根目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

try:
    # 导入训练脚本中的模块和函数
    from mol_evo.core.models.v0.gcn_linear import MoleculeEvolutionGCNLinearPredictor
    from mol_evo.core.utils.molecule import MoleculeCache
    from mol_evo.core.data.data_v0 import smiles_to_graph_data, prepare_edge_features
    # from mol_evo.utils.training_utils import split_data_indices
    from mol_evo.utils.logger_utils import DualLogger
    
    # 从训练脚本中导入数据集类和相关函数
    # 由于这些是局部定义的，我们需要重新定义它们
    class MoleculePairDataset(Dataset):
        """返回 (from_graph, to_graph, edge_feature, target) 的数据集"""
        def __init__(self, from_list, to_list, edge_attrs, targets):
            assert len(from_list) == len(to_list) == len(edge_attrs) == len(targets)
            self.from_list = from_list
            self.to_list   = to_list
            self.edge_attrs = edge_attrs      # Tensor (N, edge_feature_dim)
            self.targets    = targets         # Tensor (N, 1)

        def __len__(self):
            return len(self.from_list)

        def __getitem__(self, idx):
            return self.from_list[idx], self.to_list[idx], self.edge_attrs[idx], self.targets[idx]

    def pair_collate(batch):
        """把若干 (from, to, edge, target) 合并成 batch"""
        from_list, to_list, edge_list, target_list = zip(*batch)

        # PyG 自动把多个 Data 拼成一个 Batch
        from_batch = Batch.from_data_list(list(from_list))
        to_batch   = Batch.from_data_list(list(to_list))

        # edge 特征和目标直接 stack
        edge_batch = torch.stack(edge_list, dim=0)          # (B, edge_dim)
        target_batch = torch.stack(target_list, dim=0)      # (B, 1)

        return from_batch, to_batch, edge_batch, target_batch

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
        valid_indices = []  # 记录有效样本的索引
        
        # 创建分子缓存实例，并传入logger
        cache = MoleculeCache(csv_file=csv_file, logger=logger)    # UPDATE 避免缓存破坏
        
        # 构建分子数据列表
        from_data_list = []
        to_data_list = []
        edge_attr_list = []
        
        for idx, row in df.iterrows():
            # 使用smile_to_graph_xyz函数生成起始分子和目标分子的图结构
            from_data = smiles_to_graph_data(row['smiles_from'], cache)
            to_data = smiles_to_graph_data(row['smiles_to'], cache)
            
            # 检查转换是否成功
            if from_data is None or to_data is None:
                # message = f"跳过无法处理的分子对: {row['smiles_from']} -> {row['smiles_to']}"
                # if logger:
                #     logger.warning(message)
                # else:
                #     print(message)
                continue
                
            # 准备目标属性值
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
                
            from_data_list.append(from_data)
            to_data_list.append(to_data)
            
            # 准备边特征 (操作信息特征)
            edge_feat = prepare_edge_features(row, property_stats, include_property_changes=False)
            edge_attr_list.append(edge_feat)
            
            # 记录有效样本索引
            valid_indices.append(idx)
        
        if len(edge_attr_list) > 0:
            edge_attrs = torch.FloatTensor(np.array(edge_attr_list))
            target_features = torch.FloatTensor(np.array(target_features))
        else:
            edge_attrs = torch.FloatTensor([])
            target_features = torch.FloatTensor([])
        
        # 打印缓存统计信息
        stats = cache.get_stats()
        message = f"分子处理统计: 总数={stats['total']}, 命中={stats['hits']}, 未命中={stats['misses']}, 命中率={stats['hit_rate']:.2%}"
        if logger:
            logger.info(message)
        else:
            print(message)
        
        return from_data_list, to_data_list, edge_attrs, target_features, property_stats, valid_indices

except ImportError as e:
    print(f"无法导入所需的模块: {e}")
    exit(1)


def load_model(model_path, model_params):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        model_params: 模型参数字典
        
    Returns:
        加载好的模型
    """
    model = MoleculeEvolutionGCNLinearPredictor(
        node_feature_dim=model_params["node_feature_dim"],
        edge_feature_dim=model_params["edge_feature_dim"],
        hidden_dim=model_params["hidden_dim"],
        output_dim=model_params["output_dim"],
        num_layers=model_params["num_layers"]
    )
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model


def evaluate_model(model, data_loader, device, property_stats, target_property):
    """
    在整个数据集上评估模型
    
    Args:
        model: 模型实例
        data_loader: 数据加载器
        device: 计算设备
        property_stats: 属性统计信息
        target_property: 目标属性名称
        
    Returns:
        评估指标字典
    """
    model.eval()
    with torch.no_grad():
        all_test_predictions = []
        all_test_targets = []
        
        for from_batch, to_batch, edge_batch, target_batch in data_loader:
            # 移动到设备
            from_batch = from_batch.to(device)
            to_batch = to_batch.to(device)
            edge_batch = edge_batch.to(device)
            
            # 前向传播
            test_predictions = model(from_batch, to_batch, edge_batch)
            all_test_predictions.append(test_predictions.cpu())
            all_test_targets.append(target_batch.cpu())
        
        # 合并所有测试预测和目标
        test_predictions = torch.cat(all_test_predictions, dim=0).to(device)
        test_targets = torch.cat(all_test_targets, dim=0).to(device)
        
        # 计算损失
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
        
        # 阈值准确率评估：在标准化情况下，阈值表示标准差的倍数
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
        
        # 反标准化预测值和真实值（如果需要）
        mean, std = property_stats[target_property]
        if std > 0:
            denorm_predictions = test_predictions * std + mean
            denorm_targets = test_targets * std + mean
        else:
            denorm_predictions = test_predictions
            denorm_targets = test_targets
            
        # 计算反标准化后的指标
        denorm_mse = torch.mean((denorm_predictions - denorm_targets) ** 2)
        denorm_rmse = torch.sqrt(denorm_mse)
        denorm_mae = torch.mean(torch.abs(denorm_predictions - denorm_targets))
        
        # 反标准化后的R²
        denorm_ss_res = torch.sum((denorm_targets - denorm_predictions) ** 2)
        denorm_ss_tot = torch.sum((denorm_targets - torch.mean(denorm_targets)) ** 2)
        
        if denorm_ss_tot.item() == 0:
            denorm_r2 = 0.0
        else:
            denorm_r2 = (1 - denorm_ss_res / (denorm_ss_tot + 1e-8)).item()
        
        return {
            "test_loss": test_loss.item(),
            "rmse": rmse.item(),
            "mae": mae.item(),
            "r2": r2,
            "pcc": pcc,
            "rank_loss": rank_loss,
            "threshold_accuracies": threshold_accuracies,
            "denorm_rmse": denorm_rmse.item(),
            "denorm_mae": denorm_mae.item(),
            "denorm_r2": denorm_r2,
            "predictions": test_predictions.cpu().numpy().tolist(),
            "targets": test_targets.cpu().numpy().tolist(),
            "denorm_predictions": denorm_predictions.cpu().numpy().tolist(),
            "denorm_targets": denorm_targets.cpu().numpy().tolist()
        }


def predict_full_dataset(model_path, model_dir, data_file, seed=42, batch_size=64, max_pairs=None):
    """
    使用训练好的模型对整个数据集进行预测
    
    Args:
        model_path: 模型文件路径
        model_dir: 模型目录路径
        data_file: 数据文件路径
        seed: 随机种子
        batch_size: 批处理大小
        max_pairs: 最大对数（用于调试）
    """
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, 'mol_evo', 'output', 'v0', 'predictions', f'prediction-{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = DualLogger(output_dir)
    logger.info(f"开始对整个数据集进行预测: {data_file}")
    logger.info(f"使用模型: {model_path}")
    logger.info(f"随机种子: {seed}")
    
    try:
        # 读取模型参数
        training_data_path = os.path.join(model_dir, "training_data.json")
        if os.path.exists(training_data_path):
            with open(training_data_path, 'r') as f:
                training_data = json.load(f)
                model_params = training_data.get("model_params", {})
                target_property = training_data.get("training_params", {}).get("target_property", "mu_change")
                property_stats = training_data.get("property_stats", {target_property: (0.0, 1.0)})
                logger.info("从训练数据中加载模型参数和属性统计信息")
        else:
            # 默认参数
            model_params = {
                "node_feature_dim": 11,
                "edge_feature_dim": 11,
                "hidden_dim": 128,
                "output_dim": 1,
                "num_layers": 3
            }
            target_property = "mu_change"
            property_stats = {target_property: (0.0, 1.0)}
            logger.warning("未找到训练数据文件，使用默认参数")
        
        # 构建数据集
        logger.info(f"正在构建图数据: {data_file}")
        from_data_list, to_data_list, edge_attrs, target_features, _, valid_indices = build_molecule_evolution_dataset_v0(
            data_file, max_pairs, target_property, logger)
        
        # 使用整个数据集创建数据集对象
        full_dataset = MoleculePairDataset(
            from_data_list,
            to_data_list,
            edge_attrs,
            target_features)
        
        logger.info(f"数据集构建完成 - 总样本数: {len(full_dataset)} (原始样本数: {len(valid_indices)})")
        
        # 创建数据加载器
        data_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=pair_collate)
        
        # 加载模型
        model = load_model(model_path, model_params)
        logger.info("模型加载成功")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        logger.info(f"使用设备: {device}")
        
        # 在整个数据集上评估模型
        logger.info("开始在整个数据集上进行预测和评估")
        metrics = evaluate_model(model, data_loader, device, property_stats, target_property)
        
        # 保存结果
        results = {
            "model_info": {
                "model_path": model_path,
                "model_dir": model_dir,
                "model_filename": os.path.basename(model_path)
            },
            "prediction_config": {
                "data_file": data_file,
                "seed": seed,
                "batch_size": batch_size,
                "max_pairs": max_pairs,
                "target_property": target_property
            },
            "metrics": metrics
        }
        
        results_file = os.path.join(output_dir, "full_dataset_prediction_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 打印结果摘要
        logger.info("=" * 50)
        logger.info("全量数据集预测结果摘要")
        logger.info("=" * 50)
        logger.info(f"测试损失 (MAE): {metrics['test_loss']:.6f}")
        logger.info(f"RMSE: {metrics['rmse']:.6f}")
        logger.info(f"MAE: {metrics['mae']:.6f}")
        logger.info(f"R²: {metrics['r2']:.6f}")
        logger.info(f"PCC: {metrics['pcc']:.6f}")
        logger.info(f"Rank Loss: {metrics['rank_loss']:.6f}")
        logger.info("-" * 30)
        logger.info("反标准化后的指标:")
        logger.info(f"RMSE: {metrics['denorm_rmse']:.6f}")
        logger.info(f"MAE: {metrics['denorm_mae']:.6f}")
        logger.info(f"R²: {metrics['denorm_r2']:.6f}")
        logger.info("=" * 50)
        logger.info(f"详细结果已保存到: {results_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"预测过程中发生错误: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='使用训练好的v0模型对整个数据集进行预测')
    
    parser.add_argument('--model-path', '-mp', type=str, required=True,
                       help='模型文件路径 (.pth文件)')
    parser.add_argument('--model-dir', '-md', type=str, required=True,
                       help='模型目录路径 (包含training_data.json)')
    parser.add_argument('--data-file', '-df', type=str, 
                       default=os.path.join('mol_evo', 'dataset', 'data', 'qm9-evo-pairs-step-1-with-properties-pct.csv'),
                       help='数据文件路径')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--batch-size', '-bs', type=int, default=64,
                       help='批处理大小 (默认: 64)')
    parser.add_argument('--max-pairs', '-mpairs', type=int,
                       help='最大分子对数（用于调试）')
    
    args = parser.parse_args()
    
    try:
        predict_full_dataset(
            args.model_path, args.model_dir, args.data_file, 
            args.seed, args.batch_size, args.max_pairs
        )
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()