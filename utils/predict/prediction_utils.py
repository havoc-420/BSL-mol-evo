"""
预测工具模块
包含核心预测功能的工具函数
"""

import sys
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# 设置项目根目录路径
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

# 导入自定义模块
try:
    from mol_evo.core.models.v0.gcn_linear import MoleculeEvolutionGCNLinearPredictor
    from mol_evo.core.data.processing import prepare_edge_features
    from mol_evo.core.utils.molecule import MoleculeCache
    from mol_evo.core.data.data_v0 import smiles_to_graph_data
except ImportError as e:
    print("无法导入自定义模块", e)
    raise

try:
    from .data_utils import prepare_single_prediction_data
    from .model_utils import load_property_stats, is_model_normalized, load_training_params
except ImportError as e:
    print("无法导入工具模块", e)
    raise


def get_target_property(model_dir):
    """
    从模型目录的训练数据中获取目标属性名称
    
    Args:
        model_dir: 模型目录路径
        
    Returns:
        目标属性名称
    """
    training_params = load_training_params(model_dir)
    if training_params and 'target_property' in training_params:
        return training_params['target_property']
    else:
        # 默认返回mu_change以保持向后兼容
        return 'mu_change'


def predict_property_changes(model_path, model_dir, smiles_from, smiles_to, to_atom_symbol, operation_type, prediction_mode='denormalized'):
    """
    使用训练好的模型预测属性变化
    
    Args:
        model_path: 模型文件路径
        model_dir: 模型目录路径
        smiles_from: 起始分子的SMILES
        smiles_to: 目标分子的SMILES
        to_atom_symbol: 变化涉及的原子类型
        operation_type: 操作类型
        prediction_mode: 预测模式 ('denormalized' 反标准化预测, 'standardized' 标准差预测)
        
    Returns:
        预测的属性变化值
    """
    # 获取目标属性名称
    target_prop = get_target_property(model_dir)
    
    # 准备数据
    from_data, to_data, edge_attr, property_stats = prepare_single_prediction_data(
        smiles_from, smiles_to, to_atom_symbol, operation_type, model_dir)
    
    # 创建模型
    model = MoleculeEvolutionGCNLinearPredictor(
        node_feature_dim=11,      # v0模型节点特征维度
        edge_feature_dim=11,      # 边特征维度（5原子类型 + 6操作类型）
        hidden_dim=128,
        output_dim=1,             # v0模型只预测单个属性
        num_layers=3
    )
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # TEST 添加调试日志，输出模型参数的前一小部分
    print("Debug: 预测模型参数前10个值:")
    state_dict = model.state_dict()
    for name, param in list(state_dict.items())[:3]:  # 取前几个参数
        values = param.flatten()[:10]  # 取前10个值
        print(f"Debug: 参数 {name} (shape: {param.shape}) => 前10个值: {values.tolist()}")
    
    model.eval()
    
    # 进行预测
    with torch.no_grad():
        predictions = model(from_data, to_data, edge_attr)
    
    # 获取预测结果
    predicted_changes = predictions[0].numpy()
    
    # 检查模型是否使用了标准化
    is_normalized = is_model_normalized(model_dir)
    
    if is_normalized and property_stats:
        # 如果模型使用了标准化
        standardized_changes = {}
        original_changes = {}
        
        # 使用从模型配置中获取的目标属性
        standardized_changes[target_prop] = predicted_changes[0]
        # 反标准化得到原始尺度的预测值
        if target_prop in property_stats:
            mean, std = property_stats[target_prop]
            original_changes[target_prop] = predicted_changes[0] * std + mean
        else:
            original_changes[target_prop] = predicted_changes[0]
        
        # 根据预测模式返回相应结果
        if prediction_mode == 'standardized':
            return standardized_changes, original_changes
        else:  # denormalized
            return original_changes, standardized_changes
    else:
        # 如果模型没有使用标准化，则直接返回预测值
        original_changes = {target_prop: predicted_changes[0]}
        # 在非标准化模式下，标准化值为None
        return original_changes, None  # 第二个返回值为None表示没有标准化值


def compare_with_ground_truth(model_paths, model_dirs, csv_file, row_index, prediction_mode='denormalized'):
    """
    与CSV文件中指定行的真实值进行对比（支持多模型）
    
    Args:
        model_paths: 模型文件路径列表
        model_dirs: 模型目录路径列表
        csv_file: CSV文件路径
        row_index: 行索引
        prediction_mode: 预测模式 ('denormalized' 反标准化预测, 'standardized' 标准差预测)
        
    Returns:
        预测值和真实值的对比
    """
    # 读取指定行的数据
    df = pd.read_csv(csv_file)
    if row_index >= len(df):
        raise ValueError(f"行索引 {row_index} 超出范围，CSV文件共有 {len(df)} 行")
    
    row = df.iloc[row_index]
    
    # 获取第一个模型的目标属性以确定要比较的真实值字段
    if model_dirs:
        target_prop = get_target_property(model_dirs[0])
        # 获取真实值
        true_values = {target_prop: row[target_prop]}
    else:
        true_values = {'mu_change': row['mu_change']}
    
    # 对每个模型进行预测
    predictions_list = []
    for model_path, model_dir in zip(model_paths, model_dirs):
        primary_pred, secondary_pred = predict_property_changes(
            model_path, model_dir,
            row['smiles_from'], row['smiles_to'],
            row['to_atom_symbol'], row['operation_type'],
            prediction_mode
        )
        
        # 根据预测模式使用相应的预测值
        predictions_list.append(primary_pred)
    
    return predictions_list, true_values, row


def batch_predict(model_path, model_dir, csv_file, num_samples=10, random_seed=42, prediction_mode='denormalized', logger=None):
    """
    批量预测并计算误差
    
    Args:
        model_path: 模型文件路径
        model_dir: 模型目录路径
        csv_file: CSV文件路径
        num_samples: 采样数量
        random_seed: 随机种子
        prediction_mode: 预测模式 ('denormalized' 反标准化预测, 'standardized' 标准差预测)
        logger: 日志记录器
        
    Returns:
        预测结果和误差统计
    """
    # 获取目标属性名称
    target_prop = get_target_property(model_dir)
    
    # 读取数据
    df = pd.read_csv(csv_file)
    
    # 随机采样
    np.random.seed(random_seed)
    sampled_indices = np.random.choice(len(df), min(num_samples, len(df)), replace=False)
    sampled_df = df.iloc[sampled_indices].reset_index(drop=True)
    
    # INFO 获取属性统计信息用于反标准化
    property_stats = load_property_stats(model_dir)
    
    # 创建分子缓存实例，绑定到具体的CSV文件
    cache = MoleculeCache(csv_file=csv_file)
    
    # 预处理所有数据，检查哪些样本可以成功处理
    if logger:
        logger.info(f"开始预处理 {len(sampled_df)} 个样本...")
    valid_indices = []  # 存储有效样本的索引
    valid_from_data = []  # 存储有效的起始分子数据
    valid_to_data = []  # 存储有效的目标分子数据
    valid_edge_attrs = []  # 存储有效的边特征
    valid_targets = []  # 存储有效的目标值
    
    for idx, row in sampled_df.iterrows():
        try:
            # 准备数据
            from_data = smiles_to_graph_data(row['smiles_from'], cache)
            to_data = smiles_to_graph_data(row['smiles_to'], cache)
            
            if from_data is None or to_data is None:
                # logger.warning(f"跳过无法处理的分子对: {row['smiles_from']} -> {row['smiles_to']}")
                continue
            
            # 构建边特征（不含属性变化）
            edge_feat = prepare_edge_features(row, property_stats, include_property_changes=False)
            edge_attr = torch.FloatTensor(np.array([edge_feat]))
            
            # 获取真实值
            true_changes = {target_prop: row[target_prop]}
            
            # 保存有效数据
            valid_indices.append(idx)
            valid_from_data.append(from_data)
            valid_to_data.append(to_data)
            valid_edge_attrs.append(edge_attr)
            valid_targets.append(true_changes)
            
        except Exception as e:
            if logger:
                logger.warning(f"预处理第 {idx} 个样本时出错: {e}")
            continue
    
    # 输出缓存统计信息
    if hasattr(cache, 'get_stats') and logger:
        cache_stats = cache.get_stats()
        logger.info(f"缓存统计: {cache_stats}")
    
    if not valid_indices:
        if logger:
            logger.error("没有有效的样本用于预测")
        return None, None
    
    # 更新sampled_df，只保留有效样本
    sampled_df = sampled_df.iloc[valid_indices].reset_index(drop=True)
    
    if logger:
        logger.info(f"预处理完成，{len(valid_indices)} 个有效样本，开始进行预测...")
    
    # 创建模型
    model = MoleculeEvolutionGCNLinearPredictor(
        node_feature_dim=11,      # v0模型节点特征维度
        edge_feature_dim=11,      # 边特征维度（5原子类型 + 6操作类型）
        hidden_dim=128,
        output_dim=1,             # v0模型只预测单个属性
        num_layers=3
    )
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # 存储预测结果
    predictions_list = []
    
    # 对有效样本进行预测
    with torch.no_grad():
        # 使用 tqdm 显示预测进度
        pred_pbar = tqdm(zip(valid_from_data, valid_to_data, valid_edge_attrs), 
                       total=len(valid_indices), desc="预测")
        
        for idx, (from_data, to_data, edge_attr) in enumerate(pred_pbar):
            try:
                predictions = model(from_data, to_data, edge_attr)
                
                # 获取预测结果
                predicted_changes = predictions[0].numpy()
                
                # 检查模型是否使用了标准化
                is_normalized = is_model_normalized(model_dir)
                
                if is_normalized and property_stats:
                    # 根据预测模式处理预测值
                    if prediction_mode == 'standardized':
                        # 标准差预测模式，直接使用模型输出
                        final_pred = predicted_changes[0]
                    else:  # denormalized 反标准化预测模式
                        # 反标准化得到原始尺度的预测值
                        if target_prop in property_stats:
                            mean, std = property_stats[target_prop]
                            final_pred = predicted_changes[0] * std + mean
                        else:
                            final_pred = predicted_changes[0]
                else:
                    # 如果模型没有使用标准化，则直接使用预测值
                    final_pred = predicted_changes[0]
                
                predictions_list.append({target_prop: final_pred})
                
                # 更新进度条描述
                pred_pbar.set_description(f"预测 (已完成: {idx + 1})")
                    
            except Exception as e:
                if logger:
                    logger.error(f"预测第 {idx} 个样本时出错: {e}")
                # 如果预测出错，添加一个默认值
                predictions_list.append({target_prop: 0.0})
    
    # 计算误差统计
    if not predictions_list or not valid_targets:
        if logger:
            logger.error("没有有效的预测结果用于计算误差")
        return None, None
    
    # 转换为数组便于计算
    pred_array = np.array([pred[target_prop] for pred in predictions_list])
    target_array = np.array([target[target_prop] for target in valid_targets])
    
    # 计算各种误差指标
    mse = np.mean((pred_array - target_array) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_array - target_array))
    
    # 计算R2分数
    ss_res = np.sum((target_array - pred_array) ** 2)
    ss_tot = np.sum((target_array - np.mean(target_array)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)  # 添加小值避免除零
    
    # PCC (Pearson Correlation Coefficient)
    pred_mean = np.mean(pred_array)
    target_mean = np.mean(target_array)
    pred_centered = pred_array - pred_mean
    target_centered = target_array - target_mean
    numerator = np.sum(pred_centered * target_centered)
    pred_sq_sum = np.sum(pred_centered ** 2)
    target_sq_sum = np.sum(target_centered ** 2)
    denominator = np.sqrt(pred_sq_sum * target_sq_sum)
    
    if denominator == 0:
        pcc = 0.0
    else:
        pcc = numerator / denominator
    
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
        pred_diffs = np.expand_dims(preds, 1) - np.expand_dims(preds, 0)
        target_diffs = np.expand_dims(targets, 1) - np.expand_dims(targets, 0)
        
        # 只考虑目标值不同的样本对
        mask = target_diffs != 0
        sign_diffs = np.sign(target_diffs[mask])
        
        # 计算hinge loss
        loss = np.maximum(0.0, 1.0 - sign_diffs * pred_diffs[mask])
        return np.mean(loss)
    
    rank_loss = compute_rank_loss(pred_array, target_array)
    
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
        'property_names': [target_prop],
        'mse': np.array([mse]),
        'rmse': np.array([rmse]),
        'mae': np.array([mae]),
        'r2': np.array([r2]),
        'pcc': np.array([pcc]),
        'rank_loss': np.array([rank_loss]),
        'relative_error_percent': np.array([relative_error]),
        'num_samples': len(predictions_list),
        'pred_mean': np.array([pred_mean]),
        'target_mean': np.array([target_mean]),
        'pred_std': np.array([pred_std]),
        'target_std': np.array([target_std]),
        'prediction_mode': prediction_mode
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
    pcc = error_stats['pcc']
    rank_loss = error_stats['rank_loss']
    relative_error = error_stats['relative_error_percent']
    pred_mean = error_stats['pred_mean']
    target_mean = error_stats['target_mean']
    pred_std = error_stats['pred_std']
    target_std = error_stats['target_std']
    prediction_mode = error_stats.get('prediction_mode', 'denormalized')
    
    mode_text = "标准差" if prediction_mode == 'standardized' else "反标准化"
    print("\n" * 2 + "="*100)
    print(f"批量{mode_text}预测误差统计结果")
    print("="*100)
    print(f"样本数量: {error_stats['num_samples']}")
    print(f"预测模式: {mode_text}")
    print("-"*100)
    
    # 表头
    header = f"{'属性':<12} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R2':<12} {'PCC':<12} {'Rank Loss':<12} {'相对误差(%)':<12} {'预测均值':<12} {'真实均值':<12} {'预测std':<12} {'真实std':<12}"
    print(header)
    print("-"*100)
    
    # 数据行
    for i, prop in enumerate(property_names):
        prop_name = prop.replace('_change', '')
        row = (f"{prop_name:<12} {mse[i]:<12.4f} {rmse[i]:<12.4f} {mae[i]:<12.4f} {r2[i]:<12.4f} "
               f"{pcc[i]:<12.4f} {rank_loss[i]:<12.4f} {relative_error[i]:<12.2f} {pred_mean[i]:<12.4f} {target_mean[i]:<12.4f} "
               f"{pred_std[i]:<12.4f} {target_std[i]:<12.4f}")
        print(row)
    
    if logger:
        logger.info(f"批量{mode_text}预测完成，误差统计结果已显示")
