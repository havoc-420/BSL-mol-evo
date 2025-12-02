"""
预测工具模块
包含核心预测功能的工具函数
"""

import sys
import os
import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# 设置项目根目录路径
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

# 导入自定义模块
try:
    # 不再硬编码导入特定模型类，改为在需要时动态导入
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

def load_model_config(model_dir):
    """
    从模型目录加载模型配置信息
    
    Args:
        model_dir: 模型目录路径
        
    Returns:
        模型配置字典
    """
    config_file = os.path.join(model_dir, "model_config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            return config
    else:
        print(f"模型目录中未找到配置文件 {config_file}")
        return None


def get_target_property(model_dir):
    """
    从模型目录的训练数据中获取目标属性名称
    优先从model_config.json获取，其次尝试从training_process.json获取，最后尝试其他方式
    
    Args:
        model_dir: 模型目录路径
        
    Returns:
        目标属性名称
    """
    # 1. 首先尝试从model_config.json获取
    config_file = os.path.join(model_dir, "model_config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                if 'target_property' in config:
                    return config['target_property']
        except Exception as e:
            print(f"读取model_config.json时出错: {e}")
    
    # 2. 其次尝试从training_process.json获取
    process_file = os.path.join(model_dir, "training_process.json")
    if os.path.exists(process_file):
        try:
            with open(process_file, 'r') as f:
                data = json.load(f)
                # 从property_stats中获取第一个属性作为目标属性
                if 'property_stats' in data and isinstance(data['property_stats'], dict):
                    # 返回property_stats中的第一个键作为目标属性
                    for prop in data['property_stats'].keys():
                        return prop
        except Exception as e:
            print(f"读取training_process.json时出错: {e}")
    
    # 3. 尝试从load_training_params获取
    training_params = load_training_params(model_dir)
    if training_params and 'target_property' in training_params:
        return training_params['target_property']
    
    # 4. 默认返回mu_change以保持向后兼容
    print("无法从训练数据中获取目标属性名称，使用默认值 'mu_change'")
    return 'mu_change'


def load_model(model_path, model_dir):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        model_dir: 模型目录路径
        
    Returns:
        加载好的模型实例
    """
    # 从配置文件加载维度参数
    config = load_model_config(model_dir)
    
    if config and 'model_params' in config:
        # 从配置文件获取维度参数
        node_feature_dim = config['model_params'].get('node_feature_dim', 11)
        edge_feature_dim = config['model_params'].get('edge_feature_dim', 16)
        print(f"从配置文件加载维度参数: node_feature_dim={node_feature_dim}, edge_feature_dim={edge_feature_dim}")
    else:
        # 使用默认值并发出警告
        print("无法从配置文件加载维度参数，使用默认值")
        node_feature_dim = 11
        edge_feature_dim = 16
    
    # 根据模型目录动态导入相应的模型类
    model = None
    if "visnet" in model_dir.lower():
        from mol_evo.core.models.v0.visnet_linear_linear import MoleculeEvolutionVisnetLinearPredictor
        model = MoleculeEvolutionVisnetLinearPredictor(
            node_feature_dim=node_feature_dim,      # 模型节点特征维度
            edge_feature_dim=edge_feature_dim,      # 边特征维度
            hidden_dims=config['model_params'].get('hidden_dims', [128, 256, 256]) if config and 'model_params' in config else [128, 256, 256],
            output_dim=config['model_params'].get('output_dim', 1) if config and 'model_params' in config else 1              # 预测属性数量
        )
    else:
        # 默认使用GCN模型
        from mol_evo.core.models.v0.gcn_linear_linear import MoleculeEvolutionGCNLinearPredictor
        model = MoleculeEvolutionGCNLinearPredictor(
            node_feature_dim=node_feature_dim,      # 模型节点特征维度
            edge_feature_dim=config['model_params'].get('edge_feature_dim', 11) if config and 'model_params' in config else 11,      # 边特征维度
            hidden_dim=config['model_params'].get('hidden_dim', 128) if config and 'model_params' in config else 128,
            output_dim=config['model_params'].get('output_dim', 1) if config and 'model_params' in config else 1,             # 预测属性数量
            num_layers=config['model_params'].get('num_layers', 3) if config and 'model_params' in config else 3
        )
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    model.eval()
    return model

def predict_property_changes(model_path=None, model_dir=None, smiles_from=None, smiles_to=None, 
                            to_atom_symbol=None, operation_type=None, prediction_mode='denormalized', 
                            model=None):
    """
    使用训练好的模型预测属性变化
    
    Args:
        model_path: 模型文件路径（当model为None时必需）
        model_dir: 模型目录路径（当model为None时必需）
        smiles_from: 起始分子的SMILES
        smiles_to: 目标分子的SMILES
        to_atom_symbol: 变化涉及的原子类型
        operation_type: 操作类型
        prediction_mode: 预测模式 ('denormalized' 反标准化预测, 'standardized' 标准差预测)
        model: 已加载的模型实例（可选，如果提供则忽略model_path）
        
    Returns:
        预测的属性变化值
    """
    # 获取目标属性名称
    target_prop = get_target_property(model_dir)
    
    # 准备数据
    from_data, to_data, edge_attr, property_stats = prepare_single_prediction_data(
        smiles_from, smiles_to, to_atom_symbol, operation_type, model_dir)
    
    # 添加严格的数据验证
    if from_data is None or to_data is None:
        raise ValueError("分子图数据为None")
    
    # 检查必要的图属性
    for data in [from_data, to_data]:
        if not hasattr(data, 'x') or data.x is None:
            raise ValueError("分子数据不完整，缺少节点特征x")
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            raise ValueError("分子数据不完整，缺少边索引")
        if not hasattr(data, 'edge_attr') or data.edge_attr is None:
            raise ValueError("分子数据不完整，缺少边属性")
    
    # 如果没有提供模型，则加载模型
    if model is None:
        if model_path is None or model_dir is None:
            raise ValueError("当model为None时，必须提供model_path和model_dir")
        model = load_model(model_path, model_dir)
    
    # 进行预测
    with torch.no_grad():
        # 确保输入数据是有效的张量
        for data in [from_data, to_data]:
            data.x = data.x.float()
            data.edge_index = data.edge_index.long()
            data.edge_attr = data.edge_attr.float()
        
        # 添加batch信息（对于单个分子，batch全为0）
        from_data.batch = torch.zeros(from_data.x.size(0), dtype=torch.long)
        to_data.batch = torch.zeros(to_data.x.size(0), dtype=torch.long)
        
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
    
    # 预加载所有模型，避免重复加载提高效率
    models = [load_model(model_path, model_dir) for model_path, model_dir in zip(model_paths, model_dirs)]
    
    # 对每个模型进行预测
    predictions_list = []
    for model, model_dir in zip(models, model_dirs):
        primary_pred, secondary_pred = predict_property_changes(
            smiles_from=row['smiles_from'], 
            smiles_to=row['smiles_to'],
            to_atom_symbol=row['to_atom_symbol'], 
            operation_type=row['operation_type'],
            prediction_mode=prediction_mode,
            model=model,
            model_dir=model_dir
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
    if len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=random_seed).reset_index(drop=True)
    
    # 初始化误差统计
    error_stats = []
    
    # 只加载一次模型，避免重复加载提高效率
    model = load_model(model_path, model_dir)
    
    # 逐行预测
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="批量预测"):
        try:
            primary_pred, secondary_pred = predict_property_changes(
                smiles_from=row['smiles_from'], 
                smiles_to=row['smiles_to'],
                to_atom_symbol=row['to_atom_symbol'], 
                operation_type=row['operation_type'],
                prediction_mode=prediction_mode,
                model=model,
                model_dir=model_dir
            )
            
            # 计算误差
            true_value = row[target_prop]
            pred_value = primary_pred.get(target_prop, 0.0)
            error = abs(pred_value - true_value)
            
            # 记录统计信息
            stat = {
                'index': idx,
                'smiles_from': row['smiles_from'],
                'smiles_to': row['smiles_to'],
                'true_value': true_value,
                'predicted_value': pred_value,
                'absolute_error': error,
                'relative_error': error / (abs(true_value) + 1e-8)  # 避免除零
            }
            error_stats.append(stat)
            
        except Exception as e:
            if logger:
                logger.error(f"处理行 {idx} 时出错: {e}")
            continue
    
    return error_stats, df


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
