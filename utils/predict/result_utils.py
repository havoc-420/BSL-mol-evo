"""
结果工具模块
包含与结果展示和输出相关的工具函数
"""

import os
import numpy as np
from datetime import datetime
import json


def print_multi_model_comparison_results(predictions_list, true_values, row, model_dirs, prediction_mode='denormalized'):
    """
    打印多模型预测值与真实值的对比结果
    
    Args:
        predictions_list: 多个模型的预测值列表
        true_values: 真实值字典
        row: 数据行
        model_dirs: 模型目录路径列表
        prediction_mode: 预测模式
    """
    mode_text = "标准差" if prediction_mode == 'standardized' else "反标准化"
    print(f"\n多模型{mode_text}预测结果与真实值对比:")
    print("=" * 80)
    print(f"起始分子 SMILES: {row['smiles_from']}")
    print(f"目标分子 SMILES: {row['smiles_to']}")
    print(f"变化原子类型: {row['to_atom_symbol']}")
    print(f"操作类型: {row['operation_type']}")
    print(f"数据集行号: {row.name}")
    print(f"预测模式: {mode_text}")
    print("=" * 80)
    
    # 表头
    header = f"{'属性名称':<15}"
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        header += f"{model_name:<15}"
    header += f"{'真实值':<15}"
    print(header)
    print("-" * 80)
    
    # 数据行
    prop = 'mu_change'
    prop_name = prop.replace("_change", "") if prop.endswith("_change") else prop
    row_str = f"{prop_name:<15}"
    for predictions in predictions_list:
        pred_value = predictions[prop]
        row_str += f"{pred_value:<15.4f}"
    true_value = true_values[prop]
    row_str += f"{true_value:<15.4f}"
    print(row_str)
    
    # 计算每个模型的总体指标
    print("-" * 80)
    print(f"{'模型指标':<15}", end="")
    for i in range(len(model_dirs)):
        print(f"{'MSE':<5} {'RMSE':<5} {'MAE':<5}", end="")
    print(f"{'':<15}")  # 真实值列为空
    print("-" * 80)
    
    true_array = np.array([true_values['mu_change']])
    for i, predictions in enumerate(predictions_list):
        pred_array = np.array([predictions['mu_change']])
        mse = np.mean((pred_array - true_array) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_array - true_array))
        print(f"{'':<15}{mse:<5.2f} {rmse:<5.2f} {mae:<5.2f}", end="")
    print(f"{'':<15}")  # 真实值列为空


def print_comparison_results(predicted_values, true_values, row, model_dir, prediction_mode='denormalized'):
    """
    打印预测值与真实值的对比结果
    
    Args:
        predicted_values: 预测值字典
        true_values: 真实值字典
        row: 数据行
        model_dir: 模型目录路径
        prediction_mode: 预测模式
    """
    mode_text = "标准差" if prediction_mode == 'standardized' else "反标准化"
    print(f"\n{mode_text}预测结果与真实值对比:")
    print("=" * 60)
    print(f"起始分子 SMILES: {row['smiles_from']}")
    print(f"目标分子 SMILES: {row['smiles_to']}")
    print(f"变化原子类型: {row['to_atom_symbol']}")
    print(f"操作类型: {row['operation_type']}")
    print(f"使用模型: {os.path.basename(model_dir)}")
    print(f"数据集行号: {row.name}")
    print(f"预测模式: {mode_text}")
    print("=" * 60)
    
    # 表头
    print(f"{'属性名称':<15} {'预测值':<15} {'真实值':<15} {'差值':<15}")
    print("-" * 60)
    
    # 数据行
    prop = 'mu_change'
    pred_value = predicted_values[prop]
    true_value = true_values[prop]
    diff = pred_value - true_value
    prop_name = prop.replace("_change", "") if prop.endswith("_change") else prop
    print(f"{prop_name:<15} {pred_value:<15.4f} {true_value:<15.4f} {diff:<15.4f}")
    
    # 计算总体指标
    pred_array = np.array([predicted_values['mu_change']])
    true_array = np.array([true_values['mu_change']])
    
    mse = np.mean((pred_array - true_array) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_array - true_array))
    
    print("-" * 60)
    print(f"{'总体指标':<15} {'MSE':<15} {'RMSE':<15} {'MAE':<15}")
    print(f"{'':<15} {mse:<15.4f} {rmse:<15.4f} {mae:<15.4f}")


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
    relative_error = error_stats['relative_error_percent']
    pred_mean = error_stats['pred_mean']
    target_mean = error_stats['target_mean']
    pred_std = error_stats['pred_std']
    target_std = error_stats['target_std']
    prediction_mode = error_stats.get('prediction_mode', 'denormalized')
    
    mode_text = "标准差" if prediction_mode == 'standardized' else "反标准化"
    print("\n" * 2 + "="*80)
    print(f"批量{mode_text}预测误差统计结果")
    print("="*80)
    print(f"样本数量: {error_stats['num_samples']}")
    print(f"预测模式: {mode_text}")
    print("-"*80)
    
    # 表头
    header = f"{'属性':<12} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R2':<12} {'相对误差(%)':<12} {'预测均值':<12} {'真实均值':<12} {'预测std':<12} {'真实std':<12}"
    print(header)
    print("-"*80)
    
    # 数据行
    for i, prop in enumerate(property_names):
        prop_name = prop.replace('_change', '')
        row = (f"{prop_name:<12} {mse[i]:<12.4f} {rmse[i]:<12.4f} {mae[i]:<12.4f} {r2[i]:<12.4f} "
               f"{relative_error[i]:<12.2f} {pred_mean[i]:<12.4f} {target_mean[i]:<12.4f} "
               f"{pred_std[i]:<12.4f} {target_std[i]:<12.4f}")
        print(row)
    
    if logger:
        logger.info(f"批量{mode_text}预测完成，误差统计结果已显示")


def save_prediction_results(error_stats, sampled_df, model_path, model_dir, csv_file, num_samples, 
                          random_seed, prediction_mode, log_level, output_dir=None):
    """
    保存预测结果到文件
    
    Args:
        error_stats: 误差统计字典
        sampled_df: 采样的数据框
        model_path: 模型文件路径
        model_dir: 模型目录路径
        csv_file: CSV文件路径
        num_samples: 采样数量
        random_seed: 随机种子
        prediction_mode: 预测模式
        log_level: 日志级别
        output_dir: 输出目录路径
        
    Returns:
        输出文件路径
    """
    # 创建输出目录
    if output_dir is None:
        # 获取项目根目录
        project_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '..')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(model_path)
        model_dir_name = os.path.basename(model_dir) if model_dir else "unknown"
        output_dir = os.path.join(project_root, 'mol_evo', 'output', 'v0', 'predictions', f'prediction-{timestamp}-{model_dir_name}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件路径
    mode_suffix = "_denormalized" if prediction_mode == 'denormalized' else "_standardized"
    output_file = os.path.join(output_dir, f"batch_prediction_results{mode_suffix}.json")
    
    # 准备要保存的数据，将numpy数组转换为列表以便JSON序列化
    error_stats_serializable = {
        "property_names": error_stats["property_names"],
        "mse": error_stats["mse"].tolist(),
        "rmse": error_stats["rmse"].tolist(),
        "mae": error_stats["mae"].tolist(),
        "r2": error_stats["r2"].tolist(),
        "pcc": error_stats["pcc"].tolist(),
        "rank_loss": error_stats["rank_loss"].tolist(),
        "relative_error_percent": error_stats["relative_error_percent"].tolist(),
        "num_samples": error_stats["num_samples"],
        "pred_mean": error_stats["pred_mean"].tolist(),
        "target_mean": error_stats["target_mean"].tolist(),
        "pred_std": error_stats["pred_std"].tolist(),
        "target_std": error_stats["target_std"].tolist(),
        "prediction_mode": error_stats["prediction_mode"]
    }
    
    prediction_data = {
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "model_path": model_path,
            "model_name": os.path.basename(model_path),
            "model_dir": model_dir,
            "model_dir_name": os.path.basename(model_dir) if model_dir else "unknown"
        },
        "prediction_config": {
            "csv_file": csv_file,
            "num_samples": num_samples,
            "random_seed": random_seed,
            "prediction_mode": prediction_mode,
            "log_level": log_level
        },
        "error_stats": error_stats_serializable,
        # "sampled_data": sampled_df.to_dict('records') if sampled_df is not None else []
    }
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prediction_data, f, ensure_ascii=False, indent=2)
    
    return output_file