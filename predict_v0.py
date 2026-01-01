#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的v0模型进行分子进化属性变化预测
"""

import sys
import os
import argparse
import logging
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# 设置项目根目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

try:
    # 导入新的工具模块
    from mol_evo.utils.predict import (
        setup_logger,
        find_model_files,
        select_models_interactively,
        predict_property_changes,
        compare_with_ground_truth,
        batch_predict,
        print_multi_model_comparison_results,
        print_comparison_results,
        print_error_statistics
    )
    from mol_evo.utils.predict.result_utils import save_prediction_results
    from mol_evo.core.data.processing import load_operation_config, get_operation_types
    from mol_evo.utils.predict.prediction_utils import get_target_property
    from mol_evo.utils.predict.model_utils import load_property_stats, is_model_normalized
    
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='使用训练好的v0模型进行分子进化属性变化预测')
    parser.add_argument('--model-path', '-mp', type=str,
                       help='模型文件路径 (.pth文件)')
    parser.add_argument('--model-dir', '-md', type=str,
                       help='模型目录路径 (包含training_data.json)')
    parser.add_argument('--smiles-from', '-sf', type=str,
                       help='起始分子的SMILES (单次预测)')
    parser.add_argument('--smiles-to', '-st', type=str,
                       help='目标分子的SMILES (单次预测)')
    parser.add_argument('--atom-symbol', '-as', type=str,
                       help='变化涉及的原子类型 (如: C, N, O, F, P) (单次预测)')
    parser.add_argument('--operation-type', '-ot', type=str,
                       help='操作类型 (单次预测)')
    parser.add_argument('--csv-file', '-cf', type=str, default='mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv',
                       help='CSV文件路径 (批量预测)')
    parser.add_argument('--json-file', '-jf', type=str, default='mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.json',
                       help='JSON文件路径 (批量预测)')
    parser.add_argument('--indices-file', '-if', type=str, default='mol_evo/dataset/data/dataset_indices/indices_20251127_122155_seed42.json',
                       help='数据集索引文件路径，用于过滤test数据')
    parser.add_argument('--use-test-indices', '-uti', action='store_true',
                       help='是否使用test_indices进行数据过滤')
    parser.add_argument('--row-index', '-ri', type=int,
                       help='CSV文件中的行索引，用于对比预测值和真实值')
    parser.add_argument('--num-samples', '-ns', type=int, default=100,
                       help='批量预测的样本数量 (默认: 100)')
    parser.add_argument('--random-seed', '-rs', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--log-level', '-ll', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='日志级别 (默认: INFO)')
    parser.add_argument('--prediction-mode', '-pm', type=str, default='denormalized',
                       choices=['denormalized', 'standardized'],
                       help='预测模式: denormalized(反标准化预测) 或 standardized(标准差预测) (默认: denormalized)')
    parser.add_argument('--sample-method', '-sm', type=str, default='random',
                       choices=['random', 'sequential'],
                       help='采样方式: random(随机采样) 或 sequential(顺序采样) (默认: random)')
    parser.add_argument('--config-file', '-cf2', type=str, 
                       default='/home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml',
                       help='配置文件路径 (默认使用qm9配置文件)')
    
    args = parser.parse_args()
    
    # 参数验证
    if args.use_test_indices and not args.indices_file:
        parser.error('--indices-file必须在使用--use-test-indices时提供')
    
    # 确保批量预测时至少提供一种数据文件
    if args.row_index is None and not args.smiles_from and not args.csv_file and not args.json_file:
        parser.error('必须提供--csv-file或--json-file用于批量预测，或提供单次预测的参数')
    
    # 设置日志记录器
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logger(log_level)
    
    # 加载操作配置，初始化原子类型和操作类型
    logger.info(f"加载配置文件: {args.config_file}")
    load_operation_config(config_path=args.config_file)
    
    # 导入获取操作类型的函数
    # 在加载配置后验证操作类型
    if args.operation_type:
        valid_operations = get_operation_types()
        if args.operation_type not in valid_operations:
            logger.error(f"无效的操作类型: {args.operation_type}")
            logger.error(f"有效操作类型: {', '.join(valid_operations)}")
            sys.exit(1)
    
    # 检查是单次预测还是批量预测
    single_prediction = args.smiles_from and args.smiles_to and args.atom_symbol and args.operation_type
    batch_prediction = (args.csv_file or args.json_file) and args.row_index is None
    comparison_prediction = args.csv_file and args.row_index is not None
    
    if not single_prediction and not batch_prediction and not comparison_prediction:
        logger.error("请指定单次预测参数或批量预测参数")
        parser.print_help()
        return
    
    try:
        # STAGE 如果没有指定模型路径，则自动查找并选择
        if not args.model_path:
            model_files = find_model_files()
            selected_models = select_models_interactively(model_files)
            
            if not selected_models:
                logger.warning("未选择模型，程序退出")
                return
            
            # 如果选择了多个模型且是对比预测，则进行多模型对比
            if len(selected_models) > 1 and comparison_prediction:
                model_paths = selected_models
                model_dirs = [os.path.dirname(model_path) for model_path in model_paths]
            else:
                # 默认使用第一个选择的模型
                args.model_path = selected_models[0]
                args.model_dir = os.path.dirname(selected_models[0])
        elif not args.model_dir:
            # 如果指定了模型路径但没有指定模型目录
            args.model_dir = os.path.dirname(args.model_path)
        
        # 检查模型目录中是否包含训练数据
        if hasattr(args, 'model_dir') and args.model_dir:
            stats_file = os.path.join(args.model_dir, "training_data.json")
            if not os.path.exists(stats_file):
                # 适配新文件结构
                stats_file = os.path.join(args.model_dir, "training_process.json")
                if not os.path.exists(stats_file):
                    logger.warning(f"模型目录中未找到训练过程文件 {stats_file}")
                    logger.warning("将无法进行反标准化以获得原始尺度的预测值")
        
        # TAG 执行单次预测
        if single_prediction:
            primary_pred, secondary_pred = predict_property_changes(
                args.model_path, args.model_dir,
                args.smiles_from, args.smiles_to,
                args.atom_symbol, args.operation_type,
                args.prediction_mode
            )
            
            mode_text = "标准差" if args.prediction_mode == 'standardized' else "反标准化"
            print(f"\n{mode_text}预测结果:")
            print("=" * 50)
            print(f"起始分子 SMILES: {args.smiles_from}")
            print(f"目标分子 SMILES: {args.smiles_to}")
            print(f"变化原子类型: {args.atom_symbol}")
            print(f"操作类型: {args.operation_type}")
            print(f"使用模型: {os.path.basename(args.model_dir)}")
            print(f"预测模式: {mode_text}")
            print("=" * 50)
            
            # 创建表格展示预测结果
            print("\n预测结果汇总:")
            # 表头
            print(f"{'属性名称':<15} {'标准化值':<12} {'原始值':<12}")
            print("-" * 45)
            
            # 从模型配置文件中读取目标属性
            target_property = 'mu_change'  # 默认值
            try:
                config_file = os.path.join(args.model_dir, 'model_config.json')
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                        if 'model-train-config' in config_data and 'target_property' in config_data['model-train-config']:
                            target_property = config_data['model-train-config']['target_property']
                            # logger.info(f"从模型配置文件中读取到目标属性: {target_property}")
            except Exception as e:
                logger.warning(f"读取模型配置文件失败: {e}，将使用默认目标属性 'mu_change'")
            
            # 表格数据行
            prop = target_property
            # 根据预测模式确定主次预测值
            if args.prediction_mode == 'standardized':
                std_value = primary_pred.get(prop, "N/A") if primary_pred else "N/A"
                orig_value = secondary_pred.get(prop, "N/A") if secondary_pred else "N/A"
            else:  # denormalized
                orig_value = primary_pred.get(prop, "N/A") if primary_pred else "N/A"
                std_value = secondary_pred.get(prop, "N/A") if secondary_pred else "N/A"
                
            # 保留完整的属性名，不再去掉_change后缀
            prop_name = prop
            # 安全格式化输出，处理非数字类型的值
            if secondary_pred:
                # 格式化std_value
                if isinstance(std_value, (int, float)):
                    std_value_str = f"{std_value:<12.4f}"
                else:
                    std_value_str = f"{str(std_value):<12}"
                
                # 格式化orig_value
                if isinstance(orig_value, (int, float)):
                    orig_value_str = f"{orig_value:<12.4f}"
                else:
                    orig_value_str = f"{str(orig_value):<12}"
                
                print(f"{prop_name:<15} {std_value_str} {orig_value_str}")
            else:
                # 格式化orig_value
                if isinstance(orig_value, (int, float)):
                    orig_value_str = f"{orig_value:<12.4f}"
                else:
                    orig_value_str = f"{str(orig_value):<12}"
                
                print(f"{prop_name:<15} {'N/A':<12} {orig_value_str}")
                
            if not secondary_pred:
                print("\n注意: 模型未使用标准化数据进行训练")
        
        # TAG 执行预测值与真实值对比（支持多模型）  # TODO to-check
        elif comparison_prediction:
            if 'model_paths' in locals():
                # 多模型对比
                predictions_list, true_values, row = compare_with_ground_truth(
                    model_paths, model_dirs, args.csv_file, args.row_index, args.prediction_mode
                )
                print_multi_model_comparison_results(predictions_list, true_values, row, model_dirs, args.prediction_mode)
            else:
                # 单模型对比
                predictions_list, true_values, row = compare_with_ground_truth(
                    [args.model_path], [args.model_dir], args.csv_file, args.row_index, args.prediction_mode
                )
                # 对于单模型，predictions_list应该只包含一个元素
                predicted_values = predictions_list[0] if predictions_list else {}
                print_comparison_results(predicted_values, true_values, row, args.model_dir, args.prediction_mode)
        
        # TAG 执行批量预测
        elif batch_prediction:
            logger.info(f"执行批量预测")
            logger.info(f"加载模型: {args.model_path}")
            
            try:
                # 根据文件类型选择不同的预测函数
                if args.json_file:
                    logger.info(f"使用JSON数据集: {args.json_file}")
                    if args.use_test_indices:
                        logger.info(f"使用test_indices过滤数据，索引文件: {args.indices_file}")
                    
                    # 执行JSON批量预测
                    error_stats, sampled_df = batch_predict_json(
                        args.model_path,
                        args.model_dir,
                        args.json_file,
                        args.indices_file,
                        args.use_test_indices,
                        args.num_samples,
                        args.random_seed,
                        args.prediction_mode,
                        logger,
                        sample_method=args.sample_method
                    )
                else:
                    # 执行CSV批量预测
                    logger.info(f"使用CSV数据集: {args.csv_file}")
                    error_stats, sampled_df = batch_predict(
                        args.model_path,
                        args.model_dir,
                        args.csv_file,
                        args.num_samples,
                        args.random_seed,
                        args.prediction_mode,
                        logger,
                        sample_method=args.sample_method
                    )
                
                if error_stats:
                    print_error_statistics(error_stats, logger)
                    
                    # 保存预测结果
                    output_file = save_prediction_results(
                        error_stats, sampled_df,
                        args.model_path, args.model_dir,
                        args.csv_file or args.json_file,
                        args.num_samples,
                        args.random_seed, args.prediction_mode,
                        args.log_level
                    )
                    
                    logger.info(f"预测结果已保存到: {output_file}")
                
                logger.info(f"批量预测完成！")
                
            except Exception as e:
                logger.error(f"批量预测失败: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
            
    except Exception as e:
        logger.error(f"预测过程中发生错误: {e}")
        raise


def batch_predict_json(model_path, model_dir, json_file, indices_file=None, use_test_indices=False, 
                      num_samples=10, random_seed=42, prediction_mode='denormalized', logger=None, 
                      save_intermediates=True, sample_method='random'):
    """
    从JSON文件批量预测并计算误差，支持索引过滤
    
    Args:
        model_path: 模型文件路径
        model_dir: 模型目录路径
        json_file: JSON文件路径
        indices_file: 索引文件路径
        use_test_indices: 是否使用test_indices进行过滤
        num_samples: 采样数量
        random_seed: 随机种子
        prediction_mode: 预测模式
        logger: 日志记录器
        save_intermediates: 是否保存中间结果
        sample_method: 采样方法 ('random' 随机采样, 'sequential' 顺序采样)
        
    Returns:
        预测结果和误差统计
    """
    # 获取目标属性名称
    target_prop = get_target_property(model_dir)
    
    # 读取JSON数据
    if logger:
        logger.info(f"读取JSON文件: {json_file}")
    
    try:
        with open(json_file, 'r') as f:
            data_list = json.load(f)
    except Exception as e:
        if logger:
            logger.error(f"读取JSON文件失败: {e}")
        raise
    
    # 如果需要使用test_indices进行过滤
    filtered_indices = None
    if use_test_indices and indices_file:
        if logger:
            logger.info(f"使用test_indices过滤数据，索引文件: {indices_file}")
        try:
            with open(indices_file, 'r') as f:
                indices_data = json.load(f)
                filtered_indices = set(indices_data.get('test_indices', []))
                if logger:
                    logger.info(f"加载了 {len(filtered_indices)} 个测试索引")
        except Exception as e:
            if logger:
                logger.error(f"读取索引文件失败: {e}")
            raise
    
    # 过滤数据
    if filtered_indices:
        filtered_data = []
        for idx, item in enumerate(data_list):
            if idx in filtered_indices:
                filtered_data.append(item)
        data_list = filtered_data
        if logger:
            logger.info(f"过滤后的数据量: {len(data_list)}")
    
    # 转换为DataFrame便于处理
    df = pd.DataFrame(data_list)
    
    # 采样方式选择
    if len(df) > num_samples:
        # 根据sample_method参数选择采样策略
        if sample_method == 'sequential':
            # 按顺序采样，取前num_samples个样本
            df = df.iloc[:num_samples].reset_index(drop=True)
        else:
            # 随机采样（默认方式）
            df = df.sample(n=num_samples, random_state=random_seed).reset_index(drop=True)
    
    # 初始化预测统计结果
    predictions = []
    true_values = []
    error_stats = []
    
    # 用于保存中间输入、输出和标签的列表
    intermediate_results = []
    
    # 提前加载模型（避免重复加载）
    if logger:
        logger.info(f"加载模型: {model_path}")
    # 从导入的模块中使用load_model函数
    from mol_evo.utils.predict.prediction_utils import load_model
    model = load_model(model_path, model_dir)
    
    # 逐行预测
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="批量预测(JSON)"):
        try:
            # 从operations中提取atom信息
            operations = row.get('operations', [])
            if not operations:
                if logger:
                    logger.warning(f"行 {idx} 没有operations信息，跳过")
                continue
            
            # 获取第一个operation的atom作为to_atom_symbol
            to_atom_symbol = operations[0].get('atom', '')
            operation_type = operations[0].get('operation', '')
            
            # 进行预测 - 传入已加载的模型实例
            primary_pred, secondary_pred = predict_property_changes(
                model_path, model_dir,
                row['smiles_from'], row['smiles_to'],
                to_atom_symbol, operation_type,
                prediction_mode,
                model=model  # 传入已加载的模型实例
            )
            
            # 保存中间结果（输入、输出和标签）
            intermediate_data = {
                'index': idx,
                'input': {
                    'smiles_from': row['smiles_from'],
                    'smiles_to': row['smiles_to'],
                    'to_atom_symbol': to_atom_symbol,
                    'operation_type': operation_type
                },
                'output': {
                    'primary_pred': primary_pred,
                    'secondary_pred': secondary_pred
                },
                'label': row.get(target_prop, None)
            }
            intermediate_results.append(intermediate_data)
            
            # 获取预测值和真实值
            pred_value = primary_pred.get(target_prop, 0.0)
            true_value = row.get(target_prop, None)
            
            if true_value is not None:
                # 计算误差
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
                predictions.append(pred_value)
                true_values.append(true_value)
            
        except Exception as e:
            if logger:
                logger.error(f"处理行 {idx} 时出错: {e}")
            continue
    
    # 如果有预测结果，计算综合统计
    if predictions and true_values:
        # 转换为numpy数组
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        
        # 计算各种误差指标
        mse = np.mean((predictions - true_values) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - true_values))
        
        # 计算R²
        if len(predictions) > 1:
            ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
            ss_res = np.sum((true_values - predictions) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            r2 = 0
        
        # 计算PCC (Pearson相关系数)
        if len(predictions) > 1:
            pcc = np.corrcoef(predictions, true_values)[0, 1] if np.std(predictions) > 0 and np.std(true_values) > 0 else 0
        else:
            pcc = 0
        
        # 计算rank loss
        rank_loss = 0
        
        # 计算相对误差百分比
        relative_errors = np.abs((predictions - true_values) / (true_values + 1e-8)) * 100
        mean_relative_error = np.mean(relative_errors)
        
        # 构建完整的误差统计结果
        error_stats = {
            'property_names': [target_prop],
            'mse': np.array([mse]),
            'rmse': np.array([rmse]),
            'mae': np.array([mae]),
            'r2': np.array([r2]),
            'pcc': np.array([pcc]),
            'rank_loss': np.array([rank_loss]),
            'relative_error_percent': np.array([mean_relative_error]),
            'pred_mean': np.array([np.mean(predictions)]),
            'target_mean': np.array([np.mean(true_values)]),
            'pred_std': np.array([np.std(predictions)]),
            'target_std': np.array([np.std(true_values)]),
            'num_samples': len(predictions),
            'prediction_mode': prediction_mode
        }
    
    # 保存预测结果到文件，获取保存目录
    prediction_output_file = save_prediction_results(
        error_stats, df,
        model_path, model_dir,
        json_file, num_samples,
        random_seed, prediction_mode,
        logger.level if logger else logging.INFO
    )
    
    # 从预测结果文件路径中提取目录
    output_dir = os.path.dirname(prediction_output_file)
    
    # 保存中间结果到JSON文件
    if save_intermediates and intermediate_results:
        # 生成文件名
        output_file = os.path.join(output_dir, f'intermediate_results.json')
        
        # 转换numpy类型为Python原生类型以支持JSON序列化
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # 保存为JSON文件
        try:
            # 转换中间结果中的numpy类型
            serializable_results = convert_numpy_types(intermediate_results)
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            if logger:
                logger.info(f"中间结果已保存到: {output_file}")
        except Exception as e:
            if logger:
                logger.error(f"保存中间结果失败: {e}")
    
    return error_stats, df


def run4debug():
    """
    使用硬编码参数运行预测器，方便调试
    示例用法，参数来自:
    python mol_evo/predict_v0.py \
      --model-path /home/rhj/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200/last.pth \
      --model-dir /home/rhj/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200 \
      --smiles-from 'CC' \
      --smiles-to 'CO' \
      --atom-symbol '0' \
      --operation-type 'replace_atom'
    """
    # 硬编码的参数
    class Args:
        def __init__(self):
            self.model_path = "/home/rhj/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200/last.pth"
            self.model_dir = "/home/rhj/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200"
            self.smiles_from = "CC"
            self.smiles_to = "CO"
            self.atom_symbol = "0"
            self.operation_type = "replace_atom"
            self.csv_file = "mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv"
            self.json_file = "mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.json"
            self.indices_file = "mol_evo/dataset/data/dataset_indices/indices_20251127_122155_seed42.json"
            self.use_test_indices = False
            self.row_index = None
            self.num_samples = 100
            self.random_seed = 42
            self.log_level = "INFO"
            self.prediction_mode = "denormalized"
            self.config_file = "/home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml"   # UPDATE 这个或许得跟 model 本身的训练关联起来，然后根据 model-path 方便的链接；
    
    args = Args()
    
    # 设置日志记录器
    import logging
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logger(log_level)
    
    # 加载操作配置，初始化原子类型和操作类型
    logger.info(f"加载配置文件: {args.config_file}")
    load_operation_config(config_path=args.config_file)
    
    # 在加载配置后验证操作类型
    if args.operation_type:
        valid_operations = get_operation_types()
        if args.operation_type not in valid_operations:
            logger.error(f"无效的操作类型: {args.operation_type}")
            logger.error(f"有效操作类型: {', '.join(valid_operations)}")
            sys.exit(1)
    
    # 检查是单次预测还是批量预测
    single_prediction = args.smiles_from and args.smiles_to and args.atom_symbol and args.operation_type
    batch_prediction = (args.csv_file or args.json_file) and args.row_index is None
    comparison_prediction = args.csv_file and args.row_index is not None
    
    if not single_prediction and not batch_prediction and not comparison_prediction:
        logger.error("请指定单次预测参数或批量预测参数")
        return
    
    try:
        # 检查模型目录中是否包含训练数据
        if hasattr(args, 'model_dir') and args.model_dir:
            stats_file = os.path.join(args.model_dir, "training_data.json")
            if not os.path.exists(stats_file):
                logger.warning(f"模型目录中未找到训练数据文件 {stats_file}")
                logger.warning("将无法进行反标准化以获得原始尺度的预测值")
        
        # 执行单次预测
        if single_prediction:
            primary_pred, secondary_pred = predict_property_changes(
                args.model_path, args.model_dir,
                args.smiles_from, args.smiles_to,
                args.atom_symbol, args.operation_type,
                args.prediction_mode
            )
            
            mode_text = "标准差" if args.prediction_mode == 'standardized' else "反标准化"
            print(f"\n{mode_text}预测结果:")
            print("=" * 50)
            print(f"起始分子 SMILES: {args.smiles_from}")
            print(f"目标分子 SMILES: {args.smiles_to}")
            print(f"变化原子类型: {args.atom_symbol}")
            print(f"操作类型: {args.operation_type}")
            print(f"使用模型: {os.path.basename(args.model_dir)}")
            print(f"预测模式: {mode_text}")
            print("=" * 50)
            
            # 创建表格展示预测结果
            print("\n预测结果汇总:")
            # 表头
            print(f"{'属性名称':<15} {'标准化值':<12} {'原始值':<12}")
            print("-" * 45)
            
            # 从模型配置文件中读取目标属性
            import json
            target_property = 'mu_change'  # 默认值
            try:
                config_file = os.path.join(args.model_dir, 'model_config.json')
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                        if 'model-train-config' in config_data and 'target_property' in config_data['model-train-config']:
                            target_property = config_data['model-train-config']['target_property']
                            # logger.info(f"从模型配置文件中读取到目标属性: {target_property}")
            except Exception as e:
                logger.warning(f"读取模型配置文件失败: {e}，将使用默认目标属性 'mu_change'")
            
            # 表格数据行
            prop = target_property
            # 根据预测模式确定主次预测值
            if args.prediction_mode == 'standardized':
                std_value = primary_pred.get(prop, "N/A") if primary_pred else "N/A"
                orig_value = secondary_pred.get(prop, "N/A") if secondary_pred else "N/A"
            else:  # denormalized
                orig_value = primary_pred.get(prop, "N/A") if primary_pred else "N/A"
                std_value = secondary_pred.get(prop, "N/A") if secondary_pred else "N/A"
                
            prop_name = prop.replace("_change", "") if prop.endswith("_change") else prop
            # 安全格式化输出，处理非数字类型的值
            if secondary_pred:
                # 格式化std_value
                if isinstance(std_value, (int, float)):
                    std_value_str = f"{std_value:<12.4f}"
                else:
                    std_value_str = f"{str(std_value):<12}"
                
                # 格式化orig_value
                if isinstance(orig_value, (int, float)):
                    orig_value_str = f"{orig_value:<12.4f}"
                else:
                    orig_value_str = f"{str(orig_value):<12}"
                
                print(f"{prop_name:<15} {std_value_str} {orig_value_str}")
            else:
                # 格式化orig_value
                if isinstance(orig_value, (int, float)):
                    orig_value_str = f"{orig_value:<12.4f}"
                else:
                    orig_value_str = f"{str(orig_value):<12}"
                
                print(f"{prop_name:<15} {'N/A':<12} {orig_value_str}")
                
            if not secondary_pred:
                print("\n注意: 模型未使用标准化数据进行训练")
        
        # 执行预测值与真实值对比
        elif comparison_prediction:
            predictions_list, true_values, row = compare_with_ground_truth(
                [args.model_path], [args.model_dir], args.csv_file, args.row_index, args.prediction_mode
            )
            # 对于单模型，predictions_list应该只包含一个元素
            predicted_values = predictions_list[0] if predictions_list else {}
            print_comparison_results(predicted_values, true_values, row, args.model_dir, args.prediction_mode)
        
        # 执行批量预测
        elif batch_prediction:
            error_stats, sampled_df = batch_predict(
                args.model_path, args.model_dir,
                args.csv_file, args.num_samples, args.random_seed, args.prediction_mode, logger
            )
            
            if error_stats:
                print_error_statistics(error_stats, logger)
                
                # 保存预测结果
                output_file = save_prediction_results(
                    error_stats, sampled_df,
                    args.model_path, args.model_dir,
                    args.csv_file, args.num_samples,
                    args.random_seed, args.prediction_mode,
                    args.log_level
                )
                
                logger.info(f"预测结果已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"预测过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    # 检查是否有命令行参数传入
    if len(sys.argv) > 1:
        # 如果有命令行参数，则调用main函数处理
        main()
    else:
        # 如果没有命令行参数，则调用run4debug函数进行调试
        run4debug()