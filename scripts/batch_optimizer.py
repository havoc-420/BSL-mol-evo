#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量调用evolution_optimizer.py的脚本
从CSV文件读取分子数据，批量优化并保存结果到字典
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
import pandas as pd
import uuid
import traceback
from rdkit import RDLogger

# 禁用RDKit的警告信息
RDLogger.DisableLog('rdApp.*')

# 设置项目根目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
sys.path.insert(0, project_root)

try:
    # 导入必要的模块
    from mol_evo.core.evolution_optimizer import EvolutionTreeOptimizer
except ImportError as e:
    print(f"无法导入必要的模块: {e}")
    traceback.print_exc()
    sys.exit(1)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='批量优化分子属性')
    parser.add_argument('--input-csv', type=str,
                        default='mol_evo/dataset/eval-data/qm9_test_molecules.csv',
                        help='输入的CSV文件路径')
    parser.add_argument('--output-json', type=str,
                        help='输出的JSON文件路径')
    parser.add_argument('--model-path', type=str,
                        default='/home/data2/rhj/project/mol_editor/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200/last.pth',
                        help='模型文件路径')
    parser.add_argument('--model-dir', type=str,
                        default='/home/data2/rhj/project/mol_editor/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200',
                        help='模型目录路径')
    parser.add_argument('--config-file', type=str,
                        default='/home/data2/rhj/project/mol_editor/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml',
                        help='配置文件路径')
    parser.add_argument('--target-property', type=str, default='lumo',
                        help='目标属性名称')
    parser.add_argument('--optimization-mode', type=str, choices=['sub', 'pct'], default='sub',
                        help='优化模式')
    parser.add_argument('--max-depth', type=int, default=2,
                        help='最大演化深度')
    parser.add_argument('--max-branching', type=int, default=8,
                        help='最大分支数')
    parser.add_argument('--direction', type=str, choices=['increase', 'decrease'], default='decrease',
                        help='优化方向')
    parser.add_argument('--topK', type=int, default=20,
                        help='保留效果最好的K个结果')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='批处理大小')
    parser.add_argument('--start-index', type=int, default=0,
                        help='起始索引')
    parser.add_argument('--end-index', type=int, default=-1,
                        help='结束索引，-1表示处理到文件末尾')
    return parser.parse_args()

def create_output_dir():
    """创建输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # INFO 这里默认的运行根路径的上一层。
    base_dir = os.path.join(os.getcwd(), "mol_evo", "output", "evo-mo", f"batch_optimization_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def read_csv_data(csv_path, start_idx=0, end_idx=-1, target_property='lumo'):
    """读取CSV数据"""
    df = pd.read_csv(csv_path)
    if end_idx > 0:
        df = df.iloc[start_idx:end_idx]
    else:
        df = df.iloc[start_idx:]
    
    data_list = []
    for _, row in df.iterrows():
        # 从smiles列获取分子结构
        smiles = row['smiles']
        # 根据目标属性动态获取属性值
        property_value = row[target_property]
        
        data_list.append({
            'smiles': smiles,
            'property_value': property_value,
            'original_row': row.to_dict()
        })
    return data_list

def run_evolution_optimizer(optimizer, smiles, property_value, args, output_dir):
    """运行分子优化"""
    run_id = str(uuid.uuid4())[:8]
    
    print(f"\n=== 开始处理分子: {smiles} ===")
    print(f"初始属性值: {property_value}")
    
    try:
        start_time = time.time()
        
        # 设置优化器的初始属性值
        optimizer.initial_property_value = property_value
        
        # 优化进化树
        optimized_tree = optimizer.optimize_evolution_tree(
            smiles,
            args.max_depth,
            args.max_branching,
            args.direction,
            pruning_patience=3  # 使用默认值
        )
        
        # 保存优化结果
        output_json = os.path.join(output_dir, f"{smiles[:20].replace('/', '_')}_{run_id}.json")
        output_file = optimizer.save_optimized_tree(optimized_tree, output_json, output_dir)
        
        # 处理topK结果
        topK_results = optimizer.get_topK_results(optimized_tree, args.topK)
        topk_csv = os.path.splitext(output_json)[0] + "_topK.csv"
        optimizer.save_topK_results_to_csv(topK_results, topk_csv, output_dir)
        
        # 读取保存的优化结果
        with open(output_json, 'r') as f:
            optimized_result = json.load(f)
        
        end_time = time.time()
        
        # 显示部分关键信息
        print(f"处理完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"找到 {len(topK_results)} 个topK结果")
        
        return {
            'status': 'success',
            'smiles': smiles,
            'initial_property': property_value,
            'optimized_result': optimized_result,
            'topk_results': topK_results,
            'runtime': end_time - start_time
        }
        
    except Exception as e:
        # 处理错误
        error_msg = str(e)
        print(f"处理失败: {smiles}, 错误: {error_msg}")
        
        return {
            'status': 'error',
            'smiles': smiles,
            'initial_property': property_value,
            'error': error_msg
        }

def batch_process(data_list, args, output_dir):
    """批量处理数据"""
    results_dict = {}
    total_count = len(data_list)
    
    # 创建总日志文件
    total_log_file = os.path.join(output_dir, "batch_optimization_total.log")
    
    print(f"=== 开始批量处理，共 {total_count} 个分子 ===")
    print(f"总日志文件: {total_log_file}")
    
    # 记录开始时间
    batch_start_time = time.time()
    
    # 初始化EvolutionTreeOptimizer实例
    optimizer = EvolutionTreeOptimizer(
        args.model_path,
        args.model_dir,
        args.config_file,
        None,  # initial_smiles_csv
        args.target_property,
        None,  # initial_property_value (will be set per molecule)
        args.optimization_mode
    )
    optimizer.optimization_direction = args.direction
    
    for i, data in enumerate(data_list):
        smiles = data['smiles']
        property_value = data['property_value']
        
        # 计算进度百分比
        progress = (i+1) / total_count * 100
        print(f"\n=== 处理进度: {i+1}/{total_count} ({progress:.1f}%) ===")
        
        # 记录到总日志
        with open(total_log_file, 'a') as f:
            f.write(f"\n=== 开始处理分子 {i+1}/{total_count}: {smiles} ===\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"初始属性值: {property_value}\n")
        
        # 运行优化
        result = run_evolution_optimizer(
            optimizer,
            smiles, 
            property_value, 
            args, 
            output_dir
        )
        
        # 更新结果字典
        results_dict[smiles] = {
            'original_data': data['original_row'],
            'optimization_result': result
        }
        
        # 记录到总日志
        with open(total_log_file, 'a') as f:
            f.write(f"状态: {result['status']}\n")
            if result['status'] == 'success':
                f.write(f"耗时: {result['runtime']:.2f} 秒\n")
                f.write(f"优化结果数量: {len(result['optimized_result'].get('results', [])) if 'optimized_result' in result else 0}\n")
                f.write(f"topK结果数量: {len(result['topk_results']) if 'topk_results' in result else 0}\n")
            else:
                f.write(f"错误信息: {result['error']}\n")
        
        # 定期保存结果（每5个分子保存一次）
        if (i+1) % 5 == 0 or (i+1) == total_count:
            save_path = save_results(results_dict, args.output_json, output_dir)
    
    # 记录总耗时
    batch_end_time = time.time()
    total_time = batch_end_time - batch_start_time
    
    with open(total_log_file, 'a') as f:
        f.write(f"\n=== 批量处理完成 ===\n")
        f.write(f"总耗时: {total_time:.2f} 秒\n")
        f.write(f"平均每个分子耗时: {total_time/total_count:.2f} 秒\n")
    
    return results_dict

def save_results(results_dict, output_json, output_dir):
    """保存结果到JSON文件"""
    if not output_json:
        output_json = os.path.join(output_dir, "batch_results.json")
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    return output_json

def main():
    """主函数"""
    args = parse_args()
    output_dir = create_output_dir()
    
    # 创建主日志文件
    main_log_file = os.path.join(output_dir, "batch_optimization_main.log")
    
    print(f"=== 批量分子优化脚本 ===")
    print(f"输出目录: {output_dir}")
    print(f"主日志文件: {main_log_file}")
    print(f"读取CSV文件: {args.input_csv}")
    
    # 记录配置信息到主日志
    with open(main_log_file, 'w') as f:
        f.write(f"=== 批量分子优化配置 ===\n")
        f.write(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"输入CSV文件: {args.input_csv}\n")
        f.write(f"模型路径: {args.model_path}\n")
        f.write(f"模型目录: {args.model_dir}\n")
        f.write(f"配置文件: {args.config_file}\n")
        f.write(f"目标属性: {args.target_property}\n")
        f.write(f"优化模式: {args.optimization_mode}\n")
        f.write(f"最大深度: {args.max_depth}\n")
        f.write(f"最大分支数: {args.max_branching}\n")
        f.write(f"优化方向: {args.direction}\n")
        f.write(f"topK值: {args.topK}\n")
        f.write(f"起始索引: {args.start_index}\n")
        f.write(f"结束索引: {args.end_index}\n")
        f.write("\n")
    
    # 读取数据
    data_list = read_csv_data(args.input_csv, args.start_index, args.end_index, args.target_property)
    print(f"读取到 {len(data_list)} 个分子数据")
    
    # 更新主日志
    with open(main_log_file, 'a') as f:
        f.write(f"读取到 {len(data_list)} 个分子数据\n")
    
    # 执行批量处理
    start_time = time.time()
    # TAG core
    results_dict = batch_process(data_list, args, output_dir)
    end_time = time.time()
    
    # 保存最终结果
    final_output = save_results(results_dict, args.output_json, output_dir)
    
    # 统计结果
    success_count = sum(1 for r in results_dict.values() 
                       if r['optimization_result']['status'] == 'success')
    failure_count = len(results_dict) - success_count
    
    # 更新主日志
    with open(main_log_file, 'a') as f:
        f.write(f"\n=== 批量处理完成 ===\n")
        f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {end_time - start_time:.2f} 秒\n")
        f.write(f"成功: {success_count}, 失败: {failure_count}\n")
        f.write(f"成功率: {success_count/len(results_dict)*100:.1f}%\n")
        f.write(f"结果文件: {final_output}\n")
    
    # 打印最终统计信息
    print(f"\n=== 批量处理完成！ ===")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"成功: {success_count}, 失败: {failure_count}")
    print(f"成功率: {success_count/len(results_dict)*100:.1f}%")
    print(f"结果文件: {final_output}")
    print(f"详细日志已保存到: {output_dir}")

if __name__ == "__main__":
    main()