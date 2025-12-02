#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量调用evolution_optimizer.py的脚本
从CSV文件读取分子数据，批量优化并保存结果到字典
"""

import os
import sys
import json
import subprocess
import argparse
import time
from datetime import datetime
import pandas as pd
import uuid

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='批量优化分子属性')
    parser.add_argument('--input-csv', type=str, 
                        default='mol_evo/dataset/eval-data/qm9_optimization_lumo_pairs.csv',
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

def read_csv_data(csv_path, start_idx=0, end_idx=-1):
    """读取CSV数据"""
    df = pd.read_csv(csv_path)
    if end_idx > 0:
        df = df.iloc[start_idx:end_idx]
    else:
        df = df.iloc[start_idx:]
    
    data_list = []
    for _, row in df.iterrows():
        data_list.append({
            'smiles': row['A_smiles'],
            'property_value': row['A_lumo'],
            'original_row': row.to_dict()
        })
    return data_list

def run_evolution_optimizer(smiles, property_value, args, output_dir):
    """运行evolution_optimizer.py"""
    run_id = str(uuid.uuid4())[:8]
    
    cmd = [
        'python', '-m', 'mol_evo.core.evolution_optimizer',
        '--model-path', args.model_path,
        '--model-dir', args.model_dir,
        '--config-file', args.config_file,
        '--initial-smiles', smiles,
        '--initial-property-value', str(property_value),
        '--target-property', args.target_property,
        '--optimization-mode', args.optimization_mode,
        '--max-depth', str(args.max_depth),
        '--max-branching', str(args.max_branching),
        '--direction', args.direction,
        '--format', 'json',
        '--topK', str(args.topK)
    ]
    
    output_json = os.path.join(output_dir, f"{smiles[:20].replace('/', '_')}_{run_id}.json")
    cmd.extend(['--output-file', output_json])
    
    print(f"正在处理: {smiles}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        end_time = time.time()
        
        with open(output_json, 'r') as f:
            optimized_result = json.load(f)
        
        topk_csv = os.path.splitext(output_json)[0] + "_topK.csv"
        topk_results = []
        if os.path.exists(topk_csv):
            topk_df = pd.read_csv(topk_csv)
            topk_results = topk_df.to_dict('records')
        
        return {
            'status': 'success',
            'smiles': smiles,
            'initial_property': property_value,
            'optimized_result': optimized_result,
            'topk_results': topk_results,
            'runtime': end_time - start_time
        }
        
    except Exception as e:
        print(f"处理失败: {smiles}, 错误: {str(e)}")
        return {
            'status': 'error',
            'smiles': smiles,
            'initial_property': property_value,
            'error': str(e)
        }

def batch_process(data_list, args, output_dir):
    """批量处理数据"""
    results_dict = {}
    total_count = len(data_list)
    
    print(f"开始批量处理，共 {total_count} 个分子")
    
    for i, data in enumerate(data_list):
        print(f"处理进度: {i+1}/{total_count}")
        
        result = run_evolution_optimizer(
            data['smiles'], 
            data['property_value'], 
            args, 
            output_dir
        )
        
        results_dict[data['smiles']] = {
            'original_data': data['original_row'],
            'optimization_result': result
        }
        
        if (i+1) % args.batch_size == 0 or (i+1) == total_count:
            save_results(results_dict, args.output_json, output_dir)
    
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
    
    print(f"输出目录: {output_dir}")
    print(f"读取CSV文件: {args.input_csv}")
    
    data_list = read_csv_data(args.input_csv, args.start_index, args.end_index)
    print(f"读取到 {len(data_list)} 个分子数据")
    
    start_time = time.time()
    results_dict = batch_process(data_list, args, output_dir)
    end_time = time.time()
    
    final_output = save_results(results_dict, args.output_json, output_dir)
    
    success_count = sum(1 for r in results_dict.values() 
                       if r['optimization_result']['status'] == 'success')
    
    print(f"\n批量处理完成！")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"成功: {success_count}, 失败: {len(results_dict) - success_count}")
    print(f"结果文件: {final_output}")

if __name__ == "__main__":
    main()