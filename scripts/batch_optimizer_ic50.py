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
import signal
import multiprocessing

# 全局中断标志和当前进程跟踪
interrupted = False
current_process = None

# 信号处理函数
def signal_handler(sig, frame):
    """处理中断信号"""
    global interrupted, current_process
    if interrupted:
        # 如果已经中断过，直接退出
        print("再次收到中断信号，立即退出！")
        sys.exit(0)
    
    print("\n收到中断信号，正在终止处理过程...")
    interrupted = True
    
    # 终止当前正在运行的子进程
    if current_process is not None and current_process.is_alive():
        print(f"正在终止子进程 (PID: {current_process.pid})...")
        current_process.terminate()
        try:
            current_process.join(timeout=5)
            if current_process.is_alive():
                print("子进程未响应，强制终止...")
                current_process.kill()
                current_process.join()
        except Exception as e:
            print(f"终止子进程时出错: {e}")
    
    print("处理已终止。")
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)

# 禁用RDKit的警告信息
RDLogger.DisableLog('rdApp.*')

# 设置项目根目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
sys.path.insert(0, project_root)

try:
    # 导入必要的模块
    from mol_evo.core.evolution_optimizer_ic50 import EvolutionTreeOptimizer
except ImportError as e:
    print(f"无法导入必要的模块: {e}")
    traceback.print_exc()
    sys.exit(1)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='批量优化分子属性')
    parser.add_argument('--input-json', type=str,
                        default='/home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/data/gdcsv2/ic50_result_dict_20_30.json',
                        help='输入的JSON文件路径')
    parser.add_argument('--cell-name', type=str,
                        required=True,
                        help='细胞名称，用于从JSON文件中获取待测试的目标')
    parser.add_argument('--output-json', type=str,
                        help='输出的JSON文件路径')
    parser.add_argument('--model-path', type=str,
                        default='/home/rhj/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200/last.pth',
                        help='模型文件路径')
    parser.add_argument('--model-dir', type=str,
                        default='/home/rhj/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200',
                        help='模型目录路径')
    parser.add_argument('--config-file', type=str,
                        default='/home/rhj/projects/mol_opt/mol-ofo/mol_evo/core/drp_ic50_p/ic50.yaml',
                        help='配置文件路径')
    parser.add_argument('--target-property', type=str, default='ic50',
                        help='目标属性名称')
    parser.add_argument('--optimization-mode', type=str, choices=['sub', 'pct'], default='sub',
                        help='优化模式')
    parser.add_argument('--max-depth', type=int, default=2,
                        help='最大演化深度')
    parser.add_argument('--max-branching', type=int, default=8,
                        help='最大分支数')
    parser.add_argument('--direction', type=str, choices=['increase', 'decrease'], default='decrease',
                        help='优化方向')
    parser.add_argument('--pruning-patience', type=int, default=2,
                        help='剪枝耐心值')
    parser.add_argument('--logp-min', type=float, default=0.0,
                        help='logP的最小值')
    parser.add_argument('--logp-max', type=float, default=5.0,
                        help='logP的最大值')
    parser.add_argument('--logp-patience', type=int, default=3,
                        help='logP剪枝耐心值，连续多少代logP超出范围就剪枝')
    parser.add_argument('--topK', type=int, default=100,
                        help='保留效果最好的K个结果')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='批处理大小')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='从指定的输出目录恢复之前的运行，继续处理未完成的分子')
    return parser.parse_args()

def create_output_dir():
    """创建输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # INFO 这里默认的运行根路径的上一层。
    base_dir = os.path.join(os.getcwd(), "mol_evo", "output", "evo-mo", f"batch_optimization_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def read_json_data(json_path, cell_name, target_property='ic50'):
    """从JSON文件读取数据"""
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
    
    if cell_name not in data_dict:
        raise ValueError(f"未找到细胞名称: {cell_name}")
    
    data_list = data_dict[cell_name]
    
    result_list = []
    for item in data_list:
        smiles = item['smiles']
        property_value = item[target_property]
        
        result_list.append({
            'smiles': smiles,
            'property_value': property_value,
            'original_row': item
        })
    return result_list

def load_processed_results(resume_dir):
    """从恢复目录加载已处理的结果"""
    processed_smiles = set()
    results_dict = {}
    
    if not os.path.exists(resume_dir):
        return processed_smiles, results_dict
    
    # 查找所有JSON结果文件
    json_files = [f for f in os.listdir(resume_dir) if f.endswith('.json') and not f.endswith('_topK.csv')]
    
    for json_file in json_files:
        json_path = os.path.join(resume_dir, json_file)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # 提取SMILES（从initial_smiles字段）
            if 'initial_smiles' in data:
                smiles = data['initial_smiles']
                processed_smiles.add(smiles)
                
                # 尝试从batch_results.json加载完整结果
                batch_results_path = os.path.join(resume_dir, 'batch_results.json')
                if os.path.exists(batch_results_path):
                    with open(batch_results_path, 'r') as f:
                        batch_results = json.load(f)
                    if smiles in batch_results:
                        results_dict[smiles] = batch_results[smiles]
        except Exception as e:
            print(f"警告: 无法加载结果文件 {json_file}: {e}")
    
    return processed_smiles, results_dict

def _run_optimizer_in_process(args_tuple, result_queue):
    """在子进程中运行优化器（重新创建optimizer实例）"""
    try:
        args, smiles, property_value, output_dir = args_tuple
        
        from mol_evo.core.evolution_optimizer_ic50 import EvolutionTreeOptimizer
        
        optimizer = EvolutionTreeOptimizer(
            args.model_path,
            args.model_dir,
            args.config_file,
            None,
            args.target_property,
            None,
            args.optimization_mode
        )
        optimizer.optimization_direction = args.direction
        
        result = run_evolution_optimizer(optimizer, smiles, property_value, args, output_dir)
        result_queue.put(result)
    except Exception as e:
        result_queue.put({
            'status': 'error',
            'smiles': smiles,
            'initial_property': property_value,
            'error': f"Process error: {str(e)}"
        })

def run_evolution_optimizer_with_timeout(smiles, property_value, args, output_dir, timeout_seconds=2400):
    """运行分子优化，带超时控制（默认40分钟）"""
    global current_process
    
    print(f"\n=== 开始处理分子: {smiles} (超时限制: {timeout_seconds/60:.1f}分钟) ===")
    
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    ctx = multiprocessing.get_context('spawn')
    process = ctx.Process(
        target=_run_optimizer_in_process,
        args=((args, smiles, property_value, output_dir), result_queue)
    )
    
    # 设置全局当前进程引用，以便信号处理器可以访问
    current_process = process
    
    process.start()
    process.join(timeout=timeout_seconds)
    
    # 清除全局进程引用
    current_process = None
    
    if process.is_alive():
        print(f"处理超时 ({timeout_seconds/60:.1f}分钟)，终止进程: {smiles}")
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        
        manager.shutdown()
        
        return {
            'status': 'timeout',
            'smiles': smiles,
            'initial_property': property_value,
            'error': f'Optimization timeout after {timeout_seconds/60:.1f} minutes'
        }
    
    if result_queue.empty():
        manager.shutdown()
        return {
            'status': 'error',
            'smiles': smiles,
            'initial_property': property_value,
            'error': 'Process completed but no result returned'
        }
    
    result = result_queue.get()
    manager.shutdown()
    return result

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
            pruning_patience=args.pruning_patience,
            logp_range=(args.logp_min, args.logp_max),
            logp_patience=args.logp_patience
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

def batch_process(data_list, args, output_dir, resume_mode=False):
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
    
    # 如果是恢复模式，加载已处理的结果
    processed_smiles = set()
    if resume_mode:
        processed_smiles, results_dict = load_processed_results(output_dir)
        print(f"恢复模式: 已加载 {len(processed_smiles)} 个已处理的分子")
        
        # 记录到总日志
        with open(total_log_file, 'a') as f:
            f.write(f"\n=== 恢复模式: 从 {output_dir} 恢复 ===\n")
            f.write(f"已处理分子数: {len(processed_smiles)}\n")
            f.write(f"剩余待处理分子数: {total_count - len(processed_smiles)}\n")
    
    for i, data in enumerate(data_list):
        smiles = data['smiles']
        property_value = data['property_value']
        
        # 跳过已处理的分子
        if smiles in processed_smiles:
            print(f"\n=== 跳过已处理的分子 ({i+1}/{total_count}): {smiles} ===")
            continue
        
        # 计算进度百分比
        progress = (i+1) / total_count * 100
        print(f"\n=== 处理进度: {i+1}/{total_count} ({progress:.1f}%) ===")
        
        # 记录到总日志
        with open(total_log_file, 'a') as f:
            f.write(f"\n=== 开始处理分子 {i+1}/{total_count}: {smiles} ===\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"初始属性值: {property_value}\n")
        
        # 运行优化，带超时控制（40分钟）
        result = run_evolution_optimizer_with_timeout(
            smiles, 
            property_value, 
            args, 
            output_dir,
            timeout_seconds=600 * 4         # 40 min
            # timeout_seconds=600 * 6 * 2   # 2 h
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
            elif result['status'] == 'timeout':
                f.write(f"超时信息: {result['error']}\n")
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
    
    # 检查是否为恢复模式
    resume_mode = args.resume_from is not None
    
    if resume_mode:
        # 恢复模式：使用指定的输出目录
        output_dir = args.resume_from
        if not os.path.exists(output_dir):
            print(f"错误: 恢复目录不存在: {output_dir}")
            sys.exit(1)
        print(f"恢复模式: 从 {output_dir} 恢复之前的运行")
    else:
        # 正常模式：创建新的输出目录
        output_dir = create_output_dir()
    
    # 创建主日志文件
    main_log_file = os.path.join(output_dir, "batch_optimization_main.log")
    
    print(f"=== 批量分子优化脚本 ===")
    print(f"输出目录: {output_dir}")
    print(f"主日志文件: {main_log_file}")
    print(f"读取JSON文件: {args.input_json}")
    print(f"细胞名称: {args.cell_name}")
    if resume_mode:
        print(f"模式: 恢复模式")
    else:
        print(f"模式: 正常模式")
    
    # 记录配置信息到主日志
    with open(main_log_file, 'a' if resume_mode else 'w') as f:
        if not resume_mode:
            f.write(f"=== 批量分子优化配置 ===\n")
            f.write(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        else:
            f.write(f"\n=== 恢复批量分子优化 ===\n")
            f.write(f"恢复时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"输入JSON文件: {args.input_json}\n")
        f.write(f"细胞名称: {args.cell_name}\n")
        f.write(f"模型路径: {args.model_path}\n")
        f.write(f"模型目录: {args.model_dir}\n")
        f.write(f"配置文件: {args.config_file}\n")
        f.write(f"目标属性: {args.target_property}\n")
        f.write(f"优化模式: {args.optimization_mode}\n")
        f.write(f"最大深度: {args.max_depth}\n")
        f.write(f"最大分支数: {args.max_branching}\n")
        f.write(f"优化方向: {args.direction}\n")
        f.write(f"剪枝耐心值: {args.pruning_patience}\n")
        f.write(f"topK值: {args.topK}\n")
        f.write("\n")
    
    # 读取数据
    data_list = read_json_data(args.input_json, args.cell_name, args.target_property)
    print(f"读取到 {len(data_list)} 个分子数据")
    
    # 更新主日志
    with open(main_log_file, 'a') as f:
        f.write(f"读取到 {len(data_list)} 个分子数据\n")
    
    # 执行批量处理
    start_time = time.time()
    # TAG core
    results_dict = batch_process(data_list, args, output_dir, resume_mode=resume_mode)
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