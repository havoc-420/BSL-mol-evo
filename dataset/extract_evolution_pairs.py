import pandas as pd
from rdkit import Chem
import sys
import os
import argparse
import logging
from tqdm import tqdm
import json
import numpy as np
import inquirer
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import pickle
import time

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mol_evo.core.similarity import calculate_evolutionary_similarity
from mol_evo.core.evolver import MoleculeEvolverAnalysis

""" utils """

def load_dataset(file_path):
    """
    加载CSV数据集
    """
    return pd.read_csv(file_path)

def convert_types(obj):
    """
    递归转换numpy数据类型为Python原生类型，使其可以被JSON序列化
    """
    if isinstance(obj, dict):
        return {key: convert_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# 注意：旧的 `_convert_operation_dict` 函数已被删除。
# 因为 MoleculeEvolverAnalysis 的 `generate_path_dict()` 方法现在直接输出符合分析需求的标准格式，
# 所以该转换函数已不再需要。

def save_pairs_to_json(pairs, output_file, compact=False):
    """
    将配对结果保存到JSON文件
    
    Args:
        pairs: 配对结果列表
        output_file: 输出文件路径
        compact: 是否使用紧凑格式
    """
    if not pairs:
        print("未找到任何配对")
        return
        
    # 转换数据类型以确保可以被JSON序列化
    pairs = convert_types(pairs)
        
    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        if compact:
            # 紧凑格式，但保持一定的可读性
            f.write("[\n")
            for i, pair in enumerate(pairs):
                if i > 0:
                    f.write(",\n")
                json_str = json.dumps(pair, ensure_ascii=False, separators=(',', ':'))
                f.write("  " + json_str)
            f.write("\n]")
        else:
            # 标准格式，带缩进便于阅读
            json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(pairs)} 对分子到 {output_file}")


def preview_pairs_as_json(pairs, compact=False):
    """
    将配对结果以JSON格式预览到控制台
    
    Args:
        pairs: 配对结果列表
        compact: 是否使用紧凑格式
    """
    if not pairs:
        print("未找到任何配对")
        return
        
    # 转换数据类型以确保可以被JSON序列化
    pairs = convert_types(pairs)
        
    # 输出JSON格式到控制台
    if compact:
        # 紧凑格式，但保持一定的可读性
        print("[")
        for i, pair in enumerate(pairs):
            if i > 0:
                print(",")
            json_str = json.dumps(pair, ensure_ascii=False, separators=(',', ':'))
            print("  " + json_str)
        print("]")
    else:
        # 标准格式，带缩进便于阅读
        print(json.dumps(pairs, ensure_ascii=False, indent=2))


""" MAIN TASK """

def analyze_evolution_operation_dict(path1_dict, path2_dict):
    """
    基于结构化字典分析两个进化路径之间的差异
    优化：对于位置相同的操作，使用替换而不是先删后加
    """
    # 将操作转换为可比较的字符串形式
    def op_to_str(op):
        return f"{op.get('operation', '')}@{op.get('position', '')}@{op.get('atom', '')}"
    
    # 将路径转换为字符串集合
    set1_str = {op_to_str(op) for op in path1_dict}
    set2_str = {op_to_str(op) for op in path2_dict}
    
    # 找出path1中独有的操作（需要删除的）
    removed_ops_str = set1_str - set2_str
    # 找出path2中独有的操作（需要添加的）
    added_ops_str = set2_str - set1_str
    
    # 将字符串操作转换回字典形式
    removed_ops = [op for op in path1_dict if op_to_str(op) in removed_ops_str]
    added_ops = [op for op in path2_dict if op_to_str(op) in added_ops_str]
    
    operations = []
    
    # 查找可以优化为替换操作的情况
    replace_ops = []
    remaining_removed = []
    remaining_added = added_ops.copy()  # 初始化为所有添加操作
    
    # 检查每个被删除的操作是否可以与添加的操作配对成替换
    for removed_op in removed_ops:
        found_replace = False
        for added_op in added_ops:
            # 如果操作类型和位置相同，但原子不同，则可以替换
            if (removed_op.get('operation') == added_op.get('operation') and 
                removed_op.get('position') == added_op.get('position') and
                removed_op.get('atom') != added_op.get('atom')):
                
                # 创建替换操作
                replace_ops.append({
                    "position": added_op.get('position'),
                    "atom": added_op.get('atom'),
                    "operation": "replace_atom",
                    "from_atom": removed_op.get('atom')
                })
                found_replace = True
                # 从remaining_added中移除已配对的添加操作
                if added_op in remaining_added:
                    remaining_added.remove(added_op)
                break
        
        if not found_replace:
            remaining_removed.append(removed_op)
    
    # 修改：将需要删除的操作标记为remove而不是简单拼接
    # 组合操作序列：先删除，再替换，最后添加
    marked_removed_ops = []
    for op in remaining_removed:
        new_op = op.copy()
        new_op["operation"] = "remove_" + op.get("operation", "")
        marked_removed_ops.append(new_op)
    
    operations = marked_removed_ops + replace_ops + remaining_added
    
    return operations

def process_molecule_pair(args, evolver_cache_dict=None):
    """
    处理单个分子对的函数，用于多进程处理
    参数 args 是一个元组，包含 (task_id, i, j, smiles_n, smiles_m, step) 或 (i, j, smiles_n, smiles_m, step)
    """
    # 兼容两种格式的任务参数
    if len(args) == 6:
        # 新格式：包含task_id
        _, _, _, smiles_n, smiles_m, step = args
    elif len(args) == 5:
        # 旧格式：不包含task_id
        _, _, smiles_n, smiles_m, step = args
    else:
        return None
    
    mol_n = Chem.MolFromSmiles(smiles_n)
    mol_m = Chem.MolFromSmiles(smiles_m)
    
    if mol_n is None or mol_m is None:
        return None
    
    try:
        # 使用共享的evolver缓存避免重复创建MoleculeEvolverAnalysis实例
        if evolver_cache_dict is not None:
            # 从共享字典中获取或创建evolver实例
            if smiles_n in evolver_cache_dict:
                path_n_dict = evolver_cache_dict[smiles_n]
            else:
                evolver_n = MoleculeEvolverAnalysis(smiles_n)
                path_n_dict = evolver_n.get_full_path_dict()
                evolver_cache_dict[smiles_n] = path_n_dict
            
            if smiles_m in evolver_cache_dict:
                path_m_dict = evolver_cache_dict[smiles_m]
            else:
                evolver_m = MoleculeEvolverAnalysis(smiles_m)
                path_m_dict = evolver_m.get_full_path_dict()
                evolver_cache_dict[smiles_m] = path_m_dict
        else:
            # 没有共享缓存时，直接创建evolver实例
            evolver_n = MoleculeEvolverAnalysis(smiles_n)
            evolver_m = MoleculeEvolverAnalysis(smiles_m)
            
            # 获取结构化进化路径
            path_n_dict = evolver_n.get_full_path_dict()
            path_m_dict = evolver_m.get_full_path_dict()
        
        # 分析进化操作
        operation_result = analyze_evolution_operation_dict(path_n_dict, path_m_dict)
        
        # 检查操作序列长度是否等于step
        if len(operation_result) != step:
            return None
        
        # 按照指定顺序创建pair_data字典
        pair_data = {
            'smiles_from': smiles_n,
            'smiles_to': smiles_m,
            'operations': operation_result
        }
        
        return pair_data
        
    except Exception as e:
        return None


def process_molecule_pair_with_tracking(args, evolver_cache_dict=None, tracking_dict=None):
    """
    处理单个分子对的函数，用于多进程处理，同时支持进度跟踪
    参数 args 是一个元组，包含 (task_id, i, j, smiles_n, smiles_m, step) 或 (i, j, smiles_n, smiles_m, step)
    """
    # 兼容两种格式的任务参数
    if len(args) == 6:
        # 新格式：包含task_id
        task_id, _, _, smiles_n, smiles_m, step = args
    elif len(args) == 5:
        # 旧格式：不包含task_id
        _, _, smiles_n, smiles_m, step = args
        task_id = None
    else:
        return None
    
    mol_n = Chem.MolFromSmiles(smiles_n)
    mol_m = Chem.MolFromSmiles(smiles_m)
    
    if mol_n is None or mol_m is None:
        return None
    
    try:
        # 使用共享的evolver缓存避免重复创建MoleculeEvolverAnalysis实例
        if evolver_cache_dict is not None:
            # 从共享字典中获取或创建evolver实例
            if smiles_n in evolver_cache_dict:
                path_n_dict = evolver_cache_dict[smiles_n]
            else:
                evolver_n = MoleculeEvolverAnalysis(smiles_n)
                path_n_dict = evolver_n.get_full_path_dict()
                evolver_cache_dict[smiles_n] = path_n_dict
            
            if smiles_m in evolver_cache_dict:
                path_m_dict = evolver_cache_dict[smiles_m]
            else:
                evolver_m = MoleculeEvolverAnalysis(smiles_m)
                path_m_dict = evolver_m.get_full_path_dict()
                evolver_cache_dict[smiles_m] = path_m_dict
        else:
            # 没有共享缓存时，直接创建evolver实例
            evolver_n = MoleculeEvolverAnalysis(smiles_n)
            evolver_m = MoleculeEvolverAnalysis(smiles_m)
            
            # 获取结构化进化路径
            path_n_dict = evolver_n.get_full_path_dict()
            path_m_dict = evolver_m.get_full_path_dict()
        
        # 分析进化操作
        operation_result = analyze_evolution_operation_dict(path_n_dict, path_m_dict)
        
        # 检查操作序列长度是否等于step
        if len(operation_result) != step:
            return None
        
        # 按照指定顺序创建pair_data字典
        pair_data = {
            'smiles_from': smiles_n,
            'smiles_to': smiles_m,
            'operations': operation_result
        }
        
        # 如果提供了跟踪字典，则更新处理计数
        if tracking_dict is not None:
            with tracking_dict['lock']:
                tracking_dict['processed_count'].value += 1
                
        return pair_data
        
    except Exception as e:
        return None


def find_step_pairs(heavy_n_df, heavy_m_df, n_atoms, m_atoms, step, max_pairs=None, logger=None, preview_mode=False, 
                   checkpoint_file=None, start_index=0, from_heavy_atoms_checkpoint=None, to_heavy_atoms_checkpoint=None,
                   all_pairs_checkpoint=None, dataset_info=None, total_combinations=None, completed_combinations=None,
                   pairs_file=None):
    """
    寻找适合指定步数进化的SMILES对
    从n个重原子的分子到m个重原子的分子（包括n==m的情况）
    
    Args:
        heavy_n_df: 源分子数据集
        heavy_m_df: 目标分子数据集  
        n_atoms: 源分子重原子数
        m_atoms: 目标分子重原子数
        step: 编辑距离步数
        max_pairs: 最大配对数，达到此数量后停止
        logger: 日志记录器
        preview_mode: 预览模式，只提取少量数据用于预览
        checkpoint_file: 检查点文件路径，用于断点续传
        start_index: 开始索引，用于断点续传
        from_heavy_atoms_checkpoint: 起始重原子数，用于断点续传
        to_heavy_atoms_checkpoint: 目标重原子数，用于断点续传
        all_pairs_checkpoint: 用于恢复的已找到的配对
        dataset_info: 数据集信息，用于检查点保存
        total_combinations: 总组合数，用于进度跟踪
        completed_combinations: 已完成组合数，用于进度跟踪
        pairs_file: 配对结果文件路径
    
    Returns:
        配对结果列表
    """
    if logger is None:
        # 如果没有提供logger，创建一个默认的
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    
    pairs = []
    
    # 限制搜索范围以提高效率
    # 只检查前N个分子，避免计算量过大
    if preview_mode:
        search_limit_n = min(3, len(heavy_n_df))
        search_limit_m = min(3, len(heavy_m_df))
    else:
        # 全量搜索，不限制搜索范围
        search_limit_n = len(heavy_n_df)
        search_limit_m = len(heavy_m_df)
    
    if preview_mode:
        logger.info(f"正在从{search_limit_n}个{n_atoms}重原子分子和{search_limit_m}个{m_atoms}重原子分子中寻找编辑距离为{step}的配对...")
    
    # 准备所有需要处理的分子对
    tasks = []
    unique_smiles = set()  # 收集所有唯一的SMILES以进行预处理
    
    # 添加tqdm进度条以显示收集唯一SMILES的过程
    # 只有当我们正在处理与检查点相同的原子数组合时才使用start_index
    if from_heavy_atoms_checkpoint == n_atoms and to_heavy_atoms_checkpoint == m_atoms:
        start_i = start_index // search_limit_m  # 计算起始的i值
        start_j = start_index % search_limit_m   # 计算起始的j值
        logger.info(f"从索引 {start_index} (i={start_i}, j={start_j}) 开始处理{n_atoms}->{m_atoms}原子对")
    else:
        start_i = 0
        start_j = 0
        logger.info(f"开始处理{n_atoms}->{m_atoms}原子对")
    
    task_id = 0
    for i in tqdm(range(start_i, search_limit_n), desc=f"收集{n_atoms}原子分子", unit="mol"):
        smiles_n = heavy_n_df.iloc[i]['smiles']
        unique_smiles.add(smiles_n)
        for j in range(search_limit_m):
            # 如果是第一行(i==start_i)，则从start_j开始，否则从0开始
            if i == start_i and j < start_j:
                continue
            smiles_m = heavy_m_df.iloc[j]['smiles']
            unique_smiles.add(smiles_m)
            tasks.append((task_id, i, j, smiles_n, smiles_m, step))
            task_id += 1
    
    # 在预览模式下保持原有逻辑，便于调试
    if preview_mode:
        evolver_cache = {}
        # 使用tqdm显示进度条
        task_index = 0
        for i in tqdm(range(start_i, search_limit_n), desc=f"处理{n_atoms}→{m_atoms}原子对", unit="mol"):
            smiles_n = heavy_n_df.iloc[i]['smiles']
            mol_n = Chem.MolFromSmiles(smiles_n)
            
            if mol_n is None:
                continue
                
            # 内层循环也添加进度条
            inner_pbar = tqdm(range(search_limit_m), desc=f"内层循环", leave=False, unit="mol")
            for j in inner_pbar:
                # 如果是第一行(i==start_i)，则从start_j开始，否则从0开始
                if i == start_i and j < start_j:
                    continue
                    
                smiles_m = heavy_m_df.iloc[j]['smiles']
                mol_m = Chem.MolFromSmiles(smiles_m)
                
                if mol_m is None:
                    continue
                    
                # 更新内层进度条描述
                inner_pbar.set_description(f"比较 {i+1}:{smiles_n[:15]}... -> {j+1}:{smiles_m[:15]}...")
                    
                # 检查是否可以通过指定步数操作从smiles_n得到smiles_m
                try:
                    # 使用缓存避免重复创建MoleculeEvolverAnalysis实例
                    if smiles_n not in evolver_cache:
                        evolver_cache[smiles_n] = MoleculeEvolverAnalysis(smiles_n)
                    if smiles_m not in evolver_cache:
                        evolver_cache[smiles_m] = MoleculeEvolverAnalysis(smiles_m)
                    
                    evolver_n = evolver_cache[smiles_n]
                    evolver_m = evolver_cache[smiles_m]
                    
                    # 获取结构化进化路径
                    path_n_dict = evolver_n.get_full_path_dict()
                    path_m_dict = evolver_m.get_full_path_dict()
                    
                    # 分析进化操作
                    operation_result = analyze_evolution_operation_dict(path_n_dict, path_m_dict)
                    
                    # 检查操作序列长度是否等于step
                    if len(operation_result) != step:
                        continue
                    
                    # 按照指定顺序创建pair_data字典
                    pair_data = {
                        'smiles_from': smiles_n,
                        'smiles_to': smiles_m,
                        'operations': operation_result
                    }
                    
                    pairs.append(pair_data)
                    task_index += 1
                    
                    # 保存检查点
                    if checkpoint_file and task_index % 1000 == 0:  # 每1000个任务保存一次检查点
                        checkpoint_data = {
                            'processed_index': i * search_limit_m + j,
                            'from_heavy_atoms': n_atoms,
                            'to_heavy_atoms': m_atoms,
                            'pairs': all_pairs_checkpoint + pairs if all_pairs_checkpoint is not None else pairs,
                        }
                        with open(checkpoint_file, 'w') as f:
                            json.dump(checkpoint_data, f)
                    
                    # 检查是否达到最大配对数
                    if max_pairs and len(pairs) >= max_pairs:
                        logger.info(f"已达到最大配对数 {max_pairs}，停止搜索...")
                        inner_pbar.close()
                        return pairs
                            
                except Exception as e:
                    continue
            inner_pbar.close()
    else:
        # 非预览模式：使用多进程处理，并预处理常用的MoleculeEvolverAnalysis实例
        num_processes = min(cpu_count(), 8)  # 使用CPU核心数，但不超过8个
        logger.info(f"使用 {num_processes} 个进程进行并行计算")
        
        # 创建缓存文件路径
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'molecule_evolver_cache_step{step}.pkl')
        
        # 尝试加载现有的缓存文件
        evolver_cache_dict = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    evolver_cache_dict = pickle.load(f)
                logger.info(f"从缓存文件 {cache_file} 加载了 {len(evolver_cache_dict)} 个分子处理结果")
            except Exception as e:
                logger.warning(f"加载缓存文件时出错: {e}，将重新创建缓存")
                evolver_cache_dict = {}
        
        # 确定需要处理的新分子
        new_smiles = unique_smiles - set(evolver_cache_dict.keys())
        logger.info(f"总共 {len(unique_smiles)} 个唯一分子，其中 {len(new_smiles)} 个需要预处理")
        
        # 预处理所有新的唯一SMILES，创建MoleculeEvolverAnalysis实例并缓存它们的路径字典
        if new_smiles:
            for smiles in tqdm(new_smiles, desc="预处理新分子"):
                try:
                    evolver = MoleculeEvolverAnalysis(smiles)
                    path_dict = evolver.get_full_path_dict()
                    # 存储path_dict到缓存字典中
                    evolver_cache_dict[smiles] = path_dict
                except Exception as e:
                    logger.warning(f"预处理分子 {smiles} 时出错: {e}")
                    continue
            
            # 保存更新后的缓存到文件
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(evolver_cache_dict, f)
                logger.info(f"已将 {len(evolver_cache_dict)} 个分子处理结果保存到缓存文件 {cache_file}")
            except Exception as e:
                logger.warning(f"保存缓存文件时出错: {e}")
        
        # 创建一个管理器字典用于进程间共享evolver缓存
        manager = Manager()
        shared_evolver_cache_dict = manager.dict(evolver_cache_dict)
        
        # 使用partial固定evolver_cache_dict参数
        process_func = partial(process_molecule_pair, evolver_cache_dict=shared_evolver_cache_dict)
        
        # 分批处理任务以支持检查点
        batch_size = 2000  # 每批处理2000个任务
        all_results = []
        
        # 如果有起始索引，调整任务列表
        start_task_index = start_index if (from_heavy_atoms_checkpoint == n_atoms and to_heavy_atoms_checkpoint == m_atoms) else 0
        remaining_tasks = tasks[start_task_index:] if start_task_index < len(tasks) else []
        
        # 计算总批次数
        total_batches = (len(remaining_tasks) + batch_size - 1) // batch_size
        
        # 修改：使用批次索引而不是任务索引
        start_batch_index = start_index if (from_heavy_atoms_checkpoint == n_atoms and to_heavy_atoms_checkpoint == m_atoms) else 0
        # 从批次索引计算起始任务索引
        start_task_index = start_batch_index * batch_size
        remaining_tasks = tasks[start_task_index:] if start_task_index < len(tasks) else []
        # 重新计算总批次数
        total_batches = (len(remaining_tasks) + batch_size - 1) // batch_size
        
        with Pool(processes=num_processes) as pool:
            # 使用总批次数作为主进度条
            batch_pbar = tqdm(total=total_batches, desc=f"处理{n_atoms}→{m_atoms}原子对", unit="批")
            
            # 记录开始时间
            start_time = time.time()
            
            # 分批处理任务
            total_processed = start_task_index
            for i in range(0, len(remaining_tasks), batch_size):
                batch = remaining_tasks[i:i+batch_size]
                batch_num = i // batch_size + 1
                
                # 处理当前批次，显示子进度条
                batch_results = list(tqdm(pool.imap(process_func, batch), total=len(batch), 
                                         desc=f"批次 {batch_num}/{total_batches}", 
                                         leave=False, unit="pair"))
                
                # 过滤出有效的结果并添加到总结果中
                valid_results = [result for result in batch_results if result is not None]
                all_results.extend(valid_results)
                total_processed += len(batch)
                
                # 更新主进度条
                batch_pbar.update(1)
                
                # 每处理完一批，保存检查点（如果启用了检查点）
                # 只保存进度信息，不保存配对结果，减少I/O操作
                if checkpoint_file:
                    elapsed_time = time.time() - start_time
                    # 修改：保存批次索引而不是任务索引
                    batch_index = (total_processed - 1) // batch_size if total_processed > 0 else 0
                    checkpoint_info = {
                        'processed_index': batch_index,  # 保存批次索引
                        'from_heavy_atoms': n_atoms,
                        'to_heavy_atoms': m_atoms,
                        # 添加当前已找到的配对总数
                        'pairs_count': len(all_pairs_checkpoint) + len(all_results) if all_pairs_checkpoint is not None else len(all_results),
                        # 添加运行时间信息
                        'elapsed_time': elapsed_time,
                        'elapsed_time_formatted': time.strftime('%H:%M:%S', time.gmtime(elapsed_time)),
                        # 添加数据集信息
                        'dataset_info': dataset_info,
                        # 添加进度信息
                        'total_combinations': total_combinations,
                        'completed_combinations': completed_combinations,
                        # 添加当前组合信息
                        'current_combination': f"{n_atoms}->{m_atoms}",
                        # 添加批次信息
                        'total_batches': total_batches,
                        'current_batch': batch_num,
                        'batch_size': batch_size,
                        # 添加步骤信息
                        'step': step,
                        # 添加最大配对数限制
                        'max_pairs': max_pairs,
                        # 添加搜索范围限制
                        'search_limit_n': search_limit_n,
                        'search_limit_m': search_limit_m,
                        # 添加缓存文件路径信息
                        'cache_file': cache_file,
                        # 添加时间戳
                        'checkpoint_timestamp': time.time(),
                        'checkpoint_timestamp_formatted': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                    }
                    
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint_info, f, indent=2, ensure_ascii=False)
                    
                    # 同时更新配对结果文件，保持检查点和实际数据的一致性
                    try:
                        # 将当前批次的结果添加到临时文件中
                        temp_pairs = (all_pairs_checkpoint if all_pairs_checkpoint is not None else []) + all_results
                        if pairs_file:  # 只有当pairs_file被提供时才尝试写入
                            # 写入新的配对结果
                            with open(pairs_file, 'w') as f:
                                json.dump(temp_pairs, f)
                    except Exception as e:
                        logger.warning(f"更新配对结果文件时出错: {e}")
                
                # 检查是否达到最大配对数
                if max_pairs and len(all_results) >= max_pairs:
                    logger.info(f"已达到最大配对数 {max_pairs}，停止搜索...")
                    batch_pbar.close()
                    break
            
            batch_pbar.close()
        
        pairs = all_results
    
    # 如果设置了最大配对数，截取相应数量
    if max_pairs and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
                
    return pairs


# TAG Task-II
def debug_pair(smiles_from, smiles_to, logger=None):
    """
    调试特定的分子对，分析它们之间的进化操作
    
    Args:
        smiles_from: 起始分子的SMILES表示
        smiles_to: 目标分子的SMILES表示
        logger: 日志记录器
    """
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    
    logger.info(f"调试分子对: {smiles_from} -> {smiles_to}")
    
    mol_from = Chem.MolFromSmiles(smiles_from)
    mol_to = Chem.MolFromSmiles(smiles_to)
    
    if mol_from is None:
        logger.error(f"无法解析起始分子 SMILES: {smiles_from}")
        return None
        
    if mol_to is None:
        logger.error(f"无法解析目标分子 SMILES: {smiles_to}")
        return None
    
    try:
        # 计算进化相似度和路径
        distance, path_from, path_to = calculate_evolutionary_similarity(smiles_from, smiles_to)
        
        logger.info(f"编辑距离: {distance}")
        
        # 获取进化操作详情
        evolver_from = MoleculeEvolverAnalysis(smiles_from)
        evolver_to = MoleculeEvolverAnalysis(smiles_to)
        
        # 获取完整路径（包括起始操作）
        full_path_from_dict = evolver_from.get_full_path_dict()
        full_path_to_dict = evolver_to.get_full_path_dict()
        
        logger.info("起始分子进化路径（字符串形式）:")
        for i, op in enumerate(path_from):
            logger.info(f"  {i}: {op}")
            
        logger.info("目标分子进化路径（字符串形式）:")
        for i, op in enumerate(path_to):
            logger.info(f"  {i}: {op}")
        
        logger.info("起始分子完整进化路径（字典形式）:")
        for i, op in enumerate(full_path_from_dict):
            logger.info(f"  {i}: {op}")
            
        logger.info("目标分子完整进化路径（字典形式）:")
        for i, op in enumerate(full_path_to_dict):
            logger.info(f"  {i}: {op}")
        
        # 分析进化操作（基于包括起始操作的完整路径）
        operation_result = analyze_evolution_operation_dict(full_path_from_dict, full_path_to_dict)
        
        # 创建结果数据
        pair_data = {
            'smiles_from': smiles_from,
            'smiles_to': smiles_to,
            'distance': distance,
            'path_from': path_from,
            'path_to': path_to,
            'full_path_from_dict': full_path_from_dict,
            'full_path_to_dict': full_path_to_dict,
            'operations': operation_result
        }
        
        logger.info(f"操作详情:")
        for i, op in enumerate(operation_result):
            logger.info(f"  {i}: {op}")
            
        # 转换数据类型以确保可以被JSON序列化
        pair_data = convert_types(pair_data)
            
        return pair_data
        
    except Exception as e:
        logger.error(f"处理分子对时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


# TAG select task
def select_mode_interactively():
    """
    交互式选择模式
    """
    questions = [
        inquirer.List('mode',
                     message="请选择运行模式",
                     choices=[
                         ('全量处理并保存到文件', 'full'),
                         ('预览模式（仅输出到控制台）', 'preview'),
                         ('预览并保存到文件', 'preview_with_file'),
                         ('全量处理并保存到文件（紧凑格式）', 'full_compact'),
                         ('预览并保存到文件（紧凑格式）', 'preview_with_file_compact')
                     ])
    ]
    
    answers = inquirer.prompt(questions)
    return answers['mode'] if answers else None


def main(step=1, max_pairs=None, log_level=None, mode=None, debug_from=None, debug_to=None, num_processes=None, 
         resume=False):
    """
    主函数
    
    Args:
        step: 编辑距离步数，默认为1
        max_pairs: 最大配对数，达到此数量后停止搜索
        log_level: 日志等级
        mode: 运行模式
        debug_from: 调试模式下的起始分子SMILES
        debug_to: 调试模式下的目标分子SMILES
        resume: 是否从检查点恢复
    """
    # 检查是否是调试模式
    if debug_from is not None and debug_to is not None:
        logging.basicConfig(level=log_level, format='%(message)s')
        logger = logging.getLogger(__name__)
        logger.info("进入调试模式")
        result = debug_pair(debug_from, debug_to, logger)
        if result:
            logger.info("调试结果:")
            print(json.dumps([result], ensure_ascii=False, indent=2))
        else:
            logger.error("调试失败")
        return
    
    # 如果提供了mode参数为None（即--mode后没有跟值），则交互式选择
    if mode is None:
        mode = select_mode_interactively()
        if not mode:
            print("未选择模式，退出程序")
            sys.exit(1)
    
    # 解析对应的运行模式
    if mode == 'full':
        preview_mode = False
        preview_with_file = False
        compact = False
    elif mode == 'preview':
        preview_mode = True
        preview_with_file = False
        compact = False
    elif mode == 'preview_with_file':
        preview_mode = True
        preview_with_file = True
        compact = False
    elif mode == 'full_compact':
        preview_mode = False
        preview_with_file = False
        compact = True
    elif mode == 'preview_with_file_compact':
        preview_mode = True
        preview_with_file = True
        compact = True
    else:
        print(f"未知的模式: {mode}")
        return
    
    # 配置日志
    logging.basicConfig(level=log_level, format='%(message)s')
    logger = logging.getLogger(__name__)
    
    # 记录开始时间
    start_time = time.time()
    
    # 设置数据目录
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 定义所有文件路径 - 提前定义以确保全局可用
    output_file = os.path.join(data_dir, f'qm9-evo-pairs-step-{step}.json')
    checkpoint_file = os.path.join(data_dir, f'qm9-evo-pairs-step-{step}-checkpoint.json')
    pairs_file = os.path.join(data_dir, f'qm9-evo-pairs-step-{step}-pairs.json')  # 独立的配对结果文件
    
    # 检查是否已存在配对结果文件，如果存在则进行存档
    if not preview_mode and os.path.exists(pairs_file):
        try:
            # 导入存档函数
            from convert_pairs import convert_step2_to_step1
            # 调用转换函数进行存档
            convert_step2_to_step1(pairs_file, data_dir)
        except Exception as e:
            logger.warning(f"存档现有配对文件时出错: {e}")
    
    # 检查是否有检查点文件
    start_index = 0
    all_pairs = []
    from_heavy_atoms_checkpoint = None
    to_heavy_atoms_checkpoint = None
    last_pairs_count = 0  # 记录上一次保存的配对数量
    elapsed_time = 0  # 记录已运行时间
    dataset_info = {}  # 数据集信息
    total_combinations = 0  # 总组合数
    completed_combinations = 0  # 已完成组合数
    if resume and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                start_index = checkpoint_data.get('processed_index', 0) + 1
                # 从检查点文件中读取配对结果
                checkpoint_pairs = checkpoint_data.get('pairs', [])
                if checkpoint_pairs:
                    all_pairs.extend(checkpoint_pairs)
                    logger.info(f"从检查点文件恢复了 {len(checkpoint_pairs)} 对分子")
                from_heavy_atoms_checkpoint = checkpoint_data.get('from_heavy_atoms')
                to_heavy_atoms_checkpoint = checkpoint_data.get('to_heavy_atoms')
                last_pairs_count = checkpoint_data.get('pairs_count', 0) # 获取上一次的配对数量
                elapsed_time = checkpoint_data.get('elapsed_time', 0)  # 获取已运行时间
                dataset_info = checkpoint_data.get('dataset_info', {})  # 获取数据集信息
                total_combinations = checkpoint_data.get('total_combinations', 0)  # 获取总组合数
                completed_combinations = checkpoint_data.get('completed_combinations', 0)  # 获取已完成组合数
            logger.info(f"从检查点恢复，从批次 {start_index} 开始处理 ({from_heavy_atoms_checkpoint}->{to_heavy_atoms_checkpoint}原子对)")
            if elapsed_time > 0:
                logger.info(f"已运行时间: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
        except Exception as e:
            logger.warning(f"加载检查点文件时出错: {e}，将从头开始处理")
    
    # 加载所有数据集
    datasets = {}
    # 更新数据集信息用于检查点
    if resume and dataset_info:
        # 如果是恢复模式且检查点中有数据集信息，则使用检查点中的数据集大小信息
        # 但仍需要重新加载数据集本身
        logger.info("从检查点恢复数据集信息...")
        for i in range(1, 10):
            file_path = os.path.join(data_dir, f'qm9_smiles_heavy_{i}_atoms.csv')
            if os.path.exists(file_path):
                datasets[i] = load_dataset(file_path)
                logger.info(f"加载{i}重原子数据集: {len(datasets[i])} 个分子")
            else:
                logger.warning(f"警告: {file_path} 不存在")
                datasets[i] = pd.DataFrame(columns=['smiles'])
    else:
        # 正常加载数据集
        for i in range(1, 10):
            file_path = os.path.join(data_dir, f'qm9_smiles_heavy_{i}_atoms.csv')
            if os.path.exists(file_path):
                datasets[i] = load_dataset(file_path)
                logger.info(f"加载{i}重原子数据集...")
                logger.info(f"{i}重原子分子数量: {len(datasets[i])}")
            else:
                logger.warning(f"警告: {file_path} 不存在")
                datasets[i] = pd.DataFrame(columns=['smiles'])
        
        # 准备数据集信息用于检查点
        dataset_info = {
            'datasets': {i: len(datasets[i]) for i in datasets},
            'loaded_at': time.time()
        }

    # 计算总的组合数用于外层进度条（仅在非恢复模式下计算）
    if not (resume and total_combinations > 0):
        total_combinations = 0
        # 计算总的组合数用于外层进度条
        for from_heavy_atoms in range(1, 10):
            for to_heavy_atoms in range(from_heavy_atoms, min(from_heavy_atoms + step + 1, 10)):
                if from_heavy_atoms in datasets and to_heavy_atoms in datasets:
                    atom_diff = to_heavy_atoms - from_heavy_atoms
                    if atom_diff <= step:
                        total_combinations += 1
    
    # 外层进度条
    if preview_mode or preview_with_file:
        # 预览模式下不显示进度条
        pass
    else:
        outer_pbar = tqdm(total=total_combinations, desc="整体处理进度", unit="组合")
        if completed_combinations > 0:
            outer_pbar.update(completed_combinations)
            outer_pbar.set_description(f"整体处理进度 (已找到 {len(all_pairs)} 对)")
    
    # 统一处理所有可能的原子数对组合
    # 遍历所有可能的起始原子数
    resume_point_reached = False if resume and from_heavy_atoms_checkpoint is not None else True
    current_combination_index = 0
    for from_heavy_atoms in range(1, 10):
        # 遍历所有可能的目标原子数
        # 对于step=n的情况，我们考虑从from_heavy_atoms到from_heavy_atoms+n的所有可能
        for to_heavy_atoms in range(from_heavy_atoms, min(from_heavy_atoms + step + 1, 10)):
            current_combination_index += 1
            # 检查数据集是否存在
            if from_heavy_atoms in datasets and to_heavy_atoms in datasets:
                # 计算实际的原子数差值
                atom_diff = to_heavy_atoms - from_heavy_atoms
                
                # 只有当原子数差值在合理范围内时才处理
                # 1. 相邻原子数（差值为1）总是处理
                # 2. 相同原子数内部（差值为0）总是处理
                # 3. 跨越多个原子数（差值>1）只在step足够大时处理
                if atom_diff <= step:
                    # 如果我们处于恢复模式，但还没有到达恢复点，则跳过
                    if resume and not resume_point_reached:
                        if from_heavy_atoms == from_heavy_atoms_checkpoint and to_heavy_atoms == to_heavy_atoms_checkpoint:
                            resume_point_reached = True
                            # 重置起始索引，因为我们已经到达了恢复点
                            start_index = start_index if (from_heavy_atoms_checkpoint == from_heavy_atoms and to_heavy_atoms == to_heavy_atoms) else 0
                        else:
                            logger.info(f"跳过 {from_heavy_atoms}->{to_heavy_atoms} 原子对（恢复模式）")
                            completed_combinations += 1
                            if 'outer_pbar' in locals():
                                outer_pbar.update(1)
                            continue
                    
                    df_from = datasets[from_heavy_atoms]
                    df_to = datasets[to_heavy_atoms]
                    
                    # logger.info(f"查找{from_heavy_atoms}->{to_heavy_atoms}重原子的{step}步进化对...")
                    
                    # 查找指定步数进化对
                    pairs = find_step_pairs(df_from, df_to, from_heavy_atoms, to_heavy_atoms, step, 
                                          max_pairs=max_pairs, logger=logger, 
                                          preview_mode=(preview_mode or preview_with_file),
                                          checkpoint_file=checkpoint_file if not preview_mode else None,
                                          start_index=start_index if (from_heavy_atoms == from_heavy_atoms_checkpoint and to_heavy_atoms == to_heavy_atoms_checkpoint) else 0,
                                          from_heavy_atoms_checkpoint=from_heavy_atoms_checkpoint,
                                          to_heavy_atoms_checkpoint=to_heavy_atoms_checkpoint,
                                          all_pairs_checkpoint=all_pairs,
                                          dataset_info=dataset_info,
                                          total_combinations=total_combinations,
                                          completed_combinations=completed_combinations,
                                          pairs_file=pairs_file if not preview_mode else None)
                    all_pairs.extend(pairs)
                    
                    # 保存配对结果到独立文件
                    if not preview_mode:
                        try:
                            with open(pairs_file, 'w') as f:
                                json.dump(all_pairs, f)
                            logger.info(f"保存 {len(all_pairs)} 对分子到 {pairs_file}")
                        except Exception as e:
                            logger.warning(f"保存配对结果到文件时出错: {e}")
                    
                    # 重置起始索引，以便下一个组合从头开始
                    start_index = 0
                    
                    if preview_mode or preview_with_file:
                        # 预览模式下找到3个配对就停止
                        if len(all_pairs) >= 3:
                            logger.info("预览模式：已找到3个配对，停止搜索...")
                            break
                    else:
                        completed_combinations += 1
                        if 'outer_pbar' in locals():
                            # 更新外层进度条描述，显示当前已找到的配对数量
                            outer_pbar.set_description(f"整体处理进度 (已找到 {len(all_pairs)} 对)")
                            outer_pbar.update(1)
                    
                    # 检查是否达到最大配对数
                    if max_pairs and len(all_pairs) >= max_pairs:
                        logger.info(f"已达到最大配对数 {max_pairs}，停止搜索...")
                        break
        
        # 预览模式下的额外终止条件
        if preview_mode or preview_with_file:
            if len(all_pairs) >= 3:
                logger.info("预览模式：已找到3个配对，停止搜索...")
                break
        
        # 检查是否达到最大配对数
        if max_pairs and len(all_pairs) >= max_pairs:
            logger.info(f"已达到最大配对数 {max_pairs}，停止搜索...")
            break
    
    if 'outer_pbar' in locals():
        outer_pbar.close()
    
    # 计算总运行时间
    total_elapsed_time = elapsed_time + (time.time() - start_time)
    
    # 删除检查点文件（任务完成）
    if os.path.exists(checkpoint_file) and not preview_mode:
        os.remove(checkpoint_file)
        logger.info(f"任务完成，已删除检查点文件 {checkpoint_file}")
    
    # 根据模式决定输出方式
    if preview_mode and not preview_with_file:
        # 预览模式：只输出JSON格式到控制台
        logger.info("预览模式：输出JSON格式内容到控制台")
        preview_pairs_as_json(all_pairs[:3], compact)
    elif preview_mode and preview_with_file:
        # 预览并保存到文件模式：输出JSON格式到控制台并保存到文件
        logger.info("预览并保存到文件模式：输出JSON格式内容到控制台并保存到文件")
        preview_pairs_as_json(all_pairs[:3], compact)
        save_pairs_to_json(all_pairs[:3], output_file, compact)
        logger.info(f"总共找到 {len(all_pairs[:3])} 对编辑距离为{step}的进化关系")
    else:
        # 正常模式：保存结果到文件
        save_pairs_to_json(all_pairs, output_file, compact)
        logger.info(f"总共找到 {len(all_pairs)} 对编辑距离为{step}的进化关系")
        logger.info(f"总运行时间: {time.strftime('%H:%M:%S', time.gmtime(total_elapsed_time))}")
        
        # 删除配对结果文件（任务完成）
        if os.path.exists(pairs_file):
            os.remove(pairs_file)
            logger.info(f"任务完成，已删除配对结果文件 {pairs_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='提取指定步数的分子进化对')
    parser.add_argument('--step', type=int, default=1, help='编辑距离步数，默认为1')
    parser.add_argument('--max-pairs', type=int, help='最大配对数，达到此数量后停止')
    parser.add_argument('--log', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='设置日志等级：DEBUG, INFO, WARNING, ERROR, CRITICAL')
    parser.add_argument('--mode', type=str, nargs='?', const=None,
                        choices=['full', 'preview', 'preview_with_file', 'full_compact', 'preview_with_file_compact'],
                        help='运行模式。使用 --mode 不带参数可触发交互式选择')
    parser.add_argument('--debug-from', type=str, help='调试模式：起始分子的SMILES')
    parser.add_argument('--debug-to', type=str, help='调试模式：目标分子的SMILES')
    parser.add_argument('--processes', type=int, default=None, help='使用的进程数，默认使用所有可用CPU核心')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复处理')
    
    args = parser.parse_args()
    
    # 设置日志等级
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('无效的日志等级: %s' % args.log)
    
    main(step=args.step, max_pairs=args.max_pairs, log_level=numeric_level, mode=args.mode,
         debug_from=args.debug_from, debug_to=args.debug_to, num_processes=args.processes,
         resume=args.resume)