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

from rdkit import Chem
from rdkit.Chem import AllChem
from mol_evo.core.data.processing import calculate_molecular_similarity
from mol_evo.core.evolver import MoleculeEvolverAnalysis

""" utils """

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


# TODO 得完善
def generate_intermediate_smiles(start_smiles, operations):
    """
    根据起始SMILES和操作序列生成中间状态的SMILES
    
    Args:
        start_smiles: 起始分子的SMILES字符串
        operations: 操作序列，每个操作是一个字典
        
    Returns:
        包含中间状态SMILES的列表，包括起始状态
    """
    intermediates = [start_smiles]
    current_mol = Chem.MolFromSmiles(start_smiles)
    
    if current_mol is None:
        return intermediates
    
    # 为了能够进行修改，我们需要创建一个可编辑的分子
    editable_mol = Chem.RWMol(current_mol)
    
    for op in operations:
        try:
            op_type = op.get('operation', '')
            position = op.get('position', '')
            atom = op.get('atom')
            
            # 处理不同类型的操作
            if op_type == 'add_atom' and position and atom:
                # 添加原子
                parent_idx = int(position)
                if parent_idx < editable_mol.GetNumAtoms():
                    new_atom = Chem.Atom(atom)
                    new_idx = editable_mol.AddAtom(new_atom)
                    editable_mol.AddBond(parent_idx, new_idx, Chem.BondType.SINGLE)
            
            elif op_type == 'add_fragment' and position and atom:
                # 添加片段（简化处理，实际应用中可能需要更复杂的逻辑）
                parent_idx = int(position)
                if parent_idx < editable_mol.GetNumAtoms():
                    # 尝试将片段SMILES转换为分子并连接
                    frag_mol = Chem.MolFromSmiles(atom)
                    if frag_mol is not None:
                        # 简化处理：直接添加片段的原子并连接到父原子
                        for frag_atom in frag_mol.GetAtoms():
                            new_atom = Chem.Atom(frag_atom.GetSymbol())
                            new_idx = editable_mol.AddAtom(new_atom)
                            editable_mol.AddBond(parent_idx, new_idx, Chem.BondType.SINGLE)
                            parent_idx = new_idx
            
            elif op_type.startswith('form_double_') and '-' in position:
                # 形成双键
                pos1, pos2 = map(int, position.split('-'))
                if pos1 < editable_mol.GetNumAtoms() and pos2 < editable_mol.GetNumAtoms():
                    bond = editable_mol.GetBondBetweenAtoms(pos1, pos2)
                    if bond is not None:
                        editable_mol.RemoveBond(pos1, pos2)
                        editable_mol.AddBond(pos1, pos2, Chem.BondType.DOUBLE)
            
            elif op_type.startswith('form_triple_') and '-' in position:
                # 形成三键
                pos1, pos2 = map(int, position.split('-'))
                if pos1 < editable_mol.GetNumAtoms() and pos2 < editable_mol.GetNumAtoms():
                    bond = editable_mol.GetBondBetweenAtoms(pos1, pos2)
                    if bond is not None:
                        editable_mol.RemoveBond(pos1, pos2)
                        editable_mol.AddBond(pos1, pos2, Chem.BondType.TRIPLE)
            
            elif op_type.startswith('form_ring') and '-' in position:
                # 形成环（单键）
                pos1, pos2 = map(int, position.split('-'))
                if pos1 < editable_mol.GetNumAtoms() and pos2 < editable_mol.GetNumAtoms():
                    editable_mol.AddBond(pos1, pos2, Chem.BondType.SINGLE)
            
            # 尝试生成当前状态的SMILES
            # 需要先清理分子并分配环ID
            Chem.SanitizeMol(editable_mol)
            current_smiles = Chem.MolToSmiles(editable_mol)
            intermediates.append(current_smiles)
            
        except Exception as e:
            # 如果操作失败，使用当前分子状态
            current_smiles = Chem.MolToSmiles(editable_mol)
            intermediates.append(current_smiles)
    
    return intermediates


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
        return f"{op.get('operation', '')}@{op.get('position', '')}@{op.get('atom', '')}"   # INFO 没错，就是这里的 position 定位比较严格。
    
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
    参数 args 是一个元组，包含 (task_id, i, j, smiles_n, smiles_m)
    """
    # 兼容两种格式的任务参数
    if len(args) == 5:
        # 新格式：包含task_id
        _, _, _, smiles_n, smiles_m = args
    else:
        return None
    
    mol_n = Chem.MolFromSmiles(smiles_n)
    mol_m = Chem.MolFromSmiles(smiles_m)
    
    if mol_n is None or mol_m is None:
        return None
    
    # Calculate Morgan similarity and skip pairs below threshold (0.1)
    similarity = calculate_molecular_similarity(smiles_n, smiles_m)
    if similarity < 0.1:
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
        
        # 生成中间状态的SMILES
        intermediates = generate_intermediate_smiles(smiles_n, operation_result)
        
        # 按照指定顺序创建pair_data字典
        pair_data = {
            'smiles_from': smiles_n,
            'smiles_to': smiles_m,
            'operations': operation_result,
            'intermediates': intermediates
        }
        
        return pair_data
        
    except Exception as e:
        return None


def find_step_pairs(heavy_n_df, heavy_m_df, max_pairs=None, logger=None, preview_mode=False, 
                   checkpoint_file=None, start_index=0, all_pairs_checkpoint=None, dataset_info=None, 
                   total_combinations=None, completed_combinations=None, pairs_file=None):
    """
    寻找分子进化对（不再限制步数）
    从源分子数据集到目标分子数据集的所有可能分子对（包括相同分子的情况）
    
    Args:
        heavy_n_df: 源分子数据集
        heavy_m_df: 目标分子数据集  
        max_pairs: 最大配对数，达到此数量后停止
        logger: 日志记录器
        preview_mode: 预览模式，只提取少量数据用于预览
        checkpoint_file: 检查点文件路径，用于断点续传
        start_index: 开始索引，用于断点续传
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
        logger.info(f"正在从{search_limit_n}个源分子和{search_limit_m}个目标分子中寻找所有可能的配对...")
    
    # 准备所有需要处理的分子对
    tasks = []
    unique_smiles = set()  # 收集所有唯一的SMILES以进行预处理
    
    # 添加tqdm进度条以显示收集唯一SMILES的过程
    # 如果提供了start_index，则使用它来计算起始位置
    if start_index > 0:
        start_i = start_index // search_limit_m  # 计算起始的i值
        start_j = start_index % search_limit_m   # 计算起始的j值
        logger.info(f"从索引 {start_index} (i={start_i}, j={start_j}) 开始处理分子对")
    else:
        start_i = 0
        start_j = 0
        logger.info(f"开始处理分子对")
    
    task_id = 0
    for i in tqdm(range(start_i, search_limit_n), desc=f"收集源分子", unit="mol"):
        smiles_n = heavy_n_df.iloc[i]['smiles']
        unique_smiles.add(smiles_n)
        for j in range(search_limit_m):
            # 如果是第一行(i==start_i)，则从start_j开始，否则从0开始
            if i == start_i and j < start_j:
                continue
            smiles_m = heavy_m_df.iloc[j]['smiles']
            unique_smiles.add(smiles_m)
            tasks.append((task_id, i, j, smiles_n, smiles_m))  # 不再包含step参数
            task_id += 1
    
    # 在预览模式下保持原有逻辑，便于调试
    if preview_mode:
        evolver_cache = {}
        # 使用tqdm显示进度条
        task_index = 0
        for i in tqdm(range(start_i, search_limit_n), desc=f"处理分子对", unit="mol"):
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
                
                if mol_m is None or smiles_n == smiles_m:
                    continue
                
                # Calculate Morgan similarity and skip pairs below threshold (0.1)
                similarity = calculate_molecular_similarity(smiles_n, smiles_m)
                if similarity < 0.1:
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
                    
                    # 生成中间状态的SMILES
                    intermediates = generate_intermediate_smiles(smiles_n, operation_result)
                    
                    # 按照指定顺序创建pair_data字典
                    pair_data = {
                        'smiles_from': smiles_n,
                        'smiles_to': smiles_m,
                        'smiles_from_path': path_n_dict,   # TEST
                        'smiles_to_path': path_m_dict,
                        'operations': operation_result,
                        'intermediates': intermediates
                    }
                    
                    pairs.append(pair_data)
                    task_index += 1
                    
                    # 保存检查点
                    if checkpoint_file and task_index % 1000 == 0:  # 每1000个任务保存一次检查点
                        checkpoint_data = {
                            'processed_index': i * search_limit_m + j,
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
        cache_file = os.path.join(cache_dir, f'molecule_evolver_cache.pkl')
        
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
        start_task_index = start_index
        remaining_tasks = tasks[start_task_index:] if start_task_index < len(tasks) else []
        
        # 计算总批次数
        total_batches = (len(remaining_tasks) + batch_size - 1) // batch_size
        
        # 修改：使用批次索引而不是任务索引
        start_batch_index = start_index
        # 从批次索引计算起始任务索引
        start_task_index = start_batch_index * batch_size
        remaining_tasks = tasks[start_task_index:] if start_task_index < len(tasks) else []
        # 重新计算总批次数
        total_batches = (len(remaining_tasks) + batch_size - 1) // batch_size
        
        with Pool(processes=num_processes) as pool:
            # 使用总批次数作为主进度条
            batch_pbar = tqdm(total=total_batches, desc=f"处理分子对", unit="批")
            
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
                        'current_combination': f"all->all",
                        # 添加批次信息
                        'total_batches': total_batches,
                        'current_batch': batch_num,
                        'batch_size': batch_size,
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


# INTERACTIVE: select task
def select_mode_interactively():
    """
    交互式选择模式
    """
    questions = [
        inquirer.List('mode',
                     message="请选择运行模式",
                     choices=[
                         ('[full] 全量处理并保存到文件', 'full'),
                         ('[preview] 预览模式（仅输出到控制台）', 'preview'),
                         ('[preview_with_file] 预览并保存到文件', 'preview_with_file'),
                         ('[full_compact] 全量处理并保存到文件（紧凑格式）', 'full_compact'),
                         ('[preview_with_file_compact] 预览并保存到文件（紧凑格式）', 'preview_with_file_compact')
                     ])
    ]
    
    answers = inquirer.prompt(questions)
    return answers['mode'] if answers else None


def main(csv_path, max_pairs=None, log_level=None, mode=None, num_processes=None, resume=False):
    """
    主函数
    
    Args:
        csv_path: CSV文件路径，包含smiles字段
        max_pairs: 最大配对数，达到此数量后停止搜索
        log_level: 日志等级
        mode: 运行模式
        debug_from: 调试模式下的起始分子SMILES
        debug_to: 调试模式下的目标分子SMILES
        resume: 是否从检查点恢复
    """
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
    data_dir = os.path.join(os.path.dirname(__file__), 'data', "gdcsv2")    # INFO ./data/gdcsv2/
    os.makedirs(data_dir, exist_ok=True)
    
    # 定义所有文件路径 - 提前定义以确保全局可用
    # 从CSV文件名中提取基础名称，用于生成输出文件名
    csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
    output_file = os.path.join(data_dir, f'{csv_basename}-evo-pairs.json')
    checkpoint_file = os.path.join(data_dir, f'{csv_basename}-evo-pairs-checkpoint.json')
    pairs_file = os.path.join(data_dir, f'{csv_basename}-evo-pairs-pairs.json')  # 独立的配对结果文件
    
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
                last_pairs_count = checkpoint_data.get('pairs_count', 0) # 获取上一次的配对数量
                elapsed_time = checkpoint_data.get('elapsed_time', 0)  # 获取已运行时间
                dataset_info = checkpoint_data.get('dataset_info', {})  # 获取数据集信息
                total_combinations = checkpoint_data.get('total_combinations', 0)  # 获取总组合数
                completed_combinations = checkpoint_data.get('completed_combinations', 0)  # 获取已完成组合数
            logger.info(f"从检查点恢复，从批次 {start_index} 开始处理")
            if elapsed_time > 0:
                logger.info(f"已运行时间: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
        except Exception as e:
            logger.warning(f"加载检查点文件时出错: {e}，将从头开始处理")
    
    # 加载单个CSV文件中的所有分子
    logger.info(f"加载CSV文件: {csv_path}")
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 检查是否包含smiles列
    if 'smiles' not in df.columns:
        logger.error("CSV文件中没有smiles列")
        sys.exit(1)
    
    # 过滤掉无效的SMILES
    logger.info("过滤无效的SMILES...")
    valid_smiles = []
    for smiles in df['smiles']:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles.append(smiles)
        except Exception:
            pass
    
    # 如果没有有效分子，退出程序
    if not valid_smiles:
        logger.error("没有找到有效分子")
        sys.exit(1)
    
    # 创建包含所有有效SMILES的DataFrame
    all_molecules_df = pd.DataFrame({'smiles': valid_smiles})
    num_molecules = len(all_molecules_df)
    logger.info(f"总共找到 {num_molecules} 个有效分子")
    
    # 准备数据集信息用于检查点
    dataset_info = {
        'total_molecules': num_molecules,
        'loaded_at': time.time()
    }

    # 计算总的组合数用于外层进度条（仅在非恢复模式下计算）
    if not (resume and total_combinations > 0):
        # 笛卡尔积的总组合数是分子数量的平方
        total_combinations = num_molecules * num_molecules
    
    # 外层进度条
    if preview_mode or preview_with_file:
        # 预览模式下不显示进度条
        pass
    else:
        outer_pbar = tqdm(total=total_combinations, desc="整体处理进度", unit="组合")
        if completed_combinations > 0:
            outer_pbar.update(completed_combinations)
            outer_pbar.set_description(f"整体处理进度 (已找到 {len(all_pairs)} 对)")

    # 直接对所有分子进行笛卡尔积遍历
    resume_point_reached = True  # 不再需要按重原子数恢复
    
    # 查找所有分子对的进化对（不再限制步数）
    pairs = find_step_pairs(all_molecules_df, all_molecules_df, 
                          max_pairs=max_pairs, logger=logger, 
                          preview_mode=(preview_mode or preview_with_file),
                          checkpoint_file=checkpoint_file if not preview_mode else None,
                          start_index=start_index,
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
    
    # 更新进度
    if not (preview_mode or preview_with_file):
        completed_combinations += 1
        if 'outer_pbar' in locals():
            outer_pbar.set_description(f"整体处理进度 (已找到 {len(all_pairs)} 对)")
            outer_pbar.update(1)
    
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
        logger.info(f"总共找到 {len(all_pairs[:3])} 对编辑距离为的进化关系")
    else:
        # 正常模式：保存结果到文件
        save_pairs_to_json(all_pairs, output_file, compact)
        logger.info(f"总共找到 {len(all_pairs)} 对编辑距离为的进化关系")
        logger.info(f"总运行时间: {time.strftime('%H:%M:%S', time.gmtime(total_elapsed_time))}")
        
        # 删除配对结果文件（任务完成）
        if os.path.exists(pairs_file):
            os.remove(pairs_file)
            logger.info(f"任务完成，已删除配对结果文件 {pairs_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='提取分子进化对（不限定编辑步数）')
    parser.add_argument('--csv', type=str, required=True, help='CSV文件路径，包含smiles字段')
    parser.add_argument('--max-pairs', type=int, help='最大配对数，达到此数量后停止')
    parser.add_argument('--log', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='设置日志等级：DEBUG, INFO, WARNING, ERROR, CRITICAL')
    parser.add_argument('--mode', type=str, nargs='?', const=None,
                        choices=['full', 'preview', 'preview_with_file', 'full_compact', 'preview_with_file_compact'],
                        help='运行模式。使用 --mode 不带参数可触发交互式选择')
    parser.add_argument('--processes', type=int, default=None, help='使用的进程数，默认使用所有可用CPU核心')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复处理')
    
    args = parser.parse_args()
    
    # 设置日志等级
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('无效的日志等级: %s' % args.log)
    
    main(csv_path=args.csv, max_pairs=args.max_pairs, log_level=numeric_level, mode=args.mode, num_processes=args.processes, resume=args.resume)
