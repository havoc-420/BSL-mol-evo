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

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mol_evo.core.similarity import calculate_evolutionary_similarity
from mol_evo.core.evolver import MoleculeEvolver

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
# 因为 MoleculeEvolver 的 `generate_path_dict()` 方法现在直接输出符合分析需求的标准格式，
# 所以该转换函数已不再需要。

def analyze_evolution_operation_dict(path1_dict, path2_dict):
    """
    基于结构化字典分析两个进化路径之间的差异，确定操作类型和详情
    """
    # 将字典列表转换为字符串集合以便比较
    set1_str = {str(op) for op in path1_dict}
    set2_str = {str(op) for op in path2_dict}
    
    # 找出新增的操作（在path2中但不在path1中）
    added_ops = set2_str - set1_str
    # 找出删除的操作（在path1中但不在path2中）
    removed_ops = set1_str - set2_str
    
    # 将字符串操作转换回字典形式
    added_ops_dict = [op for op in path2_dict if str(op) in added_ops]
    removed_ops_dict = [op for op in path1_dict if str(op) in removed_ops]
    
    operation_type = "unknown"
    operations = []
    
    # 根据差异判断操作类型
    if len(added_ops_dict) == 1 and len(removed_ops_dict) == 0:
        operation_type = "add"
        operations = added_ops_dict  # 直接使用evolver的输出格式
        
    elif len(removed_ops_dict) == 1 and len(added_ops_dict) == 0:
        operation_type = "del"
        operations = removed_ops_dict  # 直接使用evolver的输出格式
        
    elif len(added_ops_dict) == 1 and len(removed_ops_dict) == 1:
        operation_type = "replace"
        # 对于替换操作，我们只关注新增的操作
        operations = added_ops_dict  # 直接使用evolver的输出格式
        
    elif len(added_ops_dict) > 1 and len(removed_ops_dict) == 0:
        operation_type = "add_multi"
        operations = added_ops_dict  # 直接使用evolver的输出格式
        
    elif len(removed_ops_dict) > 1 and len(added_ops_dict) == 0:
        operation_type = "del_multi"
        operations = removed_ops_dict  # 直接使用evolver的输出格式
        
    else:
        operation_type = "complex"
        # 复杂操作，添加所有新增的操作
        operations = added_ops_dict  # 直接使用evolver的输出格式
        if not operations:
            # 如果没有新增操作，使用删除的操作
            operations = removed_ops_dict  # 直接使用evolver的输出格式
    
    return {
        'type': operation_type,
        'operations': operations
    }

def find_step_pairs(heavy_n_df, heavy_m_df, n_atoms, m_atoms, step, max_pairs=None, logger=None, preview_mode=False):
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
        search_limit_n = min(500, len(heavy_n_df))
        search_limit_m = min(500, len(heavy_m_df))
    
    logger.info(f"正在从{search_limit_n}个{n_atoms}重原子分子和{search_limit_m}个{m_atoms}重原子分子中寻找编辑距离为{step}的配对...")
    
    # 使用tqdm显示进度条
    for i in tqdm(range(search_limit_n), desc=f"处理{n_atoms}→{m_atoms}原子对", unit="mol"):
        smiles_n = heavy_n_df.iloc[i]['smiles']
        mol_n = Chem.MolFromSmiles(smiles_n)
        
        if mol_n is None:
            continue
            
        # 内层循环也添加进度条
        inner_pbar = tqdm(range(search_limit_m), desc=f"内层循环", leave=False, unit="mol")
        for j in inner_pbar:
            smiles_m = heavy_m_df.iloc[j]['smiles']
            mol_m = Chem.MolFromSmiles(smiles_m)
            
            if mol_m is None:
                continue
                
            # 更新内层进度条描述
            inner_pbar.set_description(f"比较 {i+1}:{smiles_n[:15]}... -> {j+1}:{smiles_m[:15]}...")
                
            # 检查是否可以通过指定步数操作从smiles_n得到smiles_m
            try:
                distance, path_n, path_m = calculate_evolutionary_similarity(smiles_n, smiles_m)
                
                # 如果编辑距离等于指定步数，则认为是符合要求的进化
                if distance == step:
                    # 获取进化操作详情
                    evolver_n = MoleculeEvolver(smiles_n)
                    evolver_m = MoleculeEvolver(smiles_m)
                    
                    # 获取结构化进化路径
                    path_n_dict = evolver_n.generate_path_dict()
                    path_m_dict = evolver_m.generate_path_dict()
                    
                    # 分析进化操作
                    operation_result = analyze_evolution_operation_dict(path_n_dict, path_m_dict)
                    
                    # 按照指定顺序创建pair_data字典
                    pair_data = {
                        'smiles_from': smiles_n,
                        'smiles_to': smiles_m,
                        'operations': operation_result['operations']
                    }
                    
                    pairs.append(pair_data)
                    
                    logger.info(f"找到一对{step}步进化分子: {smiles_n} ({n_atoms}重原子) -> {smiles_m} ({m_atoms}重原子)，编辑距离: {distance}")
                    logger.info(f"  操作类型: {operation_result['type']}")
                    logger.info(f"  操作数量: {len(operation_result['operations'])}")
                    
                    # 检查是否达到最大配对数
                    if max_pairs and len(pairs) >= max_pairs:
                        logger.info(f"已达到最大配对数 {max_pairs}，停止搜索...")
                        inner_pbar.close()
                        return pairs
                        
            except Exception as e:
                continue
        inner_pbar.close()
                
    return pairs

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
        evolver_from = MoleculeEvolver(smiles_from)
        evolver_to = MoleculeEvolver(smiles_to)
        
        # 获取结构化进化路径
        path_from_dict = evolver_from.generate_path_dict()
        path_to_dict = evolver_to.generate_path_dict()
        
        logger.info("起始分子进化路径（字符串形式）:")
        for i, op in enumerate(path_from):
            logger.info(f"  {i}: {op}")
            
        logger.info("目标分子进化路径（字符串形式）:")
        for i, op in enumerate(path_to):
            logger.info(f"  {i}: {op}")
        
        logger.info("起始分子进化路径（字典形式）:")
        for i, op in enumerate(path_from_dict):
            logger.info(f"  {i}: {op}")
            
        logger.info("目标分子进化路径（字典形式）:")
        for i, op in enumerate(path_to_dict):
            logger.info(f"  {i}: {op}")
        
        # 分析进化操作
        operation_result = analyze_evolution_operation_dict(path_from_dict, path_to_dict)
        
        logger.info(f"操作分析详情:")
        logger.info(f"  新增操作数量: {len([op for op in path_to_dict if str(op) not in [str(o) for o in path_from_dict]])}")
        logger.info(f"  删除操作数量: {len([op for op in path_from_dict if str(op) not in [str(o) for o in path_to_dict]])}")
        
        # 创建结果数据
        pair_data = {
            'smiles_from': smiles_from,
            'smiles_to': smiles_to,
            'distance': distance,
            'path_from': path_from,
            'path_to': path_to,
            'path_from_dict': path_from_dict,
            'path_to_dict': path_to_dict,
            'operations': operation_result['operations']
        }
        
        logger.info(f"操作类型: {operation_result['type']}")
        logger.info(f"操作详情:")
        for i, op in enumerate(operation_result['operations']):
            logger.info(f"  {i}: {op}")
            
        # 转换数据类型以确保可以被JSON序列化
        pair_data = convert_types(pair_data)
            
        return pair_data
        
    except Exception as e:
        logger.error(f"处理分子对时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main(step=1, max_pairs=None, log_level=None, mode=None, debug_from=None, debug_to=None):
    """
    主函数
    
    Args:
        step: 编辑距离步数，默认为1
        max_pairs: 最大配对数，达到此数量后停止搜索
        log_level: 日志等级
        mode: 运行模式
        debug_from: 调试模式下的起始分子SMILES
        debug_to: 调试模式下的目标分子SMILES
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
    
    # 设置数据目录
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    output_file = os.path.join(data_dir, f'qm9-evo-pairs-step-{step}.json')
    
    # 创建输出目录（如果不存在）
    os.makedirs(data_dir, exist_ok=True)
    
    # 存储所有指定步数进化对
    all_pairs = []
    
    # 加载所有数据集
    datasets = {}
    for i in range(1, 10):
        file_path = os.path.join(data_dir, f'qm9_smiles_heavy_{i}_atoms.csv')
        if os.path.exists(file_path):
            datasets[i] = load_dataset(file_path)
            logger.info(f"加载{i}重原子数据集...")
            logger.info(f"{i}重原子分子数量: {len(datasets[i])}")
        else:
            logger.warning(f"警告: {file_path} 不存在")
            datasets[i] = pd.DataFrame(columns=['smiles'])
    total_combinations = 0
    # 计算总的组合数用于外层进度条
    for heavy_atoms in range(1, 9):
        if heavy_atoms in datasets and (heavy_atoms + 1) in datasets:
            total_combinations += 1
    
    for heavy_atoms in range(1, 10):
        if heavy_atoms in datasets:
            total_combinations += 1
    
    # 外层进度条
    if preview_mode or preview_with_file:
        # 预览模式下不显示进度条
        pass
    else:
        outer_pbar = tqdm(total=total_combinations, desc="整体处理进度", unit="组合")
    
    # 查找相邻原子数之间的进化对（1->2, 2->3, ..., 8->9）
    for heavy_atoms in range(1, 9):
        if heavy_atoms in datasets and (heavy_atoms + 1) in datasets:
            df_from = datasets[heavy_atoms]
            df_to = datasets[heavy_atoms + 1]
            
            logger.info(f"查找{heavy_atoms}->{heavy_atoms+1}重原子的{step}步进化对...")
            
            # 查找指定步数进化对
            pairs = find_step_pairs(df_from, df_to, heavy_atoms, heavy_atoms + 1, step, max_pairs=max_pairs, logger=logger, preview_mode=(preview_mode or preview_with_file))
            all_pairs.extend(pairs)
            
            if preview_mode or preview_with_file:
                # 预览模式下找到3个配对就停止
                if len(all_pairs) >= 3:
                    logger.info("预览模式：已找到3个配对，停止搜索...")
                    break
            else:
                outer_pbar.update(1)
            
            # 检查是否达到最大配对数
            if max_pairs and len(all_pairs) >= max_pairs:
                logger.info(f"已达到最大配对数 {max_pairs}，停止搜索...")
                break
    
    # 如果还没有达到最大配对数，继续查找相同原子数内部的进化对
    if not (max_pairs and len(all_pairs) >= max_pairs):
        if not preview_mode and not preview_with_file or len(all_pairs) < 3:
            for heavy_atoms in range(1, 10):
                if heavy_atoms in datasets:
                    df = datasets[heavy_atoms]
                    logger.info(f"查找{heavy_atoms}重原子内部的{step}步进化对...")
                    
                    # 查找指定步数进化对
                    pairs = find_step_pairs(df, df, heavy_atoms, heavy_atoms, step, max_pairs=max_pairs, logger=logger, preview_mode=(preview_mode or preview_with_file))
                    all_pairs.extend(pairs)
                    
                    if preview_mode or preview_with_file:
                        # 预览模式下找到3个配对就停止
                        if len(all_pairs) >= 3:
                            logger.info("预览模式：已找到3个配对，停止搜索...")
                            break
                    else:
                        if 'outer_pbar' in locals():
                            outer_pbar.update(1)
                    
                    # 检查是否达到最大配对数
                    if max_pairs and len(all_pairs) >= max_pairs:
                        logger.info(f"已达到最大配对数 {max_pairs}，停止搜索...")
                        break
    
    if 'outer_pbar' in locals():
        outer_pbar.close()
    
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
    
    args = parser.parse_args()
    
    # 设置日志等级
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('无效的日志等级: %s' % args.log)
    
    main(step=args.step, max_pairs=args.max_pairs, log_level=numeric_level, mode=args.mode,
         debug_from=args.debug_from, debug_to=args.debug_to)
