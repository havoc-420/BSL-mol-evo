import pandas as pd
from rdkit import Chem
import sys
import os
import argparse
import logging
from tqdm import tqdm

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mol_evo.core.similarity import calculate_evolutionary_similarity
from mol_evo.core.evolver import MoleculeEvolver

def load_dataset(file_path):
    """
    加载CSV数据集
    """
    return pd.read_csv(file_path)

def find_one_step_pairs(heavy_n_df, heavy_m_df, n_atoms, m_atoms, max_pairs=None, logger=None):
    """
    寻找适合一步进化的SMILES对
    从n个重原子的分子到m个重原子的分子（包括n==m的情况）
    
    Args:
        heavy_n_df: 源分子数据集
        heavy_m_df: 目标分子数据集  
        n_atoms: 源分子重原子数
        m_atoms: 目标分子重原子数
        max_pairs: 最大配对数，达到此数量后停止
        logger: 日志记录器
    
    Returns:
        配对结果列表
    """
    import logging
    if logger is None:
        # 如果没有提供logger，创建一个默认的
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    
    pairs = []
    
    # 限制搜索范围以提高效率
    # 只检查前N个分子，避免计算量过大
    search_limit_n = min(500, len(heavy_n_df))
    search_limit_m = min(500, len(heavy_m_df))
    
    logger.info(f"正在从{search_limit_n}个{n_atoms}重原子分子和{search_limit_m}个{m_atoms}重原子分子中寻找配对...")
    
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
                
            # 检查是否可以通过一步操作从smiles_n得到smiles_m
            # 使用编辑距离为1作为简单筛选条件
            try:
                distance, path_n, path_m = calculate_evolutionary_similarity(smiles_n, smiles_m)
                
                # 如果编辑距离为1，则认为是一步进化
                if distance == 1:
                    # 获取进化操作详情
                    evolver_n = MoleculeEvolver(smiles_n)
                    evolver_m = MoleculeEvolver(smiles_m)
                    
                    # 获取进化路径
                    path_n = evolver_n.generate_path()
                    path_m = evolver_m.generate_path()
                    
                    # 分析进化操作
                    operation = analyze_evolution_operation(path_n, path_m)
                    
                    # 按照指定顺序创建pair_data字典
                    pair_data = {
                        'smiles_from': smiles_n,
                        'smiles_to': smiles_m,
                        'from_heavy_atoms': n_atoms,
                        'to_heavy_atoms': m_atoms,
                        'index_from': heavy_n_df.iloc[i].name,
                        'index_to': heavy_m_df.iloc[j].name,
                        'distance': distance,
                        'operation_type': operation['type'],
                        'to_atom_symbol': operation['to_atom_symbol'],
                        'to_atom_position': operation['to_atom_position'],
                        'operation_detail': operation['detail'],
                        'evolved_molecule': operation['evolved_molecule']
                    }
                    
                    pairs.append(pair_data)
                    
                    logger.info(f"找到一对一步进化分子: {smiles_n} ({n_atoms}重原子) -> {smiles_m} ({m_atoms}重原子)")
                    logger.info(f"  操作类型: {operation['type']}")
                    logger.info(f"  操作详情: {operation['detail']}")
                    if operation['evolved_molecule']:
                        logger.info(f"  进化分子: {operation['evolved_molecule']}")
                    
                    # 检查是否达到最大配对数
                    if max_pairs and len(pairs) >= max_pairs:
                        logger.info(f"已达到最大配对数 {max_pairs}，停止搜索...")
                        inner_pbar.close()
                        return pairs
                        
            except Exception as e:
                continue
        inner_pbar.close()
                
    return pairs

def analyze_evolution_operation(path1, path2):
    """
    分析两个进化路径之间的差异，确定操作类型和详情
    """
    # 找出路径中的差异
    set1 = set(path1)
    set2 = set(path2)
    
    # 找出新增的操作（在path2中但不在path1中）
    added_ops = set2 - set1
    # 找出删除的操作（在path1中但不在path2中）
    removed_ops = set1 - set2
    
    operation_type = "unknown"
    operation_detail = ""
    evolved_molecule = ""
    
    # 根据差异判断操作类型
    if len(added_ops) == 1 and len(removed_ops) == 0:
        operation_type = "add"
        operation_detail = list(added_ops)[0]
        # 检查是否是添加手性操作
        if "指定手性(" in operation_detail or "指定顺反(" in operation_detail:
            operation_type = "add_stereo"
    elif len(removed_ops) == 1 and len(added_ops) == 0:
        operation_type = "del"
        operation_detail = list(removed_ops)[0]
        # 检查是否是删除手性操作
        if "指定手性(" in operation_detail or "指定顺反(" in operation_detail:
            operation_type = "del_stereo"
    elif len(added_ops) == 1 and len(removed_ops) == 1:
        operation_type = "replace"
        operation_detail = f"replace '{list(removed_ops)[0]}' with '{list(added_ops)[0]}'"
        # 检查是否是替换手性操作
        removed_op = list(removed_ops)[0]
        added_op = list(added_ops)[0]
        if ("指定手性(" in removed_op or "指定顺反(" in removed_op) and \
           ("指定手性(" in added_op or "指定顺反(" in added_op):
            operation_type = "replace_stereo"
    elif len(added_ops) > 1 and len(removed_ops) == 0:
        operation_type = "add_multi"
        operation_detail = "; ".join(list(added_ops))
    elif len(removed_ops) > 1 and len(added_ops) == 0:
        operation_type = "del_multi"
        operation_detail = "; ".join(list(removed_ops))
    else:
        operation_type = "complex"
        operation_detail = f"del: {list(removed_ops)}, add: {list(added_ops)}"
    
    # 提取操作中原子的位置和类别信息
    # 解析操作详情，提取原子信息
    from_atom_symbol = ""
    from_atom_position = ""
    to_atom_symbol = ""
    to_atom_position = ""
    
    # 解析删除的操作（原操作）
    if len(removed_ops) > 0:
        removed_op = list(removed_ops)[0]
        # 解析"添加原子(X) -> Y"格式
        if "添加原子(" in removed_op:
            # 提取原子符号
            start = removed_op.find("添加原子(") + len("添加原子(")
            end = removed_op.find(")", start)
            from_atom_symbol = removed_op[start:end]
            # 提取位置
            from_atom_position = removed_op.split(" -> ")[-1]
        # 解析"起始(X ID:Y)"格式
        elif "起始(" in removed_op:
            start = removed_op.find("起始(") + len("起始(")
            end = removed_op.find(" ", start)
            from_atom_symbol = removed_op[start:end]
            # 提取ID
            id_start = removed_op.find("ID:") + len("ID:")
            from_atom_position = removed_op[id_start:].split(")")[0]
    
    # 解析新增的操作（新操作）
    if len(added_ops) > 0:
        added_op = list(added_ops)[0]
        # 解析"添加原子(X) -> Y"格式
        if "添加原子(" in added_op:
            # 提取原子符号
            start = added_op.find("添加原子(") + len("添加原子(")
            end = added_op.find(")", start)
            to_atom_symbol = added_op[start:end]
            # 提取位置
            to_atom_position = added_op.split(" -> ")[-1]
        # 解析"起始(X ID:Y)"格式
        elif "起始(" in added_op:
            start = added_op.find("起始(") + len("起始(")
            end = added_op.find(" ", start)
            to_atom_symbol = added_op[start:end]
            # 提取ID
            id_start = added_op.find("ID:") + len("ID:")
            to_atom_position = added_op[id_start:].split(")")[0]
    
    # 如果path2比path1多一个操作，那么最终分子就是path2对应的分子
    # 这里我们简单地用path2的最后一个操作作为evolved_molecule的标识
    if len(added_ops) == 1 and len(removed_ops) == 0:
        evolved_molecule = list(added_ops)[0]
    elif len(added_ops) == 1 and len(removed_ops) == 1:
        # 对于replace操作，evolved_molecule应该是新增的操作而不是整个replace描述
        evolved_molecule = list(added_ops)[0]
    elif len(added_ops) > 0:
        evolved_molecule = "; ".join(list(added_ops))
    else:
        evolved_molecule = ""
    
    return {
        'type': operation_type,
        'detail': operation_detail,
        'evolved_molecule': evolved_molecule,
        'to_atom_symbol': to_atom_symbol,
        'to_atom_position': to_atom_position
    }

def save_pairs_to_csv(pairs, output_file):
    """
    将配对结果保存到CSV文件
    """
    if not pairs:
        print("未找到任何配对")
        return
        
    # 创建DataFrame
    df = pd.DataFrame(pairs)
    
    # 移除不需要的列
    columns_to_remove = ['from_atom_symbol', 'from_atom_position']
    existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    if existing_columns_to_remove:
        df = df.drop(columns=existing_columns_to_remove)
    
    # 重新排列列的顺序，将operation_detail和evolved_molecule移到最后
    column_order = [col for col in df.columns if col not in ['operation_detail', 'evolved_molecule']]
    column_order.extend(['operation_detail', 'evolved_molecule'])
    df = df[column_order]
    
    # 保存到CSV文件
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"已保存 {len(pairs)} 对分子到 {output_file}")

def main(max_pairs=None, log_level=None):
    """
    主函数
    
    Args:
        max_pairs: 最大配对数，达到此数量后停止搜索
        log_level: 日志等级
    """
    import logging
    
    # 配置日志
    logging.basicConfig(level=log_level, format='%(message)s')
    logger = logging.getLogger(__name__)
    
    # 设置数据目录
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    output_file = os.path.join(data_dir, 'qm9-evo-pairs-step-1.csv')
    
    # 创建输出目录（如果不存在）
    os.makedirs(data_dir, exist_ok=True)
    
    # 存储所有一步进化对
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
    outer_pbar = tqdm(total=total_combinations, desc="整体处理进度", unit="组合")
    
    # 查找相邻原子数之间的进化对（1->2, 2->3, ..., 8->9）
    for heavy_atoms in range(1, 9):
        if heavy_atoms in datasets and (heavy_atoms + 1) in datasets:
            df_from = datasets[heavy_atoms]
            df_to = datasets[heavy_atoms + 1]
            
            logger.info(f"查找{heavy_atoms}->{heavy_atoms+1}重原子的一次进化对...")
            
            # 查找一步进化对
            pairs = find_one_step_pairs(df_from, df_to, heavy_atoms, heavy_atoms + 1, max_pairs=max_pairs, logger=logger)
            all_pairs.extend(pairs)
            outer_pbar.update(1)
            
            # 检查是否达到最大配对数
            if max_pairs and len(all_pairs) >= max_pairs:
                logger.info(f"已达到最大配对数 {max_pairs}，停止搜索...")
                break
    
    # 如果还没有达到最大配对数，继续查找相同原子数内部的进化对
    if not (max_pairs and len(all_pairs) >= max_pairs):
        for heavy_atoms in range(1, 10):
            if heavy_atoms in datasets:
                df = datasets[heavy_atoms]
                logger.info(f"查找{heavy_atoms}重原子内部的一步进化对...")
                
                # 查找一步进化对
                pairs = find_one_step_pairs(df, df, heavy_atoms, heavy_atoms, max_pairs=max_pairs, logger=logger)
                all_pairs.extend(pairs)
                outer_pbar.update(1)
                
                # 检查是否达到最大配对数
                if max_pairs and len(all_pairs) >= max_pairs:
                    logger.info(f"已达到最大配对数 {max_pairs}，停止搜索...")
                    break
    
    outer_pbar.close()
    
    # 保存结果
    save_pairs_to_csv(all_pairs, output_file)
    logger.info(f"总共找到 {len(all_pairs)} 对进化关系")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='提取所有一步分子进化对')
    parser.add_argument('--max-pairs', type=int, help='最大配对数，达到此数量后停止')
    parser.add_argument('--log', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='设置日志等级：DEBUG, INFO, WARNING, ERROR, CRITICAL')
    
    args = parser.parse_args()
    
    # 设置日志等级
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('无效的日志等级: %s' % args.log)
    
    main(max_pairs=args.max_pairs, log_level=numeric_level)
