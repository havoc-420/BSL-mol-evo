import os
import json
import csv
import sys
import argparse
import pandas as pd
from datetime import datetime
from tdc import Oracle
from rdkit import Chem
from rdkit.Chem import Crippen
from tqdm import tqdm

# 添加对HOMO/LUMO计算模块的导入
sys.path.append('/home/data2/rhj/project/mol_optimzation/utils')
try:
    from calculate_homo_lumo import calculate_homo_lumo_gap_qm9, calculate_mu
except ImportError as e:
    tqdm.write(f"无法导入必要的模块: {e}")
    tqdm.write("请确保calculate_homo_lumo.py文件存在于指定路径")
    sys.exit(1)

# 初始化TDC的Oracle
# 创建单独的oracle用于计算不同的属性
try:
    qed_oracle = Oracle(name='qed')
except Exception as e:
    tqdm.write(f"初始化TDC Oracle时出错: {e}")
    tqdm.write("请确保TDC库已正确安装且网络连接正常")
    sys.exit(1)


def evaluate_molecule(smiles, source_csv=None):
    """
    评估单个分子的性质
    :param smiles: 分子的SMILES字符串
    :param source_csv: 分子来源的CSV文件名
    :return: 包含各种性质的字典
    """
    result = {}
    result['source_csv'] = source_csv
    
    # 计算HOMO/LUMO gap
    try:
        # qm9_result = calculate_homo_lumo_gap_qm9(smiles, quiet=True)
        qm9_result = {
            'homo_ev': None,
            'lumo_ev': None,
            'gap_ev': None,
            'calculation_time': None,
        }
        
        # 记录计算时间
        result['calculation_time'] = qm9_result.get('calculation_time', None)
        
        if "error" in qm9_result:
            tqdm.write(f"计算HOMO/LUMO gap时出错 ({smiles}): {qm9_result['error']}")
            result['homo'] = None
            result['lumo'] = None
            result['gap'] = None
        else:
            # 使用eV单位的结果，因为它们更直观且适合比较
            result['homo'] = qm9_result['homo_ev']
            result['lumo'] = qm9_result['lumo_ev']
            result['gap'] = qm9_result['gap_ev']
    except Exception as e:
        tqdm.write(f"计算HOMO/LUMO gap时出错 ({smiles}): {e}")
        result['homo'] = None
        result['lumo'] = None
        result['gap'] = None
        result['calculation_time'] = None
    
    # 计算偶极矩 mu (Debye)
    try:
        mu_result = calculate_mu(smiles, quiet=True)
        if "error" in mu_result:
            tqdm.write(f"计算mu时出错 ({smiles}): {mu_result['error']}")
            result['mu'] = None
        else:
            result['mu'] = mu_result.get('mu', None)
    except Exception as e:
        tqdm.write(f"计算mu时出错 ({smiles}): {e}")
        result['mu'] = None
    
    # 计算类药性(QED)和logP
    try:
        result['drug_likeness'] = qed_oracle(smiles)
        # 使用RDKit的Crippen.MolLogP方法计算logP值
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            result['logP'] = Crippen.MolLogP(mol)
        else:
            result['logP'] = None
            tqdm.write(f"无法解析SMILES字符串 ({smiles})")
    except Exception as e:
        tqdm.write(f"计算类药性和logP时出错 ({smiles}): {e}")
        result['drug_likeness'] = None
        result['logP'] = None
    
    return result


def extract_molecule_data(output_dir, item_size=None, max_files=None):
    """
    从CSV文件中提取所有需要评估的分子数据
    :param output_dir: 包含优化结果的目录
    :param item_size: 每个CSV文件中处理的分子数量
    :param max_files: 处理的CSV文件数量上限
    :return: 包含所有分子数据的列表
    """
    # 收集所有的CSV文件
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('_topK.csv')]
    # 根据max_files参数限制处理的CSV文件数量
    if max_files is not None:
        csv_files = csv_files[:max_files]
    
    # 存储所有提取的分子数据
    all_molecule_data = []
    
    for csv_file in tqdm(csv_files, desc="Extracting molecule data from CSV files"):
        csv_path = os.path.join(output_dir, csv_file)
        
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 获取起始分子
        start_smiles = df['mol_start'][0]
        start_value = df['value_start'][0]
        
        # 保存起始分子数据
        all_molecule_data.append({
            'smiles': start_smiles,
            'source_csv': csv_file,
            'type': 'start',
            'predicted_value': start_value,
            'group_id': csv_file  # 用于标识同一组的分子
        })
        
        # 提取优化后的分子
        # 根据item_size参数确定处理的分子数量
        end_idx = item_size + 1 if item_size is not None else 21    # INFO 这里设定的 size 看如何设定。
        for i in range(1, end_idx):  # 从mol_1到mol_end_idx-1
            mol_col = f'mol_{i}'
            value_col = f'value_{i}'
            
            if mol_col in df.columns and value_col in df.columns:
                optimized_smiles = df[mol_col][0]
                optimized_value = df[value_col][0]
                
                # 保存优化后分子的数据
                all_molecule_data.append({
                    'smiles': optimized_smiles,
                    'source_csv': csv_file,
                    'type': f'optimized_{i}',
                    'predicted_value': optimized_value,
                    'group_id': csv_file  # 用于标识同一组的分子
                })
    
    return all_molecule_data


def evaluate_batch_mo(output_dir, target_prop='gap', direction='decrease', item_size=None, max_files=None):
    """
    评估批量分子优化的结果
    :param output_dir: 包含优化结果的目录
    :param target_prop: 目标属性，可选值：'homo', 'lumo', 'gap', 'mu'
    :param direction: 优化方向，可选值：'increase', 'decrease'
    """
    
    # 创建输出目录，添加时间戳和关键参数
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 处理参数为None的情况，避免目录名中出现"None"
    size_str = f'size{item_size}' if item_size is not None else 'sizeall'
    max_str = f'max{max_files}' if max_files is not None else 'maxall'
    eval_output_dir = os.path.join(output_dir, f'evaluation_results_{target_prop}_{direction}', f'{timestamp}_{size_str}_{max_str}')
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # 第一步：从CSV文件中提取所有分子数据
    tqdm.write("1️⃣ 开始从CSV文件中提取分子数据...")
    molecule_data_list = extract_molecule_data(output_dir, item_size, max_files)
    tqdm.write(f"成功提取 {len(molecule_data_list)} 个分子数据")
    
    # 第二步：批量评估所有分子
    tqdm.write("2️⃣ 开始批量评估分子性质...")
    all_results = []
    
    # 先创建一个字典来存储每个group的起始分子评估结果
    group_start_results = {}
    
    # 第一遍：评估所有分子并收集结果
    for molecule_data in tqdm(molecule_data_list, desc="Evaluating molecules"):
        smiles = molecule_data['smiles']
        source_csv = molecule_data['source_csv']
        mol_type = molecule_data['type']
        predicted_value = molecule_data['predicted_value']
        group_id = molecule_data['group_id']
        
        # 评估分子
        eval_result = evaluate_molecule(smiles, source_csv)
        eval_result['smiles'] = smiles
        eval_result['type'] = mol_type
        eval_result['predicted_value'] = predicted_value
        eval_result['group_id'] = group_id
        
        # 如果是起始分子，保存到group_start_results中
        if mol_type == 'start':
            group_start_results[group_id] = eval_result
        
        all_results.append(eval_result)
    
    # 第三步：计算每个优化分子的改善情况
    tqdm.write("3️⃣ 计算分子优化改善情况...")
    for result in all_results:
        if result['type'] != 'start' and result['group_id'] in group_start_results:
            start_result = group_start_results[result['group_id']]
            if start_result[target_prop] is not None and result[target_prop] is not None:
                if direction == 'decrease':
                    # 对于decrease方向，improvement为负表示改善
                    result['improvement'] = result[target_prop] - start_result[target_prop]
                else:  # increase
                    # 对于increase方向，improvement为负表示改善
                    result['improvement'] = start_result[target_prop] - result[target_prop]
            else:
                result['improvement'] = None
        else:
            result['improvement'] = None
    
    # 将结果写入CSV文件
    results_path = os.path.join(eval_output_dir, 'batch_evaluation_results.csv')
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_path, index=False)
    tqdm.write(f"评估结果已保存到: {results_path}")
    
    # 生成统计报告
    generate_statistics(results_df, eval_output_dir, target_prop, direction)
    
    tqdm.write(f"评估完成！结果保存到: {results_path}")
    
    # 生成统计报告
    results_df = pd.DataFrame(all_results)
    generate_statistics(results_df, eval_output_dir, target_prop, direction)


def generate_statistics(results_df, output_dir, target_prop, direction):
    """
    生成统计报告
    :param results_df: 包含所有评估结果的数据框
    :param output_dir: 输出目录
    :param target_prop: 目标属性
    :param direction: 优化方向
    """
    # 分离起始分子和优化后的分子
    start_molecules = results_df[results_df['type'] == 'start']
    optimized_molecules = results_df[results_df['type'] != 'start']
    
    # 计算统计信息
    stats = {
        '总起始分子数': len(start_molecules),
        '总优化分子数': len(optimized_molecules),
        f'平均起始{target_prop}': start_molecules[target_prop].mean(),
        f'平均优化后{target_prop}': optimized_molecules[target_prop].mean(),
        '平均类药性': optimized_molecules['drug_likeness'].mean(),
        '平均logP': optimized_molecules['logP'].mean(),
    }
    
    # 计算改善情况
    # 无论方向如何，improvement为负表示改善
    improved_count = len(optimized_molecules[optimized_molecules['improvement'] < 0])
    
    stats['平均改善值'] = optimized_molecules['improvement'].mean()
    stats['改善的分子数'] = improved_count
    stats['改善的百分比'] = (improved_count / len(optimized_molecules)) * 100 if len(optimized_molecules) > 0 else 0
    
    # 保存统计信息到文件
    stats_path = os.path.join(output_dir, 'evaluation_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("分子优化效果评估统计报告\n")
        f.write("=" * 50 + "\n\n")
        for key, value in stats.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    tqdm.write(f"统计报告生成完成！保存到: {stats_path}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='评估批量分子优化的结果')
    parser.add_argument('output_dir', type=str, help='包含批量优化结果的目录路径')
    parser.add_argument('--target-prop', type=str, default='gap', choices=['lumo', 'homo', 'gap', 'mu'],
                        help='目标属性 (默认: gap)')
    parser.add_argument('--direction', type=str, default='decrease', choices=['increase', 'decrease'],
                        help='优化方向 (默认: decrease)')
    parser.add_argument('--item-size', type=int, default=None,
                        help='用于测试的优化分子数量（默认: 处理所有20个分子）')
    parser.add_argument('--max-files', type=int, default=None,
                        help='用于测试的CSV文件数量（默认: 处理所有文件）')
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.output_dir):
        tqdm.write(f"错误: 目录 {args.output_dir} 不存在")
        sys.exit(1)
    
    # 执行评估
    evaluate_batch_mo(args.output_dir, args.target_prop, args.direction, args.item_size, args.max_files)