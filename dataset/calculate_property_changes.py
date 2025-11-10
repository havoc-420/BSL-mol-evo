import pandas as pd
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
from rdkit import Chem

def load_heavy_atom_file(heavy_atoms):
    """
    加载指定重原子数的CSV文件
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    file_path = os.path.join(data_dir, f'qm9_smiles_heavy_{heavy_atoms}_atoms.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    return pd.read_csv(file_path)

def count_heavy_atoms(smiles):
    """
    计算SMILES中的重原子数
    
    Args:
        smiles: 分子的SMILES表示
        
    Returns:
        int: 重原子数
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol.GetNumHeavyAtoms()
    return 0

def load_pairs_file(pairs_file):
    """
    根据文件扩展名加载配对文件（支持CSV和JSON格式）
    
    Args:
        pairs_file: 配对文件路径
        
    Returns:
        tuple: (pandas.DataFrame, original_data) 配对数据和原始数据
    """
    _, ext = os.path.splitext(pairs_file)
    
    if ext.lower() == '.csv':
        return pd.read_csv(pairs_file), None
    elif ext.lower() == '.json':
        with open(pairs_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 从JSON数据中提取需要的字段
        processed_data = []
        for item in data:
            # 从SMILES推断重原子数
            from_heavy_atoms = count_heavy_atoms(item['smiles_from'])
            to_heavy_atoms = count_heavy_atoms(item['smiles_to'])
            
            processed_data.append({
                'smiles_from': item['smiles_from'],
                'smiles_to': item['smiles_to'],
                'from_heavy_atoms': from_heavy_atoms,
                'to_heavy_atoms': to_heavy_atoms,
                # JSON格式中没有index信息，暂时设为NaN
                'index_from': np.nan,
                'index_to': np.nan
            })
        
        return pd.DataFrame(processed_data), data
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

def save_pairs_file(df, output_file, original_data=None, compact=False):
    """
    根据文件扩展名保存配对文件（支持CSV和JSON格式）
    
    Args:
        df: 要保存的DataFrame
        output_file: 输出文件路径
        original_data: 原始JSON数据（如果有的话）
        compact: 是否使用紧凑模式保存JSON
    """
    _, ext = os.path.splitext(output_file)
    
    if ext.lower() == '.csv':
        df.to_csv(output_file, index=False, encoding='utf-8')
    elif ext.lower() == '.json':
        if original_data is not None:
            # 合并原始数据和新属性
            result_data = []
            property_columns = [
                'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
                'U0', 'U', 'H', 'G', 'Cv'
            ]
            
            for i, item in enumerate(original_data):
                # 复制原始条目
                new_item = item.copy()
                
                # 添加属性变化
                for prop in property_columns:
                    change_col = f'{prop}_change'
                    pct_col = f'{prop}_change_pct'
                    
                    if change_col in df.columns:
                        new_item[change_col] = df.iloc[i][change_col]
                    if pct_col in df.columns:
                        new_item[pct_col] = df.iloc[i][pct_col]
                
                result_data.append(new_item)
            
            # 保存为JSON文件
            if compact:
                # 紧凑模式：一行一个item
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write('[\n')
                    for i, item in enumerate(result_data):
                        if i > 0:
                            f.write(',\n')
                        json.dump(item, f, ensure_ascii=False, separators=(',', ':'))
                    f.write('\n]')
            else:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
        else:
            # 将DataFrame转换为字典列表并保存为JSON
            if compact:
                df.to_json(output_file, orient='records', force_ascii=False, lines=True)
            else:
                df.to_json(output_file, orient='records', force_ascii=False, indent=2)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

def calculate_property_changes(pairs_file=None, output_file=None, compact=False):
    """
    根据配对文件计算属性变化
    
    Args:
        pairs_file: 配对文件路径，默认为 qm9-evo-pairs-step-1.csv
        output_file: 输出文件路径，默认为 qm9-evo-pairs-step-1-with-properties-pct.csv
        compact: 是否使用紧凑模式保存JSON
    """
    # 设置默认文件路径
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if pairs_file is None:
        pairs_file = os.path.join(data_dir, 'qm9-evo-pairs-step-1.csv')
    if output_file is None:
        # 根据输入文件格式确定输出文件格式
        _, ext = os.path.splitext(pairs_file)
        if ext.lower() == '.json':
            output_file = os.path.join(data_dir, 'qm9-evo-pairs-step-1-with-properties-pct.json')
        else:
            output_file = os.path.join(data_dir, 'qm9-evo-pairs-step-1-with-properties-pct.csv')
    
    # 加载配对文件
    print("加载配对文件...")
    pairs_df, original_data = load_pairs_file(pairs_file)
    print(f"总共 {len(pairs_df)} 对分子")
    
    # 获取所有需要的重原子数
    unique_heavy_atoms = set(pairs_df['from_heavy_atoms'].unique())
    unique_heavy_atoms.update(pairs_df['to_heavy_atoms'].unique())
    
    # 加载所有需要的重原子文件
    heavy_files = {}
    print("加载重原子文件...")
    for heavy_atoms in tqdm(unique_heavy_atoms, desc="加载重原子文件"):
        try:
            heavy_files[heavy_atoms] = load_heavy_atom_file(heavy_atoms)
        except FileNotFoundError as e:
            print(f"警告: {e}")
    
    # 属性列名（QM9数据集的属性，不包括omega1）
    property_columns = [
        'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
        'U0', 'U', 'H', 'G', 'Cv'
    ]
    
    # 存储属性变化值的列表
    property_changes = []
    property_changes_pct = []  # 存储属性变化百分比的列表
    
    # 计算每对分子的属性变化
    print("计算属性变化...")
    for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="计算属性变化"):
        from_heavy_atoms = row['from_heavy_atoms']
        to_heavy_atoms = row['to_heavy_atoms']
        index_from = row['index_from']
        index_to = row['index_to']
        
        # 初始化属性变化字典
        prop_changes = {}
        prop_changes_pct = {}  # 属性变化百分比字典
        
        # 获取源分子和目标分子的属性
        try:
            if from_heavy_atoms in heavy_files and not np.isnan(index_from) and index_from < len(heavy_files[from_heavy_atoms]):
                from_properties = heavy_files[from_heavy_atoms].iloc[int(index_from)]
            elif from_heavy_atoms in heavy_files and np.isnan(index_from):
                # 如果没有索引信息，尝试通过SMILES匹配查找
                from_smiles = row['smiles_from']
                from_properties_df = heavy_files[from_heavy_atoms]
                matched_rows = from_properties_df[from_properties_df['smiles'] == from_smiles]
                if len(matched_rows) > 0:
                    from_properties = matched_rows.iloc[0]
                else:
                    from_properties = None
            else:
                from_properties = None
                
            if to_heavy_atoms in heavy_files and not np.isnan(index_to) and index_to < len(heavy_files[to_heavy_atoms]):
                to_properties = heavy_files[to_heavy_atoms].iloc[int(index_to)]
            elif to_heavy_atoms in heavy_files and np.isnan(index_to):
                # 如果没有索引信息，尝试通过SMILES匹配查找
                to_smiles = row['smiles_to']
                to_properties_df = heavy_files[to_heavy_atoms]
                matched_rows = to_properties_df[to_properties_df['smiles'] == to_smiles]
                if len(matched_rows) > 0:
                    to_properties = matched_rows.iloc[0]
                else:
                    to_properties = None
            else:
                to_properties = None
            
            # 计算属性变化和变化百分比
            if from_properties is not None and to_properties is not None:
                for prop in property_columns:
                    if prop in from_properties and prop in to_properties:
                        try:
                            from_val = from_properties[prop]
                            to_val = to_properties[prop]
                            
                            # 计算绝对变化
                            prop_changes[f'{prop}_change'] = to_val - from_val
                            
                            # 计算相对变化百分比
                            # 特殊处理 from_val 为 0 的情况，避免除零错误
                            if from_val == 0:
                                # 如果起始值为0，使用绝对变化值
                                prop_changes_pct[f'{prop}_change_pct'] = to_val - from_val
                            else:
                                # 计算相对变化百分比
                                prop_changes_pct[f'{prop}_change_pct'] = (to_val - from_val) / from_val
                        except (TypeError, ValueError):
                            prop_changes[f'{prop}_change'] = np.nan
                            prop_changes_pct[f'{prop}_change_pct'] = np.nan
                    else:
                        prop_changes[f'{prop}_change'] = np.nan
                        prop_changes_pct[f'{prop}_change_pct'] = np.nan
            else:
                # 如果无法获取属性，则设置为NaN
                for prop in property_columns:
                    prop_changes[f'{prop}_change'] = np.nan
                    prop_changes_pct[f'{prop}_change_pct'] = np.nan
                    
        except Exception as e:
            # 出现异常时，设置为NaN
            for prop in property_columns:
                prop_changes[f'{prop}_change'] = np.nan
                prop_changes_pct[f'{prop}_change_pct'] = np.nan
        
        property_changes.append(prop_changes)
        property_changes_pct.append(prop_changes_pct)
    
    # 将属性变化添加到配对DataFrame中
    properties_df = pd.DataFrame(property_changes)
    properties_pct_df = pd.DataFrame(property_changes_pct)
    result_df = pd.concat([pairs_df, properties_df, properties_pct_df], axis=1)
    
    # 保存结果
    print(f"保存结果到 {output_file}...")
    save_pairs_file(result_df, output_file, original_data, compact)
    print("完成!")
    
    # 显示一些统计信息
    print("\n属性变化统计信息:")
    for prop in property_columns:
        change_col = f'{prop}_change'
        if change_col in result_df.columns:
            valid_changes = result_df[change_col].dropna()
            if len(valid_changes) > 0:
                print(f"{prop}: 有效值 {len(valid_changes)}, 平均变化 {valid_changes.mean():.6f}, 标准差 {valid_changes.std():.6f}")
            else:
                print(f"{prop}: 无有效值")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='计算分子对的属性变化')
    parser.add_argument('-i', '--input', help='输入配对文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('--compact', action='store_true', help='使用紧凑模式保存JSON文件（一行一个item）')
    
    args = parser.parse_args()
    
    calculate_property_changes(args.input, args.output, args.compact)

if __name__ == "__main__":
    main()