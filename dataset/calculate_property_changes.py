import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def load_heavy_atom_file(heavy_atoms):
    """
    加载指定重原子数的CSV文件
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    file_path = os.path.join(data_dir, f'qm9_smiles_heavy_{heavy_atoms}_atoms.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    return pd.read_csv(file_path)

def calculate_property_changes(pairs_file=None, output_file=None):
    """
    根据配对文件计算属性变化
    
    Args:
        pairs_file: 配对文件路径，默认为 qm9-evo-pairs-step-1.csv
        output_file: 输出文件路径，默认为 qm9-evo-pairs-step-1-with-properties.csv
    """
    # 设置默认文件路径
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if pairs_file is None:
        pairs_file = os.path.join(data_dir, 'qm9-evo-pairs-step-1.csv')
    if output_file is None:
        output_file = os.path.join(data_dir, 'qm9-evo-pairs-step-1-with-properties.csv')
    
    # 加载配对文件
    print("加载配对文件...")
    pairs_df = pd.read_csv(pairs_file)
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
    
    # 计算每对分子的属性变化
    print("计算属性变化...")
    for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="计算属性变化"):
        from_heavy_atoms = row['from_heavy_atoms']
        to_heavy_atoms = row['to_heavy_atoms']
        index_from = row['index_from']
        index_to = row['index_to']
        
        # 初始化属性变化字典
        prop_changes = {}
        
        # 获取源分子和目标分子的属性
        try:
            if from_heavy_atoms in heavy_files and index_from < len(heavy_files[from_heavy_atoms]):
                from_properties = heavy_files[from_heavy_atoms].iloc[index_from]
            else:
                from_properties = None
                
            if to_heavy_atoms in heavy_files and index_to < len(heavy_files[to_heavy_atoms]):
                to_properties = heavy_files[to_heavy_atoms].iloc[index_to]
            else:
                to_properties = None
            
            # 计算属性变化
            if from_properties is not None and to_properties is not None:
                for prop in property_columns:
                    if prop in from_properties and prop in to_properties:
                        try:
                            prop_changes[f'{prop}_change'] = to_properties[prop] - from_properties[prop]
                        except (TypeError, ValueError):
                            prop_changes[f'{prop}_change'] = np.nan
                    else:
                        prop_changes[f'{prop}_change'] = np.nan
            else:
                # 如果无法获取属性，则设置为NaN
                for prop in property_columns:
                    prop_changes[f'{prop}_change'] = np.nan
                    
        except Exception as e:
            # 出现异常时，设置为NaN
            for prop in property_columns:
                prop_changes[f'{prop}_change'] = np.nan
        
        property_changes.append(prop_changes)
    
    # 将属性变化添加到配对DataFrame中
    properties_df = pd.DataFrame(property_changes)
    result_df = pd.concat([pairs_df, properties_df], axis=1)
    
    # 保存结果
    print(f"保存结果到 {output_file}...")
    result_df.to_csv(output_file, index=False, encoding='utf-8')
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
    calculate_property_changes()

if __name__ == "__main__":
    main()