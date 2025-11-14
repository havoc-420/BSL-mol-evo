import pandas as pd
import os
import glob

def merge_qm9_csv_files():
    """
    合并所有的qm9_smiles_heavy_*_atoms.csv文件为一个大的数据表
    """
    # 定义文件路径模式
    file_pattern = "data/qm9_smiles_heavy_*_atoms.csv"
    
    # 获取所有匹配的文件
    all_files = sorted(glob.glob(file_pattern), key=lambda x: int(x.split('_')[-2]))
    
    print(f"找到 {len(all_files)} 个文件:")
    for file in all_files:
        print(f"  - {file}")
    
    # 读取所有文件并合并
    dataframes = []
    for file in all_files:
        df = pd.read_csv(file)
        dataframes.append(df)
        print(f"文件 {file} 包含 {len(df)} 行数据")
    
    # 合并所有DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # 保存合并后的数据
    output_file = "data/qm9_smiles_all_atoms.csv"
    merged_df.to_csv(output_file, index=False)
    
    print(f"\n合并完成!")
    print(f"总行数: {len(merged_df)}")
    print(f"输出文件: {output_file}")
    
    # 显示一些基本信息
    print(f"\n各原子数量分布:")
    print(merged_df['num_atoms'].value_counts().sort_index())

if __name__ == "__main__":
    merge_qm9_csv_files()