#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
构建测试集上的 AB 优化对
根据索引文件过滤测试集，然后在过滤后的数据上构建优化对
"""

import json
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm


def process_json_data_for_optimization(data_path, indices_path):
    """
    处理JSON数据以用于优化对构建
    
    Args:
        data_path (str): 数据文件路径
        indices_path (str): 索引文件路径
        
    Returns:
        pd.DataFrame: 处理后的数据框
    """
    # 读取数据
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    
    # 读取索引
    with open(indices_path, 'r') as f:
        indices_data = json.load(f)
    
    test_indices = indices_data.get('test_indices', [])
    
    # 过滤测试集数据
    test_data = [raw_data[i] for i in test_indices if i < len(raw_data)]
    
    # 提取SMILES对和属性变化
    records = []
    for item in tqdm(test_data, desc="处理测试数据"):
        record = {
            'smiles_from': item['smiles_from'],
            'smiles_to': item['smiles_to'],
            'homo_change': item['homo_change'],
            'lumo_change': item['lumo_change'],
            'gap_change': item['gap_change'],
            'homo_change_pct': item['homo_change_pct'],
            'lumo_change_pct': item['lumo_change_pct'],
            'gap_change_pct': item['gap_change_pct']
        }
        records.append(record)
    
    return pd.DataFrame(records)


def create_molecule_dataframe(json_data_path, indices_path):
    """
    从JSON数据创建分子数据框，包含起始和目标分子的属性
    
    Args:
        json_data_path (str): JSON数据文件路径
        indices_path (str): 索引文件路径
        
    Returns:
        pd.DataFrame: 分子数据框
    """
    # 读取数据
    with open(json_data_path, 'r') as f:
        raw_data = json.load(f)
    
    # 读取索引
    with open(indices_path, 'r') as f:
        indices_data = json.load(f)
    
    # 读取QM9原始属性数据
    qm9_df = pd.read_csv('mol_evo/dataset/data/qm9_smiles_all_atoms.csv')
    # 创建SMILES到属性的映射
    qm9_smiles_dict = {}
    for _, row in tqdm(qm9_df.iterrows(), desc="构建QM9属性映射", total=len(qm9_df)):
        qm9_smiles_dict[row['smiles']] = {
            'homo': row['homo'],
            'lumo': row['lumo'],
            'gap': row['gap']
        }
    
    test_indices = indices_data.get('test_indices', [])
    
    # 过滤测试集数据
    test_data = [raw_data[i] for i in test_indices if i < len(raw_data)]
    
    # 收集所有唯一的SMILES及其属性
    smiles_dict = {}
    
    for item in tqdm(test_data, desc="收集分子属性"):
        # 处理起始分子
        from_smiles = item['smiles_from']
        if from_smiles not in smiles_dict:
            # 从QM9数据中获取原始属性
            if from_smiles in qm9_smiles_dict:
                smiles_dict[from_smiles] = {
                    'smiles': from_smiles,
                    'homo': qm9_smiles_dict[from_smiles]['homo'],
                    'lumo': qm9_smiles_dict[from_smiles]['lumo'],
                    'gap': qm9_smiles_dict[from_smiles]['gap']
                }
            else:
                # 如果在QM9数据中找不到，使用默认值0
                smiles_dict[from_smiles] = {
                    'smiles': from_smiles,
                    'homo': 0.0,
                    'lumo': 0.0,
                    'gap': 0.0
                }
        
        # 处理目标分子
        to_smiles = item['smiles_to']
        if to_smiles not in smiles_dict:
            # 从QM9数据中获取原始属性
            if to_smiles in qm9_smiles_dict:
                smiles_dict[to_smiles] = {
                    'smiles': to_smiles,
                    'homo': qm9_smiles_dict[to_smiles]['homo'],
                    'lumo': qm9_smiles_dict[to_smiles]['lumo'],
                    'gap': qm9_smiles_dict[to_smiles]['gap']
                }
            else:
                # 如果在QM9数据中找不到，使用默认值0
                smiles_dict[to_smiles] = {
                    'smiles': to_smiles,
                    'homo': 0.0,
                    'lumo': 0.0,
                    'gap': 0.0
                }
    
    # 转换为列表
    molecules = list(smiles_dict.values())
    return pd.DataFrame(molecules)


def main():
    # 导入时间模块用于生成时间戳
    import time
    import os
    
    # 文件路径
    indices_path = 'mol_evo/dataset/data/dataset_indices/indices_20251122_213847_seed42.json'
    data_path = 'mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.json'
    
    # 创建分子数据框
    print("正在处理数据...")
    df = create_molecule_dataframe(data_path, indices_path)
    print(f"处理完成，共 {len(df)} 个唯一分子")
    
    # 生成时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 输出目录 - 创建以时间戳命名的子文件夹
    output_base_dir = 'mol_evo/dataset/eval-data'
    output_dir = os.path.join(output_base_dir, timestamp)
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存处理后的数据
    df.to_csv(os.path.join(output_dir, 'qm9_test_molecules.csv'), index=False)
    print(f"分子数据已保存到 {os.path.join(output_dir, 'qm9_test_molecules.csv')}")
    
    # 从处理后的CSV创建优化对提取器
    from ex_ab import QM9OptimizationPairs
    extractor = QM9OptimizationPairs(csv_path=os.path.join(output_dir, 'qm9_test_molecules.csv'))
    
    # 为所有属性创建优化对
    print("开始构建优化对...")
    homo_pairs, lumo_pairs, gap_pairs = extractor.create_all_optimization_pairs(
        output_prefix=os.path.join(output_dir, 'qm9_test_ab_pairs')
    )
    
    # 分析结果
    extractor.analyze_pairs(homo_pairs, 'HOMO')
    extractor.analyze_pairs(lumo_pairs, 'LUMO') 
    extractor.analyze_pairs(gap_pairs, 'GAP')
    
    print("优化对构建完成！")


if __name__ == "__main__":
    main()