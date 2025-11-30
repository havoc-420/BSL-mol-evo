#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成与训练数据格式一致的分子演化数据（适应全量扩展树生成，已移除 add_fragment 操作）
"""

import sys
import os
import json
import random
from rdkit import Chem

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.evolver import MolecularEvolutionExpansion, MoleculeEvolverAnalysis

def generate_evolution_pair(start_smiles, target_smiles):
    """
    生成一对分子演化数据
    
    Args:
        start_smiles (str): 起始分子SMILES
        target_smiles (str): 目标分子SMILES
        
    Returns:
        dict: 演化数据
    """
    try:
        # 分析目标分子的构建路径
        analyzer = MoleculeEvolverAnalysis(target_smiles)
        operations = analyzer.get_full_path_dict()
        
        # 确保不包含 add_fragment 操作
        filtered_operations = [op for op in operations if op.get('operation') != 'add_fragment']
        
        # 生成演化对
        pair = {
            "smiles_from": start_smiles,
            "smiles_to": target_smiles,
            "operations": filtered_operations  # 使用过滤后的操作
        }
        
        return pair
    except Exception as e:
        print(f"生成演化对时出错: {e}")
        return None

def generate_multiple_evolution_pairs(initial_smiles_list, num_pairs=10):
    """
    生成多个演化对（适应全量扩展树生成）
    
    Args:
        initial_smiles_list (list): 初始分子SMILES列表
        num_pairs (int): 要生成的对数
        
    Returns:
        list: 演化对列表
    """
    pairs = []
    
    # 创建演化扩展器
    expander = MolecularEvolutionExpansion()
    
    for i in range(num_pairs):
        # 随机选择一个初始分子
        initial_smiles = random.choice(initial_smiles_list)
        
        try:
            # 生成演化路径（使用全量扩展）
            paths = expander.generate_multiple_paths(
                num_paths=1,
                steps_per_path=random.randint(1, 5),
                max_branching=None  # None表示全量扩展
            )
            
            if paths:
                path = paths[0]
                target_smiles = path['final_smiles']
                
                # 生成演化对
                pair = generate_evolution_pair(initial_smiles, target_smiles)
                if pair:
                    pairs.append(pair)
                    
        except Exception as e:
            print(f"生成第 {i+1} 对演化数据时出错: {e}")
            continue
    
    return pairs

def convert_to_training_format(evolution_pairs):
    """
    将演化对转换为训练数据格式（已移除 add_fragment 操作）
    
    Args:
        evolution_pairs (list): 演化对列表
        
    Returns:
        list: 训练数据
    """
    training_data = []
    
    for pair in evolution_pairs:
        # 过滤掉 add_fragment 操作
        filtered_operations = [op for op in pair["operations"] if op.get("operation") != "add_fragment"]
        
        # 基本结构与数据集一致
        training_item = {
            "smiles_from": pair["smiles_from"],
            "smiles_to": pair["smiles_to"],
            "operations": filtered_operations,  # 使用过滤后的操作
            # 添加占位符属性变化数据（实际应用中需要计算真实的分子属性变化）
            "A_change": 0.0,
            "A_change_pct": 0.0,
            "B_change": 0.0,
            "B_change_pct": 0.0,
            "C_change": 0.0,
            "C_change_pct": 0.0,
            "mu_change": 0.0,
            "mu_change_pct": 0.0,
            "alpha_change": 0.0,
            "alpha_change_pct": 0.0,
            "homo_change": 0.0,
            "homo_change_pct": 0.0,
            "lumo_change": 0.0,
            "lumo_change_pct": 0.0,
            "gap_change": 0.0,
            "gap_change_pct": 0.0,
            "r2_change": 0.0,
            "r2_change_pct": 0.0,
            "zpve_change": 0.0,
            "zpve_change_pct": 0.0,
            "U0_change": 0.0,
            "U0_change_pct": 0.0,
            "U_change": 0.0,
            "U_change_pct": 0.0,
            "H_change": 0.0,
            "H_change_pct": 0.0,
            "G_change": 0.0,
            "G_change_pct": 0.0,
            "Cv_change": 0.0,
            "Cv_change_pct": 0.0
        }
        
        training_data.append(training_item)
    
    return training_data

def save_training_data(training_data, output_file):
    """
    保存训练数据
    
    Args:
        training_data (list): 训练数据
        output_file (str): 输出文件路径
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        print(f"训练数据已保存到: {output_file}")
    except Exception as e:
        print(f"保存训练数据时出错: {e}")

def main():
    print("生成训练数据（适应全量扩展树生成，已移除 add_fragment 操作）")
    print("=" * 50)
    
    # 定义初始分子列表
    initial_molecules = [
        "C",    # 甲烷
        "O",    # 水
        "N",    # 氨
        "CC",   # 乙烷
        "CO",   # 甲醇
        "CN",   # 乙腈
        "C=C",  # 乙烯
        "C#C",  # 乙炔
        "C=O",  # 甲醛
        "CCO",  # 乙醇
        "CCC"   # 丙烷
    ]
    
    print("1. 生成演化对...")
    evolution_pairs = generate_multiple_evolution_pairs(initial_molecules, num_pairs=20)
    print(f"生成了 {len(evolution_pairs)} 对演化数据")
    
    # 显示几个示例
    print("\n2. 演化对示例:")
    for i, pair in enumerate(evolution_pairs[:3]):
        print(f"   示例 {i+1}:")
        print(f"     从: {pair['smiles_from']}")
        print(f"     到: {pair['smiles_to']}")
        print(f"     操作数: {len(pair['operations'])}")
        if pair['operations']:
            print(f"     第一个操作: {pair['operations'][0]['operation']}")
        print()
    
    print("3. 转换为训练数据格式...")
    training_data = convert_to_training_format(evolution_pairs)
    
    print("4. 保存训练数据...")
    output_file = "generated_training_data_full_expansion.json"
    save_training_data(training_data, output_file)
    
    print("\n生成完成!")
    print(f"总共生成了 {len(training_data)} 条训练数据")
    print(f"数据已保存到: {output_file}")
    print("注意：已移除所有 add_fragment 操作，确保与数据集格式一致")
    print("注意：使用全量扩展树生成，不依赖操作权重")

if __name__ == "__main__":
    main()