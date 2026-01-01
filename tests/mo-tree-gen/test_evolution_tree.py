#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子进化树生成和测试脚本
用于测试从起始分子A经过若干步操作后生成目标分子B'的过程
"""

import sys
import os
import argparse
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.molecular_evolution_expansion import MolecularEvolutionExpansion
from core.evolver import MoleculeEvolverAnalysis

def test_single_molecule_analysis(smiles):
    """
    测试单个分子的路径分析
    
    Args:
        smiles (str): 分子的SMILES表示
    """
    print(f"正在分析分子: {smiles}")
    
    try:
        analyzer = MoleculeEvolverAnalysis(smiles)
        path = analyzer.get_full_path_dict()
        
        print("\n分子构建路径:")
        for i, step in enumerate(path):
            print(f"  {i+1}. 操作: {step['operation']}, 位置: {step['position']}, 原子/片段: {step['atom']}")
        
        return path
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        return []

# INFO 
def test_evolution_generation(initial_smiles, num_paths=3, steps_per_path=3, config_file=None):
    """
    测试生成分子进化路径（全量扩展）
    
    Args:
        initial_smiles (str): 初始分子的SMILES
        num_paths (int): 要生成的路径数
        steps_per_path (int): 每条路径的步数
        config_file (str): 配置文件路径
    """
    print(f"初始分子: {initial_smiles}")
    print(f"生成路径数: {num_paths}, 每条路径步数: {steps_per_path}")
    if config_file:
        print(f"配置文件: {config_file}")
    
    try:
        expander = MolecularEvolutionExpansion(initial_smiles, config_file=config_file)
        paths = expander.generate_multiple_paths(
            num_paths=num_paths,
            steps_per_path=steps_per_path
        )
        
        print("\n生成的进化路径:")
        for i, path in enumerate(paths):
            print(f"\n路径 {i+1}:")
            print(f"  最终分子: {path['final_smiles']}")
            print(f"  路径长度: {path['path_length']} 步")
            
            print("  操作步骤:")
            for step in path["steps"]:
                print(f"    步骤 {step['step']}: {step['operation']} -> {step['smiles']}")
        
        return paths
    except Exception as e:
        print(f"生成进化路径时发生错误: {e}")
        return []

def test_evolution_tree(initial_smiles, max_depth=3, max_branching=None, config_file=None):
    """
    测试生成分子进化树（全量扩展）
    
    Args:
        initial_smiles (str): 初始分子的SMILES
        max_depth (int): 最大进化深度
        max_branching (int): 每个节点的最大分支数（为None表示全量扩展）
        config_file (str): 配置文件路径
    """
    print(f"初始分子: {initial_smiles}")
    print(f"最大深度: {max_depth}, 分支数: {'全量扩展' if max_branching is None else max_branching}")
    if config_file:
        print(f"配置文件: {config_file}")
    
    try:
        expander = MolecularEvolutionExpansion(initial_smiles, config_file=config_file)
        tree = expander.generate_evolution_tree(
            max_depth=max_depth,
            max_branching=max_branching  # None表示全量扩展
        )
        
        print(f"\n进化树统计:")
        print(f"  节点数: {len(tree['nodes'])}")
        print(f"  边数: {len(tree['edges'])}")
        
        if tree['error_stats']:
            print(f"  错误统计:")
            for error_type, count in tree['error_stats'].items():
                print(f"    {error_type}: {count}")
        
        print("\n进化树边 (演化步骤):")
        for i, edge in enumerate(tree['edges'][:10]):  # 只显示前10个
            print(f"  {i+1}. {edge['from']} -> {edge['to']}")
            print(f"     操作: {edge['operation']}")
            if edge['details']:
                print(f"     详情: {edge['details']}")
        
        if len(tree['edges']) > 10:
            print(f"  ... 还有 {len(tree['edges']) - 10} 个边")
        
        return tree
    except Exception as e:
        print(f"生成进化树时发生错误: {e}")
        return {}

def compare_with_dataset_format(path):
    """
    检查路径格式是否与数据集一致
    
    Args:
        path (list): 路径操作列表
    """
    print("\n检查路径格式是否与数据集一致:")
    required_keys = {"position", "atom", "operation"}
    
    is_consistent = True
    for i, step in enumerate(path):
        if not isinstance(step, dict):
            print(f"  步骤 {i+1}: 格式不正确，应为字典类型")
            is_consistent = False
            continue
            
        if not required_keys.issubset(step.keys()):
            missing_keys = required_keys - step.keys()
            print(f"  步骤 {i+1}: 缺少必要键 {missing_keys}")
            is_consistent = False
            
        # 检查操作类型是否在已知列表中（已移除 add_fragment）
        known_operations = {
            "replace_atom", "add_atom", "form_double_bond", 
            "form_triple_bond", "form_ring", "form_double_ring", "form_triple_ring",
            "form_aromatic_ring", "add_stereo", "init_atom", "remove_atom",
            "remove_form_double_bond", "remove_form_triple_bond",
            "remove_form_ring", "remove_form_double_ring", "remove_form_triple_ring",
            "remove_form_aromatic_ring", "remove_add_stereo"
        }
        
        if step.get("operation") not in known_operations:
            print(f"  步骤 {i+1}: 未知操作类型 '{step.get('operation')}'")
            is_consistent = False
    
    if is_consistent:
        print("  路径格式与数据集一致")
    else:
        print("  路径格式与数据集不一致")
    
    return is_consistent

def main():
    parser = argparse.ArgumentParser(description='分子进化树测试脚本')
    parser.add_argument('smiles', help='起始分子的SMILES')
    
    parser.add_argument('--test-type', choices=['analysis', 'evolution', 'tree', 'format'], 
                       default='evolution', help='测试类型')
    
    parser.add_argument('--num-paths', type=int, default=3, 
                       help='要生成的路径数 (用于evolution)')
    
    parser.add_argument('--steps-per-path', type=int, default=3,
                       help='每条路径的步数 (用于evolution)')
    
    parser.add_argument('--max-depth', type=int, default=3,
                       help='最大进化深度 (用于tree)')
    
    parser.add_argument('--max-branching', type=int, default=None,
                       help='每个节点的最大分支数，None表示全量扩展 (用于tree)')
    
    parser.add_argument('--config-file', type=str, default=None,
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    if args.test_type == 'analysis':
        path = test_single_molecule_analysis(args.smiles)
        compare_with_dataset_format(path)
    elif args.test_type == 'evolution':
        test_evolution_generation(args.smiles, args.num_paths, args.steps_per_path, args.config_file)
    elif args.test_type == 'tree':
        test_evolution_tree(args.smiles, args.max_depth, args.max_branching, args.config_file)
    elif args.test_type == 'format':
        # 格式检查测试
        analyzer = MoleculeEvolverAnalysis(args.smiles)
        path = analyzer.get_full_path_dict()
        compare_with_dataset_format(path)

if __name__ == "__main__":
    # 示例用法
    if len(sys.argv) == 1:
        print("分子进化树测试脚本")
        print("=" * 50)
        
        # 示例1: 分析单个分子
        print("\n1. 分析单个分子 (苯分子):")
        path = test_single_molecule_analysis("c1ccccc1")
        compare_with_dataset_format(path)
        
        # 示例2: 生成进化路径
        print("\n\n2. 生成进化路径:")
        test_evolution_generation("CC", num_paths=2, steps_per_path=2)
        
        # 示例3: 生成进化树（全量扩展）
        print("\n\n3. 生成进化树（全量扩展）:")
        test_evolution_tree("CC", max_depth=2, max_branching=None)
        
        print("\n\n使用命令行参数运行以获得更多信息:")
        print("  python test_evolution_tree.py --help")
    else:
        main()