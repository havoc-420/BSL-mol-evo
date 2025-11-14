#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 MolecularEvolutionExpansion 类从分子A生成更多可操作的分子B
"""

import sys
import os
import json

# 添加项目根目录到路径中，以便可以导入 mol_evo 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# 禁用RDKit的详细日志输出
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from mol_evo.core.evolver import MolecularEvolutionExpansion


def test_basic_evolution():
    """测试基本的分子进化功能"""
    print("=== 测试基本分子进化功能 ===")
    
    # 使用简单的分子作为起始点
    initial_smiles = "CC"  # 乙烷
    expander = MolecularEvolutionExpansion(initial_smiles)
    
    print(f"初始分子: {initial_smiles}")
    
    # 生成多条进化路径
    paths = expander.generate_multiple_paths(
        num_paths=3,
        steps_per_path=3,
        max_branching=2,
        diversity_threshold=0.7
    )
    
    print(f"生成了 {len(paths)} 条路径")
    
    for i, path in enumerate(paths):
        print(f"\n路径 {i+1}:")
        print(f"  初始分子: {path['initial_smiles']}")
        print(f"  最终分子: {path['final_smiles']}")
        print(f"  路径长度: {path['path_length']}")
        print("  步骤:")
        for step in path['steps']:
            print(f"    步骤 {step['step']}: {step['operation']} -> {step['smiles']}")


def test_different_starting_molecules():
    """测试不同的起始分子"""
    print("\n=== 测试不同的起始分子 ===")
    
    test_molecules = ["CCO", "c1ccccc1", "CC(=O)O"]  # 乙醇, 苯, 乙酸
    
    for smiles in test_molecules:
        print(f"\n测试起始分子: {smiles}")
        try:
            expander = MolecularEvolutionExpansion(smiles)
            paths = expander.generate_multiple_paths(
                num_paths=2,
                steps_per_path=2,
                max_branching=2
            )
            
            print(f"  成功生成 {len(paths)} 条路径")
            for i, path in enumerate(paths):
                print(f"    路径 {i+1} 最终分子: {path['final_smiles']}")
        except Exception as e:
            print(f"  错误: {e}")


def test_evolution_tree():
    """测试进化树生成功能"""
    print("\n=== 测试进化树生成功能 ===")
    
    expander = MolecularEvolutionExpansion("CC")
    tree = expander.generate_evolution_tree(max_depth=3, max_branching=2)
    
    print(f"进化树统计:")
    print(f"  节点数: {len(tree['nodes'])}")
    print(f"  边数: {len(tree['edges'])}")
    print(f"  根节点: {tree['root']}")
    
    # 显示前几个节点和边
    print("\n前5个节点:")
    for i, (smiles, node_info) in enumerate(list(tree['nodes'].items())[:5]):
        print(f"  {i+1}. {smiles} (深度: {node_info['depth']})")
    
    print("\n前5条边:")
    for i, edge in enumerate(tree['edges'][:5]):
        print(f"  {i+1}. {edge['from']} --{edge['operation']}--> {edge['to']}")


def test_path_diversity():
    """测试路径多样性分析"""
    print("\n=== 测试路径多样性分析 ===")
    
    expander = MolecularEvolutionExpansion("CCO")
    paths = expander.generate_multiple_paths(
        num_paths=5,
        steps_per_path=3
    )
    
    diversity = expander.analyze_path_diversity(paths)
    
    print("多样性分析结果:")
    print(f"  多样性分数: {diversity['diversity_score']:.3f}")
    print(f"  唯一分子数: {diversity['unique_molecules']}")
    print(f"  总路径数: {diversity['total_paths']}")
    print(f"  平均路径长度: {diversity['average_path_length']:.2f}")


def test_possible_operations():
    """测试获取可能操作的功能"""
    print("\n=== 测试获取可能操作 ===")
    
    expander = MolecularEvolutionExpansion("CC")
    mol = expander.initial_mol
    
    operations = expander._get_possible_operations(mol)
    
    print(f"对于分子 CC，找到 {len(operations)} 种可能的操作:")
    
    # 统计不同类型的操作数量
    operation_counts = {}
    for op in operations:
        op_type = op['type']
        operation_counts[op_type] = operation_counts.get(op_type, 0) + 1
    
    for op_type, count in operation_counts.items():
        print(f"  {op_type}: {count} 种")


def main():
    """主测试函数"""
    print("MolecularEvolutionExpansion 测试脚本")
    print("=" * 50)
    
    try:
        test_basic_evolution()
        test_different_starting_molecules()
        test_evolution_tree()
        test_path_diversity()
        test_possible_operations()
        
        print("\n" + "=" * 50)
        print("所有测试完成!")
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()