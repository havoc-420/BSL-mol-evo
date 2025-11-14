#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从分子A生成可操作的分子B的测试脚本
重点关注分子操作和转换过程
"""

import sys
import os
import json
from collections import defaultdict

# 添加项目根目录到路径中，以便可以导入 mol_evo 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from rdkit import Chem
from rdkit import RDLogger
from mol_evo.core.evolver import MolecularEvolutionExpansion

# 禁用RDKit的详细日志输出
RDLogger.DisableLog('rdApp.*')


def demonstrate_molecule_operations():
    """
    演示从一个起始分子生成多个可操作的分子B
    展示不同的分子操作类型
    """
    print("分子操作演示")
    print("=" * 40)
    
    # 定义起始分子A
    initial_smiles = "CCO"  # 乙醇
    print(f"起始分子A: {initial_smiles}")
    
    # 创建MolecularEvolutionExpansion实例
    expander = MolecularEvolutionExpansion(initial_smiles)
    
    # 获取起始分子
    initial_mol = expander.initial_mol
    print(f"起始分子原子数: {initial_mol.GetNumAtoms()}")
    
    # 展示不同类型的分子操作
    
    # 1. 添加原子操作
    print("\n1. 添加原子操作:")
    new_mol = expander._add_atom_operation(initial_mol, atom_symbol='C', connect_to=0)
    if new_mol:
        new_smiles = Chem.MolToSmiles(new_mol)
        print(f"   添加碳原子后: {new_smiles}")
    
    # 2. 替换原子操作
    print("\n2. 替换原子操作:")
    new_mol = expander._replace_atom_operation(initial_mol, atom_idx=0, new_symbol='N')
    if new_mol:
        new_smiles = Chem.MolToSmiles(new_mol)
        print(f"   将第一个原子替换为氮: {new_smiles}")
    
    # 3. 形成键操作
    print("\n3. 形成键操作:")
    # 先添加一个原子，然后形成键
    temp_mol = expander._add_atom_operation(initial_mol, atom_symbol='O', connect_to=0)
    if temp_mol and temp_mol.GetNumAtoms() > initial_mol.GetNumAtoms():
        # 尝试形成额外的键
        new_mol = expander._form_bond_operation(temp_mol, "form_double_bond")
        if new_mol:
            new_smiles = Chem.MolToSmiles(new_mol)
            print(f"   形成双键后: {new_smiles}")
    
    # 4. 添加片段操作
    print("\n4. 添加片段操作:")
    fragment_names = list(expander.common_fragments.keys())[:3]  # 只取前3个片段演示
    for frag_name in fragment_names:
        new_mol = expander._add_fragment_operation(initial_mol, fragment_name=frag_name, connect_to=0)
        if new_mol:
            new_smiles = Chem.MolToSmiles(new_mol)
            print(f"   添加{frag_name}片段后: {new_smiles}")
    
    # 5. 断开键操作 (只在有足够键的情况下)
    print("\n5. 断开键操作:")
    if initial_mol.GetNumBonds() > 0:
        new_mol = expander._break_bond_operation(initial_mol, atom1_idx=0)
        if new_mol:
            new_smiles = Chem.MolToSmiles(new_mol)
            print(f"   断开键后: {new_smiles}")


def generate_evolution_paths():
    """
    生成从分子A到多个分子B的进化路径
    """
    print("\n\n分子进化路径生成")
    print("=" * 40)
    
    # 使用不同的起始分子
    test_molecules = ["CCO", "c1ccccc1", "CC(=O)O"]
    
    for start_mol in test_molecules:
        print(f"\n起始分子: {start_mol}")
        expander = MolecularEvolutionExpansion(start_mol)
        
        # 生成进化路径
        paths = expander.generate_multiple_paths(
            num_paths=2,           # 生成2条路径
            steps_per_path=3,      # 每条路径3步
            max_branching=2,       # 最大分支数
            diversity_threshold=0.7  # 多样性阈值
        )
        
        print(f"生成 {len(paths)} 条路径:")
        for i, path in enumerate(paths):
            print(f"  路径 {i+1}:")
            print(f"    最终分子: {path['final_smiles']}")
            print(f"    路径长度: {path['path_length']} 步")
            # 显示操作步骤
            for step in path['steps']:
                if step['step'] > 0:  # 跳过初始步骤
                    print(f"      步骤 {step['step']}: {step['operation']}")


def explore_operation_space():
    """
    探索分子的操作空间，展示可能的操作类型和结果
    """
    print("\n\n操作空间探索")
    print("=" * 40)
    
    # 选择一个测试分子
    smiles = "CCOC"  # 甲乙醚
    expander = MolecularEvolutionExpansion(smiles)
    mol = expander.initial_mol
    
    print(f"分析分子: {smiles}")
    print(f"原子数: {mol.GetNumAtoms()}")
    print(f"键数: {mol.GetNumBonds()}")
    
    # 获取所有可能的操作
    operations = expander._get_possible_operations(mol)
    print(f"\n可能的操作总数: {len(operations)}")
    
    # 按类型统计操作
    operation_types = {}
    for op in operations:
        op_type = op['type']
        if op_type not in operation_types:
            operation_types[op_type] = []
        operation_types[op_type].append(op)
    
    print("\n按类型分类的操作:")
    for op_type, ops in operation_types.items():
        print(f"  {op_type}: {len(ops)} 种可能")
        # 显示前2个示例
        for i, op in enumerate(ops[:2]):
            params = op.get('params', {})
            print(f"    示例 {i+1}: {params}")


def generate_evolution_tree_with_stats():
    """
    生成进化树并显示错误统计信息
    """
    print("\n\n进化树生成与错误统计")
    print("=" * 40)
    
    # 使用一个简单的起始分子
    initial_smiles = "CCO"
    print(f"起始分子: {initial_smiles}")
    
    expander = MolecularEvolutionExpansion(initial_smiles)
    
    # 生成进化树
    tree = expander.generate_evolution_tree(max_depth=3, max_branching=2)
    
    print(f"进化树统计:")
    print(f"  节点数: {len(tree['nodes'])}")
    print(f"  边数: {len(tree['edges'])}")
    print(f"  根节点: {tree['root']}")
    
    # 显示错误统计
    error_stats = tree.get('error_stats', {})
    if error_stats:
        print(f"\n错误统计:")
        print(f"  总尝试次数: {error_stats.get('total_attempts', 0)}")
        print(f"  价态错误: {error_stats.get('valence_errors', 0)}")
        print(f"  Kekulization错误: {error_stats.get('kekulization_errors', 0)}")
        print(f"  其他错误: {error_stats.get('other_errors', 0)}")
        
        # 计算成功率
        total_attempts = error_stats.get('total_attempts', 0)
        total_errors = (error_stats.get('valence_errors', 0) + 
                       error_stats.get('kekulization_errors', 0) + 
                       error_stats.get('other_errors', 0))
        if total_attempts > 0:
            success_rate = ((total_attempts - total_errors) / total_attempts) * 100
            print(f"  操作成功率: {success_rate:.2f}%")
    
    # 显示前几个节点和边
    print("\n前5个节点:")
    for i, (smiles, node_info) in enumerate(list(tree['nodes'].items())[:5]):
        print(f"  {i+1}. {smiles} (深度: {node_info['depth']})")
    
    print("\n前5条边:")
    for i, edge in enumerate(tree['edges'][:5]):
        print(f"  {i+1}. {edge['from']} --{edge['operation']}--> {edge['to']}")


def main():
    """
    主函数 - 运行所有测试
    """
    print("MolecularEvolutionExpansion 测试: 从分子A生成可操作的分子B")
    
    try:
        demonstrate_molecule_operations()
        generate_evolution_paths()
        explore_operation_space()
        generate_evolution_tree_with_stats()
        
        print("\n" + "=" * 50)
        print("测试完成!")
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()