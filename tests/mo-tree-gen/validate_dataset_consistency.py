#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证evolver.py生成的路径与数据集格式一致性的脚本（适应全量扩展树生成）
"""

import sys
import os
import json
from collections import Counter

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.evolver import MolecularEvolutionExpansion, MoleculeEvolverAnalysis

def load_sample_dataset(dataset_path):
    """
    加载数据集样本
    
    Args:
        dataset_path (str): 数据集文件路径
        
    Returns:
        list: 数据集样本
    """
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            # 由于文件可能很大，只读取前几个样本
            content = f.read()
            # 找到前几个完整的JSON对象
            samples = []
            brace_count = 0
            start = 0
            
            for i, char in enumerate(content):
                if char == '{':
                    if brace_count == 0:
                        start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # 找到一个完整的JSON对象
                        try:
                            sample = json.loads(content[start:i+1])
                            samples.append(sample)
                            if len(samples) >= 5:  # 只需要几个样本进行分析
                                break
                        except json.JSONDecodeError:
                            continue
            return samples
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return []

def analyze_dataset_operations(dataset_samples):
    """
    分析数据集中的操作类型
    
    Args:
        dataset_samples (list): 数据集样本列表
        
    Returns:
        dict: 操作类型统计信息
    """
    operation_counter = Counter()
    position_formats = Counter()
    
    for sample in dataset_samples:
        if 'operations' in sample:
            for op in sample['operations']:
                operation_counter[op['operation']] += 1
                # 分析位置格式
                pos = op.get('position', '')
                if '-' in pos:
                    position_formats['bond'] += 1
                elif pos.isdigit():
                    position_formats['atom'] += 1
                else:
                    position_formats['other'] += 1
    
    return {
        'operations': operation_counter,
        'positions': position_formats
    }

def analyze_generated_path(mol_smiles):
    """
    分析生成的路径
    
    Args:
        mol_smiles (str): 分子SMILES
        
    Returns:
        dict: 生成路径的统计信息
    """
    try:
        analyzer = MoleculeEvolverAnalysis(mol_smiles)
        path = analyzer.get_full_path_dict()
        
        operation_counter = Counter()
        position_formats = Counter()
        
        for step in path:
            operation_counter[step['operation']] += 1
            # 分析位置格式
            pos = step.get('position', '')
            if '-' in pos:
                position_formats['bond'] += 1
            elif pos.isdigit() or (pos and pos.isdigit()):
                position_formats['atom'] += 1
            else:
                position_formats['other'] += 1
                
        return {
            'operations': operation_counter,
            'positions': position_formats,
            'total_steps': len(path)
        }
    except Exception as e:
        print(f"分析生成路径时出错: {e}")
        return {}

def compare_formats(dataset_stats, generated_stats):
    """
    比较数据集和生成路径的格式
    
    Args:
        dataset_stats (dict): 数据集统计信息
        generated_stats (dict): 生成路径统计信息
    """
    print("数据集与生成路径格式对比:")
    print("=" * 50)
    
    print("\n1. 操作类型对比:")
    print("   数据集操作类型:")
    for op, count in dataset_stats['operations'].most_common():
        print(f"     {op}: {count}")
        
    print("   生成路径操作类型:")
    for op, count in generated_stats['operations'].most_common():
        print(f"     {op}: {count}")
    
    # 检查是否有数据集中有但生成路径中没有的操作类型
    dataset_ops = set(dataset_stats['operations'].keys())
    generated_ops = set(generated_stats['operations'].keys())
    
    missing_in_generated = dataset_ops - generated_ops
    extra_in_generated = generated_ops - dataset_ops
    
    if missing_in_generated:
        print(f"   数据集中有但生成路径中缺少的操作类型: {missing_in_generated}")
    
    if extra_in_generated:
        print(f"   生成路径中有但数据集中没有的操作类型: {extra_in_generated}")
    
    print("\n2. 位置格式对比:")
    print("   数据集位置格式:")
    for fmt, count in dataset_stats['positions'].items():
        print(f"     {fmt}: {count}")
        
    print("   生成路径位置格式:")
    for fmt, count in generated_stats['positions'].items():
        print(f"     {fmt}: {count}")

def validate_operation_consistency(op_list):
    """
    验证操作的一致性（已移除 add_fragment 操作）
    
    Args:
        op_list (list): 操作列表
        
    Returns:
        bool: 是否一致
    """
    required_fields = {'position', 'atom', 'operation'}
    valid = True
    
    for i, op in enumerate(op_list):
        if not isinstance(op, dict):
            print(f"操作 {i} 不是字典类型")
            valid = False
            continue
            
        if not required_fields.issubset(op.keys()):
            missing = required_fields - op.keys()
            print(f"操作 {i} 缺少字段: {missing}")
            valid = False
            
        # 验证操作类型（已移除 add_fragment）
        valid_operations = {
            'replace_atom', 'add_atom', 'form_double_bond',
            'form_triple_bond', 'form_ring', 'form_double_ring', 'form_triple_ring',
            'form_aromatic_ring', 'add_stereo', 'init_atom', 'remove_atom',
            'remove_form_double_bond', 'remove_form_triple_bond',
            'remove_form_ring', 'remove_form_double_ring', 'remove_form_triple_ring',
            'remove_form_aromatic_ring', 'remove_add_stereo', 'break_bond', 'form_bond',
            'form_dative_bond', 'form_aromatic_bond'
        }
        
        if op.get('operation') not in valid_operations:
            print(f"操作 {i} 有无效操作类型: {op.get('operation')}")
            valid = False
    
    return valid

def main():
    # 数据集路径
    dataset_path = "../../dataset/data/tmp/qm9-evo-pairs-step-1-pairs-v0-81005-with-properties-pct.json"
    
    print("验证evolver.py生成路径与数据集格式一致性（适应全量扩展树生成）")
    print("=" * 60)
    
    # 1. 加载数据集样本
    print("1. 加载数据集样本...")
    dataset_samples = load_sample_dataset(dataset_path)
    if not dataset_samples:
        print("无法加载数据集样本")
        return
    
    print(f"成功加载 {len(dataset_samples)} 个样本")
    
    # 2. 分析数据集
    print("2. 分析数据集格式...")
    dataset_stats = analyze_dataset_operations(dataset_samples)
    
    # 3. 生成测试分子路径
    print("3. 生成测试分子路径...")
    test_molecules = ["CC", "CO", "CCO", "c1ccccc1", "CC(=O)O"]
    all_generated_ops = []
    
    for mol in test_molecules:
        print(f"   分析分子: {mol}")
        stats = analyze_generated_path(mol)
        if stats:
            all_generated_ops.extend(stats['operations'].keys())
    
    # 4. 汇总生成路径统计
    generated_stats = {
        'operations': Counter(all_generated_ops),
        'positions': Counter({'atom': len([op for op in all_generated_ops])})  # 简化处理
    }
    
    # 5. 比较格式
    print("4. 比较格式...")
    compare_formats(dataset_stats, generated_stats)
    
    # 6. 验证一致性
    print("\n5. 验证操作一致性...")
    # 测试几个分子
    test_smiles_list = ["CCC", "CCO", "C=O", "c1ccccc1"]
    
    for smiles in test_smiles_list:
        print(f"\n   验证分子 {smiles}:")
        try:
            analyzer = MoleculeEvolverAnalysis(smiles)
            path = analyzer.get_full_path_dict()
            
            is_consistent = validate_operation_consistency(path)
            if is_consistent:
                print("     格式一致")
            else:
                print("     格式不一致")
                
        except Exception as e:
            print(f"     分析失败: {e}")
    
    print("\n验证完成!")

if __name__ == "__main__":
    main()