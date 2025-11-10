#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子进化路径生成器演示示例

这个示例展示了如何使用 MoleculeEvolverAnalysis 类来生成分子的进化路径，
以及如何计算两个分子之间的进化相似度。
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rdkit import Chem
from core.evolver import MoleculeEvolverAnalysis
from core.similarity import calculate_evolutionary_similarity


def demo_single_molecule_evolution():
    """演示单个分子的进化路径生成"""
    print("=" * 60)
    print("单个分子进化路径生成演示")
    print("=" * 60)
    
    # 测试分子列表
    test_molecules = [
        "CCO",              # 乙醇
        "c1ccccc1",         # 苯
        "CC(=O)O",          # 乙酸
        "CC(C)C",           # 异丁烷
        "C1CC1",            # 环丙烷
    ]
    
    for smiles in test_molecules:
        try:
            # 验证SMILES有效性
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"无效的SMILES: {smiles}")
                continue
                
            print(f"\n分子: {smiles}")
            print(f"分子名称: {Chem.MolToName(mol) if hasattr(Chem, 'MolToName') else 'N/A'}")
            
            # 生成进化路径
            evolver = MoleculeEvolverAnalysis(smiles)
            path = evolver.generate_path()
            
            print(f"进化步骤数量: {len(path)}")
            print("进化路径:")
            for i, step in enumerate(path, 1):
                print(f"  {i:2d}. {step}")
                
        except Exception as e:
            print(f"处理分子 {smiles} 时发生错误: {str(e)}")
    
    print("\n")


def demo_molecular_similarity():
    """演示分子间相似度计算"""
    print("=" * 60)
    print("分子间进化相似度计算演示")
    print("=" * 60)
    
    # 定义分子对进行比较
    molecule_pairs = [
        ("CCC", "CCCC"),           # 丙烷 vs 丁烷
        ("CCC", "CCO"),            # 丙烷 vs 乙醇
        ("c1ccccc1", "c1ccc(O)cc1"), # 苯 vs 苯酚
        ("CC(=O)O", "CC(=O)N"),    # 乙酸 vs 乙酰胺
        ("C1CC1", "C1CCC1"),       # 环丙烷 vs 环丁烷
    ]
    
    for smiles1, smiles2 in molecule_pairs:
        try:
            print(f"\n比较分子对: {smiles1} vs {smiles2}")
            
            # 验证SMILES有效性
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            
            if mol1 is None or mol2 is None:
                print("  一个或两个SMILES无效")
                continue
            
            # 计算进化相似度
            distance, path1, path2 = calculate_evolutionary_similarity(smiles1, smiles2, verbose=False)
            
            print(f"  路径编辑距离: {distance}")
            print(f"  分子1路径长度: {len(path1)}")
            print(f"  分子2路径长度: {len(path2)}")
            
        except Exception as e:
            print(f"  比较 {smiles1} 和 {smiles2} 时发生错误: {str(e)}")
    
    print("\n")


def demo_complex_molecule():
    """演示复杂分子的处理"""
    print("=" * 60)
    print("复杂分子处理演示")
    print("=" * 60)
    
    # 复杂分子示例
    complex_molecules = [
        "CC1=C(C(=NO1)C2=CC=CC=C2)C(=O)NC3CCCC3",  # 一个较为复杂的分子
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",           # 咖啡因
    ]
    
    for smiles in complex_molecules:
        try:
            print(f"\n复杂分子: {smiles[:50]}{'...' if len(smiles) > 50 else ''}")
            
            # 生成进化路径
            evolver = MoleculeEvolverAnalysis(smiles)
            path = evolver.generate_path()
            
            print(f"进化步骤数量: {len(path)}")
            
            # 显示前10步和后10步
            if len(path) <= 20:
                print("完整进化路径:")
                for i, step in enumerate(path, 1):
                    print(f"  {i:2d}. {step}")
            else:
                print("前10步进化路径:")
                for i, step in enumerate(path[:10], 1):
                    print(f"  {i:2d}. {step}")
                print("  ...")
                print("后10步进化路径:")
                for i, step in enumerate(path[-10:], len(path)-9):
                    print(f"  {i:2d}. {step}")
                    
        except Exception as e:
            print(f"处理复杂分子时发生错误: {str(e)}")
    
    print("\n")


def main():
    """主函数"""
    print("分子进化路径生成器演示")
    print("该演示展示了如何使用分子进化路径生成器来分析分子结构")
    
    # 运行各个演示
    demo_single_molecule_evolution()
    demo_molecular_similarity()
    demo_complex_molecule()
    
    print("=" * 60)
    print("演示结束")
    print("=" * 60)


if __name__ == "__main__":
    main()