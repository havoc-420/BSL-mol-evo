#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子进化路径生成器测试用例

这个文件包含了针对特定分子的测试用例，专门测试CCCCCC和C1CCCCC1，
用于验证MoleculeEvolver在这两个特定分子结构上的表现。

## example
python mol_evo/examples/molecule_evolution_test_cases.py

"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, project_root)

try:
    from rdkit import Chem
    from mol_evo.core.evolver import MoleculeEvolver
except Exception as e:
    print(f"导入模块时发生错误: {e}")
    sys.exit(1)


def test_linear_alkane():
    """测试直链烷烃: CCCCCC"""
    smiles = "CCCCCC"
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("无效的SMILES")
            return
        
        # 生成进化路径
        evolver = MoleculeEvolver(smiles)
        path = evolver.generate_path()
        
        print(f"直链烷烃 {smiles} 进化路径 (步骤数: {len(path)}):")
        has_ring_closure = any("成环" in step for step in path)
        print(f"包含成环操作: {has_ring_closure}")
        for i, step in enumerate(path, 1):
            print(f"  {i:2d}. {step}")
            
    except Exception as e:
        print(f"处理分子 {smiles} 时发生错误: {str(e)}")
    
    print()


def test_cyclic_alkane():
    """测试环烷烃: C1CCCCC1"""
    smiles = "C1CCCCC1"
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("无效的SMILES")
            return
        
        # 生成进化路径
        evolver = MoleculeEvolver(smiles)
        path = evolver.generate_path()
        
        print(f"环烷烃 {smiles} 进化路径 (步骤数: {len(path)}):")
        has_ring_closure = any("成环" in step for step in path)
        print(f"包含成环操作: {has_ring_closure}")
        for i, step in enumerate(path, 1):
            print(f"  {i:2d}. {step}")
            
    except Exception as e:
        print(f"处理分子 {smiles} 时发生错误: {str(e)}")
    
    print()


def main():
    """主函数"""
    print("开始测试分子进化路径生成器...")
    
    # 运行指定的测试用例
    test_linear_alkane()        # CCCCCC
    test_cyclic_alkane()        # C1CCCCC1
    
    print("所有测试完成")


if __name__ == "__main__":
    main()