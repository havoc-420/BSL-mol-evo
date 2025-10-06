#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本用法示例

这是一个简单的示例，展示如何使用分子进化路径生成器的核心功能。
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.evolver import MoleculeEvolver
from core.similarity import calculate_evolutionary_similarity


def main():
    # 示例1: 生成单个分子的进化路径
    print("示例1: 生成乙醇分子的进化路径")
    smiles = "CCO"  # 乙醇的SMILES表示
    evolver = MoleculeEvolver(smiles)
    path = evolver.generate_path()
    
    print(f"分子: {smiles}")
    print("进化路径:")
    for i, step in enumerate(path, 1):
        print(f"  {i}. {step}")
    
    print("\n" + "="*50 + "\n")
    
    # 示例2: 计算两个分子之间的相似度
    print("示例2: 计算丙烷和丁烷的进化相似度")
    smiles1 = "CCC"   # 丙烷
    smiles2 = "CCCC"  # 丁烷
    
    distance, path1, path2 = calculate_evolutionary_similarity(smiles1, smiles2, verbose=True)
    
    print(f"路径编辑距离: {distance}")


if __name__ == "__main__":
    main()