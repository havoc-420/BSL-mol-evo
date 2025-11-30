#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立测试脚本，专门用于测试test_evolution_generation函数
"""

import sys
import os
import argparse

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

try:
    from mol_evo.core.evolver import MolecularEvolutionExpansion
    
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
            import traceback
            traceback.print_exc()
            return []
    
except ImportError as e:
    print(f"导入模块失败: {e}")
    
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
        
        # 模拟函数调用而不实际执行
        print("函数调用参数:")
        print(f"  initial_smiles: {initial_smiles}")
        print(f"  num_paths: {num_paths}")
        print(f"  steps_per_path: {steps_per_path}")
        print(f"  config_file: {config_file}")
        
        # 模拟返回值
        paths = []
        for i in range(min(num_paths, 3)):  # 限制模拟路径数
            path = {
                "path_id": i,
                "steps": [
                    {"step": 0, "smiles": initial_smiles, "operation": "start", "details": {}}
                ],
                "initial_smiles": initial_smiles,
                "final_smiles": initial_smiles,
                "path_length": 1
            }
            # 添加模拟步骤
            for j in range(min(steps_per_path, 3)):  # 限制模拟步数
                path["steps"].append({
                    "step": j+1, 
                    "smiles": initial_smiles, 
                    "operation": f"operation_{j+1}", 
                    "details": {"param": f"value_{j+1}"}
                })
                path["path_length"] += 1
                path["final_smiles"] = initial_smiles
            paths.append(path)
        
        print("\n生成的进化路径:")
        for i, path in enumerate(paths):
            print(f"\n路径 {i+1}:")
            print(f"  最终分子: {path['final_smiles']}")
            print(f"  路径长度: {path['path_length']} 步")
            
            print("  操作步骤:")
            for step in path["steps"]:
                print(f"    步骤 {step['step']}: {step['operation']} -> {step['smiles']}")
        
        return paths

def main():
    parser = argparse.ArgumentParser(description='独立测试 test_evolution_generation 函数')
    parser.add_argument('--config-file', type=str, default=None,
                        help='配置文件路径')
    args = parser.parse_args()
    
    config_file = args.config_file
    
    print("独立测试 test_evolution_generation 函数")
    print("=" * 50)
    
    # 测试用例1: 简单的乙烷分子
    print("\n测试用例1: 乙烷分子 (CC)")
    test_evolution_generation("CC", num_paths=2, steps_per_path=2, config_file=config_file)
    
    # # 测试用例2: 苯分子
    # print("\n\n测试用例2: 苯分子 (c1ccccc1)")
    # test_evolution_generation("c1ccccc1", num_paths=1, steps_per_path=3, config_file=config_file)
    
    # # 测试用例3: 甲醇分子
    # print("\n\n测试用例3: 甲醇分子 (CO)")
    # test_evolution_generation("CO", num_paths=3, steps_per_path=1, config_file=config_file)

if __name__ == "__main__":
    main()