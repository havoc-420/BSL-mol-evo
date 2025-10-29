#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoleculeEvolver类的测试套件

这个文件包含了针对MoleculeEvolver类的多个测试用例，
用于验证其在各种分子结构上的表现，包括：
1. 基本链状分子
2. 环状分子
3. 带支链的分子
4. 含杂原子的分子
5. 含多重键的分子
6. 含手性中心的分子
"""

import sys
import os
import unittest
import json

# 添加项目根目录到Python路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, project_root)

try:
    from rdkit import Chem
    from mol_evo.core.evolver import MoleculeEvolver
except Exception as e:
    print(f"导入模块时发生错误: {e}")
    sys.exit(1)


def load_test_data():
    """从JSON文件加载测试数据"""
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data.json')
    with open(test_data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class TestMoleculeEvolver(unittest.TestCase):
    """MoleculeEvolver测试类"""
    
    @classmethod
    def setUpClass(cls):
        """加载测试数据"""
        cls.test_data = load_test_data()
    
    def test_all_cases_from_json(self):
        """测试所有来自JSON的测试用例"""
        for i, test_case in enumerate(self.test_data):
            with self.subTest(case=test_case['name']):
                smiles = test_case['smiles']
                expectations = test_case['expectations']
                
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    self.assertIsNotNone(mol, f"无效的SMILES: {smiles}")
                    
                    evolver = MoleculeEvolver(smiles)
                    path = evolver.generate_path()
                    
                    # 基本验证
                    self.assertGreater(len(path), 0)
                    self.assertIn("起始", path[0])
                    
                    # 验证各项期望
                    has_ring_closure = any("成环" in step for step in path)
                    self.assertEqual(has_ring_closure, expectations['has_ring_closure'], 
                                     f"成环操作验证失败: 期望 {expectations['has_ring_closure']}, 实际 {has_ring_closure}")
                    
                    has_attachment = any("添加附件" in step for step in path)
                    self.assertEqual(has_attachment, expectations.get('has_attachment', False), 
                                     f"附件操作验证失败: 期望 {expectations.get('has_attachment', False)}, 实际 {has_attachment}")
                    
                    has_double_bond = any("形成双键" in step for step in path)
                    self.assertEqual(has_double_bond, expectations.get('has_double_bond', False), 
                                     f"双键操作验证失败: 期望 {expectations.get('has_double_bond', False)}, 实际 {has_double_bond}")
                    
                    has_aromatic_bond = any("形成芳香键" in step for step in path)
                    self.assertEqual(has_aromatic_bond, expectations.get('has_aromatic_bond', False), 
                                     f"芳香键操作验证失败: 期望 {expectations.get('has_aromatic_bond', False)}, 实际 {has_aromatic_bond}")
                                     
                    # 测试generate_path_dict方法
                    path_dict = evolver.generate_path_dict()
                    self.assertEqual(len(path_dict), len(path), 
                                     "字典格式路径长度应与字符串格式一致")
                    
                except Exception as e:
                    self.fail(f"处理分子 {smiles} 时发生错误: {str(e)}")


def run_single_test(smiles, name):
    """运行单个分子的测试并打印结果"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"无效的SMILES: {smiles}")
            return
        
        # 生成进化路径
        evolver = MoleculeEvolver(smiles)
        path = evolver.generate_path()
        
        print(f"{name} {smiles} 进化路径 (步骤数: {len(path)}):")
        for i, step in enumerate(path, 1):
            print(f"  {i:2d}. {step}")
        print()
        
        # 生成字典格式的进化路径
        path_dict = evolver.generate_path_dict()
        print(f"{name} {smiles} 字典格式进化路径 (步骤数: {len(path_dict)}):")
        for i, step in enumerate(path_dict, 1):
            print(f"  {i:2d}. {step}")
        print()
            
    except Exception as e:
        print(f"处理分子 {smiles} 时发生错误: {str(e)}")
        print()


def main():
    """主函数 - 运行示例测试"""
    print("MoleculeEvolver测试套件")
    print("=" * 50)
    
    # 从JSON文件加载测试用例
    test_data = load_test_data()
    test_cases = [(item['smiles'], item['name']) for item in test_data]
    
    for smiles, name in test_cases:
        run_single_test(smiles, name)
    
    # 运行单元测试
    print("运行单元测试:")
    print("-" * 30)
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2)


if __name__ == "__main__":
    main()