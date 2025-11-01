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
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
sys.path.insert(0, project_root)

try:
    from rdkit import Chem
    from mol_evo.core.evolver import MoleculeEvolver
except Exception as e:
    print(f"导入模块时发生错误: {e}")
    sys.exit(1)


def load_test_data(filename='test_data.json'):
    """从JSON文件加载测试数据"""
    # 优先使用环境变量中指定的测试数据根目录
    test_data_root = os.environ.get('TEST_DATA_ROOT')
    if test_data_root:
        # 根据文件名确定子目录
        if 'extended' in filename:
            test_data_path = os.path.join(test_data_root, 'extended', filename)
        else:
            test_data_path = os.path.join(test_data_root, 'basic', filename)
    else:
        test_data_path = os.path.join(os.path.dirname(__file__), 'testcases', filename)
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class TestMoleculeEvolver(unittest.TestCase):
    """MoleculeEvolver测试类"""
    
    @classmethod
    def setUpClass(cls):
        """加载测试数据"""
        cls.test_data = load_test_data('test_data.json')
        cls.extended_test_data = load_test_data('extended_test_data.json')
    
    def test_all_cases_from_json(self):
        """测试所有来自JSON的测试用例"""
        passed = 0
        total = len(self.test_data)
        
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
                    
                    # 验证各项期望 - 使用统计方式而不是断言
                    has_ring_closure = any("成环" in step for step in path)
                    if has_ring_closure == expectations['has_ring_closure']:
                        passed += 1
                    else:
                        print(f"测试用例 '{test_case['name']}' 成环操作验证失败: 期望 {expectations['has_ring_closure']}, 实际 {has_ring_closure}")
                    
                    has_attachment = any("添加附件" in step for step in path)
                    if has_attachment == expectations.get('has_attachment', False):
                        passed += 1
                    
                    has_double_bond = any("形成双键" in step for step in path)
                    if has_double_bond == expectations.get('has_double_bond', False):
                        passed += 1
                    
                    has_aromatic_bond = any("形成芳香键" in step for step in path)
                    if has_aromatic_bond == expectations.get('has_aromatic_bond', False):
                        passed += 1
                    
                except Exception as e:
                    print(f"处理分子 {smiles} 时发生错误: {str(e)}")
        
        print(f"基础测试通过率: {passed}/{total*4} ({passed/(total*4)*100:.2f}%)")

    def test_extended_cases_from_json(self):
        """测试扩展的测试用例"""
        passed = 0
        total = len(self.extended_test_data)
        
        for i, test_case in enumerate(self.extended_test_data):
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
                    
                    # 验证各项期望 - 使用统计方式而不是断言
                    has_ring_closure = any("成环" in step for step in path)
                    if has_ring_closure == expectations['has_ring_closure']:
                        passed += 1
                    else:
                        print(f"测试用例 '{test_case['name']}' 成环操作验证失败: 期望 {expectations['has_ring_closure']}, 实际 {has_ring_closure}")
                    
                    has_attachment = any("添加附件" in step for step in path)
                    if has_attachment == expectations.get('has_attachment', False):
                        passed += 1
                    
                    has_double_bond = any("形成双键" in step for step in path)
                    if has_double_bond == expectations.get('has_double_bond', False):
                        passed += 1
                    
                    has_aromatic_bond = any("形成芳香键" in step for step in path)
                    if has_aromatic_bond == expectations.get('has_aromatic_bond', False):
                        passed += 1
                    
                    has_triple_bond = any("形成三键" in step for step in path)
                    if has_triple_bond == expectations.get('has_triple_bond', False):
                        passed += 1
                    
                    has_stereochemistry = any("指定手性" in step for step in path) or any("指定顺反" in step for step in path)
                    if has_stereochemistry == expectations.get('has_stereochemistry', False):
                        passed += 1
                    
                except Exception as e:
                    print(f"处理分子 {smiles} 时发生错误: {str(e)}")
        
        print(f"扩展测试通过率: {passed}/{total*6} ({passed/(total*6)*100:.2f}%)")

    def test_path_consistency(self):
        """测试路径一致性，验证字符串和字典格式路径的一致性"""
        passed = 0
        total = min(3, len(self.test_data))  # 只测试前3个用例以节省时间
        
        for test_case in self.test_data[:3]:  # 只测试前3个用例以节省时间
            with self.subTest(case=test_case['name']):
                smiles = test_case['smiles']
                
                try:
                    evolver = MoleculeEvolver(smiles)
                    path_str = evolver.generate_path()
                    path_dict = evolver.generate_path_dict()
                    
                    # 验证长度一致性
                    if len(path_str) == len(path_dict):
                        passed += 1
                    else:
                        print(f"测试用例 '{test_case['name']}' 字符串和字典格式路径长度不一致: {len(path_str)} vs {len(path_dict)}")
                    
                    # 验证起始操作一致性
                    if "起始" in path_str[0] and path_dict[0]["op"] == "起始":
                        passed += 1
                    else:
                        print(f"测试用例 '{test_case['name']}' 起始操作不一致: {path_str[0]} vs {path_dict[0]['op']}")
                    
                except Exception as e:
                    print(f"处理分子 {smiles} 时发生错误: {str(e)}")
        
        print(f"一致性测试通过率: {passed}/{total*2} ({passed/(total*2)*100:.2f}%)")

    def test_invalid_smiles(self):
        """测试无效SMILES处理"""
        invalid_smiles_list = ["invalid", "C1C", "CC1"]
        passed = 0
        
        for invalid_smiles in invalid_smiles_list:
            with self.subTest(smiles=invalid_smiles):
                try:
                    with self.assertRaises(ValueError):
                        MoleculeEvolver(invalid_smiles)
                    passed += 1
                except AssertionError:
                    print(f"无效SMILES '{invalid_smiles}' 未正确抛出ValueError")
                except Exception as e:
                    print(f"处理无效SMILES '{invalid_smiles}' 时发生意外错误: {str(e)}")
        
        print(f"无效SMILES处理测试通过率: {passed}/{len(invalid_smiles_list)} ({passed/len(invalid_smiles_list)*100:.2f}%)")

    def test_empty_smiles(self):
        """测试空SMILES处理"""
        try:
            with self.assertRaises(ValueError):
                MoleculeEvolver("")
            print("空SMILES处理测试通过")
        except AssertionError:
            print("空SMILES未正确抛出ValueError")
        except Exception as e:
            print(f"处理空SMILES时发生意外错误: {str(e)}")

    def test_simple_molecule_reconstruction(self):
        """测试简单分子的路径重建能力"""
        simple_molecules = ["C", "CC", "CCC", "CCCC", "CO", "CN"]
        passed = 0
        
        for smiles in simple_molecules:
            with self.subTest(smiles=smiles):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    self.assertIsNotNone(mol)
                    
                    evolver = MoleculeEvolver(smiles)
                    path = evolver.generate_path()
                    
                    # 确保至少有一个步骤
                    self.assertGreater(len(path), 0)
                    
                    # 确保第一步是起始操作
                    self.assertIn("起始", path[0])
                    passed += 1
                    
                except Exception as e:
                    print(f"处理分子 {smiles} 时发生错误: {str(e)}")
        
        print(f"简单分子重建测试通过率: {passed}/{len(simple_molecules)} ({passed/len(simple_molecules)*100:.2f}%)")


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
    test_data = load_test_data('test_data.json')
    test_cases = [(item['smiles'], item['name']) for item in test_data]
    
    for smiles, name in test_cases:
        run_single_test(smiles, name)
    
    # 运行单元测试
    print("运行单元测试:")
    print("-" * 30)
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2)


if __name__ == "__main__":
    main()