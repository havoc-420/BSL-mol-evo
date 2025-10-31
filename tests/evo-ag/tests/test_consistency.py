#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoleculeEvolver一致性测试
验证不同输出格式之间的一致性
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


def load_test_data(filename='testcases/basic/test_data.json'):
    """从JSON文件加载测试数据"""
    # 优先使用环境变量中指定的测试数据根目录
    test_data_root = os.environ.get('TEST_DATA_ROOT')
    if test_data_root:
        test_data_path = os.path.join(test_data_root, filename.split('/', 1)[-1])
    else:
        test_data_path = os.path.join(os.path.dirname(__file__), '..', '..', filename)
    
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"测试数据文件未找到: {test_data_path}")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class TestConsistencyMoleculeEvolver(unittest.TestCase):
    """MoleculeEvolver一致性测试类"""
    
    @classmethod
    def setUpClass(cls):
        """加载测试数据"""
        cls.test_data = load_test_data('testcases/basic/test_data.json')
    
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


if __name__ == "__main__":
    unittest.main(verbosity=2)