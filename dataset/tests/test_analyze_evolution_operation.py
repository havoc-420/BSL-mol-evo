#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 analyze_evolution_operation_dict 函数的单元测试
重点测试起始原子不同的情况
"""

import sys
import os
import unittest

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from mol_evo.dataset.extract_evolution_pairs import analyze_evolution_operation_dict


import unittest

class TestAnalyzeEvolutionOperation(unittest.TestCase):
    """测试 analyze_evolution_operation_dict 函数"""

    def test_identical_paths(self):
        """测试完全相同的路径"""
        path1_dict = [
            {'position': '0', 'atom': 'C', 'operation': 'init_atom'},
            {'position': '1', 'atom': 'C', 'operation': 'add_atom'},
            {'position': '2', 'atom': 'O', 'operation': 'add_atom'},
            {'position': '2-3', 'atom': None, 'operation': 'form_double_bond'}
        ]
        
        path2_dict = [
            {'position': '0', 'atom': 'C', 'operation': 'init_atom'},
            {'position': '1', 'atom': 'C', 'operation': 'add_atom'},
            {'position': '2', 'atom': 'O', 'operation': 'add_atom'},
            {'position': '2-3', 'atom': None, 'operation': 'form_double_bond'}
        ]
        
        result = analyze_evolution_operation_dict(path1_dict, path2_dict)
        self.assertEqual(len(result), 0)

    def test_start_atom_replacement(self):
        """测试起始原子替换"""
        # CNC=O -> COC=O (N替换为O)
        path1_dict = [
            {'position': '0', 'atom': 'N', 'operation': 'init_atom'},
            {'position': '0', 'atom': 'C', 'operation': 'add_atom'},
            {'position': '1', 'atom': 'C', 'operation': 'add_atom'},
            {'position': '2', 'atom': 'O', 'operation': 'add_atom'},
            {'position': '2-3', 'atom': None, 'operation': 'form_double_bond'}
        ]
        
        path2_dict = [
            {'position': '0', 'atom': 'O', 'operation': 'init_atom'},
            {'position': '0', 'atom': 'C', 'operation': 'add_atom'},
            {'position': '1', 'atom': 'C', 'operation': 'add_atom'},
            {'position': '2', 'atom': 'O', 'operation': 'add_atom'},
            {'position': '2-3', 'atom': None, 'operation': 'form_double_bond'}
        ]
        
        result = analyze_evolution_operation_dict(path1_dict, path2_dict)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['position'], '0')
        self.assertEqual(result[0]['atom'], 'O')
        self.assertEqual(result[0]['operation'], 'replace_atom')
        self.assertEqual(result[0]['from_atom'], 'N')

    def test_middle_atom_replacement(self):
        """测试中间原子替换"""
        # CH4 -> CF4 (H替换为F)
        path1_dict = [
            {'position': '0', 'atom': 'C', 'operation': 'init_atom'},
            {'position': '1', 'atom': 'H', 'operation': 'add_atom'},
            {'position': '2', 'atom': 'H', 'operation': 'add_atom'},
            {'position': '3', 'atom': 'H', 'operation': 'add_atom'},
            {'position': '4', 'atom': 'H', 'operation': 'add_atom'}
        ]
        
        path2_dict = [
            {'position': '0', 'atom': 'C', 'operation': 'init_atom'},
            {'position': '1', 'atom': 'F', 'operation': 'add_atom'},
            {'position': '2', 'atom': 'F', 'operation': 'add_atom'},
            {'position': '3', 'atom': 'F', 'operation': 'add_atom'},
            {'position': '4', 'atom': 'F', 'operation': 'add_atom'}
        ]
        
        result = analyze_evolution_operation_dict(path1_dict, path2_dict)
        self.assertEqual(len(result), 4)
        for op in result:
            self.assertEqual(op['operation'], 'replace_atom')
            self.assertEqual(op['from_atom'], 'H')
            self.assertEqual(op['atom'], 'F')

    def test_add_operation(self):
        """测试添加操作"""
        # CH3 -> CH4 (添加一个H原子)
        path1_dict = [
            {'position': '0', 'atom': 'C', 'operation': 'init_atom'},
            {'position': '1', 'atom': 'H', 'operation': 'add_atom'},
            {'position': '2', 'atom': 'H', 'operation': 'add_atom'},
            {'position': '3', 'atom': 'H', 'operation': 'add_atom'}
        ]
        
        path2_dict = [
            {'position': '0', 'atom': 'C', 'operation': 'init_atom'},
            {'position': '1', 'atom': 'H', 'operation': 'add_atom'},
            {'position': '2', 'atom': 'H', 'operation': 'add_atom'},
            {'position': '3', 'atom': 'H', 'operation': 'add_atom'},
            {'position': '4', 'atom': 'H', 'operation': 'add_atom'}
        ]
        
        result = analyze_evolution_operation_dict(path1_dict, path2_dict)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['position'], '4')
        self.assertEqual(result[0]['atom'], 'H')
        self.assertEqual(result[0]['operation'], 'add_atom')

    def test_remove_operation(self):
        """测试删除操作"""
        # CH4 -> CH3 (删除一个H原子)
        path1_dict = [
            {'position': '0', 'atom': 'C', 'operation': 'init_atom'},
            {'position': '1', 'atom': 'H', 'operation': 'add_atom'},
            {'position': '2', 'atom': 'H', 'operation': 'add_atom'},
            {'position': '3', 'atom': 'H', 'operation': 'add_atom'},
            {'position': '4', 'atom': 'H', 'operation': 'add_atom'}
        ]
        
        path2_dict = [
            {'position': '0', 'atom': 'C', 'operation': 'init_atom'},
            {'position': '1', 'atom': 'H', 'operation': 'add_atom'},
            {'position': '2', 'atom': 'H', 'operation': 'add_atom'},
            {'position': '3', 'atom': 'H', 'operation': 'add_atom'}
        ]
        
        result = analyze_evolution_operation_dict(path1_dict, path2_dict)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['position'], '4')
        self.assertEqual(result[0]['atom'], 'H')
        self.assertEqual(result[0]['operation'], 'add_atom')  # 注意：删除操作在数据结构中也是add_atom，但会被执行删除

    def test_bond_change(self):
        """测试键的变化"""
        # 单键变双键
        path1_dict = [
            {'position': '0', 'atom': 'C', 'operation': 'init_atom'},
            {'position': '1', 'atom': 'O', 'operation': 'add_atom'},
            {'position': '0-1', 'atom': None, 'operation': 'form_single_bond'}
        ]
        
        path2_dict = [
            {'position': '0', 'atom': 'C', 'operation': 'init_atom'},
            {'position': '1', 'atom': 'O', 'operation': 'add_atom'},
            {'position': '0-1', 'atom': None, 'operation': 'form_double_bond'}
        ]
        
        result = analyze_evolution_operation_dict(path1_dict, path2_dict)
        self.assertEqual(len(result), 2)
        # 应该先删除单键，再添加双键
        self.assertEqual(result[0]['position'], '0-1')
        self.assertEqual(result[0]['operation'], 'form_single_bond')
        self.assertEqual(result[1]['position'], '0-1')
        self.assertEqual(result[1]['operation'], 'form_double_bond')


    def test_complex_mixed_operations(self):
        """测试复杂的混合操作"""
        # 同时有替换、添加和删除
        path1_dict = [
            {'position': '0', 'atom': 'C', 'operation': 'init_atom'},
            {'position': '1', 'atom': 'N', 'operation': 'add_atom'},
            {'position': '2', 'atom': 'O', 'operation': 'add_atom'},
            {'position': '0-1', 'atom': None, 'operation': 'form_single_bond'}
        ]
    
        path2_dict = [
            {'position': '0', 'atom': 'C', 'operation': 'init_atom'},
            {'position': '1', 'atom': 'O', 'operation': 'add_atom'},  # 这个应该与path1的N形成替换
            {'position': '2', 'atom': 'H', 'operation': 'add_atom'},  # 这个应该是替换操作而不是添加
            {'position': '0-1', 'atom': None, 'operation': 'form_double_bond'}  # 键类型变化
        ]
        
        result = analyze_evolution_operation_dict(path1_dict, path2_dict)
        
        # 检查操作类型
        operations = [op['operation'] for op in result]
        
        # 应该包括：删除单键、N->O的替换、O->H的替换、添加双键
        # 注意：位置'2'的O->H操作也会被优化为替换操作，而不是单独的删除和添加
        self.assertIn('form_single_bond', operations)  # 删除
        self.assertIn('replace_atom', operations)      # 替换(N->O 和 O->H)
        self.assertIn('form_double_bond', operations)  # 添加双键
        
        # 具体检查替换操作
        replace_ops = [op for op in result if op['operation'] == 'replace_atom']
        self.assertEqual(len(replace_ops), 2)  # 现在应该有两个替换操作
        
        # 第一个替换操作:N->O
        n_to_o_ops = [op for op in replace_ops if op['from_atom'] == 'N' and op['atom'] == 'O']
        self.assertEqual(len(n_to_o_ops), 1)
        self.assertEqual(n_to_o_ops[0]['position'], '1')
        
        # 第二个替换操作:O->H
        o_to_h_ops = [op for op in replace_ops if op['from_atom'] == 'O' and op['atom'] == 'H']
        self.assertEqual(len(o_to_h_ops), 1)
        self.assertEqual(o_to_h_ops[0]['position'], '2')

    def test_empty_path(self):
        """测试空路径"""
        # 从空路径到有内容的路径
        path1_dict = []
        path2_dict = [
            {'position': '0', 'atom': 'C', 'operation': 'init_atom'},
            {'position': '1', 'atom': 'H', 'operation': 'add_atom'}
        ]
        
        result = analyze_evolution_operation_dict(path1_dict, path2_dict)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['operation'], 'init_atom')
        self.assertEqual(result[1]['operation'], 'add_atom')

if __name__ == '__main__':
    unittest.main()