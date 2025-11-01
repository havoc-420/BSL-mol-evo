#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩展测试执行脚本
用于运行MoleculeEvolver的所有测试并生成报告
"""

import sys
import os
import unittest
import json
import time
from io import StringIO

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
    test_data_path = os.path.join(os.path.dirname(__file__), '..', filename)
    with open(test_data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_performance_test(smiles, iterations=5):
    """
    运行性能测试
    返回平均执行时间和内存使用情况
    """
    times = []
    
    for i in range(iterations):
        start_time = time.time()
        try:
            evolver = MoleculeEvolver(smiles)
            path = evolver.generate_path()
            end_time = time.time()
            times.append(end_time - start_time)
        except Exception as e:
            print(f"性能测试中发生错误: {e}")
            return None, None
    
    avg_time = sum(times) / len(times) if times else None
    max_time = max(times) if times else None
    min_time = min(times) if times else None
    
    return avg_time, max_time, min_time


class TestResultCollector(unittest.TextTestResult):
    """自定义测试结果收集器"""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.results = []
    
    def addSuccess(self, test):
        super().addSuccess(test)
        self.results.append({
            'test': str(test),
            'status': 'PASS',
            'output': ''
        })
    
    def addError(self, test, err):
        super().addError(test, err)
        self.results.append({
            'test': str(test),
            'status': 'ERROR',
            'output': self._exc_info_to_string(err, test)
        })
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.results.append({
            'test': str(test),
            'status': 'FAIL',
            'output': self._exc_info_to_string(err, test)
        })


def generate_test_report(results, test_data, extended_test_data):
    """生成测试报告"""
    passed = len([r for r in results if r['status'] == 'PASS'])
    failed = len([r for r in results if r['status'] == 'FAIL'])
    errors = len([r for r in results if r['status'] == 'ERROR'])
    total = len(results)
    
    print("\n" + "="*60)
    print("MoleculeEvolver 测试报告")
    print("="*60)
    
    print(f"总测试数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"错误: {errors}")
    print(f"通过率: {passed/total*100:.2f}%" if total > 0 else "N/A")
    
    # 按测试类型分类统计
    basic_tests = [r for r in results if 'test_all_cases_from_json' in r['test']]
    extended_tests = [r for r in results if 'test_extended_cases_from_json' in r['test']]
    other_tests = [r for r in results if 'test_all_cases_from_json' not in r['test'] and 'test_extended_cases_from_json' not in r['test']]
    
    print(f"\n基础测试: {len(basic_tests)} 通过: {len([r for r in basic_tests if r['status'] == 'PASS'])}")
    print(f"扩展测试: {len(extended_tests)} 通过: {len([r for r in extended_tests if r['status'] == 'PASS'])}")
    print(f"其他测试: {len(other_tests)} 通过: {len([r for r in other_tests if r['status'] == 'PASS'])}")
    
    # 显示失败和错误的测试
    failed_tests = [r for r in results if r['status'] != 'PASS']
    if failed_tests:
        print("\n失败/错误的测试:")
        print("-" * 40)
        for result in failed_tests:
            print(f"{result['status']}: {result['test']}")
            if result['output']:
                print(f"  错误信息: {result['output'][:100]}...")
    
    # 性能测试
    print("\n性能测试:")
    print("-" * 40)
    
    # 选择几个代表性分子进行性能测试
    performance_molecules = [
        ("甲烷", "C"),
        ("环己烷", "C1CCCCC1"),
        ("苯", "c1ccccc1"),
        ("乙醇", "CCO")
    ]
    
    for name, smiles in performance_molecules:
        avg_time, max_time, min_time = run_performance_test(smiles)
        if avg_time is not None:
            print(f"{name} ({smiles}): 平均 {avg_time*1000:.2f}ms (最小: {min_time*1000:.2f}ms, 最大: {max_time*1000:.2f}ms)")
        else:
            print(f"{name} ({smiles}): 测试失败")


def main():
    """主函数"""
    print("开始运行MoleculeEvolver扩展测试...")
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(__file__), pattern='../tests/test_*.py')
    
    # 运行测试
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=TestResultCollector)
    result = runner.run(suite)
    
    # 加载测试数据用于报告
    test_data = load_test_data('testcases/basic/test_data.json')
    extended_test_data = load_test_data('testcases/extended/extended_test_data.json')
    
    # 生成报告
    generate_test_report(result.results, test_data, extended_test_data)
    
    print("\n详细测试输出:")
    print("="*60)
    print(stream.getvalue())


if __name__ == "__main__":
    main()