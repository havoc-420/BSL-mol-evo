#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一测试入口
用于运行所有MoleculeEvolverAnalysis相关的测试
"""

import sys
import os
import unittest
import argparse
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
sys.path.insert(0, project_root)

def load_config(config_file="config/test_config.yaml"):
    """
    加载测试配置文件
    
    Args:
        config_file (str): 配置文件路径
    
    Returns:
        dict: 配置信息
    """
    config_path = Path(__file__).parent.parent / config_file
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

def run_tests(test_pattern="tests/test_*.py", verbose=1, fail_fast=False):
    """
    运行指定的测试
    
    Args:
        test_pattern (str): 测试文件模式
        verbose (int): 详细级别
        fail_fast (bool): 是否在第一个失败时停止
    
    Returns:
        bool: 测试是否全部通过
    """
    # 获取当前目录
    test_dir = Path(__file__).parent.parent
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern=test_pattern)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=verbose, failfast=fail_fast)
    result = runner.run(suite)
    
    # 返回测试结果
    return result.wasSuccessful()

def run_test_suite(suite_name, config):
    """
    运行指定的测试套件
    
    Args:
        suite_name (str): 测试套件名称
        config (dict): 配置信息
    """
    if 'test_suites' not in config or suite_name not in config['test_suites']:
        print(f"未知的测试套件: {suite_name}")
        return False
    
    suite_config = config['test_suites'][suite_name]
    print(f"运行测试套件: {suite_config['name']}")
    print(f"描述: {suite_config['description']}")
    
    options = config.get('options', {})
    verbose = options.get('verbosity', 1)
    fail_fast = options.get('fail_fast', False)
    
    return run_tests(suite_config['pattern'], verbose, fail_fast)

def list_test_suites(config):
    """列出所有可用的测试套件"""
    if 'test_suites' not in config:
        print("没有找到测试套件配置")
        return
    
    print("可用的测试套件:")
    for name, suite in config['test_suites'].items():
        print(f"  {name}: {suite['name']} - {suite['description']}")

def main():
    """主函数"""
    # 加载配置
    config = load_config()
    
    parser = argparse.ArgumentParser(description="MoleculeEvolverAnalysis统一测试入口")
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="显示详细测试信息"
    )
    parser.add_argument(
        "-q", "--quiet", 
        action="store_true", 
        help="静默模式，只显示最终结果"
    )
    parser.add_argument(
        "--pattern", 
        default="tests/test_*.py",
        help="测试文件模式 (默认: tests/test_*.py)"
    )
    parser.add_argument(
        "--suite",
        help="运行指定的测试套件"
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="列出所有测试套件"
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="在第一个失败时停止测试"
    )
    
    args = parser.parse_args()
    
    # 处理列表套件请求
    if args.list_suites:
        list_test_suites(config)
        return 0
    
    # 处理套件运行请求
    if args.suite:
        success = run_test_suite(args.suite, config)
        return 0 if success else 1
    
    # 确定详细级别
    if args.quiet:
        verbose = 0
    elif args.verbose:
        verbose = 2
    else:
        verbose = 1
    
    # 获取fail_fast选项
    fail_fast = args.fail_fast or config.get('options', {}).get('fail_fast', False)
    
    print("=" * 60)
    print("MoleculeEvolverAnalysis 统一测试入口")
    print("=" * 60)
    print(f"测试目录: {Path(__file__).parent.parent}")
    print(f"测试模式: {args.pattern}")
    print(f"详细级别: {verbose}")
    print(f"快速失败: {fail_fast}")
    print("-" * 60)
    
    # 设置测试数据目录环境变量
    test_data_dir = Path(__file__).parent.parent / "testcases"
    os.environ["TEST_DATA_DIR"] = str(test_data_dir)
    print(f"测试数据目录: {test_data_dir}")
    
    # 运行测试
    success = run_tests(args.pattern, verbose, fail_fast)
    
    print("-" * 60)
    if success:
        print("所有测试通过!")
        return 0
    else:
        print("部分测试失败!")
        return 1

if __name__ == "__main__":
    sys.exit(main())