#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从分子进化数据集中提取操作类型和原子类型配置
"""

import json
import yaml
import os
import argparse
from collections import Counter
from typing import Set, Dict, Any


def extract_operations_and_atoms(json_file_path: str) -> Dict[str, Any]:
    """
    从JSON数据集中提取所有操作类型和原子类型
    
    Args:
        json_file_path: JSON文件路径
        
    Returns:
        包含操作类型和原子类型的配置字典
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # 收集所有操作类型和原子类型
    operation_types: Set[str] = set()
    atom_types: Set[str] = set()
    
    # 统计操作类型频率
    operation_counter = Counter()
    
    for item in data:
        if 'operations' in item and isinstance(item['operations'], list):
            for operation in item['operations']:
                if 'operation' in operation:
                    op_type = operation['operation']
                    operation_types.add(op_type)
                    operation_counter[op_type] += 1
                    
                if 'atom' in operation and operation['atom']:
                    atom_types.add(operation['atom'])
    
    # 转换为排序列表
    sorted_operations = sorted(list(operation_types))
    sorted_atoms = sorted(list(atom_types))
    
    # 构建配置字典
    config = {
        'operation_types': sorted_operations,
        'atom_types': sorted_atoms,
        'operation_stats': dict(operation_counter.most_common())
    }
    
    return config


def generate_config_filename(json_file_path: str) -> str:
    """
    根据JSON文件名生成对应的YAML配置文件名
    
    Args:
        json_file_path: JSON文件路径
        
    Returns:
        YAML配置文件路径
    """
    # 获取文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    # 生成YAML文件路径（与JSON文件在同一目录下）
    yaml_file_path = os.path.join(os.path.dirname(json_file_path), f"{base_name}-config.yaml")
    return yaml_file_path


def save_config_to_yaml(config: Dict[str, Any], yaml_file_path: str):
    """
    将配置保存为YAML文件
    
    Args:
        config: 配置字典
        yaml_file_path: YAML文件路径
    """
    with open(yaml_file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"配置已保存到: {yaml_file_path}")
    print(f"操作类型数量: {len(config['operation_types'])}")
    print(f"原子类型数量: {len(config['atom_types'])}")
    print("\n操作类型统计:")
    for op, count in list(config['operation_stats'].items())[:10]:
        print(f"  {op}: {count}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从分子进化数据集中提取操作类型和原子类型配置')
    parser.add_argument('input', help='输入JSON文件路径')
    parser.add_argument('-o', '--output', help='输出YAML配置文件路径（可选，默认与输入文件同目录）')
    
    args = parser.parse_args()
    
    # 输入JSON文件路径
    json_file = args.input
    
    # 根据JSON文件名生成对应的YAML文件名或使用指定的输出路径
    if args.output:
        yaml_file = args.output
    else:
        yaml_file = generate_config_filename(json_file)
    
    # 提取配置
    config = extract_operations_and_atoms(json_file)
    
    # 保存配置
    save_config_to_yaml(config, yaml_file)
    
    print("\n配置提取完成!")


if __name__ == '__main__':
    main()