#!/usr/bin/env python3
"""
使用strings命令提取QM9*数据集中的数据
"""

import subprocess
import sys
import os
import argparse
import csv
import re
from typing import List, Dict

def extract_data_with_strings(sql_file: str, output_csv: str, max_records: int = None):
    """
    使用strings命令提取数据
    
    Args:
        sql_file: SQL文件路径
        output_csv: 输出CSV文件路径
        max_records: 最大记录数
    """
    print(f"正在使用strings命令提取数据: {sql_file}")
    
    try:
        # 使用strings命令提取可读文本
        result = subprocess.run(['strings', sql_file], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"错误: strings命令执行失败: {result.stderr}")
            return False
        
        # 解析提取的数据
        records = parse_strings_output(result.stdout, max_records)
        
        # 保存为CSV
        if records:
            print(f"正在保存 {len(records)} 条记录到 {output_csv}")
            save_to_csv(records, output_csv)
            print("完成!")
            return True
        else:
            print("未找到任何记录")
            return False
            
    except FileNotFoundError:
        print("错误: 未找到strings命令")
        return False
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return False

def parse_strings_output(output: str, max_records: int = None) -> List[Dict]:
    """
    解析strings命令的输出
    
    Args:
        output: strings命令的输出
        max_records: 最大记录数
        
    Returns:
        解析后的记录列表
    """
    records = []
    record_count = 0
    
    # 元素列表
    elements = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr"
    ]
    
    lines = output.split('\n')
    
    # 查找包含INSERT INTO formula VALUES的行
    for i, line in enumerate(lines):
        if 'INSERT INTO formula VALUES' in line and '(' in line and ')' in line:
            if max_records and record_count >= max_records:
                break
                
            # 提取VALUES部分
            match = re.search(r'INSERT INTO formula VALUES\s*(\(.*\))', line)
            if match:
                values_part = match.group(1)
                
                # 解析值
                values = parse_values(values_part)
                
                # 构建记录
                if len(values) == len(elements) + 1:  # +1 是因为formula_string
                    record = {"formula_string": values[0]}
                    for j, element in enumerate(elements):
                        try:
                            record[element] = int(values[j+1]) if values[j+1] is not None else 0
                        except (ValueError, IndexError):
                            record[element] = 0
                    
                    records.append(record)
                    record_count += 1
                    
                    # 显示进度
                    if record_count % 1000 == 0:
                        print(f"已处理 {record_count} 条记录")
    
    return records

def parse_values(values_str: str) -> List:
    """
    解析VALUES字符串
    
    Args:
        values_str: VALUES字符串，如 ('H2O', 2, 0, ...)
        
    Returns:
        解析后的值列表
    """
    # 移除括号
    values_str = values_str.strip('()')
    
    # 分割值，考虑字符串中的逗号
    values = []
    current_value = ""
    in_string = False
    escape_next = False
    
    i = 0
    while i < len(values_str):
        char = values_str[i]
        
        if escape_next:
            current_value += char
            escape_next = False
        elif char == '\\':
            current_value += char
            escape_next = True
        elif char == "'" and not escape_next:
            in_string = not in_string
            current_value += char
        elif char == ',' and not in_string:
            values.append(current_value.strip())
            current_value = ""
        else:
            current_value += char
            
        i += 1
    
    # 添加最后一个值
    if current_value:
        values.append(current_value.strip())
    
    # 清理值
    cleaned_values = []
    for v in values:
        # 移除引号
        if v.startswith("'") and v.endswith("'"):
            v = v[1:-1]
        # 处理NULL值
        if v.upper() == 'NULL':
            v = None
        # 处理数字
        elif v is not None and v.isdigit():
            v = int(v)
        cleaned_values.append(v)
    
    return cleaned_values

def save_to_csv(records: List[Dict], output_file: str):
    """
    将记录保存为CSV文件
    
    Args:
        records: 记录列表
        output_file: 输出文件路径
    """
    if not records:
        return
        
    # 获取所有字段名
    fieldnames = list(records[0].keys())
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

def main():
    parser = argparse.ArgumentParser(description='使用strings命令提取QM9*数据集中的数据')
    parser.add_argument('--sql-file', type=str, required=True, 
                        help='QM9* SQL文件路径')
    parser.add_argument('--output-csv', type=str, required=True,
                        help='输出CSV文件路径')
    parser.add_argument('--max-records', type=int, 
                        help='最大处理记录数')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.sql_file):
        print(f"错误: 文件 {args.sql_file} 不存在")
        sys.exit(1)
    
    # 提取数据
    success = extract_data_with_strings(
        sql_file=args.sql_file,
        output_csv=args.output_csv,
        max_records=args.max_records
    )
    
    if success:
        print(f"成功将数据保存到 {args.output_csv}")
    else:
        print("处理失败")
        sys.exit(1)

if __name__ == "__main__":
    main()