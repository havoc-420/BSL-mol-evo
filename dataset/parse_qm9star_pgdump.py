#!/usr/bin/env python3
"""
处理QM9*数据集的PostgreSQL自定义格式数据库转储文件
"""

import subprocess
import sys
import os
import argparse
import csv
import json
import tempfile
from typing import List, Dict

def convert_pg_dump_to_csv(pg_dump_file: str, output_csv: str, max_records: int = None):
    """
    将PostgreSQL自定义格式的数据库转储文件转换为CSV格式
    
    Args:
        pg_dump_file: PostgreSQL自定义格式的数据库转储文件路径
        output_csv: 输出CSV文件路径
        max_records: 最大记录数，None表示处理所有记录
    """
    print(f"正在处理PostgreSQL自定义格式数据库转储文件: {pg_dump_file}")
    
    try:
        # 使用pg_restore将自定义格式转换为纯文本SQL格式
        print("正在转换数据库格式...")
        result = subprocess.run([
            'pg_restore', 
            '--data-only', 
            '--table=formula', 
            '--format=custom',
            pg_dump_file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"错误: pg_restore执行失败: {result.stderr}")
            return False
        
        # 解析SQL INSERT语句
        sql_content = result.stdout
        print("正在解析数据...")
        
        # 写入临时文件以便处理
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as temp_file:
            temp_file.write(sql_content)
            temp_sql_file = temp_file.name
        
        # 解析INSERT语句
        records = parse_insert_statements(temp_sql_file, max_records)
        
        # 删除临时文件
        os.unlink(temp_sql_file)
        
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
        print("错误: 未找到pg_restore命令，请确保已安装PostgreSQL客户端工具")
        return False
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return False

def parse_insert_statements(sql_file: str, max_records: int = None) -> List[Dict]:
    """
    解析SQL INSERT语句
    
    Args:
        sql_file: SQL文件路径
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
    
    with open(sql_file, 'r') as f:
        for line in f:
            if line.startswith('INSERT INTO formula VALUES'):
                if max_records and record_count >= max_records:
                    break
                    
                # 提取VALUES部分
                values_part = line[len('INSERT INTO formula VALUES '):].strip().rstrip(';')
                
                # 简单解析值（这是一个简化的解析方法）
                # 移除括号
                values_part = values_part.strip('()')
                
                # 分割值（简单按逗号分割，不处理字符串中的逗号）
                values = []
                in_string = False
                current_value = ""
                i = 0
                while i < len(values_part):
                    char = values_part[i]
                    if char == "'" and (i == 0 or values_part[i-1] != '\\'):
                        in_string = not in_string
                        current_value += char
                    elif char == ',' and not in_string:
                        values.append(current_value.strip())
                        current_value = ""
                    else:
                        current_value += char
                    i += 1
                values.append(current_value.strip())
                
                # 清理并转换值
                cleaned_values = []
                for v in values:
                    # 移除引号
                    if v.startswith("'") and v.endswith("'"):
                        v = v[1:-1]
                    # 处理NULL值
                    if v.upper() == 'NULL':
                        v = None
                    cleaned_values.append(v)
                
                # 构建记录
                if len(cleaned_values) == len(elements) + 1:  # +1 是因为formula_string
                    record = {"formula_string": cleaned_values[0]}
                    for i, element in enumerate(elements):
                        try:
                            record[element] = int(cleaned_values[i+1]) if cleaned_values[i+1] is not None else 0
                        except (ValueError, IndexError):
                            record[element] = 0
                    
                    records.append(record)
                    record_count += 1
                    
                    # 显示进度
                    if record_count % 1000 == 0:
                        print(f"已处理 {record_count} 条记录")
    
    return records

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
    parser = argparse.ArgumentParser(description='处理QM9*数据集的PostgreSQL自定义格式数据库转储文件')
    parser.add_argument('--pg-dump-file', type=str, required=True, 
                        help='PostgreSQL自定义格式数据库转储文件路径')
    parser.add_argument('--output-csv', type=str, required=True,
                        help='输出CSV文件路径')
    parser.add_argument('--max-records', type=int, 
                        help='最大处理记录数')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.pg_dump_file):
        print(f"错误: 文件 {args.pg_dump_file} 不存在")
        sys.exit(1)
    
    # 转换文件
    success = convert_pg_dump_to_csv(
        pg_dump_file=args.pg_dump_file,
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