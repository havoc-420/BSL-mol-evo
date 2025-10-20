#!/usr/bin/env python3
"""
简单解析QM9*数据集SQL文件
"""

import re
import csv
import argparse

def simple_parse_qm9star(sql_file, output_csv, max_records=100000):
    """
    简单解析QM9*数据集
    
    Args:
        sql_file: SQL文件路径
        output_csv: 输出CSV文件路径
        max_records: 最大记录数
    """
    print(f"正在解析 {sql_file}")
    print(f"最多提取 {max_records} 条记录")
    
    # 元素列表
    elements = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr"
    ]
    
    # 所有字段
    fieldnames = ["formula_string"] + elements
    
    record_count = 0
    
    # 使用二进制模式读取文件，避免编码问题
    with open(sql_file, 'rb') as f:
        # 读取文件内容
        content = f.read()
        # 尝试解码，忽略错误
        text_content = content.decode('utf-8', errors='ignore')
    
    # 写入CSV文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 查找INSERT语句
        # 使用简单的正则表达式匹配
        pattern = rb'INSERT INTO.*?formula.*?VALUES\s*\((.*?)\)'
        matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            if record_count >= max_records:
                break
                
            # 提取值部分
            values_str = match.group(1)
            
            try:
                # 简单解析值（按逗号分割）
                values_parts = values_str.split(b',')
                values = []
                for part in values_parts:
                    # 清理引号和空格
                    part = part.strip()
                    if part.startswith(b"'") and part.endswith(b"'"):
                        part = part[1:-1]
                    values.append(part.decode('utf-8', errors='ignore'))
                
                # 构建记录
                if len(values) == len(elements) + 1:
                    record = {}
                    record["formula_string"] = values[0]
                    for i, element in enumerate(elements):
                        try:
                            record[element] = int(values[i+1]) if values[i+1].lower() != b'null' else 0
                        except:
                            record[element] = 0
                    
                    writer.writerow(record)
                    record_count += 1
                    
                    if record_count % 10000 == 0:
                        print(f"已处理 {record_count} 条记录")
                        
            except Exception as e:
                # 忽略解析错误的行
                continue
    
    print(f"完成！共提取 {record_count} 条记录到 {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='简单解析QM9*数据集SQL文件')
    parser.add_argument('--sql-file', type=str, required=True, 
                        help='QM9* SQL文件路径')
    parser.add_argument('--output-csv', type=str, required=True,
                        help='输出CSV文件路径')
    parser.add_argument('--max-records', type=int, default=100000,
                        help='最大处理记录数（默认：100000）')
    
    args = parser.parse_args()
    
    simple_parse_qm9star(
        sql_file=args.sql_file,
        output_csv=args.output_csv,
        max_records=args.max_records
    )

if __name__ == "__main__":
    main()