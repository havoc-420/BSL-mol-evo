#!/usr/bin/env python3
"""
CSV查找工具 - 根据行索引和列名查找目标属性

该脚本提供便捷的方法来在CSV文件中根据行索引(row-index)和列名(col-name)查找目标属性。
支持命令行参数和交互式两种使用方式。
"""

import argparse
import csv
import os
import sys
from typing import Optional, List

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("注意: pandas未安装，将使用内置csv模块")

try:
    import inquirer
    INQUIRER_AVAILABLE = True
except ImportError:
    INQUIRER_AVAILABLE = False
    print("注意: inquirer未安装，将使用基础交互模式")

def find_value_pandas(csv_file: str, row_index: int, col_name: str) -> Optional[str]:
    """
    使用pandas在CSV文件中查找值
    
    Args:
        csv_file: CSV文件路径
        row_index: 行索引 (0-based)
        col_name: 列名
        
    Returns:
        找到的值或None
    """
    try:
        df = pd.read_csv(csv_file)
        if col_name not in df.columns:
            print(f"错误: 列名 '{col_name}' 不存在于文件中")
            print(f"可用列名: {list(df.columns)}")
            return None
            
        if row_index >= len(df):
            print(f"错误: 行索引 {row_index} 超出范围，文件共有 {len(df)} 行")
            return None
            
        return df.at[row_index, col_name]
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def find_value_builtin(csv_file: str, row_index: int, col_name: str) -> Optional[str]:
    """
    使用内置csv模块在CSV文件中查找值
    
    Args:
        csv_file: CSV文件路径
        row_index: 行索引 (0-based)
        col_name: 列名
        
    Returns:
        找到的值或None
    """
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            # 获取列索引
            headers = reader.fieldnames
            if col_name not in headers:
                print(f"错误: 列名 '{col_name}' 不存在于文件中")
                print(f"可用列名: {headers}")
                return None
                
            # 遍历到目标行
            for i, row in enumerate(reader):
                if i == row_index:
                    return row[col_name]
                    
            print(f"错误: 行索引 {row_index} 超出范围")
            return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def find_value(csv_file: str, row_index: int, col_name: str) -> Optional[str]:
    """
    在CSV文件中查找值的主函数
    
    Args:
        csv_file: CSV文件路径
        row_index: 行索引 (0-based)
        col_name: 列名
        
    Returns:
        找到的值或None
    """
    if PANDAS_AVAILABLE:
        return find_value_pandas(csv_file, row_index, col_name)
    else:
        return find_value_builtin(csv_file, row_index, col_name)

def list_columns(csv_file: str) -> List[str]:
    """
    列出CSV文件中的所有列名
    
    Args:
        csv_file: CSV文件路径
        
    Returns:
        列名列表
    """
    try:
        if PANDAS_AVAILABLE:
            df = pd.read_csv(csv_file)
            return list(df.columns)
        else:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                return list(reader.fieldnames)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return []

def select_column_interactive(columns: List[str]) -> Optional[str]:
    """
    使用交互式方式选择列名，支持方向键和模糊搜索
    
    Args:
        columns: 列名列表
        
    Returns:
        选中的列名或None
    """
    if INQUIRER_AVAILABLE:
        # 使用inquirer进行高级交互
        questions = [
            inquirer.Text(
                'filter',
                message="输入关键词过滤列名 (留空显示所有列)"
            )
        ]
        
        filtered_columns = columns[:]
        filter_text = ""
        
        while True:
            # 显示当前过滤状态
            if filter_text:
                print(f"\n当前过滤: '{filter_text}' (匹配 {len(filtered_columns)} 项)")
            else:
                print(f"\n显示全部 {len(columns)} 个列名")
            
            # 如果过滤后结果为空，提示用户
            if not filtered_columns:
                print("没有匹配的列名")
                # 重新获取过滤条件
                answers = inquirer.prompt(questions)
                if answers is None:
                    return None
                filter_text = answers['filter'].strip().lower()
                filtered_columns = [col for col in columns if filter_text in col.lower()]
                continue
            
            # 创建选择列表
            choices = filtered_columns[:]
            choices.append("────────────────────────────────")
            choices.append("🔄 重新输入过滤条件")
            choices.append("📋 显示所有列名")
            choices.append("❌ 取消选择")
            
            # 列选择问题
            column_question = [
                inquirer.List(
                    'column',
                    message=f"选择列名 (显示 {len(filtered_columns)} / {len(columns)} 项)",
                    choices=choices,
                    carousel=True
                )
            ]
            
            # 获取用户选择
            column_answer = inquirer.prompt(column_question)
            if column_answer is None:
                return None
                
            selection = column_answer['column']
            
            # 处理特殊选项
            if selection == "🔄 重新输入过滤条件":
                answers = inquirer.prompt(questions)
                if answers is None:
                    return None
                filter_text = answers['filter'].strip().lower()
                if filter_text:
                    filtered_columns = [col for col in columns if filter_text in col.lower()]
                else:
                    filtered_columns = columns[:]
                continue
            elif selection == "📋 显示所有列名":
                filter_text = ""
                filtered_columns = columns[:]
                continue
            elif selection == "❌ 取消选择":
                return None
            elif selection == "────────────────────────────────":
                continue
            else:
                # 选择了具体的列
                print(f"已选择列: {selection}")
                return selection
    else:
        # 基础交互模式
        print(f"\n文件中共有 {len(columns)} 列:")
        for i, col in enumerate(columns):
            print(f"  {i+1}. {col}")
        
        # 获取列名
        while True:
            col_input = input("\n请输入列名或列索引 (输入 'list' 重新显示列名): ").strip()
            if col_input.lower() == 'list':
                for i, col in enumerate(columns):
                    print(f"  {i+1}. {col}")
                continue
                
            if col_input in columns:
                col_name = col_input
                break
            elif col_input.isdigit() and 1 <= int(col_input) <= len(columns):
                col_name = columns[int(col_input) - 1]
                break
            else:
                print("无效输入，请输入有效的列名或列索引")
        
        return col_name

def interactive_mode(csv_file: Optional[str] = None):
    """
    交互式模式，允许用户选择CSV文件、行索引和列名
    
    Args:
        csv_file: CSV文件路径（可选）
    """
    print("=== CSV 查找工具 - 交互式模式 ===")
    
    # 获取CSV文件路径
    if csv_file is None:
        csv_file = input("请输入CSV文件路径: ").strip()
    else:
        print(f"使用文件: {csv_file}")
        
    if not os.path.exists(csv_file):
        print(f"错误: 文件 '{csv_file}' 不存在")
        return
    
    # 显示列名并选择
    columns = list_columns(csv_file)
    if not columns:
        print("错误: 无法读取列名")
        return
    
    # 选择列名
    col_name = select_column_interactive(columns)
    if col_name is None:
        print("已取消选择")
        return
    
    # 获取行索引
    while True:
        row_input = input(f"\n请输入行索引 (0-{get_row_count(csv_file)-1}): ").strip()
        if row_input.isdigit():
            row_index = int(row_input)
            break
        else:
            print("无效输入，请输入数字")
    
    # 查找并显示结果
    result = find_value(csv_file, row_index, col_name)
    if result is not None:
        print(f"\n结果: 在第 {row_index} 行 '{col_name}' 列的值为: {result}")
    else:
        print("\n未找到结果")

def get_row_count(csv_file: str) -> int:
    """
    获取CSV文件的行数
    
    Args:
        csv_file: CSV文件路径
        
    Returns:
        行数
    """
    try:
        if PANDAS_AVAILABLE:
            df = pd.read_csv(csv_file)
            return len(df)
        else:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                return sum(1 for row in reader) - 1  # 减去标题行
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return 0

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="在CSV文件中根据行索引和列名查找目标属性")
    parser.add_argument("csv_file", nargs='?', help="CSV文件路径")
    parser.add_argument("-r", "--row", type=int, help="行索引 (0-based)")
    parser.add_argument("-c", "--column", help="列名")
    parser.add_argument("-i", "--interactive", action="store_true", 
                        help="交互式模式")
    parser.add_argument("--list-columns", action="store_true",
                        help="列出CSV文件中的所有列名")
    
    args = parser.parse_args()
    
    # 交互式模式
    if args.interactive:
        interactive_mode(args.csv_file)
        return
    
    # 如果提供了CSV文件但没有其他参数，则进入交互模式
    if args.csv_file and args.row is None and not args.column and not args.list_columns:
        interactive_mode(args.csv_file)
        return
    
    # 检查文件是否存在
    if args.csv_file and not os.path.exists(args.csv_file):
        print(f"错误: 文件 '{args.csv_file}' 不存在")
        sys.exit(1)
    
    # 列出列名
    if args.list_columns and args.csv_file:
        columns = list_columns(args.csv_file)
        if columns:
            print("列名列表:")
            for i, col in enumerate(columns):
                print(f"  {i+1}. {col}")
        return
    
    # 检查必需参数
    if args.csv_file and (args.row is None or not args.column):
        print("错误: 必须指定行索引和列名")
        parser.print_help()
        sys.exit(1)
    
    # 查找并显示结果
    if args.csv_file and args.row is not None and args.column:
        result = find_value(args.csv_file, args.row, args.column)
        if result is not None:
            print(result)
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()