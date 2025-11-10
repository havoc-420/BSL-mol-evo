import json
import os
import argparse

def convert_step2_to_step1(input_file, output_dir):
    """
    将 step-2-pairs.json 文件转换为 step-1.json 的单行格式
    并在文件名中添加行数标签
    """
    # 读取输入文件
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 获取数据总数
    total_count = len(data)
    
    # 创建输出文件名，包含行数标签
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}-{total_count}.json")
    
    # 写入转换后的数据，每个item占一行
    with open(output_file, 'w') as f:
        f.write('[\n')
        for i, item in enumerate(data):
            json.dump(item, f, separators=(',', ':'))
            if i < len(data) - 1:  # 不是最后一个元素
                f.write(',\n')
            else:  # 最后一个元素
                f.write('\n')
        f.write(']')
    
    print(f"转换完成！输出文件: {output_file}")
    print(f"总行数: {total_count}")
    
    return output_file, total_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将step-2格式的JSON文件转换为step-1格式')
    parser.add_argument('--input', '-i', type=str, 
                        default="/home/data2/rhj/project/mol_editor/mol_evo/dataset/data/qm9-evo-pairs-step-2-pairs.json",
                        help='输入文件路径')
    parser.add_argument('--output', '-o', type=str,
                        default="/home/data2/rhj/project/mol_editor/mol_evo/dataset/data",
                        help='输出目录路径')
    
    args = parser.parse_args()
    
    convert_step2_to_step1(args.input, args.output)