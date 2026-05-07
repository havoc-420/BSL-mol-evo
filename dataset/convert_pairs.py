import json
import os
import argparse

def convert_step2_to_step1(input_file, output_dir):
    """
    将 step-2-pairs.json 文件转换为 step-1.json 的单行格式
    并在文件名中添加行数标签
    """
    print(f"正在读取文件: {input_file}")
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 获取文件大小
    file_size = os.path.getsize(input_file)
    print(f"文件大小: {file_size} 字节 ({file_size / (1024*1024):.2f} MB)")
    
    # 如果文件太大，使用流式处理
    if file_size > 100 * 1024 * 1024:  # 大于100MB
        print("检测到大文件，使用流式处理...")
        return convert_large_file(input_file, output_dir)
    
    # 读取输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        handle_json_decode_error(input_file, e)
        raise
    except UnicodeDecodeError as e:
        print(f"文件编码错误: {e}")
        print("尝试使用其他编码...")
        # 尝试其他常见编码
        encodings = ['utf-8-sig', 'gbk', 'gb2312']
        data = None
        for encoding in encodings:
            try:
                with open(input_file, 'r', encoding=encoding) as f:
                    data = json.load(f)
                print(f"成功使用编码 {encoding} 读取文件")
                break
            except UnicodeDecodeError:
                print(f"编码 {encoding} 也无法读取文件")
                continue
        
        if data is None:
            raise Exception("无法使用任何编码读取文件")
    
    # 获取数据总数
    total_count = len(data)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
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

def convert_large_file(input_file, output_dir):
    """
    流式处理大文件
    """
    # 对于大文件，我们假设它是JSON数组格式
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}-processed.json")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w') as outfile:
        
        # 写入开头
        outfile.write('[\n')
        
        decoder = json.JSONDecoder()
        content = infile.read()
        pos = 0
        count = 0
        
        # 跳过开始的空白字符
        while pos < len(content) and content[pos].isspace():
            pos += 1
            
        # 应该是一个数组开始
        if pos < len(content) and content[pos] == '[':
            pos += 1
            
        # 跳过空白字符
        while pos < len(content) and content[pos].isspace():
            pos += 1
            
        first_item = True
        while pos < len(content):
            # 检查是否到达数组结尾
            if content[pos] == ']':
                break
                
            # 添加分隔符（除了第一个元素）
            if not first_item:
                outfile.write(',\n')
                
            try:
                # 解析下一个对象
                obj, end_pos = decoder.raw_decode(content, pos)
                # 写入对象
                json.dump(obj, outfile, separators=(',', ':'))
                pos += end_pos
                count += 1
                first_item = False
                
                if count % 1000 == 0:
                    print(f"已处理 {count} 条记录...")
                    
            except json.JSONDecodeError as e:
                print(f"在处理第 {count+1} 条记录时发生JSON解析错误: {e}")
                handle_json_decode_error_with_position(content, pos, e)
                raise
                
            # 跳过空白字符和逗号
            while pos < len(content) and (content[pos].isspace() or content[pos] == ','):
                pos += 1
                
        outfile.write('\n]')
        
    print(f"大文件转换完成！输出文件: {output_file}")
    print(f"总记录数: {count}")
    return output_file, count

def handle_json_decode_error(input_file, e):
    """
    处理JSON解码错误并打印详细信息
    """
    print(f"JSON解析错误: {e}")
    print(f"错误位置: 行 {e.lineno}, 列 {e.colno}")
    print(f"错误消息: {e.msg}")
    
    # 尝试定位错误附近的文本
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    error_line_idx = e.lineno - 1  # lineno 是1-indexed，lines是0-indexed
    if 0 <= error_line_idx < len(lines):
        print(f"错误行内容:")
        print(repr(lines[error_line_idx]))
        
        # 显示前后几行以便调试
        start_idx = max(0, error_line_idx - 2)
        end_idx = min(len(lines), error_line_idx + 3)
        print(f"\n上下文内容 (行 {start_idx+1}-{end_idx}):")
        for i in range(start_idx, end_idx):
            marker = ">>> " if i == error_line_idx else "    "
            print(f"{marker}{i+1:4d}: {repr(lines[i])}")

def handle_json_decode_error_with_position(content, pos, e):
    """
    处理带有位置信息的JSON解码错误
    """
    # 计算行号和列号
    lines = content[:pos].split('\n')
    line_num = len(lines)
    col_num = len(lines[-1]) if lines else 0
    
    print(f"JSON解析错误: {e}")
    print(f"错误位置: 行 {line_num}, 列 {col_num}")
    print(f"错误消息: {e.msg}")
    
    # 显示错误位置附近的上下文
    start_pos = max(0, pos - 50)
    end_pos = min(len(content), pos + 50)
    print(f"\n上下文内容 (位置 {start_pos}-{end_pos}):")
    print(repr(content[start_pos:end_pos]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将step-2格式的JSON文件转换为step-1格式')
    parser.add_argument('--input', '-i', type=str, 
                        default="/home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-2-pairs.json",
                        help='输入文件路径')
    parser.add_argument('--output', '-o', type=str,
                        default="/home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/data",
                        help='输出目录路径')
    
    args = parser.parse_args()
    
    convert_step2_to_step1(args.input, args.output)