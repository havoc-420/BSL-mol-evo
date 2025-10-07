import pandas as pd
import sys
import os
from tqdm import tqdm

def update_operation_type(row):
    """
    根据操作详情更新操作类型
    """
    operation_type = row['operation_type']
    operation_detail = row['operation_detail']
    
    # 只对特定类型进行更新
    if operation_type == "add" and ("指定手性(" in operation_detail or "指定顺反(" in operation_detail):
        return "add_stereo"
    elif operation_type == "del" and ("指定手性(" in operation_detail or "指定顺反(" in operation_detail):
        return "del_stereo"
    elif operation_type == "replace":
        # 检查是否是手性替换操作
        if ("指定手性(" in operation_detail or "指定顺反(" in operation_detail):
            # 解析replace操作的详情，检查被替换和替换的内容
            try:
                # 提取被替换的部分和替换的部分
                parts = operation_detail.split("' with '")
                if len(parts) == 2:
                    removed_part = parts[0].replace("replace '", "")
                    added_part = parts[1].replace("'", "")
                    
                    # 如果两边都涉及手性信息，则是replace_stereo
                    if ("指定手性(" in removed_part or "指定顺反(" in removed_part) and \
                       ("指定手性(" in added_part or "指定顺反(" in added_part):
                        return "replace_stereo"
            except:
                pass
        return operation_type
    else:
        return operation_type

def main():
    # 设置数据目录
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    input_file = os.path.join(data_dir, 'qm9-evo-pairs-step-1.csv')
    backup_file = os.path.join(data_dir, 'qm9-evo-pairs-step-1.backup.csv')
    output_file = os.path.join(data_dir, 'qm9-evo-pairs-step-1.csv')
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        sys.exit(1)
    
    # 创建备份
    print("创建备份文件...")
    import shutil
    shutil.copy2(input_file, backup_file)
    print(f"备份文件已创建: {backup_file}")
    
    # 读取数据
    print("读取数据...")
    df = pd.read_csv(input_file)
    print(f"总共读取 {len(df)} 行数据")
    
    # 显示更新前的操作类型分布
    print("\n更新前的操作类型分布:")
    print(df['operation_type'].value_counts())
    
    # 更新操作类型
    print("\n更新操作类型...")
    # 使用tqdm显示进度条
    updated_types = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="更新操作类型"):
        updated_types.append(update_operation_type(row))
    
    df['operation_type'] = updated_types
    
    # 修复to_atom_position列的数据类型问题，确保是整数而不是浮点数
    if 'to_atom_position' in df.columns:
        print("\n修复to_atom_position列的数据类型...")
        # 将to_atom_position列转换为整数类型，空值会保持为NaN
        df['to_atom_position'] = pd.to_numeric(df['to_atom_position'], errors='coerce').astype('Int64')
    
    # 重新排列列的顺序，将operation_detail和evolved_molecule移到最后
    print("\n调整列顺序...")
    column_order = [col for col in df.columns if col not in ['operation_detail', 'evolved_molecule']]
    column_order.extend(['operation_detail', 'evolved_molecule'])
    df = df[column_order]
    
    # 显示更新后的操作类型分布
    print("\n更新后的操作类型分布:")
    print(df['operation_type'].value_counts())
    
    # 保存更新后的数据
    print("\n保存更新后的数据...")
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"数据已保存到 {output_file}")

if __name__ == "__main__":
    main()