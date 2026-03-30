#!/usr/bin/env python3
"""
convert_qm9_evo_to_paths.py

将 qm9-evo step-1/step-2 pairs 数据转换为 v0.3 路径格式。

使用方式：
    # 转换单个文件
    python convert_qm9_evo_to_paths.py \
        -i ../dataset/data/qm9-evo-pairs-step-2-pairs-26529.json \
        -o ../dataset/data/qm9-evo-paths-step-2.json \
        -p lumo_change

    # 合并转换 step-1 + step-2
    python convert_qm9_evo_to_paths.py \
        --merge \
        -o ../dataset/data/qm9-evo-paths-all.json \
        -p lumo_change \
        --max-total 50000
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from core.data import (
    convert_pairs_file_to_paths,
    convert_step_files_to_path_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="转换 qm9-evo pairs 数据为 v0.3 路径格式"
    )
    
    # 输入输出
    parser.add_argument(
        "-i", "--input",
        help="输入 pairs JSON 文件（单文件模式）"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="输出 paths JSON 文件"
    )
    
    # 合并模式
    parser.add_argument(
        "--merge",
        action="store_true",
        help="合并多个 step 文件"
    )
    parser.add_argument(
        "--step-files",
        nargs="+",
        help="要合并的 step 文件列表（按步骤数顺序）"
    )
    
    # 属性配置
    parser.add_argument(
        "-p", "--property",
        default="lumo_change",
        help="目标属性名 (default: lumo_change)"
    )
    parser.add_argument(
        "--property-changes",
        help="属性变化值文件（JSON格式）"
    )
    
    # 样本控制
    parser.add_argument(
        "--max-samples",
        type=int,
        help="最大处理样本数（单文件模式）"
    )
    parser.add_argument(
        "--max-total",
        type=int,
        help="最大总样本数（合并模式）"
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        default=True,
        help="平衡不同步骤数的样本（合并模式）"
    )
    
    # 步骤过滤
    parser.add_argument(
        "--min-steps",
        type=int,
        default=1,
        help="最小步骤数"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="最大步骤数"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    if args.merge:
        # 合并模式
        if not args.step_files:
            # 自动发现 step 文件
            data_dir = project_root / "dataset" / "data"
            step_files = {}
            
            for f in sorted(data_dir.glob("qm9-evo-pairs-step-*-pairs-*.json")):
                # 从文件名提取步骤数
                parts = f.stem.split("-")
                if "step" in parts:
                    idx = parts.index("step")
                    step_count = int(parts[idx + 1])
                    step_files[step_count] = str(f)
            
            if not step_files:
                print("未找到任何 step 文件")
                sys.exit(1)
            
            print(f"自动发现 {len(step_files)} 个文件:")
            for k, v in sorted(step_files.items()):
                print(f"  step-{k}: {v}")
        else:
            # 从命令行参数构建
            step_files = {
                i + 1: f
                for i, f in enumerate(args.step_files)
            }
        
        stats = convert_step_files_to_path_dataset(
            step_files=step_files,
            output_file=args.output,
            target_property=args.property,
            property_changes_file=args.property_changes,
            max_total_samples=args.max_total,
            balance_steps=args.balance,
        )
        
        print("\n=== 合并统计 ===")
        print(f"总路径数: {stats['total_paths']}")
        for step, step_stats in stats.get("per_step", {}).items():
            print(f"\nstep-{step}:")
            for k, v in step_stats.items():
                print(f"  {k}: {v}")
    
    else:
        # 单文件模式
        if not args.input:
            print("错误: 单文件模式需要 -i/--input 参数")
            sys.exit(1)
        
        stats = convert_pairs_file_to_paths(
            input_file=args.input,
            output_file=args.output,
            target_property=args.property,
            property_changes_file=args.property_changes,
            max_samples=args.max_samples,
            min_steps=args.min_steps,
            max_steps=args.max_steps,
        )
        
        print("\n=== 转换统计 ===")
        for k, v in stats.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
