#!/usr/bin/env python3
"""
对实验结果 CSV 表格按分区计算排名并回填。

分区规则：以"目标属性"列（第2列）值区分，每个非空值开启一个新分区。
排名规则：
  1. 每个分区内，按"平均改善值"绝对值从大到小排名（改善越大排名越小=越好）
  2. 每个分区内，按"改善百分比"从大到小排名
  3. rank 加权 = 改善值rank + 改善率rank
  4. 总排名 = 按 rank 加权从小到大排名（相同值取平均排名）

用法:
    python3 rank_table.py input.csv [output.csv]

    # 原地更新（覆盖输入文件）
    python3 rank_table.py input.csv

    # 指定输出
    python3 rank_table.py input.csv output.csv
"""

import csv
import sys
import os
from datetime import datetime


def parse_pct(val: str) -> float:
    """解析百分比字符串，如 '93.06%' -> 93.06"""
    return float(val.strip().replace("%", ""))


def rank_values(values: list[float], higher_is_better: bool = True) -> list[int]:
    """计算排名，支持并列排名（取平均排名）。

    Args:
        values: 数值列表
        higher_is_better: True 则越大排名越前（rank越小），False 则越小排名越前

    Returns:
        排名列表（1-indexed），与 values 等长
    """
    n = len(values)
    if n == 0:
        return []

    # 构造 (value, original_index) 列表
    indexed = [(v, i) for i, v in enumerate(values)]
    # 排序
    if higher_is_better:
        indexed.sort(key=lambda x: -x[0])  # 降序
    else:
        indexed.sort(key=lambda x: x[0])   # 升序

    # 计算排名（处理并列）
    ranks = [0] * n
    i = 0
    while i < n:
        j = i
        # 找出所有与 indexed[i] 值相同的项
        while j < n and indexed[j][0] == indexed[i][0]:
            j += 1
        # 竞赛排名（"1 2 2 4"风格）：并列取相同名次，后续跳过
        competition_rank = i + 1
        for k in range(i, j):
            ranks[indexed[k][1]] = competition_rank
        i = j

    return ranks


def compute_ranks(csv_path: str, output_path: str | None = None) -> str:
    """读取 CSV，按分区计算排名，回填 rank 列，写出 CSV。"""
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # 定位列
    # 表头: 优化算法, 目标属性, ..., 平均改善值(14), rank(15), 改善百分比(16), rank(17), rank 加权(18), 总排名(19)
    col_names = [h.strip() for h in header]
    n_cols = len(col_names)

    # 找到关键列索引
    try:
        idx_algo = col_names.index("优化算法")
        idx_target = col_names.index("目标属性")
        idx_improve = col_names.index("平均改善值")
        idx_pct = col_names.index("改善百分比")
    except ValueError as e:
        print(f"找不到必需列: {e}", file=sys.stderr)
        sys.exit(1)

    # rank 列: 在 idx_improve 之后和 idx_pct 之后
    # 格式: ...平均改善值 | rank | 改善百分比 | rank | rank 加权 | 总排名
    idx_rank_improve = idx_improve + 1
    idx_rank_pct = idx_pct + 1
    idx_rank_weighted = idx_rank_pct + 1
    idx_rank_total = idx_rank_weighted + 1

    # 分区：按目标属性分组
    groups: list[list[int]] = []  # 每组是行索引列表
    current_group: list[int] = []

    for i, row in enumerate(rows):
        # 空行分隔组
        if all(c.strip() == "" for c in row):
            if current_group:
                groups.append(current_group)
                current_group = []
            continue

        target_val = row[idx_target].strip() if idx_target < len(row) else ""
        algo_val = row[idx_algo].strip() if idx_algo < len(row) else ""

        # 跳过空算法行（不应该出现，但防御）
        if not algo_val:
            continue

        # 新目标属性值 => 新分区
        if target_val:
            if current_group:
                groups.append(current_group)
            current_group = [i]
        else:
            current_group.append(i)

    if current_group:
        groups.append(current_group)

    # 对每个分区计算排名
    for group in groups:
        if len(group) == 0:
            continue

        # 提取改善值（取绝对值，因为 LUMO/HOMO 改善是负值越大越好，GAP/U 是正值越大越好）
        # 统一用绝对值衡量改善幅度
        improvements: list[float] = []
        percentages: list[float] = []

        # 先检查分区是否有有效数据
        has_data = False
        for ri in group:
            row = rows[ri]
            imp_str = row[idx_improve].strip() if idx_improve < len(row) else ""
            if imp_str:
                has_data = True
                break

        if not has_data:
            continue  # 跳过无数据分区

        for ri in group:
            row = rows[ri]
            # 补齐行
            while len(row) < n_cols:
                row.append("")

            imp_str = row[idx_improve].strip()
            pct_str = row[idx_pct].strip()

            try:
                imp_val = abs(float(imp_str))
            except (ValueError, TypeError):
                imp_val = 0.0

            try:
                pct_val = parse_pct(pct_str)
            except (ValueError, TypeError):
                pct_val = 0.0

            improvements.append(imp_val)
            percentages.append(pct_val)

        # 计算排名（越大越好 => rank 越小）
        imp_ranks = rank_values(improvements, higher_is_better=True)
        pct_ranks = rank_values(percentages, higher_is_better=True)

        # 加权排名
        weighted = [imp_ranks[i] + pct_ranks[i] for i in range(len(group))]
        total_ranks = rank_values(weighted, higher_is_better=False)  # 加权和越小越好

        # 回填
        for i, ri in enumerate(group):
            row = rows[ri]
            # 确保行足够长
            while len(row) < n_cols:
                row.append("")

            row[idx_rank_improve] = str(int(imp_ranks[i]))
            row[idx_rank_pct] = str(int(pct_ranks[i]))
            row[idx_rank_weighted] = str(int(weighted[i]))
            row[idx_rank_total] = str(int(total_ranks[i]))

            rows[ri] = row

    # 写出
    if output_path is None:
        base_dir = os.path.dirname(csv_path) or "."
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(base_dir, f"{base_name}-ranked-{ts}.csv")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

    print(f"排名完成: {csv_path} -> {output_path}")
    return output_path


def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  python3 rank_table.py input.csv [output.csv]")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    compute_ranks(csv_path, output_path)


if __name__ == "__main__":
    main()
