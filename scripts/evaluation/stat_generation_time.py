#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 batch_optimizer.py 输出目录下每个分子 JSON 文件中
`statistics.generation_time`（单位：秒）。

Usage:
    python mol_evo/scripts/evaluation/stat_generation_time.py \
        --input-dir mol_evo/output/evo-mo/batch_optimization_20260418_010636

    # 同时导出逐文件明细到 CSV
    python mol_evo/scripts/evaluation/stat_generation_time.py \
        --input-dir mol_evo/output/evo-mo/batch_optimization_20260418_010636 \
        --output-csv gen_time_stats.csv
"""

import os
import json
import argparse
import csv
import statistics as _stats
from glob import glob


def collect(input_dir):
    """遍历目录下所有 *.json（排除 *_topK.csv 等非优化结果），提取 generation_time。"""
    records = []
    skipped = []
    pattern = os.path.join(input_dir, '*.json')
    for path in sorted(glob(pattern)):
        fname = os.path.basename(path)
        # 跳过非单分子结果的 JSON（例如 batch_results.json）
        if fname in ('batch_results.json',):
            skipped.append((fname, 'aggregated file'))
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            skipped.append((fname, f'read error: {e}'))
            continue

        stats = data.get('statistics') or {}
        gt = stats.get('generation_time')
        if gt is None:
            skipped.append((fname, 'no statistics.generation_time'))
            continue

        records.append({
            'file': fname,
            'initial_smiles': data.get('initial_smiles', ''),
            'generation_time_sec': float(gt),
            'attempt_count': stats.get('attempt_count'),
        })
    return records, skipped


def summarize(records):
    times = [r['generation_time_sec'] for r in records]
    if not times:
        return None
    return {
        'count': len(times),
        'total_sec': sum(times),
        'mean_sec': _stats.mean(times),
        'median_sec': _stats.median(times),
        'min_sec': min(times),
        'max_sec': max(times),
        'std_sec': _stats.pstdev(times) if len(times) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description='统计每个分子 JSON 的 generation_time（秒）')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='batch_optimizer 输出目录')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='可选：逐文件明细输出 CSV 路径')
    parser.add_argument('--top', type=int, default=5,
                        help='打印耗时最长/最短的前 N 条，默认 5')
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    print(f'[输入目录] {input_dir}')

    records, skipped = collect(input_dir)
    summary = summarize(records)

    if not summary:
        print('❌ 未在目录下找到任何含 statistics.generation_time 的 JSON 文件')
        for f, why in skipped:
            print(f'  skip: {f}  ({why})')
        return

    print('')
    print('===== 生成时间统计（单位：秒） =====')
    print(f"样本数 (count)      : {summary['count']}")
    print(f"总耗时 (total)      : {summary['total_sec']:.3f} s "
          f"({summary['total_sec']/60:.2f} min)")
    print(f"平均   (mean)       : {summary['mean_sec']:.3f} s")
    print(f"中位数 (median)     : {summary['median_sec']:.3f} s")
    print(f"最小   (min)        : {summary['min_sec']:.3f} s")
    print(f"最大   (max)        : {summary['max_sec']:.3f} s")
    print(f"总体标准差 (pstdev) : {summary['std_sec']:.3f} s")

    # Top-N 最长/最短
    sorted_desc = sorted(records, key=lambda r: r['generation_time_sec'], reverse=True)
    n = max(0, min(args.top, len(sorted_desc)))
    if n:
        print(f'\n>> 耗时最长 Top {n}:')
        for r in sorted_desc[:n]:
            print(f"  {r['generation_time_sec']:8.3f} s | "
                  f"attempt={r['attempt_count']} | {r['file']}")
        print(f'\n>> 耗时最短 Top {n}:')
        for r in sorted_desc[-n:][::-1]:
            print(f"  {r['generation_time_sec']:8.3f} s | "
                  f"attempt={r['attempt_count']} | {r['file']}")

    if skipped:
        print(f'\n(跳过 {len(skipped)} 个文件)')
        for f, why in skipped[:10]:
            print(f'  skip: {f}  ({why})')
        if len(skipped) > 10:
            print(f'  ... 以及其它 {len(skipped) - 10} 个')

    # 导出明细 CSV
    if args.output_csv:
        out = os.path.abspath(args.output_csv)
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['file', 'initial_smiles', 'generation_time_sec', 'attempt_count'],
            )
            writer.writeheader()
            for r in sorted_desc:
                writer.writerow(r)
        print(f'\n📄 明细已导出: {out}')


if __name__ == '__main__':
    main()
