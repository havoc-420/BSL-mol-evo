#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 JSON 数据中绘制 2(property) x 3(split) 的拼接散点图。

Usage:
    conda activate plot_env && cd /home/ubuntu/mol_opt && \
    python mol-ofo/mol_evo/scripts/visualization/plot_scatter_grid.py \
        --input-dir mol-ofo/mol_evo/output/scatter_plots/20260417_075604

    # 指定属性
    python mol-ofo/mol_evo/scripts/visualization/plot_scatter_grid.py \
        --input-dir mol-ofo/mol_evo/output/scatter_plots/20260417_075604 \
        --properties lumo_change homo_change

    # 指定输出路径
    python mol-ofo/mol_evo/scripts/visualization/plot_scatter_grid.py \
        --input-dir mol-ofo/mol_evo/output/scatter_plots/20260417_075604 \
        --output scatter_grid.png
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SPLITS = ['train', 'val', 'test']
SPLIT_DISPLAY = {'train': 'Train Dataset', 'val': 'Val Dataset', 'test': 'Test Dataset'}
SPLIT_COLORS = {'train': '#4CAF50', 'val': '#FFC107', 'test': '#2196F3'}


def load_json_data(input_dir, properties, splits):
    """加载所有 JSON 数据文件"""
    data = {}
    for prop in properties:
        for split in splits:
            filename = f'results_{prop}_{split}.json'
            filepath = os.path.join(input_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data[(prop, split)] = json.load(f)
            else:
                print(f"  ⚠️  文件不存在: {filepath}")
    return data


def plot_scatter_grid(data, properties, splits, output_path):
    """绘制 2(properties) x 3(splits) 的拼接散点图"""
    n_rows = len(properties)
    n_cols = len(splits)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows), dpi=150)

    # 统一 axes 为 2D 数组
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # 统一所有数据的坐标范围（按属性统一）
    for row, prop in enumerate(properties):
        all_preds = []
        all_trues = []
        for split in splits:
            key = (prop, split)
            if key in data:
                all_preds.extend(data[key]['predictions'])
                all_trues.extend(data[key]['true_values'])

        if all_preds:
            all_vals = np.concatenate([all_trues, all_preds])
            vmin, vmax = np.min(all_vals), np.max(all_vals)
            margin = (vmax - vmin) * 0.05
            xlim = (vmin - margin, vmax + margin)
        else:
            xlim = (-1, 1)

        for col, split in enumerate(splits):
            ax = axes[row, col]
            key = (prop, split)

            if key not in data:
                ax.set_visible(False)
                continue

            d = data[key]
            preds = np.array(d['predictions'])
            trues = np.array(d['true_values'])
            metrics = d['metrics']
            color = SPLIT_COLORS.get(split, '#2196F3')

            # 散点
            ax.scatter(trues, preds, alpha=0.3, s=5, c=color, edgecolors='none', label='Predictions')

            # 对角线
            ax.plot(xlim, xlim, 'r--', linewidth=1.5, alpha=0.7, label='y = x')
            ax.set_xlim(xlim)
            ax.set_ylim(xlim)
            ax.set_aspect('equal', adjustable='box')

            # 指标文本（不显示 N）
            metrics_text = (
                f"R² = {metrics['R²']:.4f}\n"
                f"PCC = {metrics['PCC']:.4f}\n"
                f"RMSE = {metrics['RMSE']:.4f}\n"
                f"MAE = {metrics['MAE']:.4f}"
            )
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

            # 标签
            prop_display = prop.replace('_change', '').upper()
            split_display = SPLIT_DISPLAY.get(split, split.capitalize())

            if row == n_rows - 1:
                # 统一 X 轴标签：列出所有属性缩写
                all_prop_names = '/'.join(p.replace('_change', '').upper() for p in properties)
                ax.set_xlabel(f'True {all_prop_names} Change', fontsize=18, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'Predicted {prop_display} Change', fontsize=18, fontweight='bold')

            # 第一行显示列标题（split 名）
            if row == 0:
                ax.set_title(split_display, fontsize=20, fontweight='bold')

            # 最右列在右侧标注属性名
            if col == n_cols - 1:
                ax.annotate(f'{prop_display} Change', xy=(1.02, 0.5), xycoords='axes fraction',
                            fontsize=20, fontweight='bold', rotation=-90,
                            ha='left', va='center')

            ax.tick_params(axis='both', labelsize=12)
            ax.legend(loc='lower right', fontsize=11)
            ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # 右侧留空间给行标签
    plt.savefig(output_path, bbox_inches='tight')

    # 同时保存 PDF 版本
    pdf_path = os.path.splitext(output_path)[0] + '.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', format='pdf')

    plt.close()
    print(f"📊 拼接散点图已保存: {output_path}")
    print(f"📄 PDF 版本已保存: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description='从 JSON 数据绘制 2x3 拼接散点图')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='包含 results_*.json 的目录')
    parser.add_argument('--properties', type=str, nargs='+',
                        default=['lumo_change', 'homo_change'],
                        help='属性列表 (默认: lumo_change homo_change)')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=SPLITS,
                        help='分割列表 (默认: train val test)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径 (默认: <input-dir>/scatter_grid.png)')

    args = parser.parse_args()

    output_path = args.output or os.path.join(args.input_dir, 'scatter_grid.png')

    print(f"📂 输入目录: {args.input_dir}")
    print(f"📋 属性: {args.properties}")
    print(f"📋 分割: {args.splits}")

    data = load_json_data(args.input_dir, args.properties, args.splits)
    print(f"📊 加载了 {len(data)} 个数据文件")

    if not data:
        print("❌ 没有可用的数据，退出")
        return

    plot_scatter_grid(data, args.properties, args.splits, output_path)


if __name__ == '__main__':
    main()
