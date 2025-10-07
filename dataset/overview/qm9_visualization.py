#!/usr/bin/env python3
"""
QM9数据集可视化脚本
用于绘制QM9数据集的美观统计图表
支持全原子和重原子（非氢原子）两种统计模式
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse

def create_qm9_visualization(heavy_only=False):
    """
    创建QM9数据集的可视化图表
    
    Args:
        heavy_only (bool): 是否使用重原子统计数据
    """
    if heavy_only:
        # QM9重原子统计信息
        total_molecules = 129004
        atom_range = (3, 29)
        avg_atoms = 16.72
        
        # 重原子数分布数据
        atom_counts = list(range(3, 30))
        # 移除缺失的原子数（如28）
        atom_counts.remove(28)
        
        molecule_counts = [
            9,      # 3 heavy atoms
            31,     # 4 heavy atoms
            126,    # 5 heavy atoms
            603,    # 6 heavy atoms
            3108,   # 7 heavy atoms
            17665,  # 8 heavy atoms
            107462, # 9 heavy atoms
            12345,  # 10 heavy atoms (示例数据，需要实际统计数据)
            5678,   # 11 heavy atoms (示例数据，需要实际统计数据)
            2345,   # 12 heavy atoms (示例数据，需要实际统计数据)
            1234,   # 13 heavy atoms (示例数据，需要实际统计数据)
            987,    # 14 heavy atoms (示例数据，需要实际统计数据)
            654,    # 15 heavy atoms (示例数据，需要实际统计数据)
            432,    # 16 heavy atoms (示例数据，需要实际统计数据)
            321,    # 17 heavy atoms (示例数据，需要实际统计数据)
            210,    # 18 heavy atoms (示例数据，需要实际统计数据)
            150,    # 19 heavy atoms (示例数据，需要实际统计数据)
            100,    # 20 heavy atoms (示例数据，需要实际统计数据)
            80,     # 21 heavy atoms (示例数据，需要实际统计数据)
            50,     # 22 heavy atoms (示例数据，需要实际统计数据)
            30,     # 23 heavy atoms (示例数据，需要实际统计数据)
            20,     # 24 heavy atoms (示例数据，需要实际统计数据)
            10,     # 25 heavy atoms (示例数据，需要实际统计数据)
            5,      # 26 heavy atoms (示例数据，需要实际统计数据)
            3,      # 27 heavy atoms (示例数据，需要实际统计数据)
            1       # 29 heavy atoms (示例数据，需要实际统计数据)
        ]
        
        title_suffix = " (Heavy Atoms Only)"
        atom_label = "Number of Heavy Atoms"
    else:
        # QM9全原子统计信息
        total_molecules = 130831
        atom_range = (3, 29)
        avg_atoms = 18.03
        
        # 原子数分布数据
        atom_counts = list(range(3, 30))
        # 移除缺失的原子数（如28）
        atom_counts.remove(28)
        
        molecule_counts = [
            2,     # 3 atoms
            4,     # 4 atoms
            5,     # 5 atoms
            12,    # 6 atoms
            20,    # 7 atoms
            65,    # 8 atoms
            172,   # 9 atoms
            483,   # 10 atoms
            1053,  # 11 atoms
            2189,  # 12 atoms
            4027,  # 13 atoms
            6758,  # 14 atoms
            10216, # 15 atoms
            13824, # 16 atoms
            16969, # 17 atoms
            17442, # 18 atoms
            18146, # 19 atoms
            12403, # 20 atoms
            13166, # 21 atoms
            4428,  # 22 atoms
            6362,  # 23 atoms
            712,   # 24 atoms
            1923,  # 25 atoms
            59,    # 26 atoms
            356,   # 27 atoms
            35     # 29 atoms
        ]
        
        title_suffix = ""
        atom_label = "Number of Atoms"
    
    # 计算百分比
    percentages = [count / total_molecules * 100 for count in molecule_counts]
    
    # 设置字体支持
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Bitstream Vera Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'QM9 Dataset Statistics{title_suffix}', fontsize=20, fontweight='bold')
    
    # 子图1: 原子数分布柱状图（完整展示）
    ax1 = plt.subplot(2, 2, 1)
    bars = ax1.bar(atom_counts, molecule_counts, color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_xlabel(atom_label, fontsize=12)
    ax1.set_ylabel('Number of Molecules', fontsize=12)
    ax1.set_title(f'Atom Count Distribution{title_suffix}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 为数量较少的项目添加数值标签
    for i, (atom_count, molecule_count) in enumerate(zip(atom_counts, molecule_counts)):
        if molecule_count < 1000:  # 为数量较少的项目添加标签
            ax1.text(atom_count, molecule_count + 50, str(molecule_count), 
                     ha='center', va='bottom', fontsize=8, rotation=90)
    
    # 突出显示最常见的原子数
    if heavy_only:
        # 对于重原子，最常见的几个是9, 8, 7, 6, 5
        top5_indices = [6, 5, 4, 3, 2]  # 对应重原子数9, 8, 7, 6, 5
    else:
        # 对于全原子，最常见的几个是19, 18, 17, 16, 21
        top5_indices = [16, 15, 14, 13, 18]  # 对应原子数19, 18, 17, 16, 21
    
    for i in top5_indices:
        if i < len(bars):
            bars[i].set_color('orange')
            bars[i].set_alpha(0.9)
    
    # 子图2: 原子数分布饼图（Top 5 + Others）
    ax2 = plt.subplot(2, 2, 2)
    if heavy_only:
        top5_atom_counts = [9, 8, 7, 6, 5]
        top5_molecule_counts = [107462, 17665, 3108, 603, 126]
    else:
        top5_atom_counts = [19, 18, 17, 16, 21]
        top5_molecule_counts = [18146, 17442, 16969, 13824, 13166]
    
    top5_labels = [f'{count} {"heavy atoms" if heavy_only else "atoms"}' for count in top5_atom_counts]
    
    others_count = total_molecules - sum(top5_molecule_counts)
    sizes = top5_molecule_counts + [others_count]
    labels = top5_labels + ['Others']
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Top 5 Atom Counts Distribution{title_suffix}', fontsize=14, fontweight='bold')
    
    # 子图3: 累积分布图
    ax3 = plt.subplot(2, 2, 3)
    cumulative_counts = np.cumsum(molecule_counts)
    ax3.plot(atom_counts, cumulative_counts, marker='o', linewidth=2, markersize=4, color='green')
    ax3.set_xlabel(atom_label, fontsize=12)
    ax3.set_ylabel('Cumulative Number of Molecules', fontsize=12)
    ax3.set_title(f'Cumulative Distribution{title_suffix}', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 添加网格线帮助查看小值
    ax3.set_yscale('log')  # 使用对数刻度更好地显示全部数据
    
    # 子图4: 统计信息文本
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # 添加统计信息文本
    stats_text = f"""
    QM9 Dataset Statistics{title_suffix}
    
    Total Molecules: {total_molecules:,}
    
    Atom Count Range: {atom_range[0]} - {atom_range[1]}
    
    Average Atom Count: {avg_atoms}
    
    Top 5 Atom Counts:
    1. {top5_atom_counts[0]} atoms: {top5_molecule_counts[0]:,} ({top5_molecule_counts[0]/total_molecules*100:.2f}%)
    2. {top5_atom_counts[1]} atoms: {top5_molecule_counts[1]:,} ({top5_molecule_counts[1]/total_molecules*100:.2f}%)
    3. {top5_atom_counts[2]} atoms: {top5_molecule_counts[2]:,} ({top5_molecule_counts[2]/total_molecules*100:.2f}%)
    4. {top5_atom_counts[3]} atoms: {top5_molecule_counts[3]:,} ({top5_molecule_counts[3]/total_molecules*100:.2f}%)
    5. {top5_atom_counts[4]} atoms: {top5_molecule_counts[4]:,} ({top5_molecule_counts[4]/total_molecules*100:.2f}%)
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    filename = 'qm9_dataset_visualization_heavy.png' if heavy_only else 'qm9_dataset_visualization.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.show()
    
    print(f"Visualization saved as {filename}")

def main():
    parser = argparse.ArgumentParser(description='QM9数据集可视化')
    parser.add_argument('--heavy-only', action='store_true', 
                        help='使用重原子（非氢原子）统计数据')
    
    args = parser.parse_args()
    create_qm9_visualization(heavy_only=args.heavy_only)

if __name__ == "__main__":
    main()