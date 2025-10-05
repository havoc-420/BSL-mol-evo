#!/usr/bin/env python3
"""
QM9数据集可视化脚本
用于绘制QM9数据集的美观统计图表
"""

import matplotlib.pyplot as plt
import numpy as np

def create_qm9_visualization():
    """
    创建QM9数据集的可视化图表
    """
    # QM9数据集统计信息
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
    
    # 计算百分比
    percentages = [count / total_molecules * 100 for count in molecule_counts]
    
    # 设置字体支持
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Bitstream Vera Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('QM9 Dataset Statistics', fontsize=20, fontweight='bold')
    
    # 子图1: 原子数分布柱状图（完整展示）
    ax1 = plt.subplot(2, 2, 1)
    bars = ax1.bar(atom_counts, molecule_counts, color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_xlabel('Number of Atoms', fontsize=12)
    ax1.set_ylabel('Number of Molecules', fontsize=12)
    ax1.set_title('Atom Count Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 为数量较少的项目添加数值标签
    for i, (atom_count, molecule_count) in enumerate(zip(atom_counts, molecule_counts)):
        if molecule_count < 1000:  # 为数量较少的项目添加标签
            ax1.text(atom_count, molecule_count + 50, str(molecule_count), 
                     ha='center', va='bottom', fontsize=8, rotation=90)
    
    # 突出显示最常见的原子数
    top5_indices = [16, 15, 14, 13, 18]  # 对应原子数19, 18, 17, 16, 21
    for i in top5_indices:
        bars[i].set_color('orange')
        bars[i].set_alpha(0.9)
    
    # 子图2: 原子数分布饼图（Top 5 + Others）
    ax2 = plt.subplot(2, 2, 2)
    top5_atom_counts = [19, 18, 17, 16, 21]
    top5_molecule_counts = [18146, 17442, 16969, 13824, 13166]
    top5_labels = [f'{count} atoms' for count in top5_atom_counts]
    
    others_count = total_molecules - sum(top5_molecule_counts)
    sizes = top5_molecule_counts + [others_count]
    labels = top5_labels + ['Others']
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Top 5 Atom Counts Distribution', fontsize=14, fontweight='bold')
    
    # 子图3: 累积分布图
    ax3 = plt.subplot(2, 2, 3)
    cumulative_counts = np.cumsum(molecule_counts)
    ax3.plot(atom_counts, cumulative_counts, marker='o', linewidth=2, markersize=4, color='green')
    ax3.set_xlabel('Number of Atoms', fontsize=12)
    ax3.set_ylabel('Cumulative Number of Molecules', fontsize=12)
    ax3.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 添加网格线帮助查看小值
    ax3.set_yscale('log')  # 使用对数刻度更好地显示全部数据
    
    # 子图4: 统计信息文本
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # 添加统计信息文本
    stats_text = f"""
    QM9 Dataset Statistics
    
    Total Molecules: {total_molecules:,}
    
    Atom Count Range: {atom_range[0]} - {atom_range[1]}
    
    Average Atom Count: {avg_atoms}
    
    Top 5 Atom Counts:
    1. 19 atoms: 18,146 ({18146/total_molecules*100:.2f}%)
    2. 18 atoms: 17,442 ({17442/total_molecules*100:.2f}%)
    3. 17 atoms: 16,969 ({16969/total_molecules*100:.2f}%)
    4. 16 atoms: 13,824 ({13824/total_molecules*100:.2f}%)
    5. 21 atoms: 13,166 ({13166/total_molecules*100:.2f}%)
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('qm9_dataset_visualization.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("Visualization saved as qm9_dataset_visualization.png")

def main():
    create_qm9_visualization()

if __name__ == "__main__":
    main()