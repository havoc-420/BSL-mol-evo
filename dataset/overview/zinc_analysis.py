#!/usr/bin/env python3
"""
ZINC数据集分析脚本
用于分析ZINC数据集的重原子分布情况
"""

import os
import sys
import argparse
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def analyze_zinc_dataset(max_molecules=10000):
    """
    分析ZINC数据集的统计信息
    
    Args:
        max_molecules (int): 最大处理分子数
    """
    try:
        # 检查是否安装了必要的库
        from torch_geometric.datasets import ZINC
        import torch
        
        print(f"正在加载ZINC数据集...")
        
        # 加载ZINC数据集的子集
        # 注意：ZINC数据集很大，我们只加载训练集的一个小子集进行演示
        dataset = ZINC(root='/Users/havoc420/Documents/Projects/python/mol-evo/data/ZINC', 
                      subset=True, split='train')
        
        print(f"数据集加载完成，共 {len(dataset)} 个分子")
        
        # 统计重原子数分布
        heavy_atom_counts = []
        
        # 确定要处理的分子数量
        process_count = min(max_molecules, len(dataset))
        print(f"正在分析前 {process_count} 个分子...")
        
        for i in range(process_count):
            data = dataset[i]
            # 获取重原子数（ZINC数据集中的num_nodes就是重原子数）
            num_heavy_atoms = data.num_nodes
            heavy_atom_counts.append(num_heavy_atoms)
            
            # 显示进度
            if (i + 1) % 1000 == 0:
                print(f"已处理 {i + 1}/{process_count} 个分子")
        
        # 统计重原子数分布
        heavy_atom_count_distribution = Counter(heavy_atom_counts)
        
        print("\n=== ZINC数据集统计信息 ===")
        print(f"总分子数: {len(heavy_atom_counts)}")
        print(f"重原子数范围: {min(heavy_atom_counts)} - {max(heavy_atom_counts)}")
        print(f"平均重原子数: {sum(heavy_atom_counts) / len(heavy_atom_counts):.2f}")
        
        print("\n=== 重原子数分布 ===")
        # 按重原子数排序显示
        sorted_distribution = sorted(heavy_atom_count_distribution.items())
        for heavy_atom_num, count in sorted_distribution:
            percentage = (count / len(heavy_atom_counts)) * 100
            print(f"重原子数 {heavy_atom_num}: {count} 个分子 ({percentage:.2f}%)")
        
        # 显示最常见的重原子数
        most_common = heavy_atom_count_distribution.most_common(5)
        print("\n=== 最常见的重原子数Top 5 ===")
        for i, (heavy_atom_num, count) in enumerate(most_common):
            percentage = (count / len(heavy_atom_counts)) * 100
            print(f"{i+1}. 重原子数 {heavy_atom_num}: {count} 个分子 ({percentage:.2f}%)")
        
        # 绘制可视化图表
        create_visualization(heavy_atom_count_distribution, len(heavy_atom_counts))
        
        return heavy_atom_count_distribution
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装必要的库:")
        print("pip install torch torch-geometric matplotlib")
        print("\n如果尚未安装ZINC数据集，首次运行时会自动下载")
        return None
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_visualization(distribution, total_molecules):
    """
    创建ZINC数据集的可视化图表
    
    Args:
        distribution (Counter): 重原子数分布统计
        total_molecules (int): 总分子数
    """
    try:
        # 设置字体支持
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Bitstream Vera Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 按重原子数排序
        sorted_distribution = sorted(distribution.items())
        heavy_atom_nums, counts = zip(*sorted_distribution)
        
        # 计算百分比
        percentages = [count / total_molecules * 100 for count in counts]
        
        # 创建图表
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('ZINC Dataset Statistics (Heavy Atoms)', fontsize=20, fontweight='bold')
        
        # 子图1: 重原子数分布柱状图
        ax1 = plt.subplot(2, 2, 1)
        bars = ax1.bar(heavy_atom_nums, counts, color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_xlabel('Number of Heavy Atoms', fontsize=12)
        ax1.set_ylabel('Number of Molecules', fontsize=12)
        ax1.set_title('Heavy Atom Count Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 为数量较少的项目添加数值标签
        for i, (atom_count, molecule_count) in enumerate(zip(heavy_atom_nums, counts)):
            if molecule_count < max(counts) * 0.1:  # 为数量较少的项目添加标签
                ax1.text(atom_count, molecule_count + max(counts) * 0.01, str(molecule_count), 
                         ha='center', va='bottom', fontsize=8, rotation=90)
        
        # 突出显示最常见的重原子数
        most_common_indices = [i for i, _ in sorted(enumerate(counts), key=lambda x: x[1], reverse=True)[:5]]
        for i in most_common_indices:
            if i < len(bars):
                bars[i].set_color('orange')
                bars[i].set_alpha(0.9)
        
        # 子图2: 重原子数分布饼图（Top 5 + Others）
        ax2 = plt.subplot(2, 2, 2)
        top5_heavy_atom_counts = [heavy_atom_nums[i] for i in most_common_indices[:5]]
        top5_molecule_counts = [counts[i] for i in most_common_indices[:5]]
        top5_labels = [f'{count} heavy atoms' for count in top5_heavy_atom_counts]
        
        others_count = total_molecules - sum(top5_molecule_counts)
        sizes = top5_molecule_counts + [others_count]
        labels = top5_labels + ['Others']
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Top 5 Heavy Atom Counts Distribution', fontsize=14, fontweight='bold')
        
        # 子图3: 累积分布图
        ax3 = plt.subplot(2, 2, 3)
        cumulative_counts = np.cumsum(counts)
        ax3.plot(heavy_atom_nums, cumulative_counts, marker='o', linewidth=2, markersize=4, color='green')
        ax3.set_xlabel('Number of Heavy Atoms', fontsize=12)
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
        ZINC Dataset Statistics (Heavy Atoms)
        
        Total Molecules: {total_molecules:,}
        
        Heavy Atom Count Range: {min(heavy_atom_nums)} - {max(heavy_atom_nums)}
        
        Average Heavy Atom Count: {sum(heavy_atom_nums[i] * counts[i] for i in range(len(heavy_atom_nums))) / total_molecules:.2f}
        
        Top 5 Heavy Atom Counts:
        """
        
        for i in range(min(5, len(top5_heavy_atom_counts))):
            stats_text += f"{i+1}. {top5_heavy_atom_counts[i]} heavy atoms: {top5_molecule_counts[i]:,} ({top5_molecule_counts[i]/total_molecules*100:.2f}%)\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        filename = 'zinc_dataset_visualization.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n可视化图表已保存为 {filename}")
        
    except Exception as e:
        print(f"生成可视化图表时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='ZINC数据集分析')
    parser.add_argument('--max-molecules', type=int, default=10000, 
                        help='最大处理分子数 (默认: 10000)')
    
    args = parser.parse_args()
    
    # 分析数据集
    distribution = analyze_zinc_dataset(args.max_molecules)
    
    if distribution is not None:
        print("\n分析完成!")
    else:
        print("\n分析失败，请检查错误信息。")

if __name__ == "__main__":
    main()