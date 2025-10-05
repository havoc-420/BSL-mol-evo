#!/usr/bin/env python3
"""
QM9数据集综合分析脚本
通过参数切换不同功能模式
"""

import os
import sys
import argparse
from collections import Counter

def qm9_overview(root_dir):
    """
    展示QM9数据集的概览信息
    
    Args:
        root_dir (str): QM9数据集根目录路径
    """
    try:
        # 检查是否安装了必要的库
        from torch_geometric.datasets import QM9
        import torch
        
        print("=== QM9数据集概览 ===")
        
        # 加载QM9数据集
        print(f"正在加载QM9数据集...")
        dataset = QM9(root=root_dir)
        
        print(f"\n数据集基本信息:")
        print(f"  - 总分子数: {len(dataset)}")
        print(f"  - 节点特征数: {dataset.num_features}")
        print(f"  - 边特征数: {dataset.num_edge_features}")
        
        # 查看第一个分子样本
        print(f"\n第一个分子样本信息:")
        data = dataset[0]
        print(f"  - 节点数: {data.num_nodes}")
        print(f"  - 边数: {data.num_edges}")
        
        # 检查y属性
        if hasattr(data, 'y') and data.y is not None:
            print(f"  - 目标属性形状: {data.y.shape}")
            print(f"  - 目标属性数: {data.y.shape[1] if len(data.y.shape) > 1 else 1}")
        
        # 显示目标属性名称和含义
        print(f"\nQM9数据集的19个目标属性:")
        target_names = [
            'mu (偶极矩, D)',
            'alpha (各向同性极化率, a0^3)',
            'homo (最高占据分子轨道能量, eV)',
            'lumo (最低未占据分子轨道能量, eV)',
            'gap (HOMO-LUMO能隙, eV)',
            'r2 (电子空间范围, a0^2)',
            'zpve (零点振动能, eV)',
            'U0 (0K时的内能, eV)',
            'U (298.15K时的内能, eV)',
            'H (298.15K时的焓, eV)',
            'G (298.15K时的自由能, eV)',
            'Cv (298.15K时的热容, cal/(mol*K))',
            'U0_atom (0K时的原子化能, eV)',
            'U_atom (298.15K时的原子化能, eV)',
            'H_atom (298.15K时的原子化焓, eV)',
            'G_atom (298.15K时的原子化自由能, eV)',
            'A (旋转常数, GHz)',
            'B (旋转常数, GHz)',
            'C (旋转常数, GHz)'
        ]
        
        for i, name in enumerate(target_names):
            print(f"  [{i:2d}] {name}")
        
        # 显示第一个分子的目标属性值
        print(f"\n第一个分子的目标属性值:")
        if hasattr(data, 'y') and data.y is not None and data.y.shape[1] >= 19:
            for i, (name, value) in enumerate(zip(target_names, data.y[0])):
                if isinstance(value, torch.Tensor):
                    value = value.item()
                print(f"  [{i:2d}] {name.split()[0]}: {value:.6f}")
        
        # 显示SMILES字符串（在y属性的第3列，索引为2）
        if hasattr(data, 'y') and data.y is not None and data.y.shape[1] >= 3:
            smiles = data.y[0, 2]
            if isinstance(smiles, torch.Tensor):
                smiles = str(smiles.item())
            print(f"\n第一个分子的SMILES: {smiles}")
        
        # 统计原子数分布（仅显示前1000个分子以节省时间）
        print(f"\n原子数分布 (前1000个分子):")
        atom_counts = {}
        for i, data in enumerate(dataset):
            if i >= 1000:  # 只统计前1000个分子
                break
            num_atoms = data.num_nodes
            atom_counts[num_atoms] = atom_counts.get(num_atoms, 0) + 1
        
        # 按原子数排序显示
        sorted_counts = sorted(atom_counts.items())
        for atom_num, count in sorted_counts:
            print(f"  原子数 {atom_num}: {count} 个分子")
            
        print(f"\n=== 概览完成 ===")
        
        return dataset
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装必要的库:")
        print("pip install torch torch-geometric")
        sys.exit(1)
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def analyze_qm9_dataset(root_dir):
    """
    分析QM9数据集的统计信息
    
    Args:
        root_dir (str): QM9数据集根目录路径
    """
    try:
        # 检查是否安装了必要的库
        from torch_geometric.datasets import QM9
        import torch
        
        # 尝试导入matplotlib
        try:
            import matplotlib.pyplot as plt
            MATPLOTLIB_AVAILABLE = True
        except ImportError:
            MATPLOTLIB_AVAILABLE = False
            print("注意: 未安装matplotlib，将跳过图表生成")
        
        print(f"正在加载QM9数据集...")
        
        # 加载QM9数据集
        dataset = QM9(root=root_dir)
        
        print(f"数据集加载完成，共 {len(dataset)} 个分子")
        
        # 统计原子数分布
        atom_counts = []
        
        print("正在分析数据集...")
        for i, data in enumerate(dataset):
            # 获取原子数
            num_atoms = data.num_nodes
            atom_counts.append(num_atoms)
            
            # 显示进度
            if (i + 1) % 10000 == 0:
                print(f"已处理 {i + 1}/{len(dataset)} 个分子")
        
        # 统计原子数分布
        atom_count_distribution = Counter(atom_counts)
        
        print("\n=== QM9数据集统计信息 ===")
        print(f"总分子数: {len(dataset)}")
        print(f"原子数范围: {min(atom_counts)} - {max(atom_counts)}")
        print(f"平均原子数: {sum(atom_counts) / len(atom_counts):.2f}")
        
        print("\n=== 原子数分布 ===")
        # 按原子数排序显示
        sorted_distribution = sorted(atom_count_distribution.items())
        for atom_num, count in sorted_distribution:
            percentage = (count / len(dataset)) * 100
            print(f"原子数 {atom_num}: {count} 个分子 ({percentage:.2f}%)")
        
        # 显示最常见的原子数
        most_common = atom_count_distribution.most_common(5)
        print("\n=== 最常见的原子数Top 5 ===")
        for i, (atom_num, count) in enumerate(most_common):
            percentage = (count / len(dataset)) * 100
            print(f"{i+1}. 原子数 {atom_num}: {count} 个分子 ({percentage:.2f}%)")
        
        # 显示一些示例分子的SMILES
        print("\n=== 示例分子的SMILES ===")
        sample_count = 0
        for i, data in enumerate(dataset):
            if sample_count >= 10:  # 只显示前10个示例
                break
                
            if hasattr(data, 'y') and data.y is not None and data.y.shape[1] >= 3:
                # 提取SMILES (在y属性的第3列，索引为2)
                smiles = data.y[0, 2] 
                # 如果smiles是tensor类型，需要转换为字符串
                if isinstance(smiles, torch.Tensor):
                    smiles = str(smiles.item())
                
                print(f"分子 {i} (原子数: {data.num_nodes}): {smiles}")
                sample_count += 1
        
        # 尝试生成图表
        if MATPLOTLIB_AVAILABLE:
            try:
                # 绘制原子数分布直方图
                plt.figure(figsize=(12, 6))
                
                # 子图1: 完整分布
                plt.subplot(1, 2, 1)
                atom_nums, counts = zip(*sorted_distribution)
                plt.bar(atom_nums, counts, width=1.0)
                plt.xlabel('Number of Atoms')
                plt.ylabel('Number of Molecules')
                plt.title('QM9 Dataset Atom Count Distribution')
                plt.grid(True, alpha=0.3)
                
                # 子图2: 累积分布
                plt.subplot(1, 2, 2)
                cumulative_counts = []
                cumulative_sum = 0
                for atom_num, count in sorted_distribution:
                    cumulative_sum += count
                    cumulative_counts.append(cumulative_sum)
                
                plt.plot(atom_nums, cumulative_counts, marker='o')
                plt.xlabel('Number of Atoms')
                plt.ylabel('Cumulative Number of Molecules')
                plt.title('QM9 Dataset Cumulative Atom Count Distribution')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('qm9_atom_distribution.png', dpi=300, bbox_inches='tight')
                print("\nDistribution chart saved as qm9_atom_distribution.png")
                
            except Exception as e:
                print(f"\nError generating chart: {e}")
                print("图表生成失败")
        else:
            print("\n跳过图表生成（未安装matplotlib）")
        
        return atom_count_distribution
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装必要的库:")
        print("pip install torch torch-geometric matplotlib")
        sys.exit(1)
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='QM9数据集综合分析')
    parser.add_argument('--dir', type=str, default='/Users/havoc420/Documents/Projects/python/mol-evo/data/QM9', 
                        help='QM9数据集目录 (默认: /Users/havoc420/Documents/Projects/python/mol-evo/data/QM9)')
    parser.add_argument('--mode', type=str, choices=['overview', 'statistics', 'all'], default='all',
                        help='运行模式: overview(概览), statistics(统计), all(全部) (默认: all)')
    
    args = parser.parse_args()
    
    # 检查数据集目录是否存在
    if not os.path.exists(args.dir):
        print(f"错误: 目录 {args.dir} 不存在")
        print("请先使用 download_qm9.py 脚本下载数据集:")
        print(f"python download_qm9.py --dir {args.dir}")
        sys.exit(1)
    
    # 根据模式执行相应功能
    if args.mode == 'overview':
        qm9_overview(args.dir)
    elif args.mode == 'statistics':
        analyze_qm9_dataset(args.dir)
    elif args.mode == 'all':
        dataset = qm9_overview(args.dir)
        print("\n" + "="*50 + "\n")
        analyze_qm9_dataset(args.dir)

if __name__ == "__main__":
    main()