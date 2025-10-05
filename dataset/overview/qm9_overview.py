#!/usr/bin/env python3
"""
QM9数据集概览脚本
用于简单查看QM9数据集的特征和结构
"""

import os
import sys
import argparse

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

def main():
    parser = argparse.ArgumentParser(description='QM9数据集概览')
    parser.add_argument('--dir', type=str, default='/Users/havoc420/Documents/Projects/python/mol-evo/data/QM9', 
                        help='QM9数据集目录 (默认: /Users/havoc420/Documents/Projects/python/mol-evo/data/QM9)')
    
    args = parser.parse_args()
    
    # 检查数据集目录是否存在
    if not os.path.exists(args.dir):
        print(f"错误: 目录 {args.dir} 不存在")
        print("请先使用 download_qm9.py 脚本下载数据集:")
        print(f"python download_qm9.py --dir {args.dir}")
        sys.exit(1)
    
    # 展示QM9数据集概览
    qm9_overview(args.dir)

if __name__ == "__main__":
    main()