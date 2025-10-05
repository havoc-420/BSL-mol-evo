#!/usr/bin/env python3
"""
从QM9数据集中提取分子特征并保存到CSV文件
"""

import os
import sys
import csv
import argparse
from pathlib import Path

def extract_smiles_with_atom_count(root_dir, atom_count, output_file):
    """
    从QM9数据集中提取具有指定原子数的分子特征
    
    Args:
        root_dir (str): QM9数据集根目录路径
        atom_count (int): 指定的原子数（如果为None，则提取所有分子）
        output_file (str): 输出CSV文件路径
    """
    try:
        # 检查是否安装了必要的库
        from torch_geometric.datasets import QM9
        from torch_geometric.utils import to_networkx
        import torch
        import networkx as nx
        
        # 尝试导入RDKit
        try:
            from rdkit import Chem
            from rdkit.Chem import rdmolops
            RDKIT_AVAILABLE = True
        except ImportError:
            RDKIT_AVAILABLE = False
            print("警告: 未安装RDKit，将使用数据集中的SMILES字符串")
        
        print(f"正在加载QM9数据集...")
        
        # 加载QM9数据集
        dataset = QM9(root=root_dir)
        
        print(f"数据集加载完成，共 {len(dataset)} 个分子")
        
        if atom_count is not None:
            print(f"开始提取具有 {atom_count} 个原子的分子...")
        else:
            print(f"开始提取所有分子...")
        
        # 存储符合条件的分子
        molecules = []
        
        # QM9数据集包含19个目标属性
        target_names = [
            'mu',           # 偶极矩 (Dipole moment)
            'alpha',        # 各向同性极化率 (Isotropic polarizability)
            'homo',         # 最高占据分子轨道能量 (Highest occupied molecular orbital energy)
            'lumo',         # 最低未占据分子轨道能量 (Lowest unoccupied molecular orbital energy)
            'gap',          # HOMO-LUMO能隙 (Gap between HOMO and LUMO)
            'r2',           # 电子空间范围 (Electronic spatial extent)
            'zpve',         # 零点振动能 (Zero point vibrational energy)
            'U0',           # 0K时的内能 (Internal energy at 0K)
            'U',            # 298.15K时的内能 (Internal energy at 298.15K)
            'H',            # 298.15K时的焓 (Enthalpy at 298.15K)
            'G',            # 298.15K时的自由能 (Free energy at 298.15K)
            'Cv',           # 298.15K时的热容 (Heat capacity at 298.15K)
            'U0_atom',      # 0K时的原子化能 (Atomization energy at 0K)
            'U_atom',       # 298.15K时的原子化能 (Atomization energy at 298.15K)
            'H_atom',       # 298.15K时的原子化焓 (Atomization enthalpy at 298.15K)
            'G_atom',       # 298.15K时的原子化自由能 (Atomization free energy at 298.15K)
            'A',            # 旋转常数 A
            'B',            # 旋转常数 B
            'C'             # 旋转常数 C
        ]
        
        # 遍历数据集，查找具有指定原子数的分子
        for i, data in enumerate(dataset):
            # 获取原子数
            num_atoms = data.num_nodes
            
            # 如果指定了原子数，则只提取匹配的分子；否则提取所有分子
            if atom_count is None or num_atoms == atom_count:
                try:
                    # 提取所有19个目标属性
                    targets = {}
                    for j, name in enumerate(target_names):
                        value = 0.0
                        if hasattr(data, 'y') and data.y is not None and data.y.shape[1] > j:
                            value = data.y[0, j]
                            if isinstance(value, torch.Tensor):
                                value = value.item()
                        targets[name] = value
                    
                    molecule_data = {
                        'index': i,
                        'smiles': data.smiles,
                        'num_atoms': num_atoms,
                        **targets
                    }
                    
                    molecules.append(molecule_data)
                    
                    # 显示进度
                    if atom_count is not None and (len(molecules) % 100 == 0):
                        print(f"已找到 {len(molecules)} 个具有 {atom_count} 个原子的分子")
                    elif atom_count is None and ((i + 1) % 10000 == 0):
                        print(f"已处理 {i + 1}/{len(dataset)} 个分子")
                except Exception as e:
                    print(f"处理分子 {i} 时出错: {e}")
                    continue
        
        if atom_count is not None:
            print(f"共找到 {len(molecules)} 个具有 {atom_count} 个原子的分子")
        else:
            print(f"共处理 {len(molecules)} 个分子")
        
        # 保存到CSV文件
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['index', 'smiles', 'num_atoms'] + target_names
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for mol in molecules:
                writer.writerow(mol)
        
        print(f"结果已保存到 {output_file}")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装必要的库:")
        print("pip install torch torch-geometric")
        if not RDKIT_AVAILABLE:
            print("pip install rdkit-pypi")
        sys.exit(1)
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def nx_graph_to_mol(nx_graph, node_features):
    """
    将NetworkX图转换为RDKit分子对象
    
    Args:
        nx_graph: NetworkX图对象
        node_features: 节点特征张量
        
    Returns:
        RDKit分子对象
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdchem
        
        # 创建一个新的RDKit分子
        mol = Chem.RWMol()
        
        # 原子类型映射 (QM9数据集中的原子类型)
        # QM9数据集中节点特征的前5个维度对应原子类型:
        # [0] = H (1), [1] = C (6), [2] = N (7), [3] = O (8), [4] = F (9)
        atom_types = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9}
        
        # 添加原子
        for i in sorted(nx_graph.nodes()):  # 确保节点顺序一致
            # 获取节点特征中的原子类型
            if i < node_features.shape[0]:  # 确保索引有效
                atom_type_idx = node_features[i].argmax().item()
                atomic_num = atom_types.get(atom_type_idx, 0)
                if atomic_num > 0:
                    atom = Chem.Atom(atomic_num)
                    mol.AddAtom(atom)
                else:
                    return None  # 未知原子类型
            else:
                return None  # 节点索引超出特征范围
        
        # 添加键
        for (u, v) in nx_graph.edges():
            # 确保节点索引有效
            if int(u) < mol.GetNumAtoms() and int(v) < mol.GetNumAtoms():
                mol.AddBond(int(u), int(v), Chem.BondType.SINGLE)
        
        # 转换为分子对象
        mol = mol.GetMol()
        
        # 更新分子结构
        if mol.GetNumAtoms() > 0:
            mol.UpdatePropertyCache()
            Chem.SanitizeMol(mol)
            return mol
        else:
            return None
            
    except Exception as e:
        # print(f"图转换为分子时出错: {e}")  # 调试时可以启用
        return None


def main():
    parser = argparse.ArgumentParser(description='从QM9数据集中提取分子特征')
    parser.add_argument('--dir', type=str, default='/Users/havoc420/Documents/Projects/python/mol-evo/data/QM9', 
                        help='QM9数据集目录 (默认: /Users/havoc420/Documents/Projects/python/mol-evo/data/QM9)')
    parser.add_argument('--atom', type=int, default=None, 
                        help='指定原子数 (默认: None，提取所有分子)')
    parser.add_argument('--output', type=str, default=None, 
                        help='输出CSV文件名 (默认: qm9_smiles_{atom_count}_atoms.csv 或 qm9_smiles_all.csv)')
    
    args = parser.parse_args()
    
    # 如果未指定输出文件名，则根据是否指定原子数生成文件名
    if args.output is None:
        if args.atom is not None:
            args.output = f'qm9_smiles_{args.atom}_atoms.csv'
        else:
            args.output = 'qm9_smiles_all.csv'
    
    # 检查数据集目录是否存在
    if not os.path.exists(args.dir):
        print(f"错误: 目录 {args.dir} 不存在")
        print("请先使用 download_qm9.py 脚本下载数据集:")
        print(f"python download_qm9.py --dir {args.dir}")
        sys.exit(1)
    
    # 提取SMILES并保存到CSV
    extract_smiles_with_atom_count(args.dir, args.atom, args.output)

if __name__ == "__main__":
    main()