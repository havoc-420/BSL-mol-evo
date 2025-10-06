#!/usr/bin/env python3
"""
从QM9数据集中提取分子特征并保存到CSV文件
"""

import os
import sys
import csv
import argparse
from pathlib import Path

def parse_atom_list(atom_str):
    """
    解析原子数参数字符串，支持多种格式:
    - 单个数字: "5"
    - 逗号分隔: "3,4,5"
    - 范围表示: "3-5"
    - 组合形式: "3,4-6,8"
    
    Args:
        atom_str (str): 原子数参数字符串
        
    Returns:
        list: 原子数列表
    """
    if not atom_str:
        return None
        
    atom_list = []
    parts = atom_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # 处理范围格式，如 "3-5"
            try:
                start, end = map(int, part.split('-'))
                atom_list.extend(range(start, end + 1))
            except ValueError:
                raise ValueError(f"无效的范围格式: {part}")
        else:
            # 处理单个数字
            try:
                atom_list.append(int(part))
            except ValueError:
                raise ValueError(f"无效的原子数格式: {part}")
    
    # 去重并排序
    return sorted(list(set(atom_list)))

def extract_smiles_with_atom_count(root_dir, atom_count, output_file, validate_smiles=False, heavy_atoms_only=False):
    """
    从QM9数据集中提取具有指定原子数的分子特征
    
    Args:
        root_dir (str): QM9数据集根目录路径
        atom_count (int or list): 指定的原子数（如果为None，则提取所有分子）
                                  如果是列表，则提取多个原子数的分子
        output_file (str): 输出CSV文件路径
        validate_smiles (bool): 是否使用RDKit验证并重新生成SMILES
        heavy_atoms_only (bool): 是否只计算非氢原子（重原子）数量
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
            
            # 禁用RDKit的警告信息
            from rdkit import RDLogger
            lg = RDLogger.logger()
            lg.setLevel(RDLogger.CRITICAL)
        except ImportError:
            RDKIT_AVAILABLE = False
            print("警告: 未安装RDKit，将使用数据集中的SMILES字符串")
        
        print(f"正在加载QM9数据集...")
        
        # 加载QM9数据集
        dataset = QM9(root=root_dir)
        
        print(f"数据集加载完成，共 {len(dataset)} 个分子")
        
        if atom_count is not None:
            if isinstance(atom_count, list):
                if heavy_atoms_only:
                    print(f"开始提取具有 {', '.join(map(str, atom_count))} 个重原子的分子...")
                else:
                    print(f"开始提取具有 {', '.join(map(str, atom_count))} 个原子的分子...")
            else:
                if heavy_atoms_only:
                    print(f"开始提取具有 {atom_count} 个重原子的分子...")
                else:
                    print(f"开始提取具有 {atom_count} 个原子的分子...")
        else:
            if heavy_atoms_only:
                print(f"开始提取所有分子（按重原子计数）...")
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
        
        def has_invalid_valence(mol):
            """
            检查分子中是否存在不合理的价态
            
            Args:
                mol: RDKit分子对象
                
            Returns:
                bool: 如果存在不合理的价态返回True，否则返回False
            """
            try:
                # 检查常见元素的价态
                for atom in mol.GetAtoms():
                    symbol = atom.GetSymbol()
                    valence = atom.GetTotalValence()
                    
                    # 检查价态是否合理
                    if symbol == 'C' and valence > 4:
                        return True
                    elif symbol == 'N' and valence > 4:
                        return True
                    elif symbol == 'O' and valence > 2:
                        return True
                    elif symbol == 'F' and valence > 1:
                        return True
                return False
            except:
                return True  # 如果检查过程中出错，认为是无效的
        
        # 遍历数据集，查找具有指定原子数的分子
        for i, data in enumerate(dataset):
            try:
                num_atoms = data.num_nodes  # 默认使用数据集中的原子数（包括氢原子）
                valid_molecule = True  # 假设分子是有效的
                
                # 如果RDKit可用且需要验证SMILES，尝试验证
                if RDKIT_AVAILABLE and (validate_smiles or heavy_atoms_only):
                    # 临时启用RDKit警告以检查严重错误
                    lg.setLevel(RDLogger.WARNING)
                    mol = Chem.MolFromSmiles(data.smiles)
                    # 立即禁用警告
                    lg.setLevel(RDLogger.CRITICAL)
                    
                    if mol is not None:
                        # 根据heavy_atoms_only参数决定如何计算原子数
                        if heavy_atoms_only:
                            # 只计算重原子（非氢原子）
                            num_atoms = mol.GetNumHeavyAtoms()
                        elif validate_smiles:
                            # 计算所有原子（包括氢原子）
                            num_atoms = mol.GetNumAtoms()
                        
                        # 检查价态是否合理
                        if has_invalid_valence(mol):
                            # 如果价态不合理，跳过该分子
                            valid_molecule = False
                    else:
                        # 如果RDKit无法解析SMILES，跳过该分子
                        valid_molecule = False
                
                # 如果分子无效，则跳过
                if not valid_molecule:
                    continue
                
                # 如果指定了原子数，则只提取匹配的分子；否则提取所有分子
                if atom_count is None or \
                   (isinstance(atom_count, int) and num_atoms == atom_count) or \
                   (isinstance(atom_count, list) and num_atoms in atom_count):
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
                        if isinstance(atom_count, list):
                            print(f"已找到 {len(molecules)} 个符合要求的分子")
                        else:
                            if heavy_atoms_only:
                                print(f"已找到 {len(molecules)} 个具有 {atom_count} 个重原子的分子")
                            else:
                                print(f"已找到 {len(molecules)} 个具有 {atom_count} 个原子的分子")
                    elif atom_count is None and ((i + 1) % 10000 == 0):
                        print(f"已处理 {i + 1}/{len(dataset)} 个分子")
            except Exception as e:
                print(f"处理分子 {i} 时出错: {e}")
                continue
        
        if atom_count is not None:
            if isinstance(atom_count, list):
                print(f"共找到 {len(molecules)} 个符合要求的分子")
            else:
                if heavy_atoms_only:
                    print(f"共找到 {len(molecules)} 个具有 {atom_count} 个重原子的分子")
                else:
                    print(f"共找到 {len(molecules)} 个具有 {atom_count} 个原子的分子")
        else:
            print(f"共处理 {len(molecules)} 个分子")
        
        # 如果指定了多个原子数，分别保存到不同的文件中
        if isinstance(atom_count, list) and len(atom_count) > 1:
            # 按原子数分组分子
            molecules_by_atom_count = {}
            for mol in molecules:
                count = mol['num_atoms']
                if count not in molecules_by_atom_count:
                    molecules_by_atom_count[count] = []
                molecules_by_atom_count[count].append(mol)
            
            # 为每个原子数分别保存文件
            for count, mols in molecules_by_atom_count.items():
                # 生成文件名
                base_name = os.path.splitext(output_file)[0]
                if '_atoms' in base_name:
                    dir_name = os.path.dirname(output_file)
                    if heavy_atoms_only:
                        base_name = os.path.join(dir_name, f"qm9_smiles_heavy_{count}_atoms")
                    else:
                        base_name = os.path.join(dir_name, f"qm9_smiles_{count}_atoms")
                else:
                    if heavy_atoms_only:
                        base_name = f"{base_name}_heavy_{count}"
                    else:
                        base_name = f"{base_name}_{count}"
                separate_output_file = f"{base_name}.csv"
                
                # 保存到CSV文件
                with open(separate_output_file, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['index', 'smiles', 'num_atoms'] + target_names
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    for mol in mols:
                        writer.writerow(mol)
                
                if heavy_atoms_only:
                    print(f"重原子数为 {count} 的分子已保存到 {separate_output_file}")
                else:
                    print(f"原子数为 {count} 的分子已保存到 {separate_output_file}")
                print(f"有效分子数量: {len(mols)}")
        else:
            # 保存到CSV文件
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['index', 'smiles', 'num_atoms'] + target_names
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for mol in molecules:
                    writer.writerow(mol)
            
            print(f"结果已保存到 {output_file}")
            print(f"有效分子数量: {len(molecules)}")
        
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
        
        # 获取图的边特征（如果存在）
        edge_features = None
        if 'edge_attr' in nx_graph.graph and nx_graph.graph['edge_attr'] is not None:
            # 构建边到特征的映射
            edge_features = {}
            if 'edge_index' in nx_graph.graph and nx_graph.graph['edge_index'] is not None:
                edge_index = nx_graph.graph['edge_index']
                edge_attrs = nx_graph.graph['edge_attr']
                for i, (u, v) in enumerate(edge_index.T):
                    edge_features[(int(u), int(v))] = edge_attrs[i]
                    edge_features[(int(v), int(u))] = edge_attrs[i]  # 反向边
        
        # 添加键
        for (u, v) in nx_graph.edges():
            u_idx, v_idx = int(u), int(v)
            # 确保节点索引有效
            if u_idx < mol.GetNumAtoms() and v_idx < mol.GetNumAtoms():
                bond_type = Chem.BondType.SINGLE  # 默认单键
                
                # 如果有边特征，根据特征确定键类型
                if edge_features and (u_idx, v_idx) in edge_features:
                    # QM9边特征: [0]单键, [1]双键, [2]三键
                    edge_attr = edge_features[(u_idx, v_idx)]
                    if hasattr(edge_attr, 'argmax'):
                        bond_idx = edge_attr.argmax().item()
                    else:
                        bond_idx = int(edge_attr)
                    
                    if bond_idx == 0:
                        bond_type = Chem.BondType.SINGLE
                    elif bond_idx == 1:
                        bond_type = Chem.BondType.DOUBLE
                    elif bond_idx == 2:
                        bond_type = Chem.BondType.TRIPLE
                
                mol.AddBond(u_idx, v_idx, bond_type)
        
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
    parser.add_argument('--atom', type=str, default=None, 
                        help='指定原子数，支持格式: 单个数字(5)、逗号分隔(3,4,5)、范围(3-5)、组合(3,4-6,8) (默认: None，提取所有分子)')
    parser.add_argument('--output', type=str, default=None, 
                        help='输出CSV文件路径 (默认: data/qm9_smiles_{atom_count}_atoms.csv 或 data/qm9_smiles_all.csv)')
    parser.add_argument('--validate', action='store_true', 
                        help='是否使用RDKit验证并重新生成SMILES (更准确但较慢)')
    parser.add_argument('--heavy-only', action='store_true',
                        help='是否只计算重原子（非氢原子）数量 (默认: False，计算所有原子)')
    
    args = parser.parse_args()
    
    # 解析原子数参数
    atom_count = None
    if args.atom is not None:
        try:
            atom_count = parse_atom_list(args.atom)
            # 如果只有一个原子数，就用int类型以保持向后兼容
            if isinstance(atom_count, list) and len(atom_count) == 1:
                atom_count = atom_count[0]
        except ValueError as e:
            print(f"错误: {e}")
            sys.exit(1)
    
    # 如果未指定输出文件名，则根据是否指定原子数生成文件名
    if args.output is None:
        output_dir = 'data'
        os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
        if atom_count is not None:
            if isinstance(atom_count, list):
                # 对于多个原子数，使用范围或列表形式命名
                if len(atom_count) > 1:
                    if args.heavy_only:
                        args.output = f'{output_dir}/qm9_smiles_heavy_atoms.csv'  # 基础文件名，实际会分别保存
                    else:
                        args.output = f'{output_dir}/qm9_smiles_atoms.csv'  # 基础文件名，实际会分别保存
                else:
                    if args.heavy_only:
                        args.output = f'{output_dir}/qm9_smiles_heavy_{atom_count[0]}_atoms.csv'
                    else:
                        args.output = f'{output_dir}/qm9_smiles_{atom_count[0]}_atoms.csv'
            else:
                if args.heavy_only:
                    args.output = f'{output_dir}/qm9_smiles_heavy_{atom_count}_atoms.csv'
                else:
                    args.output = f'{output_dir}/qm9_smiles_{atom_count}_atoms.csv'
        else:
            if args.heavy_only:
                args.output = f'{output_dir}/qm9_smiles_heavy_all.csv'
            else:
                args.output = f'{output_dir}/qm9_smiles_all.csv'
    else:
        # 如果指定了输出路径，确保其目录存在
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查数据集目录是否存在
    if not os.path.exists(args.dir):
        print(f"错误: 目录 {args.dir} 不存在")
        print("请先使用 download_qm9.py 脚本下载数据集:")
        print(f"python -m mol_evo.dataset.download_qm9 --dir {args.dir}")
        sys.exit(1)
    
    # 提取SMILES并保存到CSV
    print(f"开始提取过程...")
    print(f"数据集目录: {args.dir}")
    if atom_count is not None:
        if isinstance(atom_count, list):
            if args.heavy_only:
                print(f"目标重原子数: {', '.join(map(str, atom_count))}")
            else:
                print(f"目标原子数: {', '.join(map(str, atom_count))}")
        else:
            if args.heavy_only:
                print(f"目标重原子数: {atom_count}")
            else:
                print(f"目标原子数: {atom_count}")
    else:
        if args.heavy_only:
            print(f"目标重原子数: 所有")
        else:
            print(f"目标原子数: 所有")
    if args.validate:
        print(f"验证模式: 开启 (将使用RDKit重新生成SMILES)")
    if args.heavy_only:
        print(f"计数模式: 仅重原子（非氢原子）")
    else:
        print(f"计数模式: 所有原子（包括氢原子）")
    
    extract_smiles_with_atom_count(args.dir, atom_count, args.output, validate_smiles=args.validate, heavy_atoms_only=args.heavy_only)

if __name__ == "__main__":
    main()