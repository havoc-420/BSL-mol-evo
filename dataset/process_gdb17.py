#!/usr/bin/env python3
"""
处理 GDB-17 数据库的脚本
支持对大型分子数据集的分析和筛选
"""

import os
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors
import argparse
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_gdb17_file(file_path, max_molecules=None, output_file=None, filter_heavy_atoms=None):
    """
    处理 GDB-17 SMI 文件
    
    Args:
        file_path (str): SMI 文件路径
        max_molecules (int): 最大处理分子数
        output_file (str): 输出文件路径
        filter_heavy_atoms (int): 筛选特定重原子数的分子
    """
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return
    
    valid_molecules = []
    invalid_count = 0
    heavy_atom_count = {}
    
    logger.info(f"开始处理文件: {file_path}")
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_molecules and i >= max_molecules:
                break
                
            if i > 0 and i % 1000000 == 0:
                logger.info(f"已处理 {i} 个分子")
            
            line = line.strip()
            if not line:
                continue
            
            # 解析 SMILES
            smiles = line.split()[0]  # 只取第一列作为 SMILES
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is not None:
                # 获取重原子数
                heavy_atoms = mol.GetNumHeavyAtoms()
                
                # 如果指定了重原子数筛选条件
                if filter_heavy_atoms is not None and heavy_atoms != filter_heavy_atoms:
                    continue
                
                # 记录重原子数分布
                if heavy_atoms not in heavy_atom_count:
                    heavy_atom_count[heavy_atoms] = 0
                heavy_atom_count[heavy_atoms] += 1
                
                # 添加到有效分子列表
                valid_molecules.append(smiles)
            else:
                invalid_count += 1
    
    logger.info(f"处理完成:")
    logger.info(f"  有效分子数: {len(valid_molecules)}")
    logger.info(f"  无效分子数: {invalid_count}")
    
    # 显示重原子数分布
    logger.info("重原子数分布:")
    for heavy_count in sorted(heavy_atom_count.keys()):
        logger.info(f"  {heavy_count} 个重原子: {heavy_atom_count[heavy_count]} 个分子")
    
    # 如果指定了输出文件，则保存有效分子
    if output_file:
        with open(output_file, 'w') as f:
            for smiles in valid_molecules:
                f.write(f"{smiles}\n")
        logger.info(f"已将有效分子保存到: {output_file}")
    
    return valid_molecules


def calculate_properties(smiles_list, max_calculate=1000):
    """
    计算分子属性示例
    
    Args:
        smiles_list (list): SMILES 列表
        max_calculate (int): 最大计算数量
    """
    logger.info(f"计算前 {min(max_calculate, len(smiles_list))} 个分子的属性")
    
    mw_list = []
    logp_list = []
    
    for i, smiles in enumerate(smiles_list[:max_calculate]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            mw_list.append(mw)
            logp_list.append(logp)
    
    if mw_list:
        logger.info(f"分子量范围: {min(mw_list):.2f} - {max(mw_list):.2f}")
        logger.info(f"平均分子量: {sum(mw_list)/len(mw_list):.2f}")
        logger.info(f"LogP范围: {min(logp_list):.2f} - {max(logp_list):.2f}")
        logger.info(f"平均LogP: {sum(logp_list)/len(logp_list):.2f}")


def main():
    parser = argparse.ArgumentParser(description="处理 GDB-17 数据库")
    parser.add_argument("input_file", help="输入的 SMI 文件路径")
    parser.add_argument("-n", "--num_molecules", type=int, 
                        help="要处理的最大分子数")
    parser.add_argument("-o", "--output", help="保存有效 SMILES 的输出文件")
    parser.add_argument("-a", "--atoms", type=int, 
                        help="筛选具有指定重原子数的分子")
    parser.add_argument("-p", "--properties", action="store_true",
                        help="计算并显示分子属性")
    
    args = parser.parse_args()
    
    # 处理文件
    valid_molecules = process_gdb17_file(
        args.input_file, 
        max_molecules=args.num_molecules,
        output_file=args.output,
        filter_heavy_atoms=args.atoms
    )
    
    # 如果需要计算属性
    if args.properties and valid_molecules:
        calculate_properties(valid_molecules)


if __name__ == "__main__":
    main()