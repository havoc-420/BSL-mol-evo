#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to calculate MCS (Maximum Common Substructure) between two molecules
and visualize the result using RDKit and matplotlib.
"""

import json
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Serif CJK JP', 'Noto Mono', 'DejaVu Sans']  # 用于正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

def load_molecules_from_json(json_path):
    """
    从JSON文件加载分子SMILES字符串
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取第一个分子对
    first_pair = data[0]
    smiles_from = first_pair['smiles_from']
    smiles_to = first_pair['smiles_to']
    
    return smiles_from, smiles_to

def calculate_mcs(mol1, mol2):
    """
    计算两个分子之间的最大公共子结构(MCS)
    """
    # 配置MCS参数
    mcs_params = rdFMCS.MCSParameters()
    mcs_params.BondCompareParameters.BondCompare = rdFMCS.BondCompare.CompareOrder
    mcs_params.AtomCompareParameters.AtomCompare = rdFMCS.AtomCompare.CompareElements
    mcs_params.Timeout = 10  # 超时时间（秒）
    mcs_params.Threshold = 0.8
    
    # 计算MCS
    mcs_result = rdFMCS.FindMCS([mol1, mol2], mcs_params)
    
    return mcs_result

def visualize_mcs(mol1, mol2, mcs_result, output_path='mcs_visualization.png'):
    """
    可视化分子对及其MCS
    """
    # 将MCS结果转换为SMARTS模式
    mcs_smarts = mcs_result.smartsString
    mcs_mol = Chem.MolFromSmarts(mcs_smarts)
    
    if mcs_mol is None:
        print(f"无法解析MCS SMARTS: {mcs_smarts}")
        return
    
    # 匹配MCS子结构
    match1 = mol1.GetSubstructMatch(mcs_mol)
    match2 = mol2.GetSubstructMatch(mcs_mol)
    
    # 创建突出显示MCS的分子
    mol1_copy = Chem.Mol(mol1)
    mol2_copy = Chem.Mol(mol2)
    
    # 为匹配的原子设置高亮颜色
    AllChem.Compute2DCoords(mol1_copy)
    AllChem.Compute2DCoords(mol2_copy)
    AllChem.Compute2DCoords(mcs_mol)
    
    # 生成分子图像
    img1 = Draw.MolToImage(mol1_copy, highlightAtoms=match1, highlightColor=(0.8, 0.8, 0.2), size=(400, 300))
    img2 = Draw.MolToImage(mol2_copy, highlightAtoms=match2, highlightColor=(0.8, 0.8, 0.2), size=(400, 300))
    img_mcs = Draw.MolToImage(mcs_mol, size=(400, 300))
    
    # 使用matplotlib创建组合图像
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(img1)
    axs[0].set_title('分子1 (smiles_from)')
    axs[0].axis('off')
    
    axs[1].imshow(img_mcs)
    axs[1].set_title(f'MCS (大小: {mcs_result.numAtoms} 原子, {mcs_result.numBonds} 键)')
    axs[1].axis('off')
    
    axs[2].imshow(img2)
    axs[2].set_title('分子2 (smiles_to)')
    axs[2].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存到: {output_path}")
    
    # 显示图像
    plt.show()

def main():
    """
    主函数
    """
    # JSON文件路径
    json_file = '/home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/data/gdcsv2/cell_687787_debug_v1-evo-pairs.json'
    
    # 加载分子SMILES
    smiles_from, smiles_to = load_molecules_from_json(json_file)
    print(f"分子1 SMILES: {smiles_from}")
    print(f"分子2 SMILES: {smiles_to}")
    
    # 转换为RDKit分子对象
    mol1 = Chem.MolFromSmiles(smiles_from)
    mol2 = Chem.MolFromSmiles(smiles_to)
    
    if mol1 is None or mol2 is None:
        print("无法解析SMILES字符串")
        return
    
    # 计算MCS
    print("正在计算MCS...")
    mcs_result = calculate_mcs(mol1, mol2)
    
    # 打印MCS信息
    print(f"MCS 原子数: {mcs_result.numAtoms}")
    print(f"MCS 键数: {mcs_result.numBonds}")
    print(f"MCS SMARTS: {mcs_result.smartsString}")
    
    # 可视化MCS
    visualize_mcs(mol1, mol2, mcs_result)

if __name__ == "__main__":
    main()