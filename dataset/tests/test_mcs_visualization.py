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
    v0.1 可视化分子对及其MCS
    """
    # 将MCS结果转换为SMARTS模式
    mcs_smarts = mcs_result.smartsString
    mcs_mol = Chem.MolFromSmarts(mcs_smarts)
    
    if mcs_mol is None:
        print(f"无法解析MCS SMARTS: {mcs_smarts}")
        return
    
    # 对MCS分子进行Kekulize处理，确保芳香环正确显示； # BUG 无用
    try:
        Chem.Kekulize(mcs_mol)
    except:
        pass
    
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
    
    # 生成分子图像，使用kekulize=True确保芳香环正确显示
    img1 = Draw.MolToImage(mol1_copy, highlightAtoms=match1, highlightColor=(0.8, 0.8, 0.2), size=(400, 300), kekulize=True)
    img2 = Draw.MolToImage(mol2_copy, highlightAtoms=match2, highlightColor=(0.8, 0.8, 0.2), size=(400, 300), kekulize=True)
    img_mcs = Draw.MolToImage(mcs_mol, size=(400, 300), kekulize=True)
    
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

def visualize_mcs_iterations(mol1_states, mol2_states, mcs_results, output_path='mcs_iterations_visualization.png'):
    """
    可视化所有迭代过程，使用n行3列的布局
    """
    n_iterations = len(mcs_results)
    
    # 创建n行3列的子图布局
    fig, axs = plt.subplots(n_iterations, 3, figsize=(15, 5 * n_iterations))
    
    # 如果只有一次迭代，确保axs是二维数组
    if n_iterations == 1:
        axs = axs.reshape(1, 3)
    
    for i in range(n_iterations):
        mol1_current = mol1_states[i]
        mol2_current = mol2_states[i]
        mcs_result = mcs_results[i]
        
        # 将MCS结果转换为SMARTS模式
        mcs_smarts = mcs_result.smartsString
        mcs_mol = Chem.MolFromSmarts(mcs_smarts)
        
        if mcs_mol is None:
            print(f"无法解析MCS SMARTS: {mcs_smarts}")
            continue
        
        # 匹配MCS子结构
        match1 = mol1_current.GetSubstructMatch(mcs_mol)
        match2 = mol2_current.GetSubstructMatch(mcs_mol)
        
        # 创建突出显示MCS的分子
        mol1_copy = Chem.Mol(mol1_current)
        mol2_copy = Chem.Mol(mol2_current)
        
        # 为匹配的原子设置高亮颜色
        AllChem.Compute2DCoords(mol1_copy)
        AllChem.Compute2DCoords(mol2_copy)
        AllChem.Compute2DCoords(mcs_mol)
        
        # 生成分子图像
        img1 = Draw.MolToImage(mol1_copy, highlightAtoms=match1, highlightColor=(0.8, 0.8, 0.2), size=(400, 300), kekulize=True)
        img2 = Draw.MolToImage(mol2_copy, highlightAtoms=match2, highlightColor=(0.8, 0.8, 0.2), size=(400, 300), kekulize=True)
        img_mcs = Draw.MolToImage(mcs_mol, size=(400, 300), kekulize=True)
        
        # 第一列：分子1
        axs[i, 0].imshow(img1)
        axs[i, 0].set_title(f'迭代 {i+1}: 分子1\n(剩余原子: {mol1_current.GetNumAtoms()})')
        axs[i, 0].axis('off')
        
        # 第二列：MCS
        axs[i, 1].imshow(img_mcs)
        axs[i, 1].set_title(f'MCS ({mcs_result.numAtoms} 原子, {mcs_result.numBonds} 键)')
        axs[i, 1].axis('off')
        
        # 第三列：分子2
        axs[i, 2].imshow(img2)
        axs[i, 2].set_title(f'分子2\n(剩余原子: {mol2_current.GetNumAtoms()})')
        axs[i, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"迭代可视化结果已保存到: {output_path}")
    
    # 显示图像
    plt.show()

def main():
    """
    主函数
    """
    # 2. 多 MCS 结构拆分
    smiles_from = "CCN1CCN(CC1)CC2=C(C=C(C=C2)NC(=O)C3=CC(=C(C=C3)C)/C=C/C4=CN=C5C(=C4OC)C=CN5)C(F)(F)F"
    smiles_to = "CCN1CCN(CC1)CC2=C(C=C(C=C2)NC(=O)C3=CC(=C(C=C3)C)OC4=C5C=CNC5=NC=C4)C(F)(F)F"
    
    # 3. 虚线？
    # smiles_from = "C[C@@]12[C@@H]([C@@H](C[C@@H](O1)N3C4=CC=CC=C4C5=C6C(=C7C8=CC=CC=C8N2C7=C53)CNC6=O)N(C)C(=O)C9=CC=CC=C9)OC"
    # smiles_to = "C1=CC(=C(C(=C1)F)N(C2=NC(=C(C=C2)C(=O)N)C3=C(C=C(C=C3)F)F)C(=O)N)F"
    
    # 转换为RDKit分子对象
    mol1 = Chem.MolFromSmiles(smiles_from)
    mol2 = Chem.MolFromSmiles(smiles_to)
    
    if mol1 is None or mol2 is None:
        print("无法解析SMILES字符串")
        return
    
    # 循环计算MCS，直到MCS为单原子
    iteration = 0
    mcs_results = []
    mol1_states = []
    mol2_states = []
    mol1_current = mol1
    mol2_current = mol2
    
    while True:
        iteration += 1
        print(f"\n=== 第 {iteration} 次迭代 ===")
        print("正在计算MCS...")
        mcs_result = calculate_mcs(mol1_current, mol2_current)
        
        # 打印MCS信息
        print(f"MCS 原子数: {mcs_result.numAtoms}")
        print(f"MCS 键数: {mcs_result.numBonds}")
        print(f"MCS SMARTS: {mcs_result.smartsString}")
        
        # 保存当前状态
        mol1_states.append(Chem.Mol(mol1_current))
        mol2_states.append(Chem.Mol(mol2_current))
        mcs_results.append(mcs_result)
        
        # 如果MCS为单原子，停止循环
        if mcs_result.numAtoms <= 1:
            print("MCS已为单原子，停止迭代")
            break
        
        # 将MCS部分从分子中移除
        mcs_smarts = mcs_result.smartsString
        mcs_mol = Chem.MolFromSmarts(mcs_smarts)
        
        if mcs_mol is None:
            print("无法解析MCS SMARTS，停止迭代")
            break
        
        # 从mol1中移除MCS
        try:
            mol1_current = Chem.DeleteSubstructs(mol1_current, mcs_mol)
            print(f"从mol1中移除MCS后剩余原子数: {mol1_current.GetNumAtoms()}")
        except Exception as e:
            print(f"从mol1移除MCS时出错: {e}")
            break
        
        # 从mol2中移除MCS
        try:
            mol2_current = Chem.DeleteSubstructs(mol2_current, mcs_mol)
            print(f"从mol2中移除MCS后剩余原子数: {mol2_current.GetNumAtoms()}")
        except Exception as e:
            print(f"从mol2移除MCS时出错: {e}")
            break
        
        # 如果任一分子为空，停止循环
        if mol1_current.GetNumAtoms() == 0 or mol2_current.GetNumAtoms() == 0:
            print("其中一个分子已为空，停止迭代")
            break
    
    print(f"\n=== 迭代完成，共找到 {len(mcs_results)} 个MCS ===")
    
    # 可视化所有迭代过程
    visualize_mcs_iterations(mol1_states, mol2_states, mcs_results)

if __name__ == "__main__":
    main()