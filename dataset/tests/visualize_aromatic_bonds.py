#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
识别并高亮显示分子中的芳香键
"""

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Serif CJK JP', 'Noto Mono', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def identify_aromatic_bonds(mol):
    """
    识别分子中的芳香键
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        aromatic_bonds: 芳香键的索引列表
    """
    aromatic_bonds = []
    
    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            aromatic_bonds.append(bond.GetIdx())
    
    return aromatic_bonds


def visualize_aromatic_bonds(smiles, output_path='aromatic_bonds_visualization.png'):
    """
    可视化分子中的芳香键
    
    Args:
        smiles: SMILES字符串
        output_path: 输出图像路径
    """
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        print(f"无法解析SMILES: {smiles}")
        return
    
    try:
        Chem.Kekulize(mol)
    except:
        pass
    
    aromatic_bonds = identify_aromatic_bonds(mol)
    
    AllChem.Compute2DCoords(mol)
    
    img = Draw.MolToImage(
        mol,
        highlightBonds=aromatic_bonds,
        highlightColor=(1.0, 0.0, 0.0),
        size=(600, 400),
        kekulize=True
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.set_title(f'芳香键高亮显示\nSMILES: {smiles}\n芳香键数量: {len(aromatic_bonds)}')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"芳香键可视化结果已保存到: {output_path}")
    plt.show()


def main():
    """
    主函数 - 测试不同的分子
    """
    test_cases = [
        "C1=CN(C(=O)N=C1N)[C@H]2C([C@@H]([C@H](O2)CO)O)(F)F",
        "C1=CN(C(=O)N=C1N)[C@H]2[C@H]([C@@H]([C@H](O2)CO)O)O",
        
        # 2. 多 MCS 结构 case
        "CCN1CCN(CC1)CC2=C(C=C(C=C2)NC(=O)C3=CC(=C(C=C3)C)/C=C/C4=CN=C5C(=C4OC)C=CN5)C(F)(F)F",
        "CCN1CCN(CC1)CC2=C(C=C(C=C2)NC(=O)C3=CC(=C(C=C3)C)OC4=C5C=CNC5=NC=C4)C(F)(F)F"    
    ]
    
    for i, smiles in enumerate(test_cases, 1):
        output_path = f'/home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/tests/image_aromatic/aromatic_bonds_{i}.png'
        print(f"\n处理分子 {i}: {smiles}")
        visualize_aromatic_bonds(smiles, output_path)


if __name__ == "__main__":
    main()
