#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用RDKit解析SMILES并生成分子结构图进行对比
专门用于对比CCCCCC和C1CCCCC1的分子结构
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, project_root)

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保已安装rdkit和matplotlib")
    sys.exit(1)


def create_molecule_image(smiles, filename, add_atom_numbers=False):
    """
    创建分子结构图像
    
    Args:
        smiles (str): SMILES字符串
        filename (str): 保存的文件名
        add_atom_numbers (bool): 是否添加原子编号
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"无效的SMILES: {smiles}")
        return False
    
    # 添加原子编号
    if add_atom_numbers:
        for atom in mol.GetAtoms():
            atom.SetProp('molAtomMapNumber', str(atom.GetIdx()))
    
    # 生成图像
    img = Draw.MolToImage(mol, size=(300, 300))
    
    # 保存图像到文件
    img.save(filename)
    print(f"分子图像已保存到: {filename}")
    return True


def main():
    """主函数"""
    # 定义要比较的分子
    molecules = {
        "CCCCCC": "正己烷 (Hexane)",
        "C1CCCCC1": "环己烷 (Cyclohexane)"
    }
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    print("生成分子结构图进行对比...")
    print("=" * 50)
    
    image_files = []
    for smiles, name in molecules.items():
        print(f"处理分子: {name} ({smiles})")
        filename = os.path.join(output_dir, f"{smiles.replace('/', '_')}.png")
        if create_molecule_image(smiles, filename, add_atom_numbers=True):
            image_files.append((filename, name, smiles))
    
    # 显示生成的图像
    if image_files:
        fig, axes = plt.subplots(1, len(image_files), figsize=(12, 6))
        if len(image_files) == 1:
            axes = [axes]
        
        for i, (filename, name, smiles) in enumerate(image_files):
            img = mpimg.imread(filename)
            axes[i].imshow(img)
            axes[i].set_title(f"{name}\n{smiles}", fontsize=12)
            axes[i].axis('off')
        
        plt.tight_layout()
        comparison_file = os.path.join(output_dir, "comparison.png")
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n对比图像已保存到: {comparison_file}")
    
    print("\n完成!")


if __name__ == "__main__":
    main()