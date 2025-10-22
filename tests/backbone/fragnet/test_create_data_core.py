#!/usr/bin/env python3
"""
测试 core/utils/fragnet_data 目录下的 CreateData 类及其子类的功能
"""

import sys
import os

# 添加项目路径，确保可以导入项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'mol_evo', 'modules', 'FragNet'))

from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from mol_evo.core.utils.fragnet_data.data import CreateData, collate_fn


def test_create_data_core():
    """
    测试 core/utils/fragnet_data 下的 CreateData 类
    """
    print("测试 core/utils/fragnet_data 下的 CreateData 类...")
    
    # 创建一个简单的分子 (乙醇)
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    
    # 创建 CreateData 实例
    creator = CreateData(data_type="exp")
    
    # 准备参数 (smiles, y, mol, conf, frag_type)
    args = (smiles, [1.0], mol, conf, "brics")  # 将y改为列表形式
    
    # 创建数据点
    data = creator.create_data_point(args)
    
    if data is not None:
        print(f"  成功创建数据点:")
        print(f"    SMILES: {data.smiles}")
        print(f"    原子特征形状: {data.x_atoms.shape}")
        print(f"    边索引形状: {data.edge_index.shape}")
        print(f"    边属性形状: {data.edge_attr.shape}")
        print(f"    片段索引形状: {data.frag_index.shape}")
        print(f"    片段数量: {data.n_frags}")
        print(f"    键图节点特征形状: {data.node_features_bonds.shape}")
        print(f"    键图边索引形状: {data.edge_index_bonds.shape}")
        print(f"    键图边属性形状: {data.edge_attr_bonds.shape}")
        return data
    else:
        print("  创建数据点失败")
        return None


def test_collate_functions_core():
    """
    测试 core/utils/fragnet_data 下的批处理函数
    """
    print("\n测试 core/utils/fragnet_data 下的批处理函数...")
    
    # 创建多个数据点进行批处理测试
    smiles_list = ["CCO", "CCN", "CCC"]
    data_list = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)
            conf = mol.GetConformer()
            
            creator = CreateData(data_type="exp", create_fragbond_graph=False)  # 禁用fragbond图以避免错误
            args = (smiles, [1.0], mol, conf, "brics")  # 将y改为列表形式
            data = creator.create_data_point(args)
            if data is not None:
                data_list.append(data)
    
    if data_list:
        batch = collate_fn(data_list)
        print(f"  成功创建批处理数据:")
        print(f"    原子特征形状: {batch['x_atoms'].shape}")
        print(f"    边索引形状: {batch['edge_index'].shape}")
        print(f"    目标值形状: {batch['y'].shape}")
        print(f"    批处理大小: {len(data_list)}")
        return batch
    else:
        print("  批处理测试失败")
        return None


def compare_with_modules_version():
    """
    比较 core/utils/fragnet_data 和 modules/FragNet/fragnet/dataset 中的 CreateData 类
    """
    print("\n比较两个版本的 CreateData 类...")
    
    # 测试 core/utils/fragnet_data 版本
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    
    # core/utils/fragnet_data 版本
    from mol_evo.core.utils.fragnet_data.data import CreateData as CoreCreateData
    core_creator = CoreCreateData(data_type="exp", create_fragbond_graph=False)
    core_args = (smiles, [1.0], mol, conf, "brics")
    core_data = core_creator.create_data_point(core_args)
    
    # modules/FragNet/fragnet/dataset 版本
    from fragnet.dataset.data import CreateData as ModuleCreateData
    module_creator = ModuleCreateData(data_type="exp", create_fragbond_graph=False)
    module_args = (smiles, [1.0], mol, conf, "brics")
    module_data = module_creator.create_data_point(module_args)
    
    if core_data is not None and module_data is not None:
        print("  两个版本都成功创建了数据点")
        print(f"    core版本原子特征形状: {core_data.x_atoms.shape}")
        print(f"    module版本原子特征形状: {module_data.x_atoms.shape}")
        print(f"    core版本边索引形状: {core_data.edge_index.shape}")
        print(f"    module版本边索引形状: {module_data.edge_index.shape}")
        print(f"    core版本片段索引形状: {core_data.frag_index.shape}")
        print(f"    module版本片段索引形状: {module_data.frag_index.shape}")
        
        # 比较原子特征是否相同
        atoms_equal = torch.allclose(core_data.x_atoms, module_data.x_atoms, equal_nan=True)
        print(f"    原子特征是否相同: {atoms_equal}")
        
        # 比较边索引是否相同
        edge_index_equal = torch.equal(core_data.edge_index, module_data.edge_index)
        print(f"    边索引是否相同: {edge_index_equal}")
        
        # 比较片段索引是否相同
        frag_index_equal = torch.equal(core_data.frag_index, module_data.frag_index)
        print(f"    片段索引是否相同: {frag_index_equal}")
        
    else:
        print("  至少有一个版本创建数据点失败")


if __name__ == "__main__":
    print("开始测试 core/utils/fragnet_data 下的 CreateData 类")
    print("=" * 60)
    
    test_create_data_core()
    test_collate_functions_core()
    compare_with_modules_version()
    
    print("\n" + "=" * 60)
    print("测试完成")