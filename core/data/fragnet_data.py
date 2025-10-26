"""
FragNet数据预处理模块

该模块提供将SMILES字符串转换为FragNet模型所需输入格式的功能。
直接使用FragNet官方的CreateData类进行数据处理，确保特征提取的一致性。
"""

import sys
import os
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

# 添加项目根目录到Python路径，解决导入问题
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, project_root)

# 使用相对导入或通过添加路径后的绝对导入
try:
    # 尝试使用绝对导入
    from mol_evo.modules.FragNet.fragnet.dataset.data import CreateData, collate_fn
except ImportError:
    # 如果失败，尝试使用相对导入
    sys.path.append(os.path.join(project_root, 'mol_evo', 'modules', 'FragNet'))
    from fragnet.dataset.data import CreateData, collate_fn


def smile_to_fragnet_features(smile):
    """
    将SMILES字符串转换为FragNet模型所需的特征格式
    
    Args:
        smile (str): SMILES字符串
        
    Returns:
        dict: 包含FragNet所需的所有特征的字典
    """
    try:
        # 创建分子对象和3D构象
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
            
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        
        # 使用 FragNet 的 CreateData 类
        creator = CreateData(data_type="exp")
        
        # 创建数据点 (smiles, y, mol, conf, frag_type)
        args = (smile, [0.0], mol, conf, "brics")
        data = creator.create_data_point(args)
        
        if data is None:
            return None
            
        # 构建返回字典，格式与原始实现保持一致
        fragnet_features = {
            'x_atoms': data.x_atoms,
            'edge_index': data.edge_index,
            'edge_attr': data.edge_attr,
            'frag_index': data.frag_index,
            'cnx_attr': data.cnx_attr,
            'x_frags': data.x_frags,
            'atom_to_frag_ids': data.atom_id_frag_id,
            'n_frags': data.n_frags,
            'smiles': data.smiles,
            'node_features_bonds': data.node_features_bonds,
            'edge_index_bonds': data.edge_index_bonds,
            'edge_attr_bonds': data.edge_attr_bonds
        }
        
        return fragnet_features
        
    except Exception as e:
        print(f"Error processing SMILES {smile}: {e}")
        import traceback
        traceback.print_exc()
        return None


def smile_to_fragnet_batch(smiles_list):
    """
    将SMILES字符串列表转换为FragNet模型所需的批处理格式
    
    Args:
        smiles_list (list): SMILES字符串列表
        
    Returns:
        dict: 包含FragNet所需的所有特征的字典，格式与collate_fn返回的批次数据一致
    """
    try:
        data_list = []
        
        for smile in smiles_list:
            # 创建分子对象和3D构象
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                continue
                
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)
            conf = mol.GetConformer()
            
            # 使用 FragNet 的 CreateData 类
            creator = CreateData(data_type="exp")
            
            # 创建数据点 (smiles, y, mol, conf, frag_type)
            args = (smile, [0.0], mol, conf, "brics")
            data = creator.create_data_point(args)
            
            if data is not None:
                data_list.append(data)
        
        if not data_list:
            return None
            
        # 使用 FragNet 的 collate_fn 进行批处理
        batch_data = collate_fn(data_list)
        
        return batch_data
        
    except Exception as e:
        print(f"Error processing SMILES list: {e}")
        import traceback
        traceback.print_exc()
        return None


def fragnet_collate(batch):
    """为FragNet模型专门设计的批处理函数。
    
    该函数不使用 PyG 的 Batch.from_data_list，因为 FragNet 模型需要接收原始的字典列表。
    直接将 from/to 数据保留为列表，仅对 edge 和 target 特征进行堆叠。
    
    Args:
        batch: 包含(from_data, to_data, edge_attr, target)元组的批次数据
        
    Returns:
        tuple: (from_batch, to_batch, edge_batch, target_batch)
    """
    from_list, to_list, edge_list, target_list = zip(*batch)
    
    # 确保数据是 FragNet 字典格式
    if len(from_list) > 0:
        assert isinstance(from_list[0], dict), f"预期 from_data 为 dict 格式，但得到 {type(from_list[0])}"
        assert isinstance(to_list[0], dict), f"预期 to_data 为 dict 格式，但得到 {type(to_list[0])}"
    
    # 对于 FragNet，直接返回列表
    from_batch = list(from_list)
    to_batch = list(to_list)
    
    # 将边特征和目标值堆叠成张量
    edge_batch = torch.stack(edge_list, dim=0)   # Shape: (B, edge_feature_dim)
    target_batch = torch.stack(target_list, dim=0)  # Shape: (B, 1)
    
    return from_batch, to_batch, edge_batch, target_batch