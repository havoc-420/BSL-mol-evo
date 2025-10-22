#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子处理工具函数
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType, BondType
import os
import hashlib
from torch_scatter import scatter_add  # 导入 scatter_add 函数

# 导入本地模块
from .fragnet_data.fragments import get_3Dcoords, FragmentedMol
from .fragnet_data.features import FeaturesEXP

# 定义需要的全局变量和辅助函数
try:
    ETKDG_PARAMS = AllChem.ETKDGv3()
    ETKDG_VERSION_USED = "ETKDGv3"
except AttributeError:
    ETKDG_PARAMS = AllChem.ETKDGv2()
    ETKDG_VERSION_USED = "ETKDGv2 (fallback)"

bonds = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3}


def one_hot(tensor, num_classes):
    """
    将整数张量转换为 one-hot 编码。
    
    Args:
        tensor: 输入张量。
        num_classes: 类别数量。
        
    Returns:
        one-hot 编码张量。
    """
    return torch.eye(num_classes, dtype=torch.float)[tensor]


def smile_to_graph_xyz(smile, types):
    """
    将SMILES字符串转换为图结构表示，包括3D坐标信息。
    
    Args:
        smile (str): SMILES字符串
        types (dict): 原子类型映射字典
        
    Returns:
        tuple: (x, z, pos, edge_index, edge_attr) 其中:
            - x: 节点特征矩阵
            - z: 原子序数
            - pos: 3D坐标
            - edge_index: 边索引
            - edge_attr: 边属性: 边类型 one-hot 编码
    """
    mol = Chem.MolFromSmiles(smile)
    
    if mol is None:
        print(f"无法解析SMILES: {smile}")
        return None, None, None, None, None
    
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, ETKDG_PARAMS)
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        pos = torch.tensor(pos, dtype=torch.float)
    except Exception:
        # UPDATE 暂时没有用到，就先不提示了。
        # print(f'无法生成3D坐标，尝试使用{ETKDG_VERSION_USED}生成坐标...', repr(e))
        # 去除立体构型
        Chem.RemoveStereochemistry(mol)
        try:
            # 第二次尝试生成坐标
            AllChem.EmbedMolecule(mol, ETKDG_PARAMS)
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)
        except Exception:
            pos = None  # 明确设置为None
    
    # 检查是否成功生成坐标
    if pos is None:
        # 返回空的结果而不是未定义的变量
        return None, None, None, None, None
        
    # 获取原子特征
    type_idx = []
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in types:
            continue
        type_idx.append(types[symbol])
        atomic_number.append(atom.GetAtomicNum())                       # 原子序数
        aromatic.append(1 if atom.GetIsAromatic() else 0)               # 芳香性
        hybridization = atom.GetHybridization()                         # 杂化类型 * 3
        sp.append(1 if hybridization == HybridizationType.SP else 0)    
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    # 获取键信息
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = one_hot(edge_type, num_classes=len(bonds))

    # 排序边
    N = mol.GetNumAtoms()
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    edge_attr = edge_attr[perm]

    # 计算氢原子数量
    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = torch.zeros(N, dtype=torch.float).scatter_add_(0, col, hs[row]).tolist()

    # 构建节点特征
    x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))        # 原子类型 one-hot 编码
    x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],  # 直接将各种属性信息作为节点特征
                        dtype=torch.float).t().contiguous()
    x = torch.cat([x1, x2], dim=-1)
    
    return x, z, pos, edge_index, edge_attr


def smile_to_fragnet_features(smile):
    """
    将SMILES字符串转换为FragNet模型所需的特征格式
    
    Args:
        smile (str): SMILES字符串
        
    Returns:
        dict: 包含FragNet所需的所有特征的字典
    """
    from .fragnet_data.fragments import get_3Dcoords, FragmentedMol
    from .fragnet_data.features import FeaturesEXP
    
    # 创建分子对象和3D构象
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    
    # 生成3D坐标，返回的是一个包含3D坐标的分子对象
    mol3d = get_3Dcoords(smile)
    if mol3d is None:
        return None
    
    # 获取构象，这个构象属于mol3d分子
    conf = mol3d.GetConformer()
    
    # 创建分片分子对象，使用mol3d和其自身的构象
    frag_mol = FragmentedMol(mol3d, conf, frag_type="brics")
    
    # 创建特征提取器
    feature_creator = FeaturesEXP()
    
    # 提取原子和键特征
    x_atoms, edge_index, edge_attr = feature_creator.get_atom_and_bond_features_atom_graph_one_hot(
        frag_mol.mol, feature_creator.use_bond_chirality
    )
    
    # 提取片段连接特征
    frag_idx = [[], []]
    cnx_attr = []
    for connection in frag_mol.connections:
        frag_idx[0] += [connection.BeginFragIdx, connection.EndFragIdx]
        frag_idx[1] += [connection.EndFragIdx, connection.BeginFragIdx]
        cnx_attr.append(feature_creator.connection_features_one_hot(connection))
        cnx_attr.append(feature_creator.connection_features_one_hot(connection))
    
    frag_idx = torch.tensor(frag_idx, dtype=torch.long)
    cnx_attr = torch.tensor(cnx_attr, dtype=torch.float)
    
    # 构建原子到片段的映射
    atom_id_frag_id = torch.tensor(
        list(frag_mol.atom_to_frag_id.values()), dtype=torch.long
    )
    
    # 聚合片段特征
    x_atoms_tensor = torch.tensor(np.array(x_atoms), dtype=torch.float)
    x_frags = scatter_add(src=x_atoms_tensor, index=atom_id_frag_id, dim=0)
    
    # 片段数量
    n_frags = torch.tensor([len(frag_mol.fragments)], dtype=torch.long)
    
    # 构建返回字典
    fragnet_features = {
        'x_atoms': x_atoms_tensor,
        'edge_index': torch.tensor(edge_index, dtype=torch.long),
        'edge_attr': torch.tensor(edge_attr, dtype=torch.float),
        'frag_index': frag_idx,
        'cnx_attr': cnx_attr,
        'x_frags': x_frags,
        'atom_id_frag_id': atom_id_frag_id,
        'n_frags': n_frags,
        'smiles': smile
    }
    
    return fragnet_features


def smile_to_fragnet_batch(smile):
    """
    将SMILES字符串转换为FragNet模型所需的输入格式
    
    Args:
        smile (str): SMILES字符串
        
    Returns:
        dict: 包含FragNet所需的所有特征的字典，格式与collate_fn返回的批次数据一致
    """
    try:
        # 导入必要的模块
        from .fragnet_data.data import CreateData
        
        # 创建分子对象和3D构象
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
            
        # 生成3D坐标
        mol3d = get_3Dcoords(smile)
        if mol3d is None:
            return None
            
        conf = mol3d.GetConformer()
        
        # 创建数据创建器
        creator = CreateData(data_type='finetune')
        
        # 创建数据点 (模拟调用 create_data_point 方法的部分逻辑)
        graph = FragmentedMol(mol3d, conf, frag_type="brics")
        
        # 获取原子和键特征
        feature_creator = FeaturesEXP()
        x_atoms, edge_index, edge_attr = feature_creator.get_atom_and_bond_features_atom_graph_one_hot(
            graph.mol, feature_creator.use_bond_chirality
        )
        np
        x_atoms = torch.tensor(np.array(x_atoms), dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # 获取片段连接特征
        frag_idx = [[], []]
        cnx_attr = []
        for connection in graph.connections:
            frag_idx[0] += [connection.BeginFragIdx, connection.EndFragIdx]
            frag_idx[1] += [connection.EndFragIdx, connection.BeginFragIdx]
            cnx_attr.append(feature_creator.connection_features_one_hot(connection))
            cnx_attr.append(feature_creator.connection_features_one_hot(connection))

        frag_idx = torch.tensor(frag_idx, dtype=torch.long)
        cnx_attr = torch.tensor(cnx_attr, dtype=torch.float)
        
        # 构建原子到片段的映射
        atom_id_frag_id = torch.tensor(
            list(graph.atom_to_frag_id.values()), dtype=torch.long
        )
        
        # 聚合片段特征
        x_frags = scatter_add(src=x_atoms, index=atom_id_frag_id, dim=0)
        
        # 片段数量
        n_frags = len(graph.fragments)
        
        # 创建单个分子的批次数据 (模拟 collate_fn 的输出结构)
        batch_data = {
            "x_atoms": x_atoms,
            "edge_index": edge_index,
            "frag_index": frag_idx,
            "x_frags": x_frags,
            "edge_attr": edge_attr,
            "cnx_attr": cnx_attr,
            "batch": torch.zeros(x_atoms.shape[0], dtype=torch.long),  # 单个分子，批次索引全为0
            "frag_batch": torch.zeros(n_frags, dtype=torch.long),      # 单个分子的所有片段，批次索引全为0
            "atom_to_frag_ids": atom_id_frag_id,
            "y": torch.tensor([0.0], dtype=torch.float),  # 默认目标值
            "smiles": smile
        }
        
        # 添加默认的键图和片段键图特征（如果模型需要）
        num_bonds = edge_index.shape[1] // 2  # 因为每条边存储了两次
        batch_data["node_features_bonds"] = torch.zeros((num_bonds, 16), dtype=torch.float)  # 默认键特征，维度应为16
        batch_data["edge_index_bonds_graph"] = torch.tensor([[], []], dtype=torch.long)  # 空的键图边索引
        batch_data["edge_attr_bonds"] = torch.zeros((0, 1), dtype=torch.float)  # 空的键图边属性，维度应为1
        batch_data["node_features_fbonds"] = torch.zeros((0, 16), dtype=torch.float)  # 空的片段键特征，维度应为16
        batch_data["edge_index_fbonds"] = torch.tensor([[], []], dtype=torch.long)  # 空的片段键图边索引
        batch_data["edge_attr_fbonds"] = torch.zeros((0, 1), dtype=torch.float)  # 空的片段键图边属性，维度应为1
        
        return batch_data
        
    except Exception as e:
        print(f"Error processing SMILES {smile}: {e}")
        return None


class MoleculeCache:
    """
    分子图结构缓存类，支持将多个SMILES存储在同一个pt文件中
    """
    
    def __init__(self, cache_name="default", csv_file=None, logger=None):
        """
        初始化缓存
        
        Args:
            cache_name (str): 缓存文件名（不包含扩展名）
            csv_file (str): 关联的CSV文件路径，如果提供则会基于此生成缓存文件名
            logger: 日志记录器
        """
        if csv_file:
            # 根据CSV文件路径生成唯一的缓存名称
            self.cache_name = self._generate_cache_name_from_csv(csv_file)
        else:
            self.cache_name = cache_name
            
        self.logger = logger
        self.cache_dir = self._get_cache_dir()
        self.cache_file = os.path.join(self.cache_dir, f"{self.cache_name}.pt")
        self.cache_data = self._load_cache()
        self.cache_hits = 0
        self.cache_misses = 0
        self.added_count = 0  # 新增计数器
    
    def _generate_cache_name_from_csv(self, csv_file):
        """
        根据CSV文件路径生成缓存名称
        
        Args:
            csv_file (str): CSV文件路径
            
        Returns:
            str: 基于CSV文件生成的缓存名称
        """
        # 获取文件的绝对路径和基本信息
        abs_path = os.path.abspath(csv_file)
        file_hash = hashlib.md5(abs_path.encode('utf-8')).hexdigest()[:16]
        basename = os.path.basename(csv_file)
        name_without_ext = os.path.splitext(basename)[0]
        
        # 生成缓存名称
        return f"csv_{name_without_ext}_{file_hash}"
    
    def _get_cache_dir(self):
        """
        获取缓存目录路径
        
        Returns:
            str: 缓存目录路径
        """
        # 使用项目根目录下的cache目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, '..', '..')
        cache_dir = os.path.join(project_root, '.cache', 'molecule_graphs')
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    
    def _log(self, message):
        """
        统一日志输出方法
        
        Args:
            message (str): 日志消息
        """
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def _load_cache(self):
        """
        加载缓存文件
        
        Returns:
            dict: 缓存数据字典
        """
        if os.path.exists(self.cache_file):
            try:
                cache_data = torch.load(self.cache_file)
                self._log(f"加载缓存文件: {self.cache_file}，包含 {len(cache_data)} 个分子")
                return cache_data
            except Exception as e:
                self._log(f"加载缓存文件失败: {e}，将创建新的缓存")
                return {}
        return {}
    
    def _save_cache(self):
        """
        保存缓存到文件
        """
        try:
            torch.save(self.cache_data, self.cache_file)
        except Exception as e:
            self._log(f"保存缓存文件失败: {e}")
    
    def get(self, smile, types):
        """
        从缓存中获取分子图结构
        
        Args:
            smile (str): SMILES字符串
            types (dict): 原子类型映射字典
            
        Returns:
            tuple or None: 分子图结构数据，如果不存在则返回None
        """
        # 创建唯一的键
        types_str = str(sorted(types.items()))
        key = f"{smile}_{types_str}"
        
        if key in self.cache_data:
            self.cache_hits += 1
            cached_item = self.cache_data[key]
            return (cached_item['x'], cached_item['z'], cached_item['pos'], 
                   cached_item['edge_index'], cached_item['edge_attr'])
        else:
            self.cache_misses += 1
            return None
    
    def put(self, smile, types, data):
        """
        将分子图结构存入缓存
        
        Args:
            smile (str): SMILES字符串
            types (dict): 原子类型映射字典
            data (tuple): 分子图结构数据
        """
        # 创建唯一的键
        types_str = str(sorted(types.items()))
        key = f"{smile}_{types_str}"
        
        # 存储数据
        self.cache_data[key] = {
            'x': data[0],
            'z': data[1], 
            'pos': data[2],
            'edge_index': data[3],
            'edge_attr': data[4]
        }
        
        # 保存到文件
        self._save_cache()
        self.added_count += 1
    
    def get_stats(self):
        """
        获取缓存统计信息
        
        Returns:
            dict: 包含缓存命中率等统计信息
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'total': total_requests,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache_data),
            'added_count': self.added_count
        }
    
    def process_smiles(self, smile, types):
        """
        处理单个SMILES，带缓存支持
        
        Args:
            smile (str): SMILES字符串
            types (dict): 原子类型映射字典
            
        Returns:
            tuple: 分子图结构数据
        """
        # 尝试从缓存获取
        cached_result = self.get(smile, types)
        if cached_result is not None:
            return cached_result
        
        # 缓存未命中，计算分子图结构
        result = smile_to_graph_xyz(smile, types)
        
        # 如果计算成功，存入缓存
        if result[0] is not None:
            self.put(smile, types, result)
        
        return result