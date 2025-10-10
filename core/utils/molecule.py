#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子处理工具函数
"""

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType, BondType
import os

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


class MoleculeCache:
    """
    分子图结构缓存类，支持将多个SMILES存储在同一个pt文件中
    """
    
    def __init__(self, cache_name="default", logger=None):
        """
        初始化缓存
        
        Args:
            cache_name (str): 缓存文件名（不包含扩展名）
            logger: 日志记录器
        """
        self.cache_name = cache_name
        self.logger = logger
        self.cache_dir = self._get_cache_dir()
        self.cache_file = os.path.join(self.cache_dir, f"{cache_name}.pt")
        self.cache_data = self._load_cache()
        self.cache_hits = 0
        self.cache_misses = 0
        self.added_count = 0  # 新增计数器
    
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