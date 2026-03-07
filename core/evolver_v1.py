#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强大的分子进化路径生成器，能够将任意SMILES解析为原子级的、可重现的构建序列。
"""

from rdkit import Chem
from rdkit.Chem import Crippen
from collections import deque
import re
import random
from typing import List, Dict, Tuple, Optional, Set, Any
import hashlib
import argparse
import json
import sys
import logging
from tqdm import tqdm
import yaml

# 设置RDKit日志级别，减少警告输出
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# 修复导入问题
try:
    from .attachment import Attachment
except ImportError:
    # 当作为脚本直接运行时的备用导入
    from attachment import Attachment

class MoleculeEvolverAnalysis: # MoleculeEvolver
    """一个封装的分子路径生成器。"""
    def __init__(self, smiles: str):
        self.smiles = smiles
        if not smiles or not isinstance(smiles, str):
            raise ValueError(f"无效的SMILES: {smiles}")
        self.mol = Chem.MolFromSmiles(smiles)
        if not self.mol:
            raise ValueError(f"无效的SMILES: {smiles}")
        
        # 核心属性，将在解析过程中被填充
        self.ranks = Chem.CanonicalRankAtoms(self.mol, breakTies=True)
        self.backbone_indices = []
        self.backbone_map = {} # old_idx -> new_idx
        self.backbone_set = set()

    def _find_canonical_start_atom(self) -> int:
        """根据分层化学规则，找到一个绝对唯一的起始原子。"""
        atoms = list(self.mol.GetAtoms())
        
        # 规则1: 优先选择杂原子，按原子序数降序
        heteroatoms = [a for a in atoms if a.GetAtomicNum() != 6]
        if heteroatoms:
            atoms = sorted(heteroatoms, key=lambda a: a.GetAtomicNum(), reverse=True)
        
        # 规则2: 按连接数降序
        max_degree = max(a.GetDegree() for a in atoms)
        atoms = [a for a in atoms if a.GetDegree() == max_degree]

        # 规则3: 按RDKit规范排名升序作为最终决胜
        start_atom = min(atoms, key=lambda a: self.ranks[a.GetIdx()])
        
        return start_atom.GetIdx()

    def _build_canonical_backbone(self, start_idx: int):
        """从规范起点开始，通过规范DFS构建唯一的分子骨架（生成树）。"""
        self.backbone_indices = []
        visited = set()
        
        def dfs(parent_idx):
            visited.add(parent_idx)
            self.backbone_indices.append(parent_idx)
            
            neighbors = sorted(self.mol.GetAtomWithIdx(parent_idx).GetNeighbors(), key=lambda n: self.ranks[n.GetIdx()])
            for neighbor in neighbors:
                if neighbor.GetIdx() not in visited:
                    dfs(neighbor.GetIdx())
        
        dfs(start_idx)
        self.backbone_set = set(self.backbone_indices)
        self.backbone_map = {old_idx: new_idx for new_idx, old_idx in enumerate(self.backbone_indices)}

    def _get_bond_operation(self, bond):
        """根据键类型返回相应的操作描述"""
        bond_type = bond.GetBondType()
        
        bond_operations = {
            Chem.BondType.SINGLE: None,  # 单键不需要特殊操作
            Chem.BondType.DOUBLE: "形成双键",
            Chem.BondType.TRIPLE: "形成三键", 
            Chem.BondType.AROMATIC: "形成芳香键",
            Chem.BondType.DATIVE: "形成配位键",
            Chem.BondType.UNSPECIFIED: "形成未指定键"
        }
        
        return bond_operations.get(bond_type, f"形成{bond_type}键")

    def _has_rings(self):
        """更准确的环检测方法"""
        try:
            # 使用RDKit的环信息检测
            if self.mol.GetRingInfo().NumRings() > 0:
                return True
        except:
            pass
            
        try:
            # 备用方法：通过SSSR检测
            if len(Chem.GetSymmSSSR(self.mol)) > 0:
                return True
        except:
            pass
            
        return False

    def _get_bond_sort_key(self, bond_op_str):
        """为键操作生成排序键"""
        # 提取原子索引用于排序
        match = re.search(r'@\((\d+)-(\d+)\)', bond_op_str)
        if match:
            return tuple(sorted(map(int, match.groups())))
        return (0, 0)

    def generate_path(self) -> list:
        """生成最终的、完整的进化路径。"""
        try:
            start_idx = self._find_canonical_start_atom()
            self._build_canonical_backbone(start_idx)
            
            # 检查骨架是否为空
            if not self.backbone_indices:
                return ["错误: 无法构建分子骨架"]
                
            # --- 骨架路径 ---
            start_atom = self.mol.GetAtomWithIdx(self.backbone_indices[0])
            path = [f"起始({start_atom.GetSymbol()} ID:{self.backbone_map[start_atom.GetIdx()]})"]
            for i in range(1, len(self.backbone_indices)):
                parent_new_idx = self.backbone_map[self.backbone_indices[i-1]]
                current_atom = self.mol.GetAtomWithIdx(self.backbone_indices[i])
                path.append(f"添加原子({current_atom.GetSymbol()}) -> {parent_new_idx}")

            # --- 附件、额外键和立体化学 ---
            non_backbone_atoms = [a.GetIdx() for a in self.mol.GetAtoms() if a.GetIdx() not in self.backbone_set]
            # 构建骨架键集合，只包含DFS遍历过程中形成的键（父子连接）
            # 这确保了即使是骨架上的环内键也会被正确识别为"额外键"
            backbone_bonds = {tuple(sorted((self.backbone_indices[i], self.backbone_indices[i-1]))) for i in range(1, len(self.backbone_indices))}
            
            # --- 附件处理 ---
            attachments = self._get_sorted_attachments(non_backbone_atoms)
            for att in attachments:
                conn_points_str = ",".join(map(str, sorted(list(att.connection_points))))
                path.append(f"添加附件({att.mol_frag_smiles}) @ {conn_points_str}")

            # --- 额外键处理 (成环、多重键) ---
            extra_bond_ops = []
            # 检查分子是否有环结构，如果没有环，则不应该有任何成环操作
            has_rings = self._has_rings()
            
            for bond in self.mol.GetBonds():
                b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bond_tuple = tuple(sorted((b, e)))
                
                # 检查此键是否是骨架、附件内部或附件到骨架的连接键
                is_backbone_bond = bond_tuple in backbone_bonds
                both_in_backbone = self.mol.GetAtomWithIdx(b).GetIdx() in self.backbone_set and self.mol.GetAtomWithIdx(e).GetIdx() in self.backbone_set

                if is_backbone_bond and bond.GetBondType() != Chem.BondType.SINGLE:
                    # 处理骨架上的多重键（保持不变）
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        op = "形成双键"
                    elif bond.GetBondType() == Chem.BondType.TRIPLE:
                        op = "形成三键"
                    elif bond.GetBondType() == Chem.BondType.AROMATIC:
                        op = "形成芳香键"
                    else:
                        op = f"形成{bond.GetBondType()}键"
                    extra_bond_ops.append(f"{op} @({self.backbone_map[b]}-{self.backbone_map[e]})")
                
                elif not is_backbone_bond and both_in_backbone and has_rings:
                    # 修复：成环操作同时考虑键的类型
                    if bond.GetBondType() == Chem.BondType.AROMATIC:
                        op = "形成芳香环键"
                    elif bond.GetBondType() == Chem.BondType.DOUBLE:
                        op = "形成双键环"
                    elif bond.GetBondType() == Chem.BondType.TRIPLE:
                        op = "形成三键环"
                    else:
                        op = "成环"  # 默认单键环
                    
                    extra_bond_ops.append(f"{op} @({self.backbone_map[b]}-{self.backbone_map[e]})")
            
            # 使用改进的排序
            extra_bond_ops.sort(key=self._get_bond_sort_key)

            # --- 立体化学处理 ---
            stereo_ops = []
            chiral_centers = Chem.FindMolChiralCenters(self.mol, includeUnassigned=False)
            for center_idx, stereo in chiral_centers:
                if center_idx in self.backbone_map:
                    stereo_ops.append(f"指定手性({stereo}) @ {self.backbone_map[center_idx]}")
            
            for bond in self.mol.GetBonds():
                if bond.GetStereo() > Chem.BondStereo.STEREOANY:
                    b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    if b in self.backbone_map and e in self.backbone_map:
                        stereo_ops.append(f"指定顺反({bond.GetStereo()}) @ ({self.backbone_map[b]}-{self.backbone_map[e]})")
            
            return path + extra_bond_ops + sorted(stereo_ops)
            
        except Exception as e:
            return [f"路径生成错误: {str(e)}"]

    # MARK
    def generate_path_dict(self) -> list:
        """
        生成格式化的进化路径，返回字典格式的操作列表。
        每个操作都是一个字典，包含操作类型、位置和相关原子等信息。
        直接输出最终格式，与qm9-evo-pairs-step-1.json中的格式一致。
        """
        try:
            start_idx = self._find_canonical_start_atom()
            self._build_canonical_backbone(start_idx)
            
            # 检查骨架是否为空
            if not self.backbone_indices:
                return [{"position": "", "atom": None, "operation": "error"}]

            # --- 骨架路径 ---
            start_atom = self.mol.GetAtomWithIdx(self.backbone_indices[0])
            path = [{
                "position": str(self.backbone_map[start_atom.GetIdx()]),
                "atom": start_atom.GetSymbol(),
                "operation": "init_atom"
            }]
            
            for i in range(1, len(self.backbone_indices)):
                parent_new_idx = self.backbone_map[self.backbone_indices[i-1]]
                current_atom = self.mol.GetAtomWithIdx(self.backbone_indices[i])
                path.append({
                    "position": str(parent_new_idx),
                    "atom": current_atom.GetSymbol(),
                    "operation": "add_atom"
                })

            # --- 附件、额外键和立体化学 ---
            non_backbone_atoms = [a.GetIdx() for a in self.mol.GetAtoms() if a.GetIdx() not in self.backbone_set]
            # 构建骨架键集合，只包含DFS遍历过程中形成的键（父子连接）
            # 这确保了即使是骨架上的环内键也会被正确识别为"额外键"
            backbone_bonds = {tuple(sorted((self.backbone_indices[i], self.backbone_indices[i-1]))) for i in range(1, len(self.backbone_indices))}
            
            # --- 附件处理 ---
            attachments = self._get_sorted_attachments(non_backbone_atoms)
            for att in attachments:
                # TODO 但是几乎没有这个能力？
                conn_points = sorted(list(att.connection_points))
                path.append({
                    "position": str(conn_points[0]) if conn_points else "",
                    "atom": att.mol_frag_smiles,
                    "operation": "add_fragment"
                })

            # --- 额外键处理 (成环、多重键) ---
            extra_bond_ops = []
            # 检查分子是否有环结构，如果没有环，则不应该有任何成环操作
            has_rings = self._has_rings()
            
            for bond in self.mol.GetBonds():
                b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bond_tuple = tuple(sorted((b, e)))
                
                # 检查此键是否是骨架、附件内部或附件到骨架的连接键
                is_backbone_bond = bond_tuple in backbone_bonds
                both_in_backbone = self.mol.GetAtomWithIdx(b).GetIdx() in self.backbone_set and self.mol.GetAtomWithIdx(e).GetIdx() in self.backbone_set

                if is_backbone_bond and bond.GetBondType() != Chem.BondType.SINGLE:
                    positions = [self.backbone_map[b], self.backbone_map[e]]
                    positions.sort()  # 确保顺序一致
                    
                    # 根据键类型确定操作类型，与数据集保持一致
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        op_type = "form_double_bond"
                    elif bond.GetBondType() == Chem.BondType.TRIPLE:
                        op_type = "form_triple_bond"
                    elif bond.GetBondType() == Chem.BondType.AROMATIC:
                        op_type = "form_aromatic_bond"
                    elif bond.GetBondType() == Chem.BondType.DATIVE:
                        op_type = "form_dative_bond"
                    else:
                        op_type = "form_bond"
                        
                    extra_bond_ops.append({
                        "position": f"{positions[0]}-{positions[1]}",
                        "atom": None,
                        "operation": op_type
                    })
                elif not is_backbone_bond and both_in_backbone and has_rings:
                    # 成环操作同时考虑键的类型，与数据集保持一致
                    positions = sorted([self.backbone_map[b], self.backbone_map[e]])
                    
                    if bond.GetBondType() == Chem.BondType.AROMATIC:
                        op_type = "form_aromatic_ring"
                    elif bond.GetBondType() == Chem.BondType.DOUBLE:
                        op_type = "form_double_ring"
                    elif bond.GetBondType() == Chem.BondType.TRIPLE:
                        op_type = "form_triple_ring"
                    else:
                        op_type = "form_ring"  # 默认单键环
                    
                    extra_bond_ops.append({
                        "position": f"{positions[0]}-{positions[1]}",
                        "atom": None,
                        "operation": op_type
                    })
            
            # 按照位置对额外键操作进行排序
            extra_bond_ops.sort(key=lambda x: x["position"])
            path += extra_bond_ops

            # --- 立体化学处理 ---
            stereo_ops = []
            chiral_centers = Chem.FindMolChiralCenters(self.mol, includeUnassigned=False)
            for center_idx, stereo in chiral_centers:
                if center_idx in self.backbone_map:
                    stereo_ops.append({
                        "position": str(self.backbone_map[center_idx]),
                        "atom": None,
                        "operation": "add_stereo"
                    })
            
            for bond in self.mol.GetBonds():
                if bond.GetStereo() > Chem.BondStereo.STEREOANY:
                    b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    if b in self.backbone_map and e in self.backbone_map:
                        positions = [self.backbone_map[b], self.backbone_map[e]]
                        positions.sort()  # 确保顺序一致
                        
                        stereo_ops.append({
                            "position": f"{positions[0]}-{positions[1]}",
                            "atom": None,
                            "operation": "add_stereo"
                        })
            
            # 按照位置对立体化学操作进行排序
            def sort_key(op_dict):
                # 分解位置字符串以进行正确的排序
                pos = op_dict["position"]
                if "-" in pos:
                    parts = pos.split("-")
                    return [int(p) for p in parts]
                else:
                    return [int(pos)]
                    
            stereo_ops.sort(key=lambda x: sort_key(x))
            path_result = path + stereo_ops
            
            # 移除起始原子操作，因为它不是"变化"操作
            # 只保留除了第一个起始操作之外的所有操作
            if len(path_result) > 1:
                return path_result[1:]
            else:
                # 如果只有起始操作，返回空列表
                return []
            
        except Exception as e:
            return [{"position": "", "atom": None, "operation": "error"}]

    def get_full_path_dict(self) -> list:
        """
        获取完整的进化路径，包括起始操作。
        返回字典格式的操作列表，包含所有的分子构建步骤。
        """
        try:
            start_idx = self._find_canonical_start_atom()
            self._build_canonical_backbone(start_idx)
            
            # 检查骨架是否为空
            if not self.backbone_indices:
                return [{"position": "", "atom": None, "operation": "error"}]

            # --- 骨架路径 ---
            start_atom = self.mol.GetAtomWithIdx(self.backbone_indices[0])
            path = [{
                "position": str(self.backbone_map[start_atom.GetIdx()]),
                "atom": start_atom.GetSymbol(),
                "operation": "init_atom"
            }]
            
            for i in range(1, len(self.backbone_indices)):
                parent_new_idx = self.backbone_map[self.backbone_indices[i-1]]
                current_atom = self.mol.GetAtomWithIdx(self.backbone_indices[i])
                path.append({
                    "position": str(parent_new_idx),
                    "atom": current_atom.GetSymbol(),
                    "operation": "add_atom"
                })

            # --- 附件、额外键和立体化学 ---
            non_backbone_atoms = [a.GetIdx() for a in self.mol.GetAtoms() if a.GetIdx() not in self.backbone_set]
            # 构建骨架键集合，只包含DFS遍历过程中形成的键（父子连接）
            # 这确保了即使是骨架上的环内键也会被正确识别为"额外键"
            backbone_bonds = {tuple(sorted((self.backbone_indices[i], self.backbone_indices[i-1]))) for i in range(1, len(self.backbone_indices))}
            
            # --- 附件处理 ---
            attachments = self._get_sorted_attachments(non_backbone_atoms)
            for att in attachments:
                conn_points = sorted(list(att.connection_points))
                path.append({
                    "position": str(conn_points[0]) if conn_points else "",
                    "atom": att.mol_frag_smiles,
                    "operation": "add_fragment"
                })

            # --- 额外键处理 (成环、多重键) ---
            extra_bond_ops = []
            # 检查分子是否有环结构，如果没有环，则不应该有任何成环操作
            has_rings = self._has_rings()
            
            for bond in self.mol.GetBonds():
                b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bond_tuple = tuple(sorted((b, e)))
                
                # 检查此键是否是骨架、附件内部或附件到骨架的连接键
                is_backbone_bond = bond_tuple in backbone_bonds
                both_in_backbone = self.mol.GetAtomWithIdx(b).GetIdx() in self.backbone_set and self.mol.GetAtomWithIdx(e).GetIdx() in self.backbone_set

                if is_backbone_bond and bond.GetBondType() != Chem.BondType.SINGLE:
                    positions = [self.backbone_map[b], self.backbone_map[e]]
                    positions.sort()  # 确保顺序一致
                    
                    # 根据键类型确定操作类型
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        op_type = "form_double_bond"
                    elif bond.GetBondType() == Chem.BondType.TRIPLE:
                        op_type = "form_triple_bond"
                    elif bond.GetBondType() == Chem.BondType.AROMATIC:
                        op_type = "form_aromatic_bond"
                    elif bond.GetBondType() == Chem.BondType.DATIVE:
                        op_type = "form_dative_bond"
                    else:
                        op_type = "form_bond"
                        
                    extra_bond_ops.append({
                        "position": f"{positions[0]}-{positions[1]}",
                        "atom": None,
                        "operation": op_type
                    })
                elif not is_backbone_bond and both_in_backbone and has_rings:
                    # 成环操作同时考虑键的类型
                    positions = sorted([self.backbone_map[b], self.backbone_map[e]])
                    
                    if bond.GetBondType() == Chem.BondType.AROMATIC:
                        op_type = "form_aromatic_ring"
                    elif bond.GetBondType() == Chem.BondType.DOUBLE:
                        op_type = "form_double_ring"
                    elif bond.GetBondType() == Chem.BondType.TRIPLE:
                        op_type = "form_triple_ring"
                    else:
                        op_type = "form_ring"  # 默认单键环
                    
                    extra_bond_ops.append({
                        "position": f"{positions[0]}-{positions[1]}",
                        "atom": None,
                        "operation": op_type
                    })
            
            # 按照位置对额外键操作进行排序
            extra_bond_ops.sort(key=lambda x: x["position"])
            path += extra_bond_ops

            # --- 立体化学处理 ---
            stereo_ops = []
            chiral_centers = Chem.FindMolChiralCenters(self.mol, includeUnassigned=False)
            for center_idx, stereo in chiral_centers:
                if center_idx in self.backbone_map:
                    stereo_ops.append({
                        "position": str(self.backbone_map[center_idx]),
                        "atom": None,
                        "operation": "add_stereo"
                    })
            
            for bond in self.mol.GetBonds():
                if bond.GetStereo() > Chem.BondStereo.STEREOANY:
                    b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    if b in self.backbone_map and e in self.backbone_map:
                        positions = [self.backbone_map[b], self.backbone_map[e]]
                        positions.sort()  # 确保顺序一致
                        
                        stereo_ops.append({
                            "position": f"{positions[0]}-{positions[1]}",
                            "atom": None,
                            "operation": "add_stereo"
                        })
            
            # 按照位置对立体化学操作进行排序
            def sort_key(op_dict):
                # 分解位置字符串以进行正确的排序
                pos = op_dict["position"]
                if "-" in pos:
                    parts = pos.split("-")
                    return [int(p) for p in parts]
                else:
                    return [int(pos)]
                    
            stereo_ops.sort(key=lambda x: sort_key(x))
            path_result = path + stereo_ops
            
            # 返回完整路径，包括起始操作
            return path_result
            
        except Exception as e:
            return [{"position": "", "atom": None, "operation": "error"}]

    def _get_sorted_attachments(self, non_backbone_atoms):
        """识别所有附件并进行绝对规范的排序。"""
        if not non_backbone_atoms: return []
        
        full_adj = Chem.GetAdjacencyMatrix(self.mol)
        q = deque(non_backbone_atoms)
        visited = set()
        attachments = []
        
        while q:
            start_node = q.popleft()
            if start_node in visited: continue
            
            component = set()
            bfs_q = deque([start_node])
            visited.add(start_node)
            component.add(start_node)
            
            while bfs_q:
                u = bfs_q.popleft()
                for v_idx in self.mol.GetAtomWithIdx(u).GetNeighbors():
                    v = v_idx.GetIdx()
                    if v in non_backbone_atoms and v not in visited:
                        visited.add(v)
                        component.add(v)
                        bfs_q.append(v)

            attachments.append(Attachment(list(component), self.mol, self.backbone_map))
        
        attachments.sort(key=lambda x: x.sort_key)
        return attachments
    

class MolecularEvolutionExpansion:
    """多路径分子进化算法，支持生成多个不同的进化路径"""
    
    def __init__(self, initial_smiles: str = "CC", config_file: str = None):
        self.initial_smiles = initial_smiles
        self.initial_mol = Chem.MolFromSmiles(initial_smiles)
        
        # 初始化错误统计
        self.error_stats = {
            "valence_errors": 0,
            "kekulization_errors": 0,
            "other_errors": 0,
            "total_attempts": 0
        }
        
        # 从配置文件加载原子类型和操作类型
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # 加载原子类型 - 如果配置项不存在，将抛出KeyError异常
                self.common_atoms = config['atom_types']
                
                # 加载操作类型键列表 - 如果配置项不存在，将抛出KeyError异常
                operation_type_keys = config['operation_types']
                
                # 为操作类型创建带中文描述的字典
                operation_type_descriptions = {
                    "add_atom": "添加原子",
                    "replace_atom": "替换原子",
                    "form_double_bond": "形成双键",
                    "form_triple_bond": "形成三键",
                    "break_bond": "断开键",
                    "add_stereo": "添加立体化学",
                    "form_ring": "成环",
                    "form_double_ring": "形成双键环",
                    "form_triple_ring": "形成三键环",
                    "form_aromatic_ring": "形成芳香环",
                    "remove_add_stereo": "移除立体化学添加操作",
                    "remove_form_double_bond": "对形成双键的反向操作",
                    "remove_form_triple_bond": "对形成三键的反向操作",
                    "remove_form_ring": "对成环的反向操作",
                    "remove_form_double_ring": "对形成双键环的反向操作",
                    "remove_form_triple_ring": "对形成三键环的反向操作",
                    "remove_form_aromatic_ring": "对形成芳香环的反向操作"
                }
                
                # 根据配置文件中的操作类型键创建实际使用的操作类型字典
                self.operation_types = {}
                for op_type in operation_type_keys:
                    if op_type in operation_type_descriptions:
                        self.operation_types[op_type] = operation_type_descriptions[op_type]
                
                # 保存操作类型键列表，用于在其他方法中确定启用哪些操作
                self.operation_type_keys = operation_type_keys
                
            except Exception as e:
                print(f"[ERROR] 加载配置文件失败: {e}, 使用默认配置")
                # 如果配置文件加载失败，使用默认值
                self._load_default_config()
        else:
            # 没有提供配置文件时使用默认值
            self._load_default_config()
            print("[WARNING] 没有提供配置文件，使用默认配置")
            
        # 与数据集保持一致的操作类型映射，移除 add_fragment
        self.reverse_operations = {
            "replace_atom": "replace_atom",
            "add_atom": "remove_atom",
            "form_double_bond": "remove_form_double_bond",
            "form_triple_bond": "remove_form_triple_bond",
            "form_ring": "remove_form_ring",
            "form_double_ring": "remove_form_double_ring",
            "form_triple_ring": "remove_form_triple_ring",
            "form_aromatic_ring": "remove_form_aromatic_ring",
            "add_stereo": "remove_add_stereo"
        }
        
        # 输出当前可用的原子和操作类型
        print("[INFO] Available atoms:", self.common_atoms)
        print("[INFO] Available operations:")
        for op_key, op_desc in self.operation_types.items():
            print(f"  {op_key}: {op_desc}")
    
    def calculate_logP(self, smiles: str) -> Optional[float]:
        """计算分子的logP值
        
        Args:
            smiles: 分子的SMILES字符串
            
        Returns:
            logP值，如果计算失败则返回None
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Crippen.MolLogP(mol)
        except Exception as e:
            print(f"计算logP时出错 ({smiles}): {e}")
            return None
    
    def validate_molecule(self, mol: Chem.Mol) -> bool:
        """验证分子是否有效"""
        self.error_stats["total_attempts"] += 1
        try:
            if mol is None:
                self.error_stats["other_errors"] += 1
                return False
            
            # 检查分子是否可以被Kekulize（芳香性检查）
            temp_mol = Chem.Mol(mol)
            Chem.SanitizeMol(temp_mol)
            
            # 检查是否有原子
            if temp_mol.GetNumAtoms() == 0:
                self.error_stats["other_errors"] += 1
                return False
                
            # 检查分子是否有多个碎片（例如断开环后产生的多部分分子如CC.N）  # UPDATE 现在 dataset 中并不包含跳过 碎片分子 的进化案例。
            # 通过将分子转换为SMILES并检查是否包含点号来判断
            smiles = Chem.MolToSmiles(temp_mol)
            if '.' in smiles:
                self.error_stats["other_errors"] += 1
                return False
                
            # 尝试计算分子量来确保分子完整性
            Chem.rdMolDescriptors.CalcExactMolWt(temp_mol)
                
            return True
        except Exception as e:
            # 记录特定类型的错误
            error_msg = str(e).lower()
            if "explicit valence" in error_msg:
                self.error_stats["valence_errors"] += 1
            elif "kekulize" in error_msg or "aromatic" in error_msg:
                self.error_stats["kekulization_errors"] += 1
            else:
                self.error_stats["other_errors"] += 1
            return False
    
    def _get_all_possible_operations(self, mol: Chem.Mol) -> List[Dict]:
        """
        获取所有可能的操作，而不是基于权重的随机选择
        实现全量扩展树生成
        """
        operations = []
        
        # 添加原子操作 - 对每个原子都尝试添加新原子
        if "add_atom" in self.operation_type_keys:
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                for element in self.common_atoms:
                    operations.append({
                        "type": "add_atom",
                        "atom_symbol": element,
                        "atom_idx": atom_idx
                    })
        
        # 替换原子操作 - 对每个原子尝试替换为其他元素
        if "replace_atom" in self.operation_type_keys:
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                current_symbol = atom.GetSymbol()
                for element in self.common_atoms:
                    if element != current_symbol:
                        operations.append({
                            "type": "replace_atom",
                            "atom_symbol": element,
                            "atom_idx": atom_idx,
                        })
        
        # 成键操作 - 尝试在合适的原子间形成各种类型的键
        atoms = list(mol.GetAtoms())
        if "form_double_bond" in self.operation_type_keys:
            for i in range(len(atoms)):
                for j in range(i+1, len(atoms)):
                    atom_idx = atoms[i].GetIdx()   # 为方便解析，这里把 atom_idx 命名为 atom_idx
                    atom2_idx = atoms[j].GetIdx()
                    
                    # 检查是否已经存在键
                    bond = mol.GetBondBetweenAtoms(atom_idx, atom2_idx)
                    if bond is None:
                        # 可以形成双键
                        operations.append({
                            "type": "form_double_bond",
                            "atom_idx": atom_idx,
                            "atom2_idx": atom2_idx
                        })
                    elif bond.GetBondType() == Chem.BondType.SINGLE:
                        # 单键可以升级为双键
                        operations.append({
                            "type": "form_double_bond",
                            "atom_idx": atom_idx,
                            "atom2_idx": atom2_idx
                        })
        
        if "form_triple_bond" in self.operation_type_keys:
            for i in range(len(atoms)):
                for j in range(i+1, len(atoms)):
                    atom_idx = atoms[i].GetIdx()
                    atom2_idx = atoms[j].GetIdx()
                    
                    # 检查是否已经存在键
                    bond = mol.GetBondBetweenAtoms(atom_idx, atom2_idx)
                    if bond is None:
                        # 可以形成三键
                        operations.append({
                            "type": "form_triple_bond",
                            "atom_idx": atom_idx,
                            "atom2_idx": atom2_idx
                        })
                    elif bond.GetBondType() == Chem.BondType.SINGLE:
                        # 单键可以升级为三键
                        operations.append({
                            "type": "form_triple_bond",
                            "atom_idx": atom_idx,
                            "atom2_idx": atom2_idx
                        })
                    elif bond.GetBondType() == Chem.BondType.DOUBLE:
                        # 双键可以升级为三键
                        operations.append({
                            "type": "form_triple_bond",
                            "atom_idx": atom_idx,
                            "atom2_idx": atom2_idx
                        })
        
        # 成环操作 - 尝试在合适的原子间形成环
        # 这里简化处理，只考虑形成5元环和6元环的可能性
        if "form_ring" in self.operation_type_keys:
            for atom in mol.GetAtoms():
                # 简化处理，只添加示例操作
                operations.append({
                    "type": "form_ring",
                    "atom_idx": atom.GetIdx()
                })
        
        return operations

    def _apply_operation(self, mol: Chem.Mol, operation_type: str, **kwargs) -> Optional[Chem.Mol]:
        """应用指定的操作到分子上"""
        if operation_type == "add_atom" and "add_atom" in self.operation_type_keys:
            return self._add_atom_operation(mol, **kwargs)
        elif operation_type == "replace_atom" and "replace_atom" in self.operation_type_keys:
            return self._replace_atom_operation(mol, **kwargs)
        elif operation_type == "form_double_bond" and "form_double_bond" in self.operation_type_keys:
            return self._form_double_bond_operation(mol, **kwargs)
        elif operation_type == "form_triple_bond" and "form_triple_bond" in self.operation_type_keys:
            return self._form_triple_bond_operation(mol, **kwargs)
        # 注释掉断开键操作，因为这个操作不太可控
        # elif operation_type == "break_bond":
        #     return self._break_bond_operation(mol, **kwargs)
        elif operation_type == "add_stereo" and "add_stereo" in self.operation_type_keys:
            return self._add_stereo_operation(mol, **kwargs)
        elif operation_type == "form_ring" and "form_ring" in self.operation_type_keys:
            return self._form_ring_operation(mol, **kwargs)
        # 移除了 add_fragment 操作的处理
        else:
            # 其他操作保持不变
            return self._apply_other_operation(mol, operation_type, **kwargs)
    
    def _add_atom_operation(self, mol: Chem.Mol, atom_symbol: str = None, atom_idx: int = None) -> Optional[Chem.Mol]:
        """添加原子操作"""
        try:
            if atom_symbol is None:
                atom_symbol = random.choice(self.common_atoms)
            
            # 创建新原子
            new_atom = Chem.Atom(atom_symbol)
            new_mol = Chem.RWMol(mol)
            new_idx = new_mol.AddAtom(new_atom)
            
            # 如果指定了连接位置，则创建键
            if atom_idx is not None and atom_idx < new_mol.GetNumAtoms() - 1:
                new_mol.AddBond(atom_idx, new_idx, Chem.BondType.SINGLE)
            
            # 更新分子 - 内联 _sanitize_mol 方法的实现
            try:
                Chem.SanitizeMol(new_mol)
                return Chem.Mol(new_mol)
            except:
                return None
            
        except Exception as e:
            self._update_error_stats(e)
            return None

    def _apply_other_operation(self, mol: Chem.Mol, operation_type: str, **kwargs) -> Optional[Chem.Mol]:
        """应用其他操作"""
        try:
            if operation_type.startswith("remove_") and operation_type in self.operation_type_keys:
                # 移除类操作先执行移除，再执行后续操作
                if operation_type == "remove_add_stereo" and "remove_add_stereo" in self.operation_type_keys:
                    return mol
                elif operation_type == "remove_form_double_bond" and "remove_form_double_bond" in self.operation_type_keys:
                    return self._break_bond_operation(mol, **kwargs)
                elif operation_type == "remove_form_triple_bond" and "remove_form_triple_bond" in self.operation_type_keys:
                    return self._break_bond_operation(mol, **kwargs)
                elif operation_type == "remove_form_ring" and "remove_form_ring" in self.operation_type_keys:
                    return self._break_bond_operation(mol, **kwargs)
                elif operation_type == "remove_form_double_ring" and "remove_form_double_ring" in self.operation_type_keys:
                    return self._break_bond_operation(mol, **kwargs)
                elif operation_type == "remove_form_triple_ring" and "remove_form_triple_ring" in self.operation_type_keys:
                    return self._break_bond_operation(mol, **kwargs)
                elif operation_type == "remove_form_aromatic_ring" and "remove_form_aromatic_ring" in self.operation_type_keys:
                    return self._break_bond_operation(mol, **kwargs)
        except Exception as e:
            self._update_error_stats(e)
            return None
    
    def _replace_atom_operation(self, mol: Chem.Mol, atom_idx: int = None, atom_symbol: str = None) -> Optional[Chem.Mol]:
        """替换原子操作"""
        try:
            if atom_idx is None:
                atom_idx = random.randint(0, mol.GetNumAtoms() - 1)
            
            if atom_symbol is None:
                current_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                available_symbols = [a for a in self.common_atoms if a != current_symbol]
                if available_symbols:
                    atom_symbol = random.choice(available_symbols)
                else:
                    return None
            
            mol_copy = Chem.Mol(mol)
            atom = mol_copy.GetAtomWithIdx(atom_idx)
            atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(atom_symbol))
            
            if self.validate_molecule(mol_copy):
                return mol_copy
            else:
                return None
        except Exception as e:
            self._update_error_stats(e)
            return None
    
    def _form_bond_operation(self, mol: Chem.Mol, operation_type: str, atom_idx: int = None, atom2_idx: int = None) -> Optional[Chem.Mol]:
        """形成键操作"""
        try:
            # 获取可能的原子对
            possible_pairs = []
            for i in range(mol.GetNumAtoms()):
                for j in range(i + 1, mol.GetNumAtoms()):
                    if not mol.GetBondBetweenAtoms(i, j):
                        possible_pairs.append((i, j))
            
            if not possible_pairs:
                return None
            
            if atom_idx is None or atom2_idx is None:
                atom_idx, atom2_idx = random.choice(possible_pairs)
            
            # 检查是否已经存在键
            if mol.GetBondBetweenAtoms(atom_idx, atom2_idx):
                return None
            
            # 映射键类型
            bond_type_map = {
                "form_single_bond": Chem.BondType.SINGLE,
                "form_double_bond": Chem.BondType.DOUBLE,
                "form_triple_bond": Chem.BondType.TRIPLE,
                "form_ring": Chem.BondType.SINGLE,
                "form_double_ring": Chem.BondType.DOUBLE,
                "form_triple_ring": Chem.BondType.TRIPLE,
                "form_aromatic_ring": Chem.BondType.AROMATIC
            }
            
            bond_type = bond_type_map.get(operation_type, Chem.BondType.SINGLE)
            
            emol = Chem.EditableMol(mol)
            emol.AddBond(atom_idx, atom2_idx, bond_type)
            new_mol = emol.GetMol()
            
            if self.validate_molecule(new_mol):
                return new_mol
            else:
                return None
        except Exception as e:
            self._update_error_stats(e)
            return None
    
    def _break_bond_operation(self, mol: Chem.Mol, atom_idx: int = None, atom2_idx: int = None) -> Optional[Chem.Mol]:
        """断开键操作"""
        try:
            bonds = []
            for bond in mol.GetBonds():
                bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            
            if not bonds:
                return None
            
            if atom_idx is None or atom2_idx is None:
                atom_idx, atom2_idx = random.choice(bonds)
            
            if not mol.GetBondBetweenAtoms(atom_idx, atom2_idx):
                return None
            
            emol = Chem.EditableMol(mol)
            emol.RemoveBond(atom_idx, atom2_idx)
            new_mol = emol.GetMol()
            
            if self.validate_molecule(new_mol):
                return new_mol
            else:
                return None
        except Exception as e:
            self._update_error_stats(e)
            return None
    
    def _add_stereo_operation(self, mol: Chem.Mol, atom_idx: int = None, stereo: str = None) -> Optional[Chem.Mol]:
        """添加立体化学操作"""
        try:
            if atom_idx is None:
                atom_idx = random.randint(0, mol.GetNumAtoms() - 1)
            
            if stereo is None:
                stereo = random.choice(['R', 'S'])
            
            mol_copy = Chem.Mol(mol)
            atom = mol_copy.GetAtomWithIdx(atom_idx)
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW if stereo == 'R' else Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            
            if self.validate_molecule(mol_copy):
                return mol_copy
            else:
                return None
        except Exception as e:
            self._update_error_stats(e)
            return None
    
    def _form_double_bond_operation(self, mol: Chem.Mol, atom_idx: int = None, atom2_idx: int = None) -> Optional[Chem.Mol]:
        """形成双键操作"""
        return self._form_bond_operation(mol, "form_double_bond", atom_idx=atom_idx, atom2_idx=atom2_idx)
    
    def _form_triple_bond_operation(self, mol: Chem.Mol, atom_idx: int = None, atom2_idx: int = None) -> Optional[Chem.Mol]:
        """形成三键操作"""
        return self._form_bond_operation(mol, "form_triple_bond", atom_idx=atom_idx, atom2_idx=atom2_idx)
    
    def _form_ring_operation(self, mol: Chem.Mol, atom_idx: int = None, atom2_idx: int = None) -> Optional[Chem.Mol]:
        """形成环操作"""
        return self._form_bond_operation(mol, "form_ring", atom_idx=atom_idx, atom2_idx=atom2_idx)
    
    def _generate_single_path(self, steps: int, max_branching: int, 
                            diversity_threshold: float, seen_molecules: set) -> List[Dict]:
        """生成单条进化路径"""
        path = []
        current_mol = self.initial_mol
        current_smiles = self.initial_smiles
        
        # 记录初始状态
        if current_smiles not in seen_molecules:
            path.append({
                "step": 0,
                "smiles": current_smiles,
                "operation": "start",
                "details": {}
            })
            seen_molecules.add(current_smiles)
        
        for step in range(1, steps + 1):
            # 获取可能的操作
            possible_operations = self._get_possible_operations(current_mol)
            
            if not possible_operations:
                break
            
            # 根据多样性选择操作
            selected_operation = self._select_diverse_operation(
                possible_operations, current_mol, seen_molecules, diversity_threshold
            )
            
            if not selected_operation:
                # 如果没有找到合适的操作，尝试随机选择
                selected_operation = random.choice(possible_operations)
            
            # 应用操作
            operation_type = selected_operation["type"]
            operation_params = selected_operation.get("params", {})
            
            new_mol = self._apply_operation(current_mol, operation_type, **operation_params)
            
            if new_mol and self.validate_molecule(new_mol):
                new_smiles = Chem.MolToSmiles(new_mol)
                
                # 检查是否已经见过这个分子
                if new_smiles not in seen_molecules:
                    current_mol = new_mol
                    current_smiles = new_smiles
                    
                    path.append({
                        "step": step,
                        "smiles": current_smiles,
                        "operation": operation_type,
                        "details": operation_params
                    })
                    
                    seen_molecules.add(current_smiles)
                else:
                    # 如果分子已存在，跳过这一步
                    continue
            else:
                # 如果操作失败，跳过这一步
                continue
        
        return path
    
    # MARK
    def _get_possible_operations(self, mol: Chem.Mol) -> List[Dict]:
        """获取当前分子所有可能的操作（全量扩展，移除权重）
        
        支持的操作类型包括：
        - 添加操作：add_atom, replace_atom, form_double_bond, form_triple_bond, form_ring
        - 移除操作：remove_add_stereo, remove_form_double_bond, remove_form_double_ring, 
                   remove_form_ring, remove_form_triple_bond
        - 立体化学操作：add_stereo
        
        返回的操作列表与generate_path_dict方法生成的操作格式一致，支持配置文件中的所有操作类型。
        """
        operations = []
        
        # 1. 添加原子操作
        if "add_atom" in self.operation_type_keys:
            for atom_symbol in self.common_atoms:
                for atom_idx in range(mol.GetNumAtoms()):
                    operations.append({
                        "type": "add_atom",
                        "params": {"atom_symbol": atom_symbol, "atom_idx": atom_idx}
                    })
        
        # 2. 替换原子操作
        if "replace_atom" in self.operation_type_keys:
            for atom_idx in range(mol.GetNumAtoms()):
                current_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                for atom_symbol in self.common_atoms:
                    if atom_symbol != current_symbol:
                        operations.append({
                            "type": "replace_atom",
                            "params": {"atom_idx": atom_idx, "atom_symbol": atom_symbol}
                        })
        
        # 3. 键操作管理 - 正向操作（形成键）
        bond_formation_map = {
            "form_double_bond": {"type": "form_double_bond", "bond_type": Chem.BondType.DOUBLE},
            "form_triple_bond": {"type": "form_triple_bond", "bond_type": Chem.BondType.TRIPLE},
            "form_ring": {"type": "form_ring", "bond_type": Chem.BondType.SINGLE},
            "form_double_ring": {"type": "form_double_ring", "bond_type": Chem.BondType.DOUBLE},
            "form_aromatic_ring": {"type": "form_aromatic_ring", "bond_type": Chem.BondType.AROMATIC}
        }
        
        # 3.1 生成形成键操作（针对非键合原子对）
        for op_key, op_info in bond_formation_map.items():
            if op_key in self.operation_type_keys:
                for i in range(mol.GetNumAtoms()):
                    for j in range(i + 1, mol.GetNumAtoms()):
                        if not mol.GetBondBetweenAtoms(i, j):
                            operations.append({
                                "type": op_info["type"],
                                "params": {"atom_idx": i, "atom2_idx": j}
                            })
        
        # 4. 键操作管理 - 反向操作（移除键）
        bond_removal_map = {
            "remove_form_double_bond": {"type": "remove_form_double_bond", "bond_type": Chem.BondType.DOUBLE},
            "remove_form_triple_bond": {"type": "remove_form_triple_bond", "bond_type": Chem.BondType.TRIPLE},
            "remove_form_ring": {"type": "remove_form_ring", "bond_type": Chem.BondType.SINGLE},
            "remove_form_double_ring": {"type": "remove_form_double_ring", "bond_type": Chem.BondType.DOUBLE}
        }
        
        # 4.1 生成移除键操作（针对已存在的特定类型键）
        for op_key, op_info in bond_removal_map.items():
            if op_key in self.operation_type_keys:
                for bond in mol.GetBonds():
                    if bond.GetBondType() == op_info["bond_type"]:
                        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                        # 确保索引顺序一致
                        atom_idx, atom2_idx = sorted([i, j])
                        operations.append({
                            "type": op_info["type"],
                            "params": {"atom_idx": atom_idx, "atom2_idx": atom2_idx}
                        })
        
        # 5. 立体化学操作
        # 5.1 添加立体化学
        if "add_stereo" in self.operation_type_keys:
            operations.append({
                "type": "add_stereo",
                "params": {}
            })
        
        # 5.2 移除立体化学
        if "remove_add_stereo" in self.operation_type_keys:
            # 查找具有立体化学的中心
            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
            for center_idx, _ in chiral_centers:
                operations.append({
                    "type": "remove_add_stereo",
                    "params": {"atom_idx": center_idx}
                })
            
            # 查找具有立体化学的双键
            for bond in mol.GetBonds():
                if bond.GetStereo() > Chem.BondStereo.STEREOANY:
                    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    atom_idx, atom2_idx = sorted([i, j])
                    operations.append({
                        "type": "remove_add_stereo",
                        "params": {"atom_idx": atom_idx, "atom2_idx": atom2_idx}
                    })
        
        return operations
    
    def _select_diverse_operation(self, operations: List[Dict], current_mol: Chem.Mol, 
                                seen_molecules: set, diversity_threshold: float) -> Optional[Dict]:
        """选择能够产生多样化分子的操作"""
        diverse_operations = []
        
        for operation in operations:
            # 尝试应用操作
            test_mol = self._apply_operation(current_mol, operation["type"], **operation["params"])
            if test_mol and self.validate_molecule(test_mol):
                test_smiles = Chem.MolToSmiles(test_mol)
                
                # 如果这个分子还没见过，优先选择
                if test_smiles not in seen_molecules:
                    diverse_operations.append(operation)
        
        if diverse_operations:
            # 优先选择产生新分子的操作
            return random.choice(diverse_operations)
        else:
            # 如果没有新分子，随机选择
            return random.choice(operations) if operations else None
    
    # TAG API for outside call
    def generate_multiple_paths(self, num_paths: int = 10, steps_per_path: int = 5, 
                               max_branching: int = 3, diversity_threshold: float = 0.7) -> List[Dict]:
        """
        生成多个不同的进化路径
        
        Args:
            num_paths: 要生成的路径数量
            steps_per_path: 每条路径的步数
            max_branching: 每个节点的最大分支数
            diversity_threshold: 多样性阈值，控制路径间的差异
        """
        all_paths = []
        seen_molecules = set()
        
        # 添加tqdm进度条以显示路径生成进度
        for path_idx in tqdm(range(num_paths), desc="生成进化路径", unit="条"):
            path = self._generate_single_path(
                steps_per_path, 
                max_branching, 
                diversity_threshold,
                seen_molecules
            )
            if path:
                all_paths.append({
                    "path_id": path_idx,
                    "steps": path,
                    "initial_smiles": self.initial_smiles,
                    "final_smiles": path[-1]["smiles"] if path else self.initial_smiles,
                    "path_length": len(path)
                })
        
        return all_paths
    
    def generate_evolution_tree(self, max_depth: int = 5, max_branching: int = 3) -> Dict:
        """
        生成进化树，展示所有可能的进化路径
        
        Args:
            max_depth: 最大进化深度
            max_branching: 每个节点的最大分支数
        """
        tree = {
            "root": self.initial_smiles,
            "nodes": {},
            "edges": [],
            "error_stats": {}
        }
        
        # 显式重置错误统计
        self.error_stats = {
            "valence_errors": 0,
            "kekulization_errors": 0,
            "other_errors": 0,
            "total_attempts": 0
        }
        
        queue = deque([(self.initial_smiles, 0)])
        visited = {self.initial_smiles}
        
        while queue:
            current_smiles, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            current_mol = Chem.MolFromSmiles(current_smiles)
            if not current_mol:
                # 分子解析失败，计入其他错误
                self.error_stats["other_errors"] += 1
                self.error_stats["total_attempts"] += 1
                continue
            
            # 记录当前节点
            tree["nodes"][current_smiles] = {
                "depth": depth,
                "molecule": current_smiles
            }
            
            # 获取可能的操作
            operations = self._get_possible_operations(current_mol)
            random.shuffle(operations)
            
            branch_count = 0
            for operation in operations:
                if branch_count >= max_branching:
                    break
                
                # 应用操作并验证
                new_mol = self._apply_operation(current_mol, operation["type"], **operation["params"])
                if new_mol is not None and self.validate_molecule(new_mol):
                    new_smiles = Chem.MolToSmiles(new_mol)
                    
                    if new_smiles not in visited:
                        visited.add(new_smiles)
                        queue.append((new_smiles, depth + 1))
                        
                        # 记录边（演化步骤）
                        tree["edges"].append({
                            "from": current_smiles,
                            "to": new_smiles,
                            "operation": operation["type"],
                            "details": operation["params"],
                            "depth": depth + 1
                        })
                        
                        branch_count += 1
        
        # 将完整的错误统计信息添加到树中
        tree["error_stats"] = self.error_stats.copy()
        return tree
    
    def analyze_path_diversity(self, paths: List[Dict]) -> Dict:
        """分析路径多样性"""
        if not paths:
            return {"diversity_score": 0, "unique_molecules": 0}
        
        all_molecules = set()
        for path in paths:
            for step in path["steps"]:
                all_molecules.add(step["smiles"])
        
        total_possible = len(paths) * (len(paths[0]["steps"]) if paths else 0)
        diversity_score = len(all_molecules) / total_possible if total_possible > 0 else 0
        
        return {
            "diversity_score": diversity_score,
            "unique_molecules": len(all_molecules),
            "total_paths": len(paths),
            "average_path_length": sum(len(p["steps"]) for p in paths) / len(paths) if paths else 0
        }

    def generate_accumulative_paths(self, num_paths: int = 10, max_steps: int = 3) -> List[Dict]:
        """
        生成从初始分子开始的连续演化路径
        
        Args:
            num_paths: 要生成的路径数量
            max_steps: 最大步骤数
            
        Returns:
            包含所有演化步骤的列表
        """
        all_results = []
        
        # 添加tqdm进度条以显示路径生成进度
        for path_idx in tqdm(range(num_paths), desc="生成进化路径", unit="条"):
            current_mol = self.initial_mol
            current_smiles = self.initial_smiles
            
            # 逐步执行操作，直到达到最大步骤数
            for step in range(max_steps):
                # 获取可能的操作
                possible_operations = self._get_possible_operations(current_mol)
                if not possible_operations:
                    break
                
                # 随机选择一个操作
                selected_operation = random.choice(possible_operations)
                
                # 应用操作
                operation_type = selected_operation["type"]
                operation_params = selected_operation.get("params", {})
                
                new_mol = self._apply_operation(current_mol, operation_type, **operation_params)
                
                if new_mol and self.validate_molecule(new_mol):
                    new_smiles = Chem.MolToSmiles(new_mol)
                    
                    # 创建符合示例格式的结果
                    result_entry = {
                        "smiles_from": current_smiles,
                        "smiles_to": new_smiles,
                        "operations": [{
                            "position": str(operation_params.get("atom_idx", "")) if operation_type == "add_atom" 
                                      else str(operation_params.get("atom_idx", "")) if operation_type == "replace_atom"
                                      else f"{operation_params.get('atom_idx', '')}-{operation_params.get('atom2_idx', '')}" if "bond" in operation_type
                                      else "",
                            "atom": operation_params.get("atom_symbol", operation_params.get("atom_symbol", "")),
                            "operation": operation_type,
                            "from_atom": current_smiles
                        }]
                    }
                    all_results.append(result_entry)
                    
                    # 更新当前分子
                    current_mol = new_mol
                    current_smiles = new_smiles
                else:
                    # 如果操作失败，跳过这一步
                    continue
        
        return all_results

    # MARK
    def generate_expansion_tree(self, max_depth: int = 3, max_branching: int = 5, 
                              predictor=None, optimization_direction='increase', 
                              pruning_patience=3, initial_property_value=None, 
                              optimization_mode=None, logp_range=(0, 5), logp_patience=3) -> Dict:
        """
        生成从初始分子开始的扩展树，用于分子优化迭代，支持生成过程中的预测和剪枝
        
        Args:
            max_depth: 最大扩展深度
            max_branching: 每个节点的最大分支数
            predictor: 预测器对象，用于预测属性变化
            optimization_direction: 优化方向 ('increase' 或 'decrease')
            pruning_patience: 剪枝耐心值，连续多少代没有改善就剪枝
            initial_property_value: 初始分子的属性值
            optimization_mode: 优化模式
            logp_range: logP值的有效范围，默认(0, 5)
            logp_patience: logP剪枝耐心值，连续多少代logP超出范围就剪枝
            
        Returns:
            包含扩展树结构的字典
        """
        # 初始化扩展树
        expansion_tree = {
            "initial_smiles": self.initial_smiles,
            "max_depth": max_depth,
            "max_branching": max_branching,
            "nodes": {},
            "edges": []
        }
        
        # 使用队列进行广度优先搜索
        queue = deque[tuple[str, int, str]]([(self.initial_smiles, 0, "0")])  # (smiles, depth, parent_id) 根节点的parent_id设为"0"
        node_counter = 0
        
        # 初始化根节点
        root_node = {
            "id": str(node_counter),
            "smiles": self.initial_smiles,
            "depth": 0,
            "parent_id": None,
            "operation": None,
            "details": {}
        }
        
        # 计算并添加根节点的logP值
        root_logp = self.calculate_logP(self.initial_smiles)
        root_node["logP"] = root_logp
        # 记录logP是否在有效范围内
        root_node["logP_in_range"] = root_logp is not None and logp_range[0] <= root_logp <= logp_range[1]
        
        # 如果提供了预测器，为根节点添加属性值
        if predictor and initial_property_value is not None:
            root_node["property_value"] = initial_property_value
            root_node["property_change"] = 0.0
            root_node["accumulated_change"] = 0.0
        
        expansion_tree["nodes"][str(node_counter)] = root_node
        node_counter += 1
        
        seen_molecules = {self.initial_smiles}  # 存储已处理的分子
        
        while queue:
            current_smiles, depth, parent_id = queue.popleft()
            
            # 获取当前节点
            current_node = expansion_tree["nodes"].get(parent_id)
            if not current_node:
                continue
            
            # 达到最大深度时停止扩展
            if depth >= max_depth:
                continue
                
            # INFO 检查是否需要剪枝该分支
            if predictor and pruning_patience > 0:
                # 从当前节点回溯到最近的属性改善点
                should_prune = False
                stagnation_count = 0
                last_improvement_depth = depth
                
                # 计算该路径的连续未改善次数
                current_backtrack_id = parent_id
                while current_backtrack_id:
                    backtrack_node = expansion_tree["nodes"].get(current_backtrack_id)
                    if not backtrack_node:
                        break
                    
                    # 如果该节点的属性变化是改善的，重置停滞计数
                    if "property_change" in backtrack_node:
                        change = backtrack_node["property_change"]
                        if (optimization_direction == 'increase' and change > 0) or \
                           (optimization_direction == 'decrease' and change < 0):
                            last_improvement_depth = backtrack_node["depth"]
                            break
                    
                    # 向前回溯
                    current_backtrack_id = backtrack_node.get("parent_id")
                
                stagnation_count = depth - last_improvement_depth
                if stagnation_count >= pruning_patience:
                    print(f"剪枝分支，起始节点: {current_smiles}，连续未改善次数: {stagnation_count}")
                    continue
            
            # INFO 检查是否需要根据logP剪枝该分支
            if logp_patience > 0:
                # 从当前节点回溯到最近的logP在范围内的节点
                logp_out_range_count = 0
                last_valid_logp_depth = depth
                
                # 计算该路径的连续logP超出范围次数
                current_backtrack_id = parent_id
                while current_backtrack_id:
                    backtrack_node = expansion_tree["nodes"].get(current_backtrack_id)
                    if not backtrack_node:
                        break
                    
                    # 检查该节点的logP是否在范围内
                    if backtrack_node.get("logP_in_range", True):
                        last_valid_logp_depth = backtrack_node["depth"]
                        break
                    
                    # 向前回溯
                    current_backtrack_id = backtrack_node.get("parent_id")
                
                logp_out_range_count = depth - last_valid_logp_depth
                if logp_out_range_count >= logp_patience:
                    print(f"根据logP剪枝分支，起始节点: {current_smiles}，连续超出范围次数: {logp_out_range_count}")
                    continue
            
            # 解析当前分子
            current_mol = Chem.MolFromSmiles(current_smiles)
            if not current_mol:
                continue
                
            # 获取可能的操作
            possible_operations = self._get_possible_operations(current_mol)
            
            # 如果有预测器，先对每个可能的操作进行预测，然后根据预测结果排序
            if predictor:
                # 收集所有需要预测的操作
                # TODO 这里是否需要 batch-size 的设定来限制呢？
                batch_from_smiles = []
                batch_to_smiles = []
                batch_operations = []
                valid_operations = []
                
                for operation in possible_operations:
                    try:
                        # 应用操作生成新分子
                        operation_type = operation["type"]
                        operation_params = operation.get("params", {})
                        new_mol = self._apply_operation(current_mol, operation_type, **operation_params)
                        
                        # 检查新分子是否有效
                        if new_mol and self.validate_molecule(new_mol):
                            new_smiles = Chem.MolToSmiles(new_mol)
                            
                            # 添加到批量预测列表
                            batch_from_smiles.append(current_smiles)
                            batch_to_smiles.append(new_smiles)
                            batch_operations.append(operation)
                            valid_operations.append((operation, new_smiles))
                    except Exception as e:
                        # 记录错误但继续处理其他操作
                        continue
                
                # 进行批量预测
                operations_with_predictions = []
                if batch_from_smiles:
                    try:
                        # 调用批量预测方法
                        property_changes = predictor.predict_batch(batch_from_smiles, batch_to_smiles, batch_operations)
                        
                        # 过滤掉预测值为None的操作
                        for i, (operation, new_smiles) in enumerate(valid_operations):
                            if i < len(property_changes):
                                property_change = property_changes[i]
                                # 确保property_change是有效的数值类型
                                if property_change is not None and not (isinstance(property_change, list) or isinstance(property_change, dict)):
                                    operations_with_predictions.append((operation, new_smiles, property_change))
                    except Exception as e:
                        print(f"批量预测出错: {e}")
                        # 如果批量预测失败，回退到逐个预测
                        for operation, new_smiles in valid_operations:
                            try:
                                property_change = predictor.predict_property_change(current_smiles, new_smiles, operation)
                                if property_change is not None:
                                    operations_with_predictions.append((operation, new_smiles, property_change))
                            except Exception as e:
                                continue
                
                # 根据优化方向排序操作
                if optimization_direction == 'increase':
                    operations_with_predictions.sort(key=lambda x: x[2], reverse=True)
                else:
                    operations_with_predictions.sort(key=lambda x: x[2])
                
                # 限制分支数量
                branch_count = 0
                for operation, new_smiles, property_change in operations_with_predictions:
                    if branch_count >= max_branching:
                        break
                    
                    # 避免重复分子
                    if new_smiles not in seen_molecules:
                        seen_molecules.add(new_smiles)
                        
                        # 计算累计变化和新属性值
                        current_accumulated = current_node.get("accumulated_change", 0)
                        new_accumulated = current_accumulated + property_change
                        current_value = current_node.get("property_value", 0)
                        new_value = current_value + property_change
                        
                        # 计算新分子的logP值
                        new_logp = self.calculate_logP(new_smiles)
                        # 检查logP是否在有效范围内
                        logp_in_range = new_logp is not None and logp_range[0] <= new_logp <= logp_range[1]
                        
                        # 添加新节点
                        node_id = str(node_counter)
                        expansion_tree["nodes"][node_id] = {
                            "id": node_id,
                            "smiles": new_smiles,
                            "depth": depth + 1,
                            "parent_id": parent_id,
                            "operation": operation["type"],
                            "details": operation.get("params", {}),
                            "property_change": property_change,
                            "accumulated_change": new_accumulated,
                            "property_value": new_value,
                            "logP": new_logp,
                            "logP_in_range": logp_in_range
                        }
                        
                        # 添加边
                        expansion_tree["edges"].append({
                            "from": parent_id,
                            "to": node_id,
                            "operation": operation["type"],
                            "details": operation.get("params", {})
                        })
                        
                        # 添加到队列继续扩展
                        queue.append((new_smiles, depth + 1, node_id))
                        node_counter += 1
                        branch_count += 1
            else:
               raise NotImplementedError("[Molecular Evolver] 预测器未定义")
                        
        return expansion_tree
    
    def generate_sequential_expansions(self, num_steps: int = 5) -> List[Dict]:
        """
        生成从初始分子开始的顺序扩展路径，每一步都是在前一步基础上进行操作
        
        Args:
            num_steps: 要执行的扩展步骤数
            
        Returns:
            包含每一步扩展结果的列表
        """
        expansions = []
        current_mol = self.initial_mol
        current_smiles = self.initial_smiles
        
        # 添加初始状态
        expansions.append({
            "step": 0,
            "smiles_from": None,
            "smiles_to": current_smiles,
            "operation": "initial",
            "details": {}
        })
        
        # 逐步执行扩展
        for step in range(1, num_steps + 1):
            # 获取可能的操作
            possible_operations = self._get_possible_operations(current_mol)
            if not possible_operations:
                break
                
            # 随机选择一个操作
            selected_operation = random.choice(possible_operations)
            
            # 应用操作
            operation_type = selected_operation["type"]
            operation_params = selected_operation.get("params", {})
            
            new_mol = self._apply_operation(current_mol, operation_type, **operation_params)
            
            if new_mol and self.validate_molecule(new_mol):
                new_smiles = Chem.MolToSmiles(new_mol)
                
                # 记录扩展步骤
                expansions.append({
                    "step": step,
                    "smiles_from": current_smiles,
                    "smiles_to": new_smiles,
                    "operation": operation_type,
                    "details": operation_params
                })
                
                # 更新当前分子
                current_mol = new_mol
                current_smiles = new_smiles
            else:
                # 如果操作失败，跳过这一步
                continue
                
        return expansions

    def generate_paths_with_max_steps(self, num_paths: int = 10, max_steps: int = 3) -> List[Dict]:
        """
        生成多个进化路径，返回所有步骤数小于等于max_steps的中间结果
        
        Args:
            num_paths: 要生成的路径数量
            max_steps: 最大步骤数，返回所有步骤数小于等于此值的中间结果
            
        Returns:
            包含所有满足条件的步骤的列表，每个步骤包含SMILES和对应操作
        """
        all_results = []
        
        # 生成多条顺序扩展路径
        for path_idx in tqdm(range(num_paths), desc="生成扩展路径", unit="条"):
            expansions = self.generate_sequential_expansions(max_steps)
            
            # 转换为所需的输出格式
            for expansion in expansions[1:]:  # 跳过初始状态
                result_entry = {
                    "smiles_from": expansion["smiles_from"],
                    "smiles_to": expansion["smiles_to"],
                    "operations": [{
                        "position": str(expansion["details"].get("atom_idx", "")) if expansion["operation"] == "add_atom" 
                                  else str(expansion["details"].get("atom_idx", "")) if expansion["operation"] == "replace_atom"
                                  else f"{expansion['details'].get('atom_idx', '')}-{expansion['details'].get('atom2_idx', '')}" if "bond" in expansion["operation"]
                                  else "",
                        "atom": expansion["details"].get("atom_symbol", expansion["details"].get("atom_symbol", "")),
                        "operation": expansion["operation"],
                        "from_atom": expansion["smiles_from"]
                    }]
                }
                all_results.append(result_entry)
        
        return all_results

        # Error Handling
   
    # TAG Error Handling: 加载默认配置
    def _load_default_config(self):
        """加载默认的原子类型和操作类型配置"""
        # 操作类型定义 - 与数据集中操作类型保持一致，移除 add_fragment
        self.operation_types = {
            "add_atom": "添加原子",
            "replace_atom": "替换原子",
            "form_double_bond": "形成双键",
            "form_triple_bond": "形成三键",
            "break_bond": "断开键", # UPDATE 数据集中好像没有这种操作；
            "add_stereo": "添加立体化学",
            "form_ring": "成环",
            "form_double_ring": "形成双键环",
            "form_triple_ring": "形成三键环",
            "form_aromatic_ring": "形成芳香环",     # 10
            "remove_add_stereo": "移除并添加立体化学",
            "remove_form_double_bond": "移除双键",
            "remove_form_triple_bond": "移除三键",
            "remove_form_ring": "移除环",
            "remove_form_double_ring": "移除双键环",    # 15
            "remove_form_triple_ring": "移除三键环",
            "remove_form_aromatic_ring": "移除芳香环"
        }
        
        # 默认操作类型键列表
        self.operation_type_keys = [
            "add_atom", "replace_atom", "form_double_bond", "form_triple_bond", 
            "break_bond", "add_stereo", "form_ring", "form_double_ring", 
            "form_triple_ring", "form_aromatic_ring", "remove_add_stereo",
            "remove_form_double_bond", "remove_form_triple_bond", "remove_form_ring",
            "remove_form_double_ring", "remove_form_triple_ring", "remove_form_aromatic_ring"
        ]
        
        # 常见的原子类型和片段（移除了片段部分，因为我们不再使用 add_fragment）
        self.common_atoms = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        
        # 确保错误统计被正确初始化
        self.error_stats = {
            "valence_errors": 0,
            "kekulization_errors": 0,
            "other_errors": 0,
            "total_attempts": 0
        }
    
    # TAG Debug
    def _update_error_stats(self, e: Exception):
        """更新错误统计"""
        # 确保 error_stats 属性存在
        if not hasattr(self, 'error_stats'):
            self.error_stats = {
                "valence_errors": 0,
                "kekulization_errors": 0,
                "other_errors": 0,
                "total_attempts": 0
            }
        
        self.error_stats["total_attempts"] += 1
        error_msg = str(e).lower()
        if "explicit valence" in error_msg:
            self.error_stats["valence_errors"] += 1
        elif "kekulize" in error_msg or "aromatic" in error_msg:
            self.error_stats["kekulization_errors"] += 1
        else:
            self.error_stats["other_errors"] += 1
    
    def _print_tree_recursive(self, nodes: Dict, children_map: Dict, node_id: str, prefix: str, is_last: bool) -> None:
        """
        递归打印树形结构
        
        Args:
            nodes: 节点字典
            children_map: 子节点映射
            node_id: 当前节点ID
            prefix: 前缀字符串
            is_last: 是否为最后一个子节点
        """
        # 获取当前节点的子节点
        children = children_map.get(node_id, [])
        
        # 打印子节点
        for i, child_id in enumerate(children):
            if child_id not in nodes:
                continue
                
            child_node = nodes[child_id]
            is_last_child = (i == len(children) - 1)
            
            # 构造操作信息
            operation_info = f" [{child_node['operation']}]" if child_node['operation'] else ""
            
            # 构造变化值信息
            change_info = ""  
            if 'property_change' in child_node:
                change_info = f" (变化: {child_node['property_change']:.4f})"
            
            # 构造属性值和深度信息
            property_info = f", 属性值: {child_node['property_value']:.6f}" if 'property_value' in child_node else ""
            depth_info = f" (深度: {child_node['depth']})"
            
            # 打印当前子节点
            if is_last_child:
                print(f"{prefix}└── {child_node['smiles']}{operation_info}{change_info}{property_info}{depth_info}")
                new_prefix = f"{prefix}    "
            else:
                print(f"{prefix}├── {child_node['smiles']}{operation_info}{change_info}{property_info}{depth_info}")
                new_prefix = f"{prefix}│   "
            
            
            # 递归打印孙节点
            self._print_tree_recursive(nodes, children_map, child_id, new_prefix, is_last_child)

    def print_expansion_tree(self, tree_data: Dict, format: str = 'text') -> None:
        """
        以文件系统树的形式打印扩展树
        
        Args:
            tree_data: generate_expansion_tree方法生成的树形数据
            format: 输出格式 ('text' 或 'json')
        """
        if format == 'json':
            print(json.dumps(tree_data, ensure_ascii=False, indent=2))
            return
            
        # 以树形结构打印
        nodes = tree_data["nodes"]
        edges = tree_data["edges"]
        initial_smiles = tree_data["initial_smiles"]
        
        print(f"初始分子: {initial_smiles}")
        print(f"最大深度: {tree_data['max_depth']}")
        print(f"最大分支数: {tree_data['max_branching']}")
        print("\n扩展树结构:")
        
        # 打印根节点
        root_node = nodes["0"]
        property_info = f", 属性值: {root_node['property_value']:.6f}" if 'property_value' in root_node else ""
        print(f"└── {root_node['smiles']} (深度: {root_node['depth']}{property_info})")
        
        # 构建父子关系映射（基于边数据）
        children_map = {}
        for edge in edges:
            parent_id = edge["from"] if edge["from"] is not None else "0"  # 根节点的from为None
            child_id = edge["to"]
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(child_id)
        
        # 递归打印树形结构，从根节点的子节点开始
        self._print_tree_recursive(nodes, children_map, "0", "    ", True)

def main():
    parser = argparse.ArgumentParser(description='分子进化路径生成器')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # MoleculeEvolverAnalysis 命令
    evolver_parser = subparsers.add_parser('analyze', help='分析单个分子的进化路径')
    evolver_parser.add_argument('smiles', help='输入的SMILES字符串')
    evolver_parser.add_argument('--format', choices=['text', 'json', 'dict'], default='text', 
                              help='输出格式 (默认: text)')
    
    # MolecularEvolutionExpansion 命令
    expansion_parser = subparsers.add_parser('expand', help='生成多条分子进化路径')
    expansion_parser.add_argument('smiles', help='初始分子的SMILES字符串')
    expansion_parser.add_argument('--num-paths', type=int, default=5, help='要生成的路径数量')
    expansion_parser.add_argument('--steps-per-path', type=int, default=3, help='每条路径的步数')
    expansion_parser.add_argument('--max-steps', type=int, help='最大步骤数，返回所有中间步骤结果')
    expansion_parser.add_argument('--max-branching', type=int, default=3, help='每个节点的最大分支数')
    expansion_parser.add_argument('--diversity-threshold', type=float, default=0.7, help='多样性阈值')
    expansion_parser.add_argument('--config-file', type=str, help='配置文件路径')
    expansion_parser.add_argument('--mode', choices=['sequential', 'tree'], default='sequential', 
                                help='扩展模式: sequential(顺序) 或 tree(树形)')
    expansion_parser.add_argument('--format', choices=['text', 'json'], default='text',
                                help='输出格式 (默认: text)')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        try:
            evolver = MoleculeEvolverAnalysis(args.smiles)
            if args.format == 'text':
                path = evolver.generate_path()
                for step in path:
                    print(step)
            elif args.format == 'json':
                path = evolver.generate_path()
                print(json.dumps(path, ensure_ascii=False, indent=2))
            elif args.format == 'dict':
                path = evolver.generate_path_dict()
                print(json.dumps(path, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.command == 'expand':
        try:
            if args.config_file:
                expander = MolecularEvolutionExpansion(args.smiles, config_file=args.config_file)
            else:
                expander = MolecularEvolutionExpansion(args.smiles)
            
            if args.mode == 'tree':
                # 使用树形扩展模式
                paths = expander.generate_expansion_tree(
                    max_depth=args.max_steps or 3,
                    max_branching=args.max_branching
                )
            elif args.max_steps:
                # 使用顺序扩展模式
                paths = expander.generate_paths_with_max_steps(
                    num_paths=args.num_paths,
                    max_steps=args.max_steps
                )
            else:
                # 使用原有功能
                paths = expander.generate_multiple_paths(
                    num_paths=args.num_paths,
                    steps_per_path=args.steps_per_path,
                    max_branching=args.max_branching,
                    diversity_threshold=args.diversity_threshold
                )
            
            if args.format == 'text':
                if args.mode == 'tree':
                    # 树形模式的文本输出格式
                    expander.print_expansion_tree(paths, 'text')
                elif args.max_steps:
                    # 顺序扩展模式的文本输出格式
                    print(f"初始分子: {args.smiles}")
                    print(f"最大步骤数: {args.max_steps}")
                    print("顺序扩展步骤:")
                    for step in paths:
                        print(f"  {step['smiles_from']} -> {step['smiles_to']}: {step['operations'][0]['operation']}")
                else:
                    # 原有功能的文本输出格式
                    for i, path in enumerate(paths):
                        print(f"路径 {i+1}:")
                        print(f"  初始分子: {path['initial_smiles']}")
                        print(f"  最终分子: {path['final_smiles']}")
                        print(f"  路径长度: {path['path_length']}")
                        print("  步骤:")
                        for step in path['steps']:
                            print(f"    步骤 {step['step']}: {step['operation']} -> {step['smiles']}")
                        print()
            elif args.format == 'json':
                print(json.dumps(paths, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        parser.print_help()

# 当作为主模块运行时执行示例代码
if __name__ == "__main__":
    # 检查是否提供了命令行参数
    if len(sys.argv) > 1:
        # 如果有参数，使用CLI模式
        main()
    else:
        # 否则运行原有的示例代码
        # 创建多路径进化器
        multi_evolver = MolecularEvolutionExpansion("CC")
        
        print("初始分子: CC")
        print("生成多条进化路径...")
        
        # 生成10条不同的进化路径，每条5步
        paths = multi_evolver.generate_multiple_paths(
            num_paths=10, 
            steps_per_path=5,
            diversity_threshold=0.8
        )
        
        # 分析多样性
        diversity = multi_evolver.analyze_path_diversity(paths)
        print(f"\n多样性分析:")
        print(f"  多样性分数: {diversity['diversity_score']:.3f}")
        print(f"  唯一分子数: {diversity['unique_molecules']}")
        print(f"  总路径数: {diversity['total_paths']}")
        print(f"  平均路径长度: {diversity['average_path_length']:.2f}")
        
        # 显示前3条路径
        print(f"\n前3条进化路径:")
        for i, path in enumerate(paths[:3]):
            print(f"\n路径 {i + 1}:")
            print(f"  最终分子: {path['final_smiles']}")
            print(f"  路径长度: {path['path_length']} 步")
            
            for step in path["steps"][:3]:  # 显示前3步
                print(f"    步骤 {step['step']}: {step['operation']} -> {step['smiles']}")
            if len(path["steps"]) > 3:
                print(f"    ... 还有 {len(path['steps']) - 3} 步")
        
        # 生成进化树
        print(f"\n生成进化树...")
        evolution_tree = multi_evolver.generate_evolution_tree(max_depth=3, max_branching=2)
        print(f"进化树包含 {len(evolution_tree['nodes'])} 个节点和 {len(evolution_tree['edges'])} 条边")
