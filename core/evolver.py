#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强大的分子进化路径生成器，能够将任意SMILES解析为原子级的、可重现的构建序列。
"""

import os
import re
import argparse
import json
import sys
import numpy as np
from collections import deque

from rdkit import Chem
from rdkit.Chem import rdFMCS
import matplotlib.pyplot as plt
from rdkit.Chem import Draw

try:
    from .utils.molecule import smile_to_graph_xyz
    from .molecule_rebuilder import MoleculeRebuilder
except ImportError:
    from mol_evo.core.utils.molecule import smile_to_graph_xyz
    from mol_evo.core.molecule_rebuilder import MoleculeRebuilder

# 设置RDKit日志级别，减少警告输出
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# 修复导入问题
try:
    from .attachment import Attachment
except ImportError:
    from attachment import Attachment

class MoleculeEvolverAnalysis: # MoleculeEvolver
    """一个封装的分子路径生成器。"""
    def __init__(self, smiles: str):
        self.ori_smiles = smiles
        if not smiles or not isinstance(smiles, str):
            raise ValueError(f"无效的SMILES: {smiles}")
        self.mol = Chem.MolFromSmiles(smiles)
        if not self.mol:
            raise ValueError(f"无效的SMILES: {smiles}")
        # INFO 先处理为 rdkit规范化 SMILES，便于和 Rebuild 同步。
        self.smiles = Chem.MolToSmiles(self.mol)
        self.mol = Chem.MolFromSmiles(self.smiles)
        
        # INFO 标准化分子，确保环内键被正确处理为单键或双键
        self.mol = Chem.RemoveHs(self.mol)
        # 仅对非芳香环执行Kekulize，保留芳香环的芳香键类型
        if not self._is_aromatic_ring():
            Chem.Kekulize(self.mol)
        
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

        # 规则3: 按RDKit规范排名升序作为最终起点
        start_atom = min(atoms, key=lambda a: self.ranks[a.GetIdx()])
        
        return start_atom.GetIdx()

    def _build_canonical_backbone(self, start_idx: int):
        """从规范起点开始，通过规范DFS构建唯一的分子骨架（生成树）。"""
        self.backbone_indices = []
        visited = set()
        parent_map = {}
        
        def dfs(current_idx, parent_idx=None):
            visited.add(current_idx)
            self.backbone_indices.append(current_idx)
            
            if parent_idx is not None:
                parent_map[current_idx] = parent_idx
            
            # neighbors = sorted(self.mol.GetAtomWithIdx(current_idx).GetNeighbors(), key=lambda n: self.ranks[n.GetIdx()])
            neighbors = self.mol.GetAtomWithIdx(current_idx).GetNeighbors()     # INFO 使用 rdkit 的解析顺序
            for neighbor in neighbors:
                if neighbor.GetIdx() not in visited:
                    dfs(neighbor.GetIdx(), current_idx)
        
        dfs(start_idx)
        self.backbone_set = set(self.backbone_indices)
        self.backbone_map = {old_idx: new_idx for new_idx, old_idx in enumerate(self.backbone_indices)}
        self.parent_map = parent_map

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

    def _is_aromatic_ring(self) -> bool:
        """判断分子是否含有芳香环"""
        for bond in self.mol.GetBonds():
            if bond.GetIsAromatic():
                return True
        return False

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
            backbone_bonds = {tuple(sorted((atom_idx, self.parent_map[atom_idx]))) for atom_idx in self.parent_map}
            
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

    def get_full_path_dict(self) -> list:
        """
        获取完整的进化路径，包括起始操作。
        返回字典格式的操作列表，包含所有的分子构建步骤。
        """
        try:
            # start_idx = self._find_canonical_start_atom()
            self._build_canonical_backbone(0)   # INFO v1.6 调整，直接从 rdkit 规范化 st-idx 作为起点。
            
            # 检查骨架是否为空
            if not self.backbone_indices:
                return [{"position": "", "atom": None, "operation": "error"}]

            # --- STAGE 骨架路径 ---
            start_atom = self.mol.GetAtomWithIdx(self.backbone_indices[0])
            path = [{
                "position": str(self.backbone_map[start_atom.GetIdx()]),
                "atom": start_atom.GetSymbol(),
                "operation": "init_atom",
                "rdkit_idx": str(start_atom.GetIdx())  # 补充 RDKit 原始 idx
            }]
            
            for i in range(1, len(self.backbone_indices)):
                current_old_idx = self.backbone_indices[i]
                parent_old_idx = self.parent_map[current_old_idx]
                parent_new_idx = self.backbone_map[parent_old_idx]
                current_atom = self.mol.GetAtomWithIdx(current_old_idx)
                current_mol = self.mol.GetAtomWithIdx(current_old_idx)
                path.append({
                    "position": str(parent_new_idx),
                    "atom": current_atom.GetSymbol(),
                    "operation": "add_atom",
                    "rdkit_idx": str(current_mol.GetIdx())  # 补充 RDKit 原始 idx
                })

            # --- 附件、额外键和立体化学 ---
            non_backbone_atoms = [a.GetIdx() for a in self.mol.GetAtoms() if a.GetIdx() not in self.backbone_set]
            # 构建骨架键集合，只包含DFS遍历过程中形成的键（父子连接）
            # 这确保了即使是骨架上的环内键也会被正确识别为"额外键"
            backbone_bonds = {tuple(sorted((atom_idx, self.parent_map[atom_idx]))) for atom_idx in self.parent_map}
            
            # --- 附件处理 ---
            attachments = self._get_sorted_attachments(non_backbone_atoms)
            for att in attachments:
                conn_points = sorted(list(att.connection_points))
                # 附件操作需要记录连接点的 RDKit idx
                rdkit_conn_points = [str(idx) for idx in conn_points]
                path.append({
                    "position": str(conn_points[0]) if conn_points else "",
                    "atom": att.mol_frag_smiles,
                    "operation": "add_fragment",
                    "rdkit_idx": ",".join(rdkit_conn_points) if conn_points else ""  # 附件连接点的 RDKit idx
                })

            # --- STAGE 额外键处理 (成环、多重键) ---
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
                        
                    # 记录原始 RDKit idx
                    rdkit_positions = [str(b), str(e)]
                    rdkit_positions.sort()
                    extra_bond_ops.append({
                        "position": f"{positions[0]}-{positions[1]}",
                        "atom": None,
                        "operation": op_type,
                        "rdkit_idx": f"{rdkit_positions[0]}-{rdkit_positions[1]}"  # 额外键的 RDKit idx
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
                    
                    # 记录原始 RDKit idx
                    rdkit_positions = [str(b), str(e)]
                    rdkit_positions.sort()
                    extra_bond_ops.append({
                        "position": f"{positions[0]}-{positions[1]}",
                        "atom": None,
                        "operation": op_type,
                        "rdkit_idx": f"{rdkit_positions[0]}-{rdkit_positions[1]}"  # 成环键的 RDKit idx
                    })
            
            # 按照位置对额外键操作进行排序 # TODO ？
            extra_bond_ops.sort(key=lambda x: x["position"])
            
            # STAGE 合并连续的芳香键和芳香环操作
            merged_ops = []
            i = 0
            while i < len(extra_bond_ops):
                current_op = extra_bond_ops[i]
                
                # 检查是否是芳香键或芳香环操作
                if current_op["operation"] in ["form_aromatic_bond", "form_aromatic_ring"]:
                    # 收集所有连续的芳香键和芳香环操作
                    aromatic_ops = []
                    j = i
                    while j < len(extra_bond_ops):
                        if extra_bond_ops[j]["operation"] in ["form_aromatic_bond", "form_aromatic_ring"]:
                            aromatic_ops.append(extra_bond_ops[j])
                            j += 1
                        else:
                            break
                    
                    # 如果有多个芳香操作，检查是否满足合并条件
                    if len(aromatic_ops) >= 2:
                        # 提取所有涉及的原子位置
                        positions = []
                        for op in aromatic_ops:
                            pos_parts = op["position"].split("-")
                            positions.extend(pos_parts)
                        
                        # 去重并排序
                        unique_positions = sorted(list(set(positions)), key=int)
                        
                        # 生成位置字符串，格式为"0-1-2-3-4-5"
                        merged_position = "-".join(unique_positions)
                        
                        # 创建合并后的操作
                        merged_op = {
                            "position": merged_position,
                            "atom": None,
                            "operation": "form_aromatic_ring"
                        }
                        merged_ops.append(merged_op)
                        i = j  # 跳过已合并的所有操作
                    else:
                        # 只有一个芳香操作，直接添加
                        merged_ops.append(current_op)
                        i += 1
                else:
                    # 其他类型操作直接添加
                    merged_ops.append(current_op)
                    i += 1
            
            # 收集所有芳香操作和非芳香操作，用于判断是否需要第二次合并
            aromatic_ops_all = []  # 所有芳香操作
            non_aromatic_ops = []  # 所有非芳香操作
            for op in merged_ops:
                if op["operation"] in ["form_aromatic_bond", "form_aromatic_ring"]:
                    aromatic_ops_all.append(op)
                else:
                    non_aromatic_ops.append(op)
            
            # 只有在前面记录了多个芳香环时，才进行第二次合并处理
            if len(aromatic_ops_all) > 1:
                # 第二次合并处理：查找所有可能的芳香操作组合（不要求连续）
                second_merged_ops = []
                
                # 2. 如果没有芳香操作，直接返回
                if not aromatic_ops_all:
                    path += merged_ops
                    return path_result
                
                # 3. 如果只有一个芳香操作，直接添加
                if len(aromatic_ops_all) == 1:
                    second_merged_ops = merged_ops
                else:
                    # 4. 尝试将所有芳香操作合并
                    # 提取所有涉及的原子位置
                    positions = []
                    for op in aromatic_ops_all:
                        pos_parts = op["position"].split("-")
                        positions.extend(pos_parts)
                    
                    # 去重并排序
                    unique_positions = sorted(list(set(positions)), key=int)
                    total_atoms = len(unique_positions)
                    
                    # 检查是否满足合并条件：
                    # 1. 总原子数不超过6
                    # 2. 所有操作之间有足够的重叠（至少有一个重叠点）
                    if total_atoms <= 6:
                        # 检查所有操作是否有足够的重叠
                        # 首先获取所有操作的原子位置集合
                        op_positions = [set(op["position"].split("-")) for op in aromatic_ops_all]
                        
                        # 检查是否有共同的原子位置（至少2个重叠）
                        has_enough_overlap = True
                        
                        # 检查任意两个操作之间是否有至少2个重叠原子
                        for i in range(len(op_positions)):
                            for j in range(i + 1, len(op_positions)):
                                if len(op_positions[i].intersection(op_positions[j])) < 2:
                                    has_enough_overlap = False
                                    break
                            if not has_enough_overlap:
                                break
                        
                        if has_enough_overlap:
                            # 生成位置字符串，格式为"0-1-2-3-4-5"
                            merged_position = "-".join(unique_positions)
                            
                            # 创建合并后的操作
                            merged_op = {
                                "position": merged_position,
                                "atom": None,
                                "operation": "form_aromatic_ring"
                            }
                            
                            # 添加合并后的操作和所有非芳香操作
                            second_merged_ops = non_aromatic_ops + [merged_op]
                            # 排序以保持一致性
                            def sort_key(op_dict):
                                pos = op_dict["position"]
                                if "-" in pos:
                                    parts = pos.split("-")
                                    return [int(p) for p in parts]
                                else:
                                    return [int(pos)]
                            second_merged_ops.sort(key=lambda x: sort_key(x))
                        else:
                            # 不满足条件，保持原操作
                            second_merged_ops = merged_ops
                    else:
                        # 原子数超过6，不合并
                        second_merged_ops = merged_ops
                
                # 将第二次合并后的操作添加到路径
                path += second_merged_ops
            else:
                # 没有多个芳香环，直接使用合并后的操作
                path += merged_ops

            # --- STAGE 立体化学处理 ---
            stereo_ops = []
            for atom in self.mol.GetAtoms():
                center_idx = atom.GetIdx()
                chiral_tag = atom.GetChiralTag()
                if chiral_tag in [Chem.CHI_TETRAHEDRAL_CCW, Chem.CHI_TETRAHEDRAL_CW]:
                    if center_idx in self.backbone_map:
                        # 根据手性标签设置不同的操作
                        if chiral_tag == Chem.CHI_TETRAHEDRAL_CCW:
                            operation = "add_stereo_ccw"
                        else:  # CHI_TETRAHEDRAL_CW
                            operation = "add_stereo_cw"
                        stereo_ops.append({
                            "position": str(self.backbone_map[center_idx]),
                            "atom": None,
                            "operation": operation,
                            "rdkit_idx": str(center_idx)  # 手性中心的 RDKit idx
                        })
                        # print('😀 [Analysis]', center_idx, chiral_tag, atom.GetSymbol(), str(self.backbone_map[center_idx]))
            
            for bond in self.mol.GetBonds():
                if bond.GetStereo() > Chem.BondStereo.STEREOANY:
                    b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    if b in self.backbone_map and e in self.backbone_map:
                        positions = [self.backbone_map[b], self.backbone_map[e]]
                        positions.sort()  # 确保顺序一致
                        
                        # 记录立体键的原始 RDKit idx
                        rdkit_positions = [str(b), str(e)]
                        rdkit_positions.sort()
                        stereo_ops.append({
                            "position": f"{positions[0]}-{positions[1]}",
                            "atom": None,
                            "operation": "add_stereo",
                            "bond_stereo": bond.GetStereo(),  # 包含原始键立体构型信息   # UPDATE
                            "rdkit_idx": f"{rdkit_positions[0]}-{rdkit_positions[1]}"  # 立体键的 RDKit idx
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
        

class PairMoleculeEvolverAnalysis:
    """一个封装的分子对路径生成器，基于MCS进行处理。"""
    def __init__(self, smiles1: str, smiles2: str):
        self.smiles1 = smiles1
        self.smiles2 = smiles2
        
        # 使用 MoleculeEvolverAnalysis 来处理分子，复用其标准化逻辑
        self.analyzer1 = MoleculeEvolverAnalysis(smiles1)
        self.analyzer2 = MoleculeEvolverAnalysis(smiles2)
        
        # 获取处理后的分子对象
        self.mol1 = self.analyzer1.mol
        self.mol2 = self.analyzer2.mol
        
        # 计算MCS
        self.mcs_result = self._calculate_mcs()
        self.mcs_mol = Chem.MolFromSmarts(self.mcs_result.smartsString) if self.mcs_result.smartsString else None
        
        # 原子映射
        self.atom_map1 = {}  # mol1的原子idx -> mcs中的idx
        self.atom_map2 = {}  # mol2的原子idx -> mcs中的idx
        
        if self.mcs_mol:
            self._find_mcs_mapping()
        else:
            print("⚠️ 警告: 两个分子没有公共结构")
    
    def _calculate_mcs(self):
        """计算两个分子之间的最大公共子结构(MCS)"""
        mcs_params = rdFMCS.MCSParameters()
        mcs_params.BondCompareParameters.BondCompare = rdFMCS.BondCompare.CompareOrder
        mcs_params.AtomCompareParameters.AtomCompare = rdFMCS.AtomCompare.CompareElements
        mcs_params.Timeout = 10
        mcs_params.Threshold = 0.8
        
        return rdFMCS.FindMCS([self.mol1, self.mol2], mcs_params)
    
    def _find_mcs_mapping(self):
        """找到MCS在两个分子中的原子映射"""
        if not self.mcs_mol:
            return
        
        # 找到mol1中匹配MCS的原子
        matches1 = self.mol1.GetSubstructMatches(self.mcs_mol)
        if matches1:
            match1 = matches1[0]
            self.atom_map1 = {atom_idx: mcs_idx for mcs_idx, atom_idx in enumerate(match1)}
        
        # 找到mol2中匹配MCS的原子
        matches2 = self.mol2.GetSubstructMatches(self.mcs_mol)
        if matches2:
            match2 = matches2[0]
            self.atom_map2 = {atom_idx: mcs_idx for mcs_idx, atom_idx in enumerate(match2)}
    
    def get_mcs_based_positioning(self):
        """获取基于MCS的position定位信息"""
        positioning_info = {
            "mcs_size": self.mcs_result.numAtoms if self.mcs_result else 0,
            "mcs_bonds": self.mcs_result.numBonds if self.mcs_result else 0,
            "atom_map1": self.atom_map1,
            "atom_map2": self.atom_map2
        }
        
        return positioning_info

    def _filter_non_mcs_path(self, full_path, atom_map):
        """过滤路径，只保留非MCS部分的路径
        
        Args:
            full_path: 完整的evo-path
            atom_map: MCS的原子映射（mol_idx -> mcs_idx）
            
        Returns:
            过滤后的路径，只包含非MCS部分的操作
        """
        if not atom_map:
            return full_path
        
        # MCS中的原子idx集合
        mcs_atom_indices = set(atom_map.keys())
        
        filtered_path = []
        
        for step in full_path:
            operation = step.get("operation", "")
            rdkit_idx = step.get("rdkit_idx")
            position = step.get("position", "")
            
            # 处理 add_fragment 操作
            if operation == "add_fragment" and rdkit_idx:
                # rdkit_idx 可能是逗号分隔的多个连接点
                conn_indices = [int(idx) for idx in rdkit_idx.split(",") if idx]
                # 如果至少有一个连接点不在MCS中，则保留该操作
                if not all(idx in mcs_atom_indices for idx in conn_indices):
                    filtered_path.append(step)
            # 处理其他操作（init_atom, add_atom等）
            elif rdkit_idx is not None and rdkit_idx:
                # 检查是否是单个原子idx
                try:
                    idx = int(rdkit_idx)
                    # 如果该原子不在MCS中，则保留
                    if idx not in mcs_atom_indices:
                        filtered_path.append(step)
                except ValueError:
                    # 处理可能是键操作的格式（如 "1-2"）
                    if "-" in rdkit_idx:
                        bond_indices = [int(idx) for idx in rdkit_idx.split("-")]
                        # 如果键的至少一个端点不在MCS中，则保留
                        if not all(idx in mcs_atom_indices for idx in bond_indices):
                            filtered_path.append(step)
                    else:
                        # 其他格式，暂时保留
                        filtered_path.append(step)
            # 处理 rdkit_idx 为空但 position 存在的情况（如合并后的芳香环操作）
            elif position and not rdkit_idx:
                # position 可能是 "0-1-2-3-4-5" 这样的格式
                if "-" in position:
                    pos_indices = [int(idx) for idx in position.split("-") if idx]
                    # 如果至少有一个原子不在MCS中，则保留该操作
                    if not all(idx in mcs_atom_indices for idx in pos_indices):
                        filtered_path.append(step)
                else:
                    # 单个位置
                    try:
                        pos_idx = int(position)
                        if pos_idx not in mcs_atom_indices:
                            filtered_path.append(step)
                    except ValueError:
                        # 无法解析的位置，暂时保留
                        filtered_path.append(step)
            else:
                # 对于没有rdkit_idx和position的操作，暂时保留
                # 可能需要根据具体操作类型做更精细的处理
                filtered_path.append(step)
        
        return filtered_path
    
    def _convert_path_to_mcs_position(self, path, backbone_map, mol_to_mcs_map):
        """将路径中的 position 从 backbone position 转换为 MCS position
        
        Args:
            path: 需要转换的路径
            backbone_map: backbone 映射 {rdkit_idx: backbone_position}
            mol_to_mcs_map: MCS 映射 {rdkit_idx: mcs_position}
            
        Returns:
            转换后的路径，新增 mcs_position 字段存储转换后的 MCS position
        """
        if not mol_to_mcs_map or not backbone_map:
            return path
        
        converted_path = []
        
        for step in path:
            converted_step = step.copy()
            rdkit_idx = step.get("rdkit_idx")
            operation = step.get("operation", "")
            
            # 处理不同类型的操作
            if operation == "add_fragment" and rdkit_idx:
                # 处理片段添加操作，可能有多个连接点
                conn_indices = [int(idx) for idx in rdkit_idx.split(",") if idx]
                mcs_positions = []
                for idx in conn_indices:
                    if str(idx) in mol_to_mcs_map:
                        mcs_positions.append(str(mol_to_mcs_map[str(idx)]))
                    else:
                        # 如果不在 MCS 中，保持原样或标记为非MCS
                        mcs_positions.append(f"non_mcs_{idx}")
                
                if mcs_positions:
                    converted_step["mcs_position"] = ",".join(mcs_positions)
            
            elif rdkit_idx and "-" not in str(rdkit_idx):
                # 处理单个原子操作
                try:
                    idx = int(rdkit_idx)
                    if str(idx) in mol_to_mcs_map:
                        converted_step["mcs_position"] = str(mol_to_mcs_map[str(idx)])
                    else:
                        # 如果不在 MCS 中，标记为非MCS
                        converted_step["mcs_position"] = f"non_mcs_{idx}"
                except ValueError:
                    pass
            
            elif rdkit_idx and "-" in str(rdkit_idx):
                # 处理键操作（如 "1-2"）
                bond_indices = [int(idx) for idx in rdkit_idx.split("-")]
                mcs_positions = []
                for idx in bond_indices:
                    if str(idx) in mol_to_mcs_map:
                        mcs_positions.append(str(mol_to_mcs_map[str(idx)]))
                    else:
                        mcs_positions.append(f"non_mcs_{idx}")
                
                if mcs_positions:
                    converted_step["mcs_position"] = "-".join(mcs_positions)
            
            elif not rdkit_idx and step.get("position"):
                # 处理没有 rdkit_idx 但有 position 的情况（如合并后的芳香环操作）
                position = step.get("position", "")
                if "-" in position:
                    # position 是基于 backbone 的，需要先转换为 rdkit_idx，再转换为 MCS position
                    pos_indices = [int(idx) for idx in position.split("-") if idx]
                    mcs_positions = []
                    for pos_idx in pos_indices:
                        # 找到对应的 rdkit_idx
                        rdkit_idx = None
                        for rdkit, backbone_pos in backbone_map.items():
                            if backbone_pos == pos_idx:
                                rdkit_idx = rdkit
                                break
                        
                        if rdkit_idx is not None and str(rdkit_idx) in mol_to_mcs_map:
                            mcs_positions.append(str(mol_to_mcs_map[str(rdkit_idx)]))
                        else:
                            mcs_positions.append(f"non_mcs_{pos_idx}")
                    
                    if mcs_positions:
                        converted_step["mcs_position"] = "-".join(mcs_positions)
                else:
                    # 单个位置
                    try:
                        pos_idx = int(position)
                        # 找到对应的 rdkit_idx
                        rdkit_idx = None
                        for rdkit, backbone_pos in backbone_map.items():
                            if backbone_pos == pos_idx:
                                rdkit_idx = rdkit
                                break
                        
                        if rdkit_idx is not None and str(rdkit_idx) in mol_to_mcs_map:
                            converted_step["mcs_position"] = str(mol_to_mcs_map[str(rdkit_idx)])
                    except ValueError:
                        pass
            
            converted_path.append(converted_step)
        
        return converted_path
    
    def generate_combined_path(self):
        """CORE: 生成基于MCS的组合进化路径
        
        新的逻辑：
        1. 分别获取 mol1 和 mol2 的完整 evo-path
        2. 根据 MCS 的 atom_map 和 path 中的 rdkit_idx 进行过滤
        3. 将 mol2_non_mcs_path 的 position 从 mol2 backbone position 转换为 MCS position
        4. 这样就能得到离散片段之间的变化关系，并保持与MCS的关系
        """
        try:
            # 直接使用已有的 analyzer 获取完整 evo-path
            mol1_full_path = self.analyzer1.get_full_path_dict()
            mol2_full_path = self.analyzer2.get_full_path_dict()
            
            # 根据 MCS 的 atom_map 过滤路径，保留非 MCS 部分
            mol1_non_mcs_path = self._filter_non_mcs_path(mol1_full_path, self.atom_map1)
            mol2_non_mcs_path = self._filter_non_mcs_path(mol2_full_path, self.atom_map2)
            
            # 将 mol2_non_mcs_path 的 position 从 mol2 backbone position 转换为 MCS position
            mol2_non_mcs_path_converted = self._convert_path_to_mcs_position(
                mol2_non_mcs_path,
                self.analyzer2.backbone_map,
                self.atom_map2
            )
            
            # 组合路径
            combined_path = []

            combined_path.append({
                "section": "mol1",
                "smiles": self.smiles1,
                "path": mol1_full_path,
                "path_non_mcs": mol1_non_mcs_path,
                "mol_to_mcs_map": self.atom_map1
            })

            combined_path.append({
                "section": "mol2",
                "smiles": self.smiles2,
                "path": mol2_full_path,
                "path_non_mcs": mol2_non_mcs_path_converted,
                "mol_to_mcs_map": self.atom_map2
            })

            # 添加 MCS 信息，便于理解片段与MCS的关系
            combined_path.append({
                "section": "mcs_info",
                "mcs_smarts": self.mcs_result.smartsString,
                "mcs_size": self.mcs_result.numAtoms,
                "mcs_bonds": self.mcs_result.numBonds
            })

            return combined_path
            
        except Exception as e:
            return [{"section": "error", "path": [f"路径生成错误: {str(e)}"]}]
    

def main():
    """分子进化路径生成器的命令行接口"""
    parser = argparse.ArgumentParser(description='分子进化路径生成器')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # MoleculeEvolverAnalysis 命令
    evolver_parser = subparsers.add_parser('analyze', help='分析单个分子的进化路径')
    evolver_parser.add_argument('smiles', help='输入的SMILES字符串')
    evolver_parser.add_argument('--format', choices=['text', 'json', 'dict'], default='text', 
                              help='输出格式 (默认: text)')
    
    # PairMoleculeEvolverAnalysis 命令
    pair_parser = subparsers.add_parser('analyze-pair', help='分析两个分子的进化路径，基于MCS')
    pair_parser.add_argument('smiles1', help='第一个SMILES字符串')
    pair_parser.add_argument('smiles2', help='第二个SMILES字符串')
    pair_parser.add_argument('--format', choices=['text', 'json', 'dict'], default='json', 
                           help='输出格式 (默认: json)')
    
    # Rebuild 命令
    rebuild_parser = subparsers.add_parser('rebuild', help='根据路径重建分子')
    rebuild_parser.add_argument('path_file', help='包含路径的JSON文件')
    rebuild_parser.add_argument('--save-images', action='store_true', help='保存每一步的图像')
    rebuild_parser.add_argument('--output-dir', default='./rebuild_steps', help='图像保存目录')
    
    # `Test 命令
    test_parser = subparsers.add_parser('test', help='测试拆分和重建功能')
    test_parser.add_argument('smiles', help='要测试的SMILES字符串')
    test_parser.add_argument('--save-images', action='store_true', help='保存每一步的图像')
    test_parser.add_argument('--output-dir', default='./test_steps', help='图像保存目录')

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
    
    elif args.command == 'rebuild':
        try:
            with open(args.path_file, 'r', encoding='utf-8') as f:
                path_data = json.load(f)
            
            if isinstance(path_data, dict):
                path = path_data.get('smiles_from_path', [])
            else:
                path = path_data
            
            rebuilder = MoleculeRebuilder(path)
            mol = rebuilder.rebuild(visualize=True, save_images=args.save_images, output_dir=args.output_dir)
            rebuilder.print_summary()
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.command == 'analyze-pair':
        try:
            pair_evolver = PairMoleculeEvolverAnalysis(args.smiles1, args.smiles2)
            combined_path = pair_evolver.generate_combined_path()
            positioning_info = pair_evolver.get_mcs_based_positioning()
            
            result = {
                "mcs_info": positioning_info,
                "combined_path": combined_path
            }
            
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elif args.command == 'test':
        try:
            print(f"\n测试 SMILES: {args.smiles}")
            print(f"{'='*60}\n")
            
            # 步骤1: 拆分
            print("步骤 1: 拆分分子...")
            evolver = MoleculeEvolverAnalysis(args.smiles)
            path = evolver.get_full_path_dict()
            
            print(f"生成的路径包含 {len(path)} 个操作")
            print("\n路径详情:")
            print(json.dumps(path, ensure_ascii=False, indent=2))
            
            # 步骤2: 重建
            print(f"\n{'='*60}")
            print("步骤 2: 根据路径重建分子...")
            print(f"{'='*60}\n")
            
            rebuilder = MoleculeRebuilder(path)
            mol = rebuilder.rebuild(visualize=True, save_images=args.save_images, output_dir=args.output_dir)
            
            # 验证
            if mol:
                rebuilt_smiles = Chem.MolToSmiles(mol)
                original_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(args.smiles))
                
                print(f"\n{'='*60}")
                print("验证结果")
                print(f"{'='*60}")
                print(f"原始 SMILES: {original_smiles}")
                print(f"重建 SMILES: {rebuilt_smiles}")
                
                if rebuilt_smiles == original_smiles:
                    print("✓ 重建成功！SMILES 完全匹配")
                else:
                    print("⚠ SMILES 不匹配，但可能代表同一分子")
                    print(f"  (不同的 SMILES 表示)")
                
                rebuilder.print_summary()
            else:
                print("✗ 重建失败")
                sys.exit(1)
                
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    else:
        parser.print_help()


# 当作为主模块运行时执行
if __name__ == "__main__":
    main()
