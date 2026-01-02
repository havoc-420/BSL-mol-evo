#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强大的分子进化路径生成器，能够将任意SMILES解析为原子级的、可重现的构建序列。
"""

from rdkit import Chem
from collections import deque
import re
import argparse
import json
import sys
import numpy as np
from rdkit.Chem import rdDepictor, AllChem, rdchem
from rdkit.Chem.rdmolfiles import MolToSmiles
import matplotlib.pyplot as plt
from rdkit.Chem import Draw
import os

try:
    from .utils.molecule import smile_to_graph_xyz
except ImportError:
    from mol_evo.core.utils.molecule import smile_to_graph_xyz

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
                "operation": "init_atom"
            }]
            
            for i in range(1, len(self.backbone_indices)):
                current_old_idx = self.backbone_indices[i]
                parent_old_idx = self.parent_map[current_old_idx]
                parent_new_idx = self.backbone_map[parent_old_idx]
                current_atom = self.mol.GetAtomWithIdx(current_old_idx)
                path.append({
                    "position": str(parent_new_idx),
                    "atom": current_atom.GetSymbol(),
                    "operation": "add_atom"
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
                path.append({
                    "position": str(conn_points[0]) if conn_points else "",
                    "atom": att.mol_frag_smiles,
                    "operation": "add_fragment"
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
            
            # 合并连续的芳香键和芳香环操作
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
                    
                    # 如果有多个芳香操作，合并为一个成苯环操作
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
            
            # 将合并后的操作添加到路径
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
                        })
                        print('😀 [Analysis]', center_idx, chiral_tag, atom.GetSymbol(), str(self.backbone_map[center_idx]))
            
            for bond in self.mol.GetBonds():
                if bond.GetStereo() > Chem.BondStereo.STEREOANY:
                    b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    if b in self.backbone_map and e in self.backbone_map:
                        positions = [self.backbone_map[b], self.backbone_map[e]]
                        positions.sort()  # 确保顺序一致
                        
                        stereo_ops.append({
                            "position": f"{positions[0]}-{positions[1]}",
                            "atom": None,
                            "operation": "add_stereo",
                            "bond_stereo": bond.GetStereo()  # 包含原始键立体构型信息   # UPDATE
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


class MoleculeRebuilder:
    """
    分子重建器，能够根据操作路径逐步重建分子并可视化每一步的状态。
    """
    
    def __init__(self, path: list, types: dict = None):
        self.path = path
        self.steps = []
        self.current_mol = None
        self.atom_map = {}
        self.atom_counter = 0
        self.types = types or self._get_default_types()
    
    def _get_default_types(self):
        """获取默认的原子类型映射"""
        return {
            'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 'S': 6, 'Cl': 7,
            'Br': 8, 'I': 9, 'B': 10, 'Si': 11, 'Se': 12, 'As': 13, 'Te': 14
        }
    
    def rebuild_step_by_step(self, analyze_xyz: bool = False):
        """
        逐步执行路径操作，记录每一步的分子状态。
        返回步骤列表，每个步骤包含操作信息和分子SMILES。
        
        Args:
            analyze_xyz: 是否使用 smile_to_graph_xyz 分析 z 和 pos 参数
        """        
        # 初始化空分子
        self.current_mol = rdchem.RWMol()
        self.atom_map = {}
        self.atom_counter = 0
        self.steps = []
        
        for step_idx, operation in enumerate(self.path):
            step_info = {
                "step": step_idx + 1,
                "operation": operation,
                "smiles_before": self._get_current_smiles(),
                "description": self._describe_operation(operation)
            }
            
            # 执行操作
            self._execute_operation(operation)
            
            # 记录操作后的状态
            step_info["smiles_after"] = self._get_current_smiles()
            
            # 如果需要，分析 z 和 pos 参数
            if analyze_xyz:
                step_info["xyz_analysis"] = self._analyze_xyz()
            
            self.steps.append(step_info)
        
        return self.steps
    
    def _analyze_xyz(self):
        """
        分析当前分子的 z 和 pos 参数。
        优先尝试通过 SMILES 解析，如果失败则直接从 RWMol 对象提取原子信息。
        
        Returns:
            dict: 包含分析结果的字典
        """
        try:
            smiles = self._get_current_smiles()
            
            if smiles != "invalid":
                x, z, pos, edge_index, edge_attr = smile_to_graph_xyz(smiles, self.types)
                
                if z is not None and pos is not None:
                    return {
                        "success": True,
                        "z": z.tolist() if hasattr(z, 'tolist') else z,
                        "pos": pos.tolist() if hasattr(pos, 'tolist') else pos,
                        "num_atoms": len(z),
                        "edge_index": edge_index.tolist() if edge_index is not None else None,
                        "edge_attr": edge_attr.tolist() if edge_attr is not None else None,
                        "method": "smiles_based"
                    }
            
            return self._analyze_from_rwmol()
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "z": None,
                "pos": None
            }
    
    def _analyze_from_rwmol(self):
        """
        直接从 RWMol 对象提取原子信息和键信息，不依赖 SMILES。
        对于不完整的分子结构，可以提取基本的原子序数和连接关系。
        
        Returns:
            dict: 包含分析结果的字典
        """
        try:
            mol = self.current_mol.GetMol()
            num_atoms = mol.GetNumAtoms()
            
            if num_atoms == 0:
                return {
                    "success": False,
                    "error": "No atoms in molecule",
                    "z": None,
                    "pos": None
                }
            
            z = []
            for atom in mol.GetAtoms():
                z.append(atom.GetAtomicNum())
            
            z = np.array(z, dtype=np.int64)
            
            edge_index = []
            edge_attr = []
            
            for bond in mol.GetBonds():
                start_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                
                edge_index.append([start_idx, end_idx])
                edge_index.append([end_idx, start_idx])
                
                bond_type = bond.GetBondType()
                bond_type_map = {
                    Chem.BondType.SINGLE: [1, 0, 0, 0],
                    Chem.BondType.DOUBLE: [0, 1, 0, 0],
                    Chem.BondType.TRIPLE: [0, 0, 1, 0],
                    Chem.BondType.AROMATIC: [0, 0, 0, 1],
                }
                bond_attr = bond_type_map.get(bond_type, [0, 0, 0, 0])
                edge_attr.append(bond_attr)
                edge_attr.append(bond_attr)
            
            if edge_index:
                edge_index = np.array(edge_index, dtype=np.int64).T
                edge_attr = np.array(edge_attr, dtype=np.float32)
            else:
                edge_index = None
                edge_attr = None
            
            try:
                mol_with_h = Chem.AddHs(mol)
                conf_id = AllChem.EmbedMolecule(mol_with_h, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
                
                if conf_id != -1:
                    conf = mol_with_h.GetConformer()
                    pos = conf.GetPositions()
                    pos = pos[:num_atoms]
                else:
                    pos = self._generate_simple_coordinates(mol)
            except:
                pos = self._generate_simple_coordinates(mol)
            
            x = self._generate_node_features(z, edge_index, edge_attr)
            
            return {
                "success": True,
                "z": z.tolist() if hasattr(z, 'tolist') else z,
                "pos": pos.tolist() if hasattr(pos, 'tolist') else pos,
                "num_atoms": len(z),
                "edge_index": edge_index.tolist() if edge_index is not None else None,
                "edge_attr": edge_attr.tolist() if edge_attr is not None else None,
                "method": "rwmol_based"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"RWMol analysis failed: {str(e)}",
                "z": None,
                "pos": None
            }
    
    def _generate_simple_coordinates(self, mol):
        """
        为分子生成简化的 2D 坐标。
        使用 RDKit 的 2D 坐标生成方法，对不完整的分子更宽容。
        
        Args:
            mol: RDKit 分子对象
            
        Returns:
            numpy.ndarray: 原子坐标数组
        """
        try:
            mol_copy = Chem.Mol(mol)
            
            try:
                rdDepictor.Compute2DCoords(mol_copy)
                conf = mol_copy.GetConformer()
                pos = conf.GetPositions()
                
                return np.array(pos, dtype=np.float32)
            except:
                num_atoms = mol_copy.GetNumAtoms()
                pos = np.zeros((num_atoms, 3), dtype=np.float32)
                
                for i in range(num_atoms):
                    pos[i] = [float(i * 1.5), 0.0, 0.0]
                
                return pos
        except:
            import numpy as np
            num_atoms = mol.GetNumAtoms()
            return np.zeros((num_atoms, 3), dtype=np.float32)
    
    def _generate_node_features(self, z, edge_index, edge_attr):
        """
        生成节点特征矩阵。
        
        Args:
            z: 原子序数数组
            edge_index: 边索引
            edge_attr: 边属性
            
        Returns:
            numpy.ndarray: 节点特征矩阵
        """
        try:
            import numpy as np
            
            num_atoms = len(z)
            max_atomic_num = max(z) if len(z) > 0 else 0
            num_types = len(self.types)
            
            x = np.zeros((num_atoms, num_types), dtype=np.float32)
            
            atomic_num_to_type = {}
            for symbol, type_idx in self.types.items():
                atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(symbol)
                atomic_num_to_type[atomic_num] = type_idx
            
            for i, atomic_num in enumerate(z):
                type_idx = atomic_num_to_type.get(atomic_num, 0)
                x[i, type_idx] = 1.0
            
            return x
        except:
            import numpy as np
            return np.zeros((len(z), len(self.types)), dtype=np.float32)
    
    def _execute_operation(self, operation):
        """执行单个操作"""
        op_type = operation.get("operation")
        position = operation.get("position")
        atom = operation.get("atom")
        
        if op_type == "init_atom":
            self._init_atom(position, atom)
        elif op_type == "add_atom":
            self._add_atom(position, atom)
        elif op_type == "form_double_bond":
            self._form_bond(position, Chem.BondType.DOUBLE)
        elif op_type == "form_triple_bond":
            self._form_bond(position, Chem.BondType.TRIPLE)
        elif op_type == "form_aromatic_bond":
            self._form_bond(position, Chem.BondType.AROMATIC)
        elif op_type == "form_ring":
            self._form_bond(position, Chem.BondType.SINGLE)
        elif op_type == "form_double_ring":
            self._form_bond(position, Chem.BondType.DOUBLE)
        elif op_type == "form_aromatic_ring" or op_type == "成苯环":
            # 处理合并的芳香环操作，位置格式为 "0-1-2-3-4-5"
            pos_parts = position.split("-")
            if len(pos_parts) == 2:
                # 如果只有两个位置，按照普通键处理
                self._form_bond(position, Chem.BondType.AROMATIC)
            else:
                # 如果有多个位置，处理为环结构
                # 首先形成相邻原子之间的键
                for i in range(len(pos_parts) - 1):
                    self._form_bond(f"{pos_parts[i]}-{pos_parts[i+1]}", Chem.BondType.AROMATIC)
                # 最后形成首尾原子之间的键（闭环）
                self._form_bond(f"{pos_parts[-1]}-{pos_parts[0]}", Chem.BondType.AROMATIC)
        elif op_type == "add_stereo_ccw" or op_type == "add_stereo_cw":
            self._add_stereo(operation, ccw_flag=(op_type == "add_stereo_ccw"))  # 传递整个操作对象，包含立体构型信息
        elif op_type == "add_fragment":
            self._add_fragment(position, atom)
    
    def _init_atom(self, position, atom_symbol):
        """初始化第一个原子"""
        atom_idx = self.current_mol.AddAtom(Chem.Atom(atom_symbol))
        self.atom_map[position] = atom_idx
        self.atom_counter += 1
    
    def _add_atom(self, position, atom_symbol):
        """添加原子到指定位置"""
        parent_pos = position
        if parent_pos not in self.atom_map:
            parent_pos = str(self.atom_counter - 1)
        
        atom_idx = self.current_mol.AddAtom(Chem.Atom(atom_symbol))
        self.atom_map[str(self.atom_counter)] = atom_idx
        
        # 添加单键连接到父原子
        if parent_pos in self.atom_map:
            parent_idx = self.atom_map[parent_pos]
            self.current_mol.AddBond(parent_idx, atom_idx, Chem.BondType.SINGLE)
        
        self.atom_counter += 1
    
    def _form_bond(self, position, bond_type):
        """形成键"""
        if "-" in position:
            pos1, pos2 = position.split("-")
            if pos1 in self.atom_map and pos2 in self.atom_map:
                idx1 = self.atom_map[pos1]
                idx2 = self.atom_map[pos2]
                
                # 检查是否已存在键
                existing_bond = self.current_mol.GetBondBetweenAtoms(idx1, idx2)
                if existing_bond:
                    existing_bond.SetBondType(bond_type)
                else:
                    self.current_mol.AddBond(idx1, idx2, bond_type)
    
    def _add_stereo(self, operation, ccw_flag):
        # """添加立体化学信息"""
        # tmp_map = { # TEST
        #     5: 2,
        #     8: 3,
        #     2: 21,
        #     4: 23
        # }
        # tmp_map = { # TEST
        #     2: 21,
        #     3: 23,
        #     21: 2,
        #     23: 3
        # }
        position = operation.get("position")
        idx = self.atom_map[position]
        # tmp_current_smiles = Chem.MolToSmiles(self.current_mol) # TEST 调整分子顺序
        # self.current_mol = rdchem.RWMol(Chem.MolFromSmiles(tmp_current_smiles))
        atom = self.current_mol.GetAtomWithIdx(idx)  # INFO ori
        # atom = self.current_mol.GetAtomWithIdx(tmp_map[idx])  # TEST
        # 使用操作类型确定手性：ccw_flag=True
        target_tag = Chem.CHI_TETRAHEDRAL_CCW if not ccw_flag else Chem.CHI_TETRAHEDRAL_CW
        atom.SetChiralTag(target_tag)
        print('😺 [rebuilder]', f"{position}->{idx}", target_tag, atom.GetSymbol(), Chem.MolToSmiles(self.current_mol))
    
    def _add_fragment(self, position, fragment_smiles):
        """添加片段"""
        try:
            frag_mol = Chem.MolFromSmiles(fragment_smiles)
            if frag_mol:
                parent_idx = self.atom_map.get(position)
                if parent_idx is not None:
                    offset = self.current_mol.GetNumAtoms()
                    
                    # 添加片段中的所有原子
                    for atom in frag_mol.GetAtoms():
                        new_atom = Chem.Atom(atom.GetAtomicNum())
                        self.current_mol.AddAtom(new_atom)
                    
                    # 添加片段中的所有键
                    for bond in frag_mol.GetBonds():
                        b_idx = bond.GetBeginAtomIdx() + offset
                        e_idx = bond.GetEndAtomIdx() + offset
                        self.current_mol.AddBond(b_idx, e_idx, bond.GetBondType())
                    
                    # 连接片段到父原子
                    if offset < self.current_mol.GetNumAtoms():
                        self.current_mol.AddBond(parent_idx, offset, Chem.BondType.SINGLE)
        except Exception as e:
            pass
    
    def _get_current_smiles(self):
        """获取当前分子的SMILES"""
        try:
            mol = self.current_mol.GetMol()
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        except:
            return "invalid"
    
    def _describe_operation(self, operation):
        """生成操作的描述"""
        op_type = operation.get("operation")
        position = operation.get("position")
        atom = operation.get("atom")
        
        descriptions = {
            "init_atom": f"初始化原子 {atom} (位置 {position})",
            "add_atom": f"添加原子 {atom} 连接到位置 {position}",
            "form_double_bond": f"在位置 {position} 形成双键",
            "form_triple_bond": f"在位置 {position} 形成三键",
            "form_aromatic_bond": f"在位置 {position} 形成芳香键",
            "form_ring": f"在位置 {position} 形成环",
            "form_aromatic_ring": f"在位置 {position} 形成芳香环（成苯环）",
            "add_stereo": f"在位置 {position} 添加立体化学信息",
            "add_stereo_ccw": f"在位置 {position} 添加CCW手性",
            "add_stereo_cw": f"在位置 {position} 添加CW手性",
            "add_fragment": f"添加片段 {atom} 到位置 {position}"
        }
        
        return descriptions.get(op_type, f"执行操作: {op_type}")
    
    def visualize_steps(self, output_file: str = None, analyze_xyz: bool = False):
        """
        可视化重建步骤，打印每一步的详细信息。
        如果指定了 output_file，将结果保存到文件。
        
        Args:
            output_file: 输出文件路径
            analyze_xyz: 是否分析 xyz 数据
        """
        if not self.steps:
            self.rebuild_step_by_step(analyze_xyz=analyze_xyz)
        
        lines = []
        lines.append("=" * 80)
        lines.append("分子逐步重建过程可视化")
        lines.append("=" * 80)
        lines.append("")
        
        for step in self.steps:
            lines.append(f"步骤 {step['step']}: {step['description']}")
            lines.append("-" * 80)
            lines.append(f"操作类型: {step['operation']['operation']}")
            lines.append(f"位置: {step['operation']['position']}")
            lines.append(f"原子: {step['operation']['atom']}")
            lines.append(f"操作前 SMILES: {step['smiles_before']}")
            lines.append(f"操作后 SMILES: {step['smiles_after']}")
            
            if analyze_xyz and "xyz_analysis" in step:
                xyz = step["xyz_analysis"]
                lines.append("")
                lines.append("XYZ 分析结果:")
                if xyz["success"]:
                    lines.append(f"  状态: 成功")
                    lines.append(f"  方法: {xyz.get('method', 'unknown')}")
                    lines.append(f"  原子数: {xyz['num_atoms']}")
                    lines.append(f"  z (原子序数): {xyz['z']}")
                    lines.append(f"  pos (坐标数): {xyz['pos']}")
                    if xyz['edge_index'] is not None:
                        lines.append(f"  edge_index: {xyz['edge_index']}")
                    if xyz['edge_attr'] is not None:
                        lines.append(f"  edge_attr: {xyz['edge_attr']}")
                else:
                    lines.append(f"  状态: 失败")
                    lines.append(f"  错误: {xyz['error']}")
            
            lines.append("")
        
        lines.append("=" * 80)
        lines.append(f"重建完成，共 {len(self.steps)} 步")
        
        if analyze_xyz:
            xyz_success_count = sum(1 for s in self.steps if s.get("xyz_analysis", {}).get("success", False))
            invalid_smiles_count = sum(1 for s in self.steps if s.get("smiles_after") == "invalid")
            lines.append(f"XYZ 分析成功率: {xyz_success_count}/{len(self.steps)}")
            lines.append(f"SMILES 为 'invalid' 的步骤数: {invalid_smiles_count}")
        
        lines.append("=" * 80)
        
        output = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"可视化结果已保存到: {output_file}")
        
        return output
    
    def visualize_with_mpl(self, output_file: str = None, cols: int = 4, figsize: tuple = (20, 15)):
        """
        使用 matplotlib 可视化每一步的分子结构。
        
        Args:
            output_file: 输出图像文件路径
            cols: 每行显示的分子数量
            figsize: 图像大小 (width, height)
        """        
        if not self.steps:
            self.rebuild_step_by_step()
        
        # 重新执行步骤以获取每一步的分子
        self.current_mol = None
        self.atom_map = {}
        self.atom_counter = 0
        
        mols = []
        legends = []
        
        self.current_mol = rdchem.RWMol()
        
        for step_idx, operation in enumerate(self.path):
            # 执行操作
            self._execute_operation(operation)
            
            # 获取当前分子
            try:
                mol = self.current_mol.GetMol()
                Chem.SanitizeMol(mol)
                mols.append(mol)
                legends.append(f"Step {step_idx + 1}\n{operation['operation']}")
            except:
                # 如果分子无效，创建一个空白分子
                mols.append(None)
                legends.append(f"Step {step_idx + 1}\n{operation['operation']}\n(invalid)")
        
        # 计算网格布局
        num_steps = len(mols)
        rows = (num_steps + cols - 1) // cols
        
        # 创建图形
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # 绘制每个分子
        for idx, mol in enumerate(mols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            if mol is not None:
                # 使用 RDKit 绘制分子
                img = Draw.MolToImage(mol, size=(300, 300))
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Invalid\nMolecule", 
                       ha='center', va='center', fontsize=12, color='red')
            
            ax.set_title(legends[idx], fontsize=10)
            ax.axis('off')
        
        # 隐藏多余的子图
        for idx in range(num_steps, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if output_file:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"分子可视化图像已保存到: {output_file}")            
        
        plt.close()
    

def main():
    """分子进化路径生成器的命令行接口"""
    parser = argparse.ArgumentParser(description='分子进化路径生成器')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # MoleculeEvolverAnalysis 命令
    evolver_parser = subparsers.add_parser('analyze', help='分析单个分子的进化路径')
    evolver_parser.add_argument('smiles', help='输入的SMILES字符串')
    evolver_parser.add_argument('--format', choices=['text', 'json', 'dict'], default='text', 
                              help='输出格式 (默认: text)')
    
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
