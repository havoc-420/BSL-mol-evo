#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强大的分子进化路径生成器，能够将任意SMILES解析为原子级的、可重现的构建序列。
"""

from rdkit import Chem
from collections import deque
import re
import random
from typing import List, Dict, Tuple, Optional, Set, Any
import hashlib


from .attachment import Attachment

class MoleculeEvolverAnalysis: # MoleculeEvolver
    """一个封装了所有逻辑的、强大的分子路径生成器。"""
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
    
    def __init__(self, initial_smiles: str = "CC"):
        self.initial_smiles = initial_smiles
        self.initial_mol = Chem.MolFromSmiles(initial_smiles)
        
        # 操作类型定义
        self.operation_types = {
            "add_atom": "添加原子",
            "remove_atom": "移除原子", 
            "replace_atom": "替换原子",
            "form_single_bond": "形成单键",
            "form_double_bond": "形成双键",
            "form_triple_bond": "形成三键",
            "add_fragment": "添加片段",
            "break_bond": "断开键"
        }
        
        # 常见的原子类型和片段
        self.common_atoms = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        self.common_fragments = {
            'methyl': 'C',
            'ethyl': 'CC', 
            'propyl': 'CCC',
            'hydroxyl': 'O',
            'amino': 'N',
            'carboxyl': 'C(=O)O',
            'aldehyde': 'C=O',
            'nitro': 'N(=O)=O',
            'cyano': 'C#N',
            'fluoro': 'F',
            'chloro': 'Cl',
            'bromo': 'Br',
            'phenyl': 'c1ccccc1',
            'benzyl': 'Cc1ccccc1'
        }
        
        # 操作权重配置
        self.operation_weights = {
            "add_atom": 0.25,
            "replace_atom": 0.2,
            "form_single_bond": 0.15,
            "form_double_bond": 0.1,
            "form_triple_bond": 0.05,
            "add_fragment": 0.2,
            "break_bond": 0.05
        }
    
    def validate_molecule(self, mol: Chem.Mol) -> bool:
        """验证分子是否有效"""
        try:
            if mol is None:
                return False
            
            # 检查分子是否可以被Kekulize（芳香性检查）
            temp_mol = Chem.Mol(mol)
            Chem.SanitizeMol(temp_mol)
            
            # 检查是否有原子
            if temp_mol.GetNumAtoms() == 0:
                return False
                
            # 尝试计算分子量来确保分子完整性
            Chem.rdMolDescriptors.CalcExactMolWt(temp_mol)
                
            return True
        except:
            return False
    
    def _apply_operation(self, mol: Chem.Mol, operation_type: str, **kwargs) -> Optional[Chem.Mol]:
        """应用单个操作到分子"""
        try:
            if operation_type == "add_atom":
                return self._add_atom_operation(mol, **kwargs)
            elif operation_type == "replace_atom":
                return self._replace_atom_operation(mol, **kwargs)
            elif operation_type.startswith("form_"):
                return self._form_bond_operation(mol, operation_type, **kwargs)
            elif operation_type == "add_fragment":
                return self._add_fragment_operation(mol, **kwargs)
            elif operation_type == "break_bond":
                return self._break_bond_operation(mol, **kwargs)
        except Exception as e:
            return None
        return None
    
    def _add_atom_operation(self, mol: Chem.Mol, atom_symbol: str = None, connect_to: int = None) -> Optional[Chem.Mol]:
        """添加原子操作"""
        try:
            if atom_symbol is None:
                atom_symbol = random.choice(self.common_atoms)
            
            emol = Chem.EditableMol(mol)
            new_atom_idx = emol.AddAtom(Chem.Atom(atom_symbol))
            
            if connect_to is None:
                available_atoms = list(range(mol.GetNumAtoms()))
                if available_atoms:
                    connect_to = random.choice(available_atoms)
                else:
                    connect_to = 0
            
            emol.AddBond(connect_to, new_atom_idx, Chem.BondType.SINGLE)
            new_mol = emol.GetMol()
            
            # 验证新分子
            if self.validate_molecule(new_mol):
                return new_mol
        except:
            pass
        return None
    
    def _replace_atom_operation(self, mol: Chem.Mol, atom_idx: int = None, new_symbol: str = None) -> Optional[Chem.Mol]:
        """替换原子操作"""
        try:
            if atom_idx is None:
                atom_idx = random.randint(0, mol.GetNumAtoms() - 1)
            
            if new_symbol is None:
                current_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                available_symbols = [a for a in self.common_atoms if a != current_symbol]
                if available_symbols:
                    new_symbol = random.choice(available_symbols)
                else:
                    return None
            
            mol_copy = Chem.Mol(mol)
            atom = mol_copy.GetAtomWithIdx(atom_idx)
            atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(new_symbol))
            
            if self.validate_molecule(mol_copy):
                return mol_copy
        except:
            pass
        return None
    
    def _form_bond_operation(self, mol: Chem.Mol, operation_type: str, atom1_idx: int = None, atom2_idx: int = None) -> Optional[Chem.Mol]:
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
            
            if atom1_idx is None or atom2_idx is None:
                atom1_idx, atom2_idx = random.choice(possible_pairs)
            
            # 检查是否已经存在键
            if mol.GetBondBetweenAtoms(atom1_idx, atom2_idx):
                return None
            
            # 映射键类型
            bond_type_map = {
                "form_single_bond": Chem.BondType.SINGLE,
                "form_double_bond": Chem.BondType.DOUBLE,
                "form_triple_bond": Chem.BondType.TRIPLE
            }
            
            bond_type = bond_type_map.get(operation_type, Chem.BondType.SINGLE)
            
            emol = Chem.EditableMol(mol)
            emol.AddBond(atom1_idx, atom2_idx, bond_type)
            new_mol = emol.GetMol()
            
            if self.validate_molecule(new_mol):
                return new_mol
        except:
            pass
        return None
    
    def _add_fragment_operation(self, mol: Chem.Mol, fragment_name: str = None, connect_to: int = None) -> Optional[Chem.Mol]:
        """添加片段操作"""
        try:
            if fragment_name is None:
                fragment_name = random.choice(list(self.common_fragments.keys()))
            
            fragment_smiles = self.common_fragments.get(fragment_name)
            if not fragment_smiles:
                return None
            
            frag_mol = Chem.MolFromSmiles(fragment_smiles)
            if not frag_mol:
                return None
            
            combined = Chem.CombineMols(mol, frag_mol)
            
            if connect_to is None:
                available_atoms = list(range(mol.GetNumAtoms()))
                if available_atoms:
                    connect_to = random.choice(available_atoms)
                else:
                    connect_to = 0
            
            frag_start_idx = mol.GetNumAtoms()
            frag_connect_point = frag_start_idx
            
            emol = Chem.EditableMol(combined)
            emol.AddBond(connect_to, frag_connect_point, Chem.BondType.SINGLE)
            new_mol = emol.GetMol()
            
            if self.validate_molecule(new_mol):
                return new_mol
        except:
            pass
        return None
    
    def _break_bond_operation(self, mol: Chem.Mol, atom1_idx: int = None, atom2_idx: int = None) -> Optional[Chem.Mol]:
        """断开键操作"""
        try:
            bonds = []
            for bond in mol.GetBonds():
                bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            
            if not bonds:
                return None
            
            if atom1_idx is None or atom2_idx is None:
                atom1_idx, atom2_idx = random.choice(bonds)
            
            if not mol.GetBondBetweenAtoms(atom1_idx, atom2_idx):
                return None
            
            emol = Chem.EditableMol(mol)
            emol.RemoveBond(atom1_idx, atom2_idx)
            new_mol = emol.GetMol()
            
            if self.validate_molecule(new_mol):
                return new_mol
        except:
            pass
        return None
    
    def _get_molecule_fingerprint(self, mol: Chem.Mol) -> str:
        """获取分子的指纹用于去重"""
        try:
            smiles = Chem.MolToSmiles(mol)
            return hashlib.md5(smiles.encode()).hexdigest()
        except:
            return ""
    
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
        
        for path_idx in range(num_paths):
            print(f"生成路径 {path_idx + 1}/{num_paths}")
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
    
    def _get_possible_operations(self, mol: Chem.Mol) -> List[Dict]:
        """获取当前分子所有可能的操作"""
        operations = []
        
        # 添加原子操作
        for atom_symbol in self.common_atoms:
            for connect_to in range(mol.GetNumAtoms()):
                operations.append({
                    "type": "add_atom",
                    "params": {"atom_symbol": atom_symbol, "connect_to": connect_to},
                    "weight": self.operation_weights["add_atom"]
                })
        
        # 替换原子操作
        for atom_idx in range(mol.GetNumAtoms()):
            current_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            for new_symbol in self.common_atoms:
                if new_symbol != current_symbol:
                    operations.append({
                        "type": "replace_atom",
                        "params": {"atom_idx": atom_idx, "new_symbol": new_symbol},
                        "weight": self.operation_weights["replace_atom"]
                    })
        
        # 形成键操作
        bond_operations = ["form_single_bond", "form_double_bond", "form_triple_bond"]
        for op_type in bond_operations:
            for i in range(mol.GetNumAtoms()):
                for j in range(i + 1, mol.GetNumAtoms()):
                    if not mol.GetBondBetweenAtoms(i, j):
                        operations.append({
                            "type": op_type,
                            "params": {"atom1_idx": i, "atom2_idx": j},
                            "weight": self.operation_weights[op_type]
                        })
        
        # 添加片段操作
        for fragment_name in self.common_fragments.keys():
            for connect_to in range(mol.GetNumAtoms()):
                operations.append({
                    "type": "add_fragment",
                    "params": {"fragment_name": fragment_name, "connect_to": connect_to},
                    "weight": self.operation_weights["add_fragment"]
                })
        
        # 断开键操作
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            operations.append({
                "type": "break_bond",
                "params": {"atom1_idx": i, "atom2_idx": j},
                "weight": self.operation_weights["break_bond"]
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
            "edges": []
        }
        
        queue = deque([(self.initial_smiles, 0)])  # (smiles, depth)
        visited = {self.initial_smiles}
        
        while queue:
            current_smiles, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            current_mol = Chem.MolFromSmiles(current_smiles)
            if not current_mol:
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
                
                new_mol = self._apply_operation(current_mol, operation["type"], **operation["params"])
                if new_mol and self.validate_molecule(new_mol):
                    new_smiles = Chem.MolToSmiles(new_mol)
                    
                    if new_smiles not in visited:
                        visited.add(new_smiles)
                        queue.append((new_smiles, depth + 1))
                        
                        # 记录边
                        tree["edges"].append({
                            "from": current_smiles,
                            "to": new_smiles,
                            "operation": operation["type"],
                            "details": operation["params"]
                        })
                        
                        branch_count += 1
        
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


# 使用示例
if __name__ == "__main__":
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
