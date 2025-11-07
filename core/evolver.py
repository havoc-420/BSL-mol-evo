#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强大的分子进化路径生成器，能够将任意SMILES解析为原子级的、可重现的构建序列。
"""

from rdkit import Chem
from collections import deque
import re

from .attachment import Attachment

class MoleculeEvolver:
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