from rdkit import Chem
from collections import deque
from .attachment import Attachment


class MoleculeEvolver:
    """一个封装了所有逻辑的、强大的分子路径生成器。"""
    def __init__(self, smiles: str):
        self.smiles = smiles
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

    def generate_path(self) -> list:
        """生成最终的、完整的进化路径。"""
        start_idx = self._find_canonical_start_atom()
        self._build_canonical_backbone(start_idx)

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
        has_rings = len(Chem.GetSymmSSSR(self.mol)) > 0
        
        for bond in self.mol.GetBonds():
            b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_tuple = tuple(sorted((b, e)))
            
            # 检查此键是否是骨架、附件内部或附件到骨架的连接键
            is_backbone_bond = bond_tuple in backbone_bonds
            both_in_backbone = self.mol.GetAtomWithIdx(b).GetIdx() in self.backbone_set and self.mol.GetAtomWithIdx(e).GetIdx() in self.backbone_set

            if is_backbone_bond and bond.GetBondType() != Chem.BondType.SINGLE:
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
                # 只有当两个原子都在骨架中，键不是骨架键，且分子有环结构时，才是成环操作
                extra_bond_ops.append(f"成环 @({self.backbone_map[b]}-{self.backbone_map[e]})")
        
        path += sorted(extra_bond_ops)

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
        
        return path + sorted(stereo_ops)

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