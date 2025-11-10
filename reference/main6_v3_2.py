from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType
import numpy as np
from collections import deque

# --- 1. 核心数据结构：附件描述符 ---
class Attachment:
    """一个数据类，用于存储关于附件（支链/环）的所有规范化信息，以便进行无歧义的排序。"""
    def __init__(self, attachment_indices, mol, backbone_map):
        self.indices = set(attachment_indices)
        
        # 为附件本身生成一个规范SMILES，作为最终的决胜排序规则
        self.mol_frag_smiles = Chem.MolFragmentToSmiles(mol, list(attachment_indices), canonical=True)
        
        self.size = len(self.indices)
        self.connection_points = self._find_connection_points(mol, backbone_map)
        
        # 核心：定义一个绝对无歧义的排序键
        # 规则1：按最小的骨架连接点排序
        # 规则2：如果连接点相同，按附件大小排序
        # 规则3：如果大小还相同，按附件的规范SMILES字典序排序
        self.sort_key = (min(self.connection_points), self.size, self.mol_frag_smiles)

    def _find_connection_points(self, mol, backbone_map):
        """找到附件连接到主骨架上的所有点（使用骨架的内在坐标）。"""
        points = set()
        for idx in self.indices:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() in backbone_map:
                    points.add(backbone_map[neighbor.GetIdx()])
        return points

# --- 2. 核心算法：分子进化路径生成器 ---
class MoleculeEvolverAnalysis:
    """一个封装了所有逻辑的、强大的分子路径生成器。"""
    def __init__(self, smiles: str):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        if not self.mol: raise ValueError(f"无效的SMILES: {smiles}")
        
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
        backbone_bonds = {tuple(sorted((self.backbone_indices[i], self.backbone_indices[i-1]))) for i in range(1, len(self.backbone_indices))}
        
        # --- 附件处理 ---
        attachments = self._get_sorted_attachments(non_backbone_atoms)
        for att in attachments:
            conn_points_str = ",".join(map(str, sorted(list(att.connection_points))))
            path.append(f"添加附件({att.mol_frag_smiles}) @ {conn_points_str}")

        # --- 额外键处理 (成环、多重键) ---
        extra_bond_ops = []
        for bond in self.mol.GetBonds():
            b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_tuple = tuple(sorted((b, e)))
            
            # 检查此键是否是骨架、附件内部或附件到骨架的连接键
            is_backbone_bond = bond_tuple in backbone_bonds
            is_attachment_bond = not (self.mol.GetAtomWithIdx(b) in self.backbone_set and self.mol.GetAtomWithIdx(e) in self.backbone_set)

            if is_backbone_bond and bond.GetBondType() != Chem.BondType.SINGLE:
                op = "形成双键" if bond.GetBondType() == Chem.BondType.DOUBLE else "形成三键"
                extra_bond_ops.append(f"{op} @({self.backbone_map[b]}-{self.backbone_map[e]})")
            elif not is_backbone_bond and not is_attachment_bond:
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

# --- 3. 相似度计算模块 ---
def calculate_path_edit_distance(path1, path2):
    len1, len2 = len(path1), len(path2)
    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)
    for i in range(len1 + 1): dp[i][0] = i
    for j in range(len2 + 1): dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if path1[i-1] == path2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[len1][len2]

def calculate_evolutionary_similarity(smiles1: str, smiles2: str, verbose=False):
    """最终的、对用户友好的顶层调用函数。"""
    try:
        path1 = MoleculeEvolverAnalysis(smiles1).generate_path()
        path2 = MoleculeEvolverAnalysis(smiles2).generate_path()
    except ValueError as e:
        print(f"错误: {e}")
        return 0.0, [], []

    distance = calculate_path_edit_distance(path1, path2)
    # similarity = 1.0 / (1.0 + distance)
    
    if verbose:
        print(f"--- 比较 '{smiles1}' vs '{smiles2}' ---")
        print(f"路径 1 (长度 {len(path1)}): {path1}")
        print(f"路径 2 (长度 {len(path2)}): {path2}")
        print(f"路径编辑距离: {distance}")
        # print(f"最终相似度 (1 / (1 + d)): {similarity:.3f}\n")
        
    return distance, path1, path2

# --- 4. 终极验证 ---
if __name__ == '__main__':
    # print("--- 验证 1: 您的核心反例 (长链 vs 长环) ---")
    calculate_evolutionary_similarity('CCCCCC', 'c1ccccc1', verbose=True)

    # print("\n--- 验证 2: 支链操作 ---")
    # calculate_evolutionary_similarity('CCCC', 'CC(C)C', verbose=True)
    
    # print("\n--- 验证 3: 复杂多环系统 (金刚烷 vs 螺环) ---")
    # adamantane = 'C1C2CC3CC1CC(C2)C3' # 桥环
    # spiro_nonane = 'C1CCC2(C1)CCCC2' # 螺环
    # calculate_evolutionary_similarity(adamantane, spiro_nonane, verbose=True)
    
    # print("\n--- 验证 4: 立体化学 ---")
    # r_alanine = 'C[C@H](N)C(=O)O'
    # s_alanine = 'C[C@@H](N)C(=O)O'
    # calculate_evolutionary_similarity(r_alanine, s_alanine, verbose=True)
    
    # print("\n--- 验证 5: 复杂药物分子 (阿托伐他汀) ---")
    atorvastatin = 'CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@H](O)C[C@H](O)CC(=O)O'
    simvastatin = 'CCC(C)C(=O)O[C@H]1C[C@@H](C)C=C2C=C(C)C[C@H](O)[C@H]21'
    calculate_evolutionary_similarity(atorvastatin, simvastatin, verbose=True)
    
    calculate_evolutionary_similarity("CCC", "CCO", verbose=True)