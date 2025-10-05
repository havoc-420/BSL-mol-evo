from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
from collections import deque
import os
from PIL import Image, ImageDraw, ImageFont

IMG_SAVE_BASE_PATH = "mol-evo/tests/data/visualization"
if not os.path.exists(IMG_SAVE_BASE_PATH):
    os.makedirs(IMG_SAVE_BASE_PATH)

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
class MoleculeEvolver:
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
        path = [f"Start({start_atom.GetSymbol()} ID:{self.backbone_map[start_atom.GetIdx()]})"]
        for i in range(1, len(self.backbone_indices)):
            parent_new_idx = self.backbone_map[self.backbone_indices[i-1]]
            current_atom = self.mol.GetAtomWithIdx(self.backbone_indices[i])
            path.append(f"Add atom({current_atom.GetSymbol()}) -> {parent_new_idx}")

        # --- 附件、额外键和立体化学 ---
        non_backbone_atoms = [a.GetIdx() for a in self.mol.GetAtoms() if a.GetIdx() not in self.backbone_set]
        backbone_bonds = {tuple(sorted((self.backbone_indices[i], self.backbone_indices[i-1]))) for i in range(1, len(self.backbone_indices))}
        
        # --- 附件处理 ---
        attachments = self._get_sorted_attachments(non_backbone_atoms)
        for att in attachments:
            conn_points_str = ",".join(map(str, sorted(list(att.connection_points))))
            path.append(f"Add fragment({att.mol_frag_smiles}) @ {conn_points_str}")

        # --- 额外键处理 (成环、多重键) ---
        extra_bond_ops = []
        for bond in self.mol.GetBonds():
            b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_tuple = tuple(sorted((b, e)))
            
            # 检查此键是否是骨架、附件内部或附件到骨架的连接键
            is_backbone_bond = bond_tuple in backbone_bonds
            is_attachment_bond = not (self.mol.GetAtomWithIdx(b) in self.backbone_set and self.mol.GetAtomWithIdx(e) in self.backbone_set)

            if is_backbone_bond and bond.GetBondType() != Chem.BondType.SINGLE:
                op = "Form double bond" if bond.GetBondType() == Chem.BondType.DOUBLE else "Form triple bond"
                extra_bond_ops.append(f"{op} @({self.backbone_map[b]}-{self.backbone_map[e]})")
            elif not is_backbone_bond and not is_attachment_bond:
                extra_bond_ops.append(f"Form ring @({self.backbone_map[b]}-{self.backbone_map[e]})")
        
        path += sorted(extra_bond_ops)

        # --- 立体化学处理 ---
        stereo_ops = []
        chiral_centers = Chem.FindMolChiralCenters(self.mol, includeUnassigned=False)
        for center_idx, stereo in chiral_centers:
            if center_idx in self.backbone_map:
                stereo_ops.append(f"Specify chirality({stereo}) @ {self.backbone_map[center_idx]}")
        
        for bond in self.mol.GetBonds():
            if bond.GetStereo() > Chem.BondStereo.STEREOANY:
                b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if b in self.backbone_map and e in self.backbone_map:
                    stereo_ops.append(f"Specify stereochemistry({bond.GetStereo()}) @ ({self.backbone_map[b]}-{self.backbone_map[e]})")
        
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

    def generate_combined_visualization(self, output_file="evolution_combined.png"):
        """生成展示完整进化过程的组合图像"""
        start_idx = self._find_canonical_start_atom()
        self._build_canonical_backbone(start_idx)
        
        # 初始化可视化步骤
        mol_steps = []
        step_descriptions = []
        
        # 创建一个空的可编辑分子
        editable_mol = Chem.RWMol()
        atom_map = {}  # old_idx -> new_idx in editable_mol
        
        # 步骤1: 起始原子
        start_atom = self.mol.GetAtomWithIdx(self.backbone_indices[0])
        new_atom = Chem.Atom(start_atom.GetAtomicNum())
        new_atom_idx = editable_mol.AddAtom(new_atom)
        atom_map[start_atom.GetIdx()] = new_atom_idx
        
        # 保存分子状态
        temp_mol = editable_mol.GetMol()
        mol_steps.append(Chem.Mol(temp_mol))
        step_descriptions.append(f"Start({start_atom.GetSymbol()} ID:{self.backbone_map[start_atom.GetIdx()]})")
        
        # 添加骨架原子
        for i in range(1, len(self.backbone_indices)):
            parent_old_idx = self.backbone_indices[i-1]
            current_old_idx = self.backbone_indices[i]
            
            parent_new_idx = atom_map[parent_old_idx]
            current_atom = self.mol.GetAtomWithIdx(current_old_idx)
            new_atom = Chem.Atom(current_atom.GetAtomicNum())
            current_new_idx = editable_mol.AddAtom(new_atom)
            atom_map[current_old_idx] = current_new_idx
            
            # 添加键
            editable_mol.AddBond(parent_new_idx, current_new_idx, Chem.BondType.SINGLE)
            
            # 保存分子状态
            temp_mol = editable_mol.GetMol()
            mol_steps.append(Chem.Mol(temp_mol))
            step_descriptions.append(f"Add atom({current_atom.GetSymbol()}) -> {self.backbone_map[parent_old_idx]}")
        
        # 添加附件原子
        non_backbone_atoms = [a.GetIdx() for a in self.mol.GetAtoms() if a.GetIdx() not in self.backbone_set]
        if non_backbone_atoms:
            attachments = self._get_sorted_attachments(non_backbone_atoms)
            for att in attachments:
                # 添加附件中的原子
                added_atoms = False
                for old_idx in att.indices:
                    if old_idx not in atom_map:
                        atom = self.mol.GetAtomWithIdx(old_idx)
                        new_atom = Chem.Atom(atom.GetAtomicNum())
                        new_idx = editable_mol.AddAtom(new_atom)
                        atom_map[old_idx] = new_idx
                        added_atoms = True
                
                # 如果添加了新原子，保存状态
                if added_atoms:
                    temp_mol = editable_mol.GetMol()
                    mol_steps.append(Chem.Mol(temp_mol))
                    conn_points_str = ",".join(map(str, sorted(list(att.connection_points))))
                    step_descriptions.append(f"Add fragment atoms({att.mol_frag_smiles}) @ {conn_points_str}")
                
                # 添加附件内部的键
                bonds_added = False
                for old_idx in att.indices:
                    atom = self.mol.GetAtomWithIdx(old_idx)
                    for neighbor in atom.GetNeighbors():
                        neighbor_idx = neighbor.GetIdx()
                        # 如果邻居也在附件中，且键还没添加
                        if neighbor_idx in att.indices and neighbor_idx > old_idx:
                            bond = self.mol.GetBondBetweenAtoms(old_idx, neighbor_idx)
                            if not editable_mol.GetBondBetweenAtoms(atom_map[old_idx], atom_map[neighbor_idx]):
                                editable_mol.AddBond(atom_map[old_idx], atom_map[neighbor_idx], bond.GetBondType())
                                bonds_added = True
                
                # 添加附件与骨架的连接键
                for old_idx in att.indices:
                    atom = self.mol.GetAtomWithIdx(old_idx)
                    for neighbor in atom.GetNeighbors():
                        neighbor_idx = neighbor.GetIdx()
                        # 如果邻居在骨架上
                        if neighbor_idx in self.backbone_set and neighbor_idx in atom_map and old_idx in atom_map:
                            # 检查键是否已存在
                            if not editable_mol.GetBondBetweenAtoms(atom_map[old_idx], atom_map[neighbor_idx]):
                                bond = self.mol.GetBondBetweenAtoms(old_idx, neighbor_idx)
                                editable_mol.AddBond(atom_map[old_idx], atom_map[neighbor_idx], bond.GetBondType())
                                bonds_added = True
                
                # 如果添加了键，保存状态
                if bonds_added:
                    temp_mol = editable_mol.GetMol()
                    mol_steps.append(Chem.Mol(temp_mol))
                    conn_points_str = ",".join(map(str, sorted(list(att.connection_points))))
                    step_descriptions.append(f"Connect fragment({att.mol_frag_smiles}) @ {conn_points_str}")
        
        # 添加额外的键（双键、三键、环）
        backbone_bonds = {tuple(sorted((self.backbone_indices[i], self.backbone_indices[i-1]))) for i in range(1, len(self.backbone_indices))}
        for bond in self.mol.GetBonds():
            b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_tuple = tuple(sorted((b, e)))
            
            is_backbone_bond = bond_tuple in backbone_bonds
            is_attachment_bond = not (self.mol.GetAtomWithIdx(b) in self.backbone_set and self.mol.GetAtomWithIdx(e) in self.backbone_set)
            
            # 修改骨架中的多重键或添加环键
            if (is_backbone_bond and bond.GetBondType() != Chem.BondType.SINGLE) or \
               (not is_backbone_bond and not is_attachment_bond):
                if b in atom_map and e in atom_map:
                    editable_bond = editable_mol.GetBondBetweenAtoms(atom_map[b], atom_map[e])
                    if editable_bond:
                        old_bond_type = editable_bond.GetBondType()
                        if old_bond_type != bond.GetBondType():
                            editable_bond.SetBondType(bond.GetBondType())
                            
                            # 保存分子状态
                            temp_mol = editable_mol.GetMol()
                            mol_steps.append(Chem.Mol(temp_mol))
                            op = "Form double bond" if bond.GetBondType() == Chem.BondType.DOUBLE else "Form triple bond"
                            step_descriptions.append(f"{op} @({self.backbone_map[b]}-{self.backbone_map[e]})")
        
        # 添加立体化学信息（这里只是示意，实际显示可能需要更复杂的处理）
        chiral_centers = Chem.FindMolChiralCenters(self.mol, includeUnassigned=False)
        for center_idx, stereo in chiral_centers:
            if center_idx in self.backbone_map:
                # 保存分子状态
                temp_mol = editable_mol.GetMol()
                mol_steps.append(Chem.Mol(temp_mol))
                step_descriptions.append(f"Specify chirality({stereo}) @ {self.backbone_map[center_idx]}")
        
        # 生成组合图像
        if mol_steps:
            # 计算图像布局
            num_steps = len(mol_steps)
            cols = min(4, num_steps)  # 每行最多4个图像
            rows = (num_steps + cols - 1) // cols  # 计算需要的行数
            
            # 图像尺寸
            img_size = 300
            padding = 20
            label_height = 50
            
            # 创建组合图像
            combined_width = cols * (img_size + padding) + padding
            combined_height = rows * (img_size + label_height + padding) + padding
            
            combined_img = Image.new('RGB', (combined_width, combined_height), 'white')
            draw = ImageDraw.Draw(combined_img)
            
            # 生成每个步骤的图像并放置在组合图像中
            for i, (mol, desc) in enumerate(zip(mol_steps, step_descriptions)):
                # 生成分子图像
                try:
                    img = Draw.MolToImage(mol, size=(img_size, img_size))
                except:
                    # 如果无法生成图像，创建一个空白图像
                    img = Image.new('RGB', (img_size, img_size), 'lightgray')
                
                # 计算位置
                row = i // cols
                col = i % cols
                x = padding + col * (img_size + padding)
                y = padding + row * (img_size + label_height + padding)
                
                # 粘贴分子图像
                combined_img.paste(img, (x, y))
                
                # 添加文字描述
                try:
                    # 尝试使用更好的字体
                    font = ImageFont.truetype("arial.ttf", 14)
                except:
                    # 如果没有可用字体，使用默认字体
                    font = ImageFont.load_default()
                
                # 文字居中
                text_x = x + img_size // 2
                text_y = y + img_size + 10
                draw.text((text_x, text_y), desc, fill='black', font=font, anchor='mt')
            
            # 保存组合图像
            combined_img.save(os.path.join(IMG_SAVE_BASE_PATH, output_file))
            print(f"Combined evolution image saved to: {output_file}")
        
        return step_descriptions

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

def calculate_evolutionary_similarity(smiles1: str, smiles2: str, verbose=False, visualize=False, combined_visualize=False):
    """最终的、对用户友好的顶层调用函数。"""
    try:
        evolver1 = MoleculeEvolver(smiles1)
        evolver2 = MoleculeEvolver(smiles2)
        path1 = evolver1.generate_path()
        path2 = evolver2.generate_path()
        
        if visualize:
            print(f"Generating visualization for '{smiles1}'...")
            evolver1.generate_visualization_steps(f"evolution_steps_{smiles1.replace('/', '_')}")
            print(f"Generating visualization for '{smiles2}'...")
            evolver2.generate_visualization_steps(f"evolution_steps_{smiles2.replace('/', '_')}")
        
        if combined_visualize:
            print(f"Generating combined visualization for '{smiles1}'...")
            evolver1.generate_combined_visualization(f"evolution_{smiles1.replace('/', '_')}.png")
            print(f"Generating combined visualization for '{smiles2}'...")
            evolver2.generate_combined_visualization(f"evolution_{smiles2.replace('/', '_')}.png")
    except ValueError as e:
        print(f"Error: {e}")
        return 0.0, [], []

    distance = calculate_path_edit_distance(path1, path2)
    # similarity = 1.0 / (1.0 + distance)
    
    if verbose:
        print(f"--- Comparing '{smiles1}' vs '{smiles2}' ---")
        print(f"Path 1 (length {len(path1)}): {path1}")
        print(f"Path 2 (length {len(path2)}): {path2}")
        print(f"Path edit distance: {distance}")
        # print(f"Final similarity (1 / (1 + d)): {similarity:.3f}\n")
        
    return distance, path1, path2

# --- 4. 主函数 ---
if __name__ == "__main__":
    print("--- 验证 1: 您的核心反例 (长链 vs 长环) ---")
    calculate_evolutionary_similarity('CCCCCC', 'c1ccccc1', verbose=True, combined_visualize=True)

    print("\n--- 验证 2: 支链操作 ---")
    calculate_evolutionary_similarity('CCCC', 'CC(C)C', verbose=True, combined_visualize=True)
    
    print("\n--- 验证 3: 复杂多环系统 (金刚烷 vs 螺环) ---")
    adamantane = 'C1C2CC3CC1CC(C2)C3' # 桥环
    spiro_nonane = 'C1CCC2(C1)CCCC2' # 螺环
    calculate_evolutionary_similarity(adamantane, spiro_nonane, verbose=True, combined_visualize=True)
    
    print("\n--- 验证 4: 立体化学 ---")
    r_alanine = 'C[C@H](N)C(=O)O'
    s_alanine = 'C[C@@H](N)C(=O)O'
    calculate_evolutionary_similarity(r_alanine, s_alanine, verbose=True, combined_visualize=True)
    
    print("\n--- 验证 5: 复杂药物分子 (阿托伐他汀) ---")
    atorvastatin = 'CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@H](O)C[C@H](O)CC(=O)O'
    simvastatin = 'CCC(C)C(=O)O[C@H]1C[C@@H](C)C=C2C=C(C)C[C@H](O)[C@H]21'
    calculate_evolutionary_similarity(atorvastatin, simvastatin, verbose=True, combined_visualize=True)
    
    calculate_evolutionary_similarity("CCC", "CCO", verbose=True, combined_visualize=True)
