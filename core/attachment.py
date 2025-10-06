from rdkit import Chem


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
        if self.connection_points:  # 确保集合不为空
            self.sort_key = (min(self.connection_points), self.size, self.mol_frag_smiles)
        else:
            self.sort_key = (0, self.size, self.mol_frag_smiles)

    def _find_connection_points(self, mol, backbone_map):
        """找到附件连接到主骨架上的所有点（使用骨架的内在坐标）。"""
        points = set()
        for idx in self.indices:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() in backbone_map:
                    points.add(backbone_map[neighbor.GetIdx()])
        return points