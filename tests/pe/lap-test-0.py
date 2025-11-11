import numpy as np
import networkx as nx
from rdkit import Chem

def get_laplacian_pe_from_smiles(smiles, k=8):
    """从SMILES获取拉普拉斯位置编码"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    # 构建分子图（只考虑重原子）
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx())
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        G.add_edge(i, j)
    
    # 计算归一化拉普拉斯矩阵
    L = nx.normalized_laplacian_matrix(G).astype(float)
    
    # 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(L.toarray())
    
    # 选择最小的k个非零特征值对应的特征向量
    # 跳过第一个特征值为0的特征向量
    pe_vectors = eigenvectors[:, 1:min(k+1, len(eigenvalues))]
    
    return pe_vectors

# 示例使用
smiles = "C#C"  # 乙炔
pe = get_laplacian_pe_from_smiles(smiles)
print(f"位置编码形状: {pe.shape}")
for i, atom_pe in enumerate(pe):
    print(f"原子 {i}: {atom_pe}")