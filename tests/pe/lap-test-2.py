import numpy as np
import networkx as nx
from rdkit import Chem

def get_laplacian_pe_from_smiles(smiles, k=8, target_position=None):
    """从SMILES获取拉普拉斯位置编码，并特别标记目标位置
    
    Args:
        smiles: 分子SMILES
        k: 位置编码维度
        target_position: 要特别标记的目标原子位置（从0开始）
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    # 构建分子图
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
    valid_indices = np.where(eigenvalues > 1e-8)[0]
    if len(valid_indices) == 0:
        # 如果所有特征值都很小，使用随机编码
        pe_vectors = np.random.normal(0, 0.1, (len(G.nodes()), k))
    else:
        k_actual = min(k, len(valid_indices))
        selected_indices = valid_indices[:k_actual]
        pe_vectors = eigenvectors[:, selected_indices]
        
        # 如果维度不够，用零填充
        if pe_vectors.shape[1] < k:
            padding = np.zeros((pe_vectors.shape[0], k - pe_vectors.shape[1]))
            pe_vectors = np.hstack([pe_vectors, padding])
    
    # 添加目标位置标记
    if target_position is not None and target_position < len(G.nodes()):
        # 方法1: 在PE后添加二进制标记
        position_marker = np.zeros((len(G.nodes()), 1))
        position_marker[target_position] = 1.0
        pe_vectors = np.hstack([pe_vectors, position_marker])
        
        # 方法2: 在PE中乘以放大因子 (取消注释使用)
        # pe_vectors[target_position] *= 10  # 放大目标位置的PE值
        
        # 方法3: 添加特殊偏移 (取消注释使用)
        # offset = np.zeros((len(G.nodes()), 1))
        # offset[target_position] = 100.0
        # pe_vectors = np.hstack([pe_vectors, offset])
    
    return pe_vectors

def get_enhanced_positional_encoding(smiles, target_position=None, k=8, method="binary_marker"):
    """增强版位置编码，提供多种标记方法
    
    Args:
        smiles: 分子SMILES
        target_position: 目标原子位置
        k: 基础PE维度
        method: 标记方法 - "binary_marker", "amplify", "offset", "separate_channel"
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    # 构建分子图
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx())
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        G.add_edge(i, j)
    
    # 计算基础拉普拉斯PE
    L = nx.normalized_laplacian_matrix(G).astype(float)
    eigenvalues, eigenvectors = np.linalg.eigh(L.toarray())
    
    valid_indices = np.where(eigenvalues > 1e-8)[0]
    if len(valid_indices) == 0:
        base_pe = np.random.normal(0, 0.1, (len(G.nodes()), k))
    else:
        k_actual = min(k, len(valid_indices))
        selected_indices = valid_indices[:k_actual]
        base_pe = eigenvectors[:, selected_indices]
        
        if base_pe.shape[1] < k:
            padding = np.zeros((base_pe.shape[0], k - base_pe.shape[1]))
            base_pe = np.hstack([base_pe, padding])
    
    # 根据选择的方法添加位置标记
    if target_position is not None and target_position < len(G.nodes()):
        if method == "binary_marker":
            # 方法1: 二进制标记 (推荐)
            marker = np.zeros((len(G.nodes()), 1))
            marker[target_position] = 1.0
            enhanced_pe = np.hstack([base_pe, marker])
            
        elif method == "amplify":
            # 方法2: 放大目标位置
            enhanced_pe = base_pe.copy()
            enhanced_pe[target_position] *= 10.0  # 放大10倍
            
        elif method == "offset":
            # 方法3: 添加特殊偏移
            offset = np.zeros((len(G.nodes()), 1))
            offset[target_position] = 100.0
            enhanced_pe = np.hstack([base_pe, offset])
            
        elif method == "separate_channel":
            # 方法4: 分离通道
            position_channel = np.zeros((len(G.nodes()), k))
            position_channel[target_position] = 1.0
            enhanced_pe = np.hstack([base_pe, position_channel])
            
        else:
            enhanced_pe = base_pe
    else:
        enhanced_pe = base_pe
    
    return enhanced_pe

# 测试示例
def test_enhanced_pe():
    """测试增强版位置编码"""
    
    test_cases = [
        ("C#C", 1, "乙炔 - 在位置1添加原子"),
        ("CCC", 1, "丙烷 - 在位置1添加支链"),
        ("C=CC", 2, "丙烯 - 在位置2进行取代"),
    ]
    
    for smiles, target_pos, description in test_cases:
        print(f"\n=== {description} ===")
        print(f"SMILES: {smiles}, 目标位置: {target_pos}")
        
        # 原始PE
        original_pe = get_laplacian_pe_from_smiles(smiles, k=4)
        print(f"原始PE形状: {original_pe.shape}")
        
        # 增强PE - 二进制标记方法
        enhanced_pe = get_enhanced_positional_encoding(
            smiles, target_position=target_pos, k=4, method="binary_marker"
        )
        print(f"增强PE形状: {enhanced_pe.shape}")
        
        # 显示每个原子的编码
        mol = Chem.MolFromSmiles(smiles)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_symbol = atom.GetSymbol()
            is_target = "★" if i == target_pos else " "
            print(f"{is_target} 原子 {i}({atom_symbol}): {enhanced_pe[i]}")
        
        print("-" * 50)

def create_position_aware_dataset(data_entries, k=6, method="binary_marker"):
    """创建位置感知的数据集
    
    Args:
        data_entries: 原始数据条目
        k: PE维度
        method: 位置标记方法
    """
    dataset = []
    
    for entry in data_entries:
        smiles = entry['smiles_from']
        target_position = int(entry['operations'][0]['position'])
        
        # 获取增强的位置编码
        enhanced_pe = get_enhanced_positional_encoding(
            smiles, target_position=target_position, k=k, method=method
        )
        
        if enhanced_pe is not None:
            # 获取原子特征
            mol = Chem.MolFromSmiles(smiles)
            atom_features = []
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum(),      # 原子类型
                    atom.GetDegree(),         # 连接数
                    atom.GetFormalCharge(),   # 形式电荷
                    int(atom.GetHybridization()),  # 杂化
                    atom.IsInRing(),          # 是否在环中
                ]
                atom_features.append(features)
            
            dataset.append({
                'smiles': smiles,
                'target_position': target_position,
                'pe': enhanced_pe,
                'atom_features': np.array(atom_features),
                'operation': entry['operations'][0]['operation']
            })
    
    return dataset

# 示例使用
if __name__ == "__main__":
    # 测试基本功能
    smiles = "C#C"  # 乙炔
    target_position = 1  # 在位置1添加原子
    
    print("=== 基本示例 ===")
    pe_original = get_laplacian_pe_from_smiles(smiles, k=4)
    print("原始PE:")
    for i, vec in enumerate(pe_original):
        print(f"  原子 {i}: {vec}")
    
    pe_enhanced = get_laplacian_pe_from_smiles(smiles, k=4, target_position=target_position)
    print(f"\n增强PE (目标位置 {target_position}):")
    for i, vec in enumerate(pe_enhanced):
        marker = " ★" if i == target_position else ""
        print(f"  原子 {i}{marker}: {vec}")
    
    # 测试不同方法
    print("\n=== 不同标记方法比较 ===")
    methods = ["binary_marker", "amplify", "offset", "separate_channel"]
    
    for method in methods:
        pe = get_enhanced_positional_encoding(smiles, target_position, k=4, method=method)
        print(f"\n方法: {method}")
        for i, vec in enumerate(pe):
            marker = " ★" if i == target_position else ""
            print(f"  原子 {i}{marker}: {vec}")
    
    # 测试完整数据集创建
    print("\n=== 数据集创建测试 ===")
    test_data = [
        {
            "smiles_from": "C#C",
            "smiles_to": "C#CC", 
            "operations": [{"position": "1", "atom": "C", "operation": "add_atom"}]
        },
        {
            "smiles_from": "CCC",
            "smiles_to": "CC(C)C",
            "operations": [{"position": "1", "atom": "C", "operation": "add_atom"}]
        },
    ]
    
    dataset = create_position_aware_dataset(test_data, k=4, method="binary_marker")
    for i, data in enumerate(dataset):
        print(f"\n样本 {i}: {data['smiles']}")
        print(f"目标位置: {data['target_position']}")
        print(f"操作: {data['operation']}")
        print(f"PE形状: {data['pe'].shape}")
        print("各原子编码:")
        for j, pe_vec in enumerate(data['pe']):
            marker = " ★" if j == data['target_position'] else ""
            print(f"  原子 {j}{marker}: {pe_vec}")
    
    # 运行详细测试
    test_enhanced_pe()