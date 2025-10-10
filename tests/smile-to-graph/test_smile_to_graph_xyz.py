import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType, BondType
import numpy as np

# 定义需要的全局变量和辅助函数
try:
    ETKDG_PARAMS = AllChem.ETKDGv3()
    ETKDG_VERSION_USED = "ETKDGv3"
except AttributeError:
    ETKDG_PARAMS = AllChem.ETKDGv2()
    ETKDG_VERSION_USED = "ETKDGv2 (fallback)"

bonds = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3}

def one_hot(tensor, num_classes):
    """
    将整数张量转换为 one-hot 编码。
    :param tensor: 输入张量。
    :param num_classes: 类别数量。
    :return: one-hot 编码张量。
    """
    return torch.eye(num_classes, dtype=torch.float)[tensor]

def smile_to_graph_xyz(smile, types):
    """
    将SMILES字符串转换为图结构表示，包括3D坐标信息。
    
    Args:
        smile (str): SMILES字符串
        types (dict): 原子类型映射字典
        
    Returns:
        tuple: (x, z, pos, edge_index, edge_attr) 其中:
            - x: 节点特征矩阵
            - z: 原子序数
            - pos: 3D坐标
            - edge_index: 边索引
            - edge_attr: 边属性
    """
    mol = Chem.MolFromSmiles(smile)
    
    if mol is None:
        print(f"无法解析SMILES: {smile}")
        return None, None, None, None, None
    
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, ETKDG_PARAMS)
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        pos = torch.tensor(pos, dtype=torch.float)
    except Exception as e:
        # 去除立体构型
        Chem.RemoveStereochemistry(mol)
        try:
            # 第二次尝试生成坐标
            AllChem.EmbedMolecule(mol, ETKDG_PARAMS)
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)
        except Exception as e:
            print(f'Cannot generate 3D coordinates for {smile}')
            pos = None  # 明确设置为None
    
    # 检查是否成功生成坐标
    if pos is None:
        # 返回空的结果而不是未定义的变量
        return None, None, None, None, None
        
    # 获取原子特征
    type_idx = []
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in types:
            continue  
        type_idx.append(types[symbol])
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    # 获取键信息
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = one_hot(edge_type, num_classes=len(bonds))

    # 排序边
    N = mol.GetNumAtoms()
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    edge_attr = edge_attr[perm]

    # 计算氢原子数量
    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = torch.zeros(N, dtype=torch.float).scatter_add_(0, col, hs[row]).tolist()

    # 构建节点特征
    x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                        dtype=torch.float).t().contiguous()
    x = torch.cat([x1, x2], dim=-1)  
    
    return x, z, pos, edge_index, edge_attr

def view_coordinates_details(smile="c1ccccc1", types={'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}):
    """查看分子坐标的详细信息"""
    print(f"分子 SMILES: {smile}")
    
    x, z, pos, edge_index, edge_attr = smile_to_graph_xyz(smile, types)
    
    if pos is not None:
        print(f"\n坐标矩阵维度: {pos.shape}")
        print("\n原子坐标详情:")
        print("原子序号\t原子类型\tX坐标\t\tY坐标\t\tZ坐标")
        print("-" * 60)
        
        # 获取原子类型列表
        atom_symbols = []
        mol = Chem.MolFromSmiles(smile)
        mol = Chem.AddHs(mol)
        for atom in mol.GetAtoms():
            atom_symbols.append(atom.GetSymbol())
        
        for i in range(pos.shape[0]):
            atom_type = atom_symbols[i] if i < len(atom_symbols) else "Unknown"
            print(f"{i+1}\t\t{atom_type}\t\t{pos[i, 0]:.4f}\t\t{pos[i, 1]:.4f}\t\t{pos[i, 2]:.4f}")
        
        print(f"\n坐标范围:")
        print(f"X轴: {torch.min(pos[:, 0]):.4f} 到 {torch.max(pos[:, 0]):.4f}")
        print(f"Y轴: {torch.min(pos[:, 1]):.4f} 到 {torch.max(pos[:, 1]):.4f}")
        print(f"Z轴: {torch.min(pos[:, 2]):.4f} 到 {torch.max(pos[:, 2]):.4f}")
        
        print(f"\n分子几何中心:")
        centroid = torch.mean(pos, dim=0)
        print(f"({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")
        
        print(f"\n原子间距离统计:")
        distances = []
        for i in range(pos.shape[0]):
            for j in range(i+1, pos.shape[0]):
                dist = torch.sqrt(torch.sum((pos[i] - pos[j])**2))
                distances.append(dist.item())
        
        print(f"最小距离: {min(distances):.4f}")
        print(f"最大距离: {max(distances):.4f}")
        print(f"平均距离: {np.mean(distances):.4f}")
    else:
        print("无法生成坐标信息")

def test_smile_to_graph_xyz():
    """测试 smile_to_graph_xyz 函数"""
    # 定义原子类型映射
    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    
    # 测试用的SMILES字符串
    test_smiles = [
        "CCO",           # 乙醇
        "c1ccccc1",      # 苯
        "CC(=O)O",       # 乙酸
        "[H]C([H])([H])[H]",  # 甲烷
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # 咖啡因
    ]
    
    print("测试 smile_to_graph_xyz 函数:")
    print("=" * 50)
    
    for i, smile in enumerate(test_smiles):
        print(f"\n测试分子 {i+1}: {smile}")
        try:
            x, z, pos, edge_index, edge_attr = smile_to_graph_xyz(smile, types)
            
            if x is not None:
                print(f"  节点数量: {x.shape[0]}")
                print(f"  节点特征维度: {x.shape[1]}")
                print(f"  x: {x}")
                print(f"  边数量: {edge_index.shape[1]}")
                print(f"  边特征维度: {edge_attr.shape[1]}")
                print(f"  坐标维度: {pos.shape}")
                print(f"  原子序数: {z.tolist()}")
                print("  成功生成图结构!")
            else:
                print("  无法生成图结构")
        except Exception as e:
            print(f"  处理过程中出现错误: {str(e)}")
    
    print("\n" + "=" * 50)
    print("测试完成!")

if __name__ == "__main__":
    test_smile_to_graph_xyz()
    print("\n" + "=" * 50)
    print("详细坐标分析:")
    # view_coordinates_details()