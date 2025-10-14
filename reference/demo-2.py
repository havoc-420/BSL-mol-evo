from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType
import torch
import numpy as np

def  smile_to_graph_xyz(smile, types):
    mol = Chem.MolFromSmiles(smile)
    
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
    

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    # Chem.MolFromSmiles('CC[C@H](F)Cl', useChirality=True)
    # mol = Chem.MolFromSmiles(smile, useChirality=True, isomericSmiles= True, kekuleSmiles = True)
    # pdb.set_trace()
    c_size = mol.GetNumAtoms()
    
    features = []
    node_dict = {}
    
    for atom in mol.GetAtoms():
        node_dict[str(atom.GetIdx())] = atom
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    
    # bonds
    num_bond_features = 5
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # if allowable_features['possible_bond_dirs'].index(bond.GetBondDir()) !=0:
                # pdb.set_trace()
                # print(smile, allowable_features['possible_bond_dirs'].index(bond.GetBondDir()))
            edge_feature = [
                bond.GetBondTypeAsDouble(), 
                bond.GetIsAromatic(),
                # 芳香键
                bond.GetIsConjugated(),
                # 是否为共轭键
                bond.IsInRing(),             
                allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    
    return c_size, features, edge_index, edge_attr