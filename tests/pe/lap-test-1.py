import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class MolecularPEDataset(Dataset):
    def __init__(self, data_entries, pe_dim=6):
        self.data_entries = data_entries
        self.pe_dim = pe_dim
        
    def __len__(self):
        return len(self.data_entries)
    
    def __getitem__(self, idx):
        entry = self.data_entries[idx]
        
        # 获取分子图和位置编码
        graph_data, atom_features = self.smiles_to_graph_data(entry['smiles_from'])
        
        # 操作位置作为标签
        position_label = int(entry['operations'][0]['position'])
        
        return {
            'atom_features': torch.FloatTensor(atom_features),
            'adj_matrix': torch.FloatTensor(graph_data['adj_matrix']),
            'pe': torch.FloatTensor(graph_data['pe']),
            'position_label': torch.LongTensor([position_label]),
            'smiles': entry['smiles_from'],
            'operation': entry['operations'][0]['operation'],
            'num_atoms': atom_features.shape[0]  # 添加原子数信息
        }
    
    def smiles_to_graph_data(self, smiles):
        """将SMILES转换为图数据"""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # 构建分子图
        G = nx.Graph()
        num_atoms = mol.GetNumAtoms()
        
        # 添加节点
        atom_features = []
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            G.add_node(atom_idx)
            
            # 原子特征: [原子序数, 度, 形式电荷, 杂化类型, 是否在环中]
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                atom.IsInRing()
            ]
            atom_features.append(features)
        
        # 添加边
        adj_matrix = np.zeros((num_atoms, num_atoms))
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            G.add_edge(i, j)
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
        
        # 计算拉普拉斯位置编码
        pe = self.get_laplacian_pe(G, self.pe_dim)
        
        graph_data = {
            'adj_matrix': adj_matrix,
            'pe': pe
        }
        
        return graph_data, np.array(atom_features)

    def get_laplacian_pe(self, graph, k):
        """计算拉普拉斯位置编码"""
        try:
            # 计算归一化拉普拉斯矩阵
            L = nx.normalized_laplacian_matrix(graph).astype(float)
            
            # 特征分解
            eigenvalues, eigenvectors = np.linalg.eigh(L.toarray())
            
            # 选择最小的k个非零特征值对应的特征向量
            # 跳过第一个特征值为0的特征向量
            valid_indices = np.where(eigenvalues > 1e-8)[0]
            if len(valid_indices) == 0:
                # 如果所有特征值都很小，使用随机编码
                return np.random.normal(0, 0.1, (len(graph.nodes()), k))
            
            k_actual = min(k, len(valid_indices))
            selected_indices = valid_indices[:k_actual]
            pe_vectors = eigenvectors[:, selected_indices]
            
            # 如果维度不够，用零填充
            if pe_vectors.shape[1] < k:
                padding = np.zeros((pe_vectors.shape[0], k - pe_vectors.shape[1]))
                pe_vectors = np.hstack([pe_vectors, padding])
            
            return pe_vectors
            
        except np.linalg.LinAlgError:
            # 如果特征分解失败，使用随机编码
            return np.random.normal(0, 0.1, (len(graph.nodes()), k))

# 自定义collate函数来处理不同大小的张量
def custom_collate_fn(batch):
    """自定义collate函数处理不同大小的邻接矩阵"""
    # 提取所有样本的原子数
    max_atoms = max(item['num_atoms'] for item in batch)
    
    # 对每个样本进行填充
    for item in batch:
        num_atoms = item['num_atoms']
        if num_atoms < max_atoms:
            # 填充原子特征
            padding = torch.zeros((max_atoms - num_atoms, item['atom_features'].shape[1]))
            item['atom_features'] = torch.cat([item['atom_features'], padding], dim=0)
            
            # 填充位置编码
            padding = torch.zeros((max_atoms - num_atoms, item['pe'].shape[1]))
            item['pe'] = torch.cat([item['pe'], padding], dim=0)
            
            # 填充邻接矩阵
            adj_padding = torch.zeros((max_atoms - num_atoms, num_atoms))
            item['adj_matrix'] = torch.cat([item['adj_matrix'], adj_padding], dim=0)
            adj_padding = torch.zeros((max_atoms, max_atoms - num_atoms))
            item['adj_matrix'] = torch.cat([item['adj_matrix'], adj_padding], dim=1)
    
    # 使用默认collate函数处理其他字段
    return {
        'atom_features': torch.stack([item['atom_features'] for item in batch]),
        'adj_matrix': torch.stack([item['adj_matrix'] for item in batch]),
        'pe': torch.stack([item['pe'] for item in batch]),
        'position_label': torch.stack([item['position_label'] for item in batch]).squeeze(),
        'smiles': [item['smiles'] for item in batch],
        'operation': [item['operation'] for item in batch],
        'num_atoms': [item['num_atoms'] for item in batch]
    }

class PositionPredictionModel(nn.Module):
    def __init__(self, atom_feat_dim, pe_dim, hidden_dim=64, max_atoms=9):
        super(PositionPredictionModel, self).__init__()
        self.max_atoms = max_atoms
        
        # 原子特征编码器
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_feat_dim + pe_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 全局注意力池化
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # 位置预测器
        self.position_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, max_atoms)  # 预测哪个位置会被操作
        )
        
    def forward(self, atom_features, pe, adj_matrix, num_atoms_list=None):
        batch_size, max_atoms, _ = atom_features.shape
        
        # 结合原子特征和位置编码
        combined_features = torch.cat([atom_features, pe], dim=-1)
        
        # 编码原子特征
        atom_embeddings = self.atom_encoder(combined_features)
        
        # 使用注意力机制获取全局信息
        attention_out, _ = self.attention(
            atom_embeddings, atom_embeddings, atom_embeddings
        )
        
        # 全局池化
        global_feature = torch.mean(attention_out, dim=1)
        
        # 为每个原子预测操作概率
        position_logits = []
        for i in range(max_atoms):
            # 结合原子特征和全局特征
            atom_global = torch.cat([atom_embeddings[:, i, :], global_feature], dim=-1)
            logits = self.position_predictor(atom_global)
            position_logits.append(logits.unsqueeze(1))
        
        position_logits = torch.cat(position_logits, dim=1)
        
        return position_logits

def generate_test_data():
    """生成测试数据"""
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
        {
            "smiles_from": "C=CC", 
            "smiles_to": "C=CCBr",
            "operations": [{"position": "2", "atom": "Br", "operation": "replace_atom"}]
        },
        {
            "smiles_from": "CCO",
            "smiles_to": "CCOC",
            "operations": [{"position": "2", "atom": "C", "operation": "add_atom"}]
        },
        {
            "smiles_from": "C1CCCC1",
            "smiles_to": "C1CCCC1C",
            "operations": [{"position": "0", "atom": "C", "operation": "add_atom"}]
        },
        {
            "smiles_from": "CC=O",
            "smiles_to": "CC(=O)C",
            "operations": [{"position": "1", "atom": "C", "operation": "add_atom"}]
        },
        {
            "smiles_from": "CCl",
            "smiles_to": "CBr", 
            "operations": [{"position": "1", "atom": "Br", "operation": "replace_atom"}]
        },
        {
            "smiles_from": "CN",
            "smiles_to": "CNC",
            "operations": [{"position": "1", "atom": "C", "operation": "add_atom"}]
        }
    ]
    return test_data

def visualize_molecule_with_pe(smiles, pe, title):
    """可视化分子和位置编码"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制分子
    img = Draw.MolToImage(mol, size=(300, 300))
    ax1.imshow(img)
    ax1.set_title(f'Molecule: {smiles}')
    ax1.axis('off')
    
    # 绘制位置编码热图
    im = ax2.imshow(pe, cmap='coolwarm', aspect='auto')
    ax2.set_title('Laplacian Positional Encoding')
    ax2.set_xlabel('PE Dimension')
    ax2.set_ylabel('Atom Index')
    plt.colorbar(im, ax=ax2)
    
    # 为每个原子添加数值标注
    for i in range(pe.shape[0]):
        for j in range(pe.shape[1]):
            ax2.text(j, i, f'{pe[i, j]:.2f}', ha='center', va='center', 
                    fontsize=8, color='black' if abs(pe[i, j]) < 0.5 else 'white')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def test_pe_calculation():
    """测试位置编码计算"""
    test_smiles = ["C#C", "CCC", "C1CCCC1", "C=CC"]
    
    print("=== 测试位置编码计算 ===\n")
    
    for smiles in test_smiles:
        mol = Chem.MolFromSmiles(smiles)
        G = nx.Graph()
        
        # 构建图
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx())
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            G.add_edge(i, j)
        
        # 计算PE
        dataset = MolecularPEDataset([])
        pe = dataset.get_laplacian_pe(G, 6)
        
        print(f"分子: {smiles}")
        print(f"原子数量: {len(G.nodes())}")
        print("位置编码:")
        for i, atom_pe in enumerate(pe):
            print(f"  原子 {i}: {atom_pe}")
        print("-" * 50)
        
        # 可视化
        visualize_molecule_with_pe(smiles, pe, f"PE for {smiles}")

def train_model():
    """训练测试模型"""
    print("\n=== 训练测试模型 ===\n")
    
    # 生成数据
    test_data = generate_test_data()
    dataset = MolecularPEDataset(test_data, pe_dim=6)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
    
    # 初始化模型
    model = PositionPredictionModel(
        atom_feat_dim=5,  # 原子特征维度
        pe_dim=6,         # PE维度
        hidden_dim=32,    # 隐藏层维度
        max_atoms=9       # 最大原子数
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    model.train()
    for epoch in range(50):
        total_loss = 0
        for batch in dataloader:
            atom_features = batch['atom_features']
            pe = batch['pe']
            adj_matrix = batch['adj_matrix']
            position_label = batch['position_label']
            num_atoms_list = batch['num_atoms']
            
            # 前向传播
            position_logits = model(atom_features, pe, adj_matrix, num_atoms_list)
            
            # 计算损失 - 只对实际存在的原子计算损失
            batch_size, max_atoms, num_classes = position_logits.shape
            loss = 0
            valid_loss_count = 0
            
            for i in range(batch_size):
                valid_atoms = num_atoms_list[i]
                if valid_atoms > 0 and position_label[i] < valid_atoms:
                    # 只考虑有效的原子
                    sample_logits = position_logits[i, :valid_atoms, :valid_atoms]
                    sample_labels = position_label[i].unsqueeze(0).repeat(valid_atoms)
                    loss += criterion(sample_logits, sample_labels)
                    valid_loss_count += 1
            
            if valid_loss_count > 0:
                loss = loss / valid_loss_count
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/50], Loss: {total_loss/len(dataloader):.4f}')
    
    # 测试预测
    print("\n=== 测试预测 ===\n")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            atom_features = batch['atom_features']
            pe = batch['pe']
            adj_matrix = batch['adj_matrix']
            true_position = batch['position_label']
            smiles = batch['smiles']
            num_atoms_list = batch['num_atoms']
            
            position_logits = model(atom_features, pe, adj_matrix, num_atoms_list)
            
            for j in range(len(smiles)):
                valid_atoms = num_atoms_list[j]
                if valid_atoms > 0:
                    # 只考虑有效的原子
                    probs = torch.softmax(position_logits[j, :valid_atoms, :valid_atoms], dim=-1)
                    pred_position = torch.argmax(probs.mean(dim=1))  # 平均所有位置的预测
                    
                    print(f"分子: {smiles[j]}")
                    print(f"真实操作位置: {true_position[j].item()}")
                    print(f"预测位置分布: {probs.mean(dim=1).squeeze().detach().numpy()}")
                    print(f"预测操作位置: {pred_position.item()}")
                    print(f"是否正确: {pred_position.item() == true_position[j].item()}")
                    print("-" * 30)

def analyze_pe_patterns():
    """分析PE模式"""
    print("\n=== 分析PE模式 ===\n")
    
    test_cases = [
        ("C#C", "线性对称分子"),
        ("CCC", "线性不对称分子"), 
        ("C1CCCC1", "环状对称分子"),
        ("CCO", "含杂原子分子")
    ]
    
    for smiles, description in test_cases:
        mol = Chem.MolFromSmiles(smiles)
        G = nx.Graph()
        
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx())
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            G.add_edge(i, j)
        
        dataset = MolecularPEDataset([])
        pe = dataset.get_laplacian_pe(G, 6)
        
        print(f"{description}: {smiles}")
        print("原子位置编码模式:")
        for i in range(len(pe)):
            atom = mol.GetAtomWithIdx(i)
            symbol = atom.GetSymbol()
            degree = atom.GetDegree()
            print(f"  原子{i}({symbol}, 度{degree}): {pe[i]}")
        
        # 计算原子间的PE距离
        print("原子间PE欧氏距离:")
        for i in range(len(pe)):
            for j in range(i+1, len(pe)):
                distance = np.linalg.norm(pe[i] - pe[j])
                print(f"  原子{i}-原子{j}: {distance:.4f}")
        print("=" * 60)

if __name__ == "__main__":
    print("分子位置编码测试脚本")
    print("=" * 50)
    
    # 测试PE计算
    test_pe_calculation()
    
    # 分析PE模式
    analyze_pe_patterns()
    
    # 训练测试模型
    train_model()
    
    print("\n=== 测试完成 ===")