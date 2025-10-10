#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于分子进化的图神经网络实现

该模块实现了基于分子进化思想的图神经网络，其中：
- 节点代表完整的SMILES分子
- 边表示分子间的进化关系
- 边特征包含原子类型、操作类型和属性变化信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, GATConv, global_mean_pool, BatchNorm
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from typing import List, Tuple, Dict


class MoleculeGNN(nn.Module):
    """
    基于分子进化的图神经网络
    
    该网络使用NNConv层，将分子作为节点，进化关系作为边，
    边特征包含位置、原子类型、操作类型和属性变化信息。
    """
    
    def __init__(self, node_feature_dim: int = 15, edge_feature_dim: int = 30,
                 hidden_dim: int = 128, output_dim: int = 64, num_layers: int = 3):
        """
        初始化分子GNN

        Args:
            node_feature_dim: 节点特征维度（默认15，对应QM9的15个量子化学属性）
            edge_feature_dim: 边特征维度（30 = 5个原子类型 + 6个操作类型 + 15个属性变化 + 3个位置特征 + 1个相似性特征）
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: GNN层数
        """
        super(MoleculeGNN, self).__init__()
        
        # 节点嵌入层
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        
        # 边网络（用于NNConv）
        self.edge_networks = nn.ModuleList()
        self.nnconv_layers = nn.ModuleList()
        
        # 第一层
        edge_net = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        self.edge_networks.append(edge_net)
        self.nnconv_layers.append(NNConv(hidden_dim, hidden_dim, edge_net, aggr='mean'))
        
        # 中间层
        for _ in range(num_layers - 2):
            edge_net = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim)
            )
            self.edge_networks.append(edge_net)
            self.nnconv_layers.append(NNConv(hidden_dim, hidden_dim, edge_net, aggr='mean'))
        
        # 最后一层
        if num_layers > 1:
            edge_net = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * output_dim)
            )
            self.edge_networks.append(edge_net)
            self.nnconv_layers.append(NNConv(hidden_dim, output_dim, edge_net, aggr='mean'))
        
        # 输出层
        self.output_dim = output_dim
        self.num_layers = num_layers

    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: 包含节点特征、边索引和边特征的图数据
            
        Returns:
            节点表示张量
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 节点嵌入
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # 多层NNConv
        for i in range(self.num_layers):
            x = self.nnconv_layers[i](x, edge_index, edge_attr)
            if i < self.num_layers - 1:  # 最后一层不加激活函数
                x = F.relu(x)
        
        # 全局池化
        if hasattr(data, 'batch'):
            x = global_mean_pool(x, data.batch)
        
        return x


class MoleculeEvolutionPredictor(nn.Module):
    """
    分子进化属性变化预测器
    
    基于源分子和边特征预测目标分子的属性变化
    """
    
    def __init__(self, node_feature_dim: int = 15, edge_feature_dim: int = 30,
                 hidden_dim: int = 128, property_dim: int = 15):
        """
        初始化预测器

        Args:
            node_feature_dim: 节点特征维度（15个QM9量子化学属性）
            edge_feature_dim: 边特征维度（30维）
            hidden_dim: 隐藏层维度
            property_dim: 属性变化维度（15个量子化学属性）
        """
        super(MoleculeEvolutionPredictor, self).__init__()
        
        self.gnn = MoleculeGNN(node_feature_dim, edge_feature_dim, hidden_dim, hidden_dim)
        
        # 属性变化预测头
        self.property_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, property_dim)
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: 图数据
            
        Returns:
            属性变化预测值
        """
        # 获取图表示
        graph_embedding = self.gnn(data)
        
        # 预测属性变化
        property_changes = self.property_predictor(graph_embedding)
        
        return property_changes


class EnhancedMoleculeGNN(nn.Module):
    """
    增强的分子进化图神经网络

    结合NNConv和GAT，添加残差连接和归一化层
    """

    def __init__(self, node_feature_dim: int = 15, edge_feature_dim: int = 30,
                 hidden_dim: int = 128, output_dim: int = 64, num_layers: int = 3,
                 heads: int = 4, dropout: float = 0.1):
        """
        初始化增强GNN

        Args:
            node_feature_dim: 节点特征维度
            edge_feature_dim: 边特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: GNN层数
            heads: 注意力头数
            dropout: Dropout率
        """
        super(EnhancedMoleculeGNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # 节点嵌入层
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)

        # NNConv层（处理边特征）
        self.nnconv_layers = nn.ModuleList()
        self.nnconv_bns = nn.ModuleList()

        # GAT层（处理节点注意力）
        self.gat_layers = nn.ModuleList()
        self.gat_bns = nn.ModuleList()

        # 边网络
        self.edge_networks = nn.ModuleList()

        # 构建多层网络
        for i in range(num_layers):
            # 边网络
            edge_net = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim)
            )
            self.edge_networks.append(edge_net)

            # NNConv层
            self.nnconv_layers.append(NNConv(hidden_dim, hidden_dim, edge_net, aggr='mean'))
            self.nnconv_bns.append(BatchNorm(hidden_dim))

            # GAT层
            if i == num_layers - 1:
                # 最后一层输出到目标维度
                self.gat_layers.append(GATConv(hidden_dim, output_dim, heads=1, concat=False, dropout=dropout))
            else:
                self.gat_layers.append(GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout))
                self.gat_bns.append(BatchNorm(hidden_dim))

        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # 残差连接
        self.residual_connections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])

    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播

        Args:
            data: 图数据

        Returns:
            节点表示张量
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 节点嵌入
        x = self.node_embedding(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 多层消息传递
        for i in range(self.num_layers):
            # 保存残差连接
            residual = x

            # NNConv处理边特征
            x = self.nnconv_layers[i](x, edge_index, edge_attr)
            x = self.nnconv_bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # GAT处理节点注意力
            x = self.gat_layers[i](x, edge_index)

            if i < self.num_layers - 1:
                x = self.gat_bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            # 残差连接
            if i > 0 and residual.shape == x.shape:
                x = x + self.residual_connections[i-1](residual)

        # 全局池化
        if hasattr(data, 'batch'):
            x = global_mean_pool(x, data.batch)

        return x


class MoleculeGNNWithFingerprint(nn.Module):
    """
    基于分子指纹的图神经网络
    
    使用Morgan指纹作为节点特征的GNN实现
    """
    
    def __init__(self, node_feature_dim: int = 2048, edge_feature_dim: int = 30,
                 hidden_dim: int = 128, output_dim: int = 64, num_layers: int = 3):
        """
        初始化分子GNN

        Args:
            node_feature_dim: 节点特征维度（默认2048，对应Morgan指纹）
            edge_feature_dim: 边特征维度（30维）
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: GNN层数
        """
        super(MoleculeGNNWithFingerprint, self).__init__()
        
        # 节点嵌入层
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        
        # 边网络（用于NNConv）
        self.edge_networks = nn.ModuleList()
        self.nnconv_layers = nn.ModuleList()
        
        # 第一层
        edge_net = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        self.edge_networks.append(edge_net)
        self.nnconv_layers.append(NNConv(hidden_dim, hidden_dim, edge_net, aggr='mean'))
        
        # 中间层
        for _ in range(num_layers - 2):
            edge_net = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim)
            )
            self.edge_networks.append(edge_net)
            self.nnconv_layers.append(NNConv(hidden_dim, hidden_dim, edge_net, aggr='mean'))
        
        # 最后一层
        if num_layers > 1:
            edge_net = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * output_dim)
            )
            self.edge_networks.append(edge_net)
            self.nnconv_layers.append(NNConv(hidden_dim, output_dim, edge_net, aggr='mean'))
        
        # 输出层
        self.output_dim = output_dim
        self.num_layers = num_layers

    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: 包含节点特征、边索引和边特征的图数据
            
        Returns:
            节点表示张量
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 节点嵌入
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # 多层NNConv
        for i in range(self.num_layers):
            x = self.nnconv_layers[i](x, edge_index, edge_attr)
            if i < self.num_layers - 1:  # 最后一层不加激活函数
                x = F.relu(x)
        
        # 全局池化
        if hasattr(data, 'batch'):
            x = global_mean_pool(x, data.batch)
        
        return x


class MoleculeEvolutionPredictorWithFingerprint(nn.Module):
    """
    基于分子指纹的分子进化属性变化预测器
    """
    
    def __init__(self, node_feature_dim: int = 2048, edge_feature_dim: int = 30,
                 hidden_dim: int = 128, property_dim: int = 15):
        """
        初始化预测器

        Args:
            node_feature_dim: 节点特征维度（2048维Morgan指纹）
            edge_feature_dim: 边特征维度（30维）
            hidden_dim: 隐藏层维度
            property_dim: 属性变化维度（15个量子化学属性）
        """
        super(MoleculeEvolutionPredictorWithFingerprint, self).__init__()
        
        self.gnn = MoleculeGNNWithFingerprint(node_feature_dim, edge_feature_dim, hidden_dim, hidden_dim)
        
        # 属性变化预测头
        self.property_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, property_dim)
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: 图数据
            
        Returns:
            属性变化预测值
        """
        # 获取图表示
        graph_embedding = self.gnn(data)
        
        # 预测属性变化
        property_changes = self.property_predictor(graph_embedding)
        
        return property_changes


class EnhancedMoleculeEvolutionPredictor(nn.Module):
    """
    增强的分子进化属性变化预测器

    使用增强GNN模型进行属性变化预测
    """

    def __init__(self, node_feature_dim: int = 15, edge_feature_dim: int = 30,
                 hidden_dim: int = 128, property_dim: int = 15, num_layers: int = 3,
                 heads: int = 4, dropout: float = 0.1):
        """
        初始化增强预测器

        Args:
            node_feature_dim: 节点特征维度
            edge_feature_dim: 边特征维度
            hidden_dim: 隐藏层维度
            property_dim: 属性变化维度
            num_layers: GNN层数
            heads: 注意力头数
            dropout: Dropout率
        """
        super(EnhancedMoleculeEvolutionPredictor, self).__init__()

        self.gnn = EnhancedMoleculeGNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout
        )

        # 属性变化预测头
        self.property_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, property_dim)
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播

        Args:
            data: 图数据

        Returns:
            属性变化预测值
        """
        # 获取图表示
        graph_embedding = self.gnn(data)

        # 预测属性变化
        property_changes = self.property_predictor(graph_embedding)

        return property_changes


class MoleculeEvolutionTransformer(nn.Module):
    """
    分子进化转换器
    
    根据起始分子特征、原子和操作类型预测目标分子特征
    """
    
    def __init__(self, node_feature_dim: int = 2048, edge_feature_dim: int = 30,
                 hidden_dim: int = 128, output_dim: int = 2048):
        """
        初始化转换器

        Args:
            node_feature_dim: 节点特征维度（默认2048，对应Morgan指纹）
            edge_feature_dim: 边特征维度（30维）
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（默认2048，对应Morgan指纹）
        """
        super(MoleculeEvolutionTransformer, self).__init__()
        
        # 起始分子特征编码器
        self.source_molecule_encoder = nn.Linear(node_feature_dim, hidden_dim)
        
        # 边特征编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 转换网络
        self.transformer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 拼接起始分子和边特征
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, source_features: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            source_features: 起始分子特征 [batch_size, node_feature_dim]
            edge_features: 边特征 [batch_size, edge_feature_dim]
            
        Returns:
            目标分子特征预测值 [batch_size, output_dim]
        """
        # 编码起始分子特征
        source_encoded = self.source_molecule_encoder(source_features)
        
        # 编码边特征
        edge_encoded = self.edge_encoder(edge_features)
        
        # 拼接特征
        combined_features = torch.cat([source_encoded, edge_encoded], dim=1)
        
        # 转换
        target_features = self.transformer(combined_features)
        
        return target_features


def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    将SMILES转换为Morgan指纹
    
    Args:
        smiles: SMILES字符串
        radius: Morgan指纹半径
        n_bits: 指纹位数
        
    Returns:
        指纹数组
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        
        # 生成Morgan指纹，使用新的API避免弃用警告
        try:
            # 尝试使用新的MorganGenerator API
            generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
            fingerprint = generator.GetFingerprint(mol)
            return np.array(fingerprint)
        except AttributeError:
            # 如果新API不可用，回退到旧API
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return np.array(fingerprint)
    except:
        return np.zeros(n_bits)


def atom_type_to_onehot(atom_symbol: str) -> List[int]:
    """
    将原子类型转换为独热编码
    
    Args:
        atom_symbol: 原子符号
        
    Returns:
        独热编码向量
    """
    atom_types = ['C', 'N', 'O', 'F', 'P']  # 常见原子类型
    onehot = [0] * len(atom_types)
    
    if atom_symbol in atom_types:
        idx = atom_types.index(atom_symbol)
        onehot[idx] = 1
        
    return onehot


def operation_type_to_onehot(operation_type: str) -> List[int]:
    """
    将操作类型转换为独热编码
    
    Args:
        operation_type: 操作类型
        
    Returns:
        独热编码向量
    """
    op_types = ['add', 'replace', 'del', 'add_multi', 'del_multi', 'complex']
    onehot = [0] * len(op_types)
    
    if operation_type in op_types:
        idx = op_types.index(operation_type)
        onehot[idx] = 1
        
    return onehot


def calculate_molecular_similarity(smiles1: str, smiles2: str) -> float:
    """
    计算两个分子的Tanimoto相似度

    Args:
        smiles1: 第一个分子的SMILES
        smiles2: 第二个分子的SMILES

    Returns:
        Tanimoto相似度 (0-1)
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return 0.0

        # 计算Morgan指纹相似度
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)

        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0


def prepare_edge_features(row: pd.Series, property_stats: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """
    准备边特征

    Args:
        row: 数据行
        property_stats: 属性统计信息（均值和标准差）

    Returns:
        边特征向量
    """
    # 原子类型特征
    atom_features = atom_type_to_onehot(row['to_atom_symbol'])

    # 操作类型特征
    op_features = operation_type_to_onehot(row['operation_type'])

    # 属性变化特征
    property_changes = []
    property_names = ['A_change', 'B_change', 'C_change', 'mu_change', 'alpha_change',
                      'homo_change', 'lumo_change', 'gap_change', 'r2_change', 'zpve_change',
                      'U0_change', 'U_change', 'H_change', 'G_change', 'Cv_change']

    for prop in property_names:
        if prop in row and not pd.isna(row[prop]):
            value = row[prop]
            property_changes.append(value)
        else:
            property_changes.append(0.0)

    # 位置敏感特征
    position_features = []
    if 'from_heavy_atoms' in row and 'to_heavy_atoms' in row:
        position_features = [
            row['from_heavy_atoms'],
            row['to_heavy_atoms'],
            row['to_heavy_atoms'] - row['from_heavy_atoms']  # 原子数变化
        ]
    else:
        position_features = [0, 0, 0]

    # 分子结构相似性特征
    similarity_features = []
    if 'smiles_from' in row and 'smiles_to' in row:
        similarity = calculate_molecular_similarity(row['smiles_from'], row['smiles_to'])
        similarity_features = [similarity]
    else:
        similarity_features = [0.0]

    # 组合所有特征
    edge_features = atom_features + op_features + property_changes + position_features + similarity_features

    return np.array(edge_features, dtype=np.float32)


def load_qm9_properties() -> Dict[str, np.ndarray]:
    """
    加载所有QM9分子的原始属性

    Returns:
        SMILES到属性向量的映射
    """
    smiles_to_properties = {}

    # QM9属性列名
    qm9_property_columns = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
                           'U0', 'U', 'H', 'G', 'Cv', 'A', 'B', 'C']

    # 加载所有重原子数的QM9数据文件
    for heavy_atoms in range(1, 10):
        file_path = f"mol_evo/dataset/data/qm9_smiles_heavy_{heavy_atoms}_atoms.csv"
        try:
            df = pd.read_csv(file_path)

            for _, row in df.iterrows():
                smiles = row['smiles']
                properties = []

                # 提取15个量子化学属性
                for prop in qm9_property_columns:
                    if prop in row and not pd.isna(row[prop]):
                        properties.append(row[prop])
                    else:
                        properties.append(0.0)

                smiles_to_properties[smiles] = np.array(properties, dtype=np.float32)

        except FileNotFoundError:
            print(f"警告: 找不到文件 {file_path}")
            continue

    return smiles_to_properties


def build_molecule_graph_with_properties(csv_file: str, max_molecules: int = None) -> Tuple[Data, Dict[str, int]]:
    """
    从CSV文件构建分子进化图（使用QM9原始属性作为节点特征）

    Args:
        csv_file: CSV文件路径
        max_molecules: 最大分子数（用于调试）

    Returns:
        图数据和SMILES到索引的映射
    """
    # 读取数据
    df = pd.read_csv(csv_file)

    if max_molecules:
        df = df.head(max_molecules)

    # 计算属性统计信息用于标准化
    property_names = ['A_change', 'B_change', 'C_change', 'mu_change', 'alpha_change',
                      'homo_change', 'lumo_change', 'gap_change', 'r2_change', 'zpve_change',
                      'U0_change', 'U_change', 'H_change', 'G_change', 'Cv_change']

    property_stats = {}
    for prop in property_names:
        if prop in df.columns:
            mean = df[prop].mean()
            std = df[prop].std()
            property_stats[prop] = (mean, std)

    # 收集所有唯一的SMILES
    all_smiles = set(df['smiles_from'].tolist() + df['smiles_to'].tolist())
    smiles_to_idx = {smiles: idx for idx, smiles in enumerate(all_smiles)}

    # 加载QM9原始属性
    print("正在加载QM9原始属性...")
    qm9_properties = load_qm9_properties()
    print(f"已加载 {len(qm9_properties)} 个分子的属性")

    # 构建节点特征 (使用QM9原始属性)
    num_nodes = len(all_smiles)
    node_features = []
    missing_smiles = []

    for smiles in all_smiles:
        if smiles in qm9_properties:
            properties = qm9_properties[smiles]
        else:
            # 如果找不到原始属性，使用默认值
            properties = np.zeros(15, dtype=np.float32)
            missing_smiles.append(smiles)
        node_features.append(properties)

    if missing_smiles:
        print(f"警告: 找不到 {len(missing_smiles)} 个分子的QM9属性")

    node_features = torch.FloatTensor(np.array(node_features))

    # 构建边索引和边特征
    edge_indices = []
    edge_features = []

    for _, row in df.iterrows():
        src_idx = smiles_to_idx[row['smiles_from']]
        dst_idx = smiles_to_idx[row['smiles_to']]

        edge_indices.append([src_idx, dst_idx])
        edge_feat = prepare_edge_features(row, property_stats)
        edge_features.append(edge_feat)

    edge_index = torch.LongTensor(edge_indices).t().contiguous()
    edge_attr = torch.FloatTensor(np.array(edge_features))

    # 创建图数据对象
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    print(f"图构建完成: {num_nodes} 个节点, {len(edge_indices)} 条边")

    return data, smiles_to_idx


def build_molecule_graph_with_fingerprints(csv_file: str, max_molecules: int = None) -> Tuple[Data, Dict[str, int]]:
    """
    从CSV文件构建分子进化图（使用Morgan指纹作为节点特征）
    
    Args:
        csv_file: CSV文件路径
        max_molecules: 最大分子数（用于调试）
        
    Returns:
        图数据和SMILES到索引的映射
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    
    if max_molecules:
        df = df.head(max_molecules)
    
    # 计算属性统计信息用于标准化
    property_names = ['A_change', 'B_change', 'C_change', 'mu_change', 'alpha_change',
                      'homo_change', 'lumo_change', 'gap_change', 'r2_change', 'zpve_change',
                      'U0_change', 'U_change', 'H_change', 'G_change', 'Cv_change']
    
    property_stats = {}
    for prop in property_names:
        if prop in df.columns:
            mean = df[prop].mean()
            std = df[prop].std()
            property_stats[prop] = (mean, std)
    
    # 收集所有唯一的SMILES
    all_smiles = set(df['smiles_from'].tolist() + df['smiles_to'].tolist())
    smiles_to_idx = {smiles: idx for idx, smiles in enumerate(all_smiles)}
    
    # 构建节点特征 (使用Morgan指纹)
    num_nodes = len(all_smiles)
    node_features = []
    
    # 为每个SMILES构建特征
    for smiles in all_smiles:
        fp = smiles_to_fingerprint(smiles)
        node_features.append(fp)
    
    node_features = torch.FloatTensor(np.array(node_features))
    
    # 构建边索引和边特征
    edge_indices = []
    edge_features = []
    
    for _, row in df.iterrows():
        src_idx = smiles_to_idx[row['smiles_from']]
        dst_idx = smiles_to_idx[row['smiles_to']]
        
        edge_indices.append([src_idx, dst_idx])
        edge_feat = prepare_edge_features(row, property_stats)
        edge_features.append(edge_feat)
    
    edge_index = torch.LongTensor(edge_indices).t().contiguous()
    edge_attr = torch.FloatTensor(np.array(edge_features))
    
    # 创建图数据对象
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    
    return data, smiles_to_idx


def prepare_evolution_data(csv_file: str, max_pairs: int = None):
    """
    准备分子进化数据用于转换器模型训练
    
    Args:
        csv_file: CSV文件路径
        max_pairs: 最大对数（用于调试）
        
    Returns:
        起始分子特征、边特征、目标分子特征和属性统计信息
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    
    if max_pairs:
        df = df.head(max_pairs)
    
    # 计算属性统计信息用于标准化
    property_names = ['A_change', 'B_change', 'C_change', 'mu_change', 'alpha_change',
                      'homo_change', 'lumo_change', 'gap_change', 'r2_change', 'zpve_change',
                      'U0_change', 'U_change', 'H_change', 'G_change', 'Cv_change']
    
    property_stats = {}
    for prop in property_names:
        if prop in df.columns:
            mean = df[prop].mean()
            std = df[prop].std()
            property_stats[prop] = (mean, std)
    
    # 准备特征
    source_features = []
    edge_features = []
    target_features = []
    
    for _, row in df.iterrows():
        # 起始分子特征
        source_fp = smiles_to_fingerprint(row['smiles_from'])
        source_features.append(source_fp)
        
        # 边特征
        edge_feat = prepare_edge_features(row, property_stats)
        edge_features.append(edge_feat)
        
        # 目标分子特征
        target_fp = smiles_to_fingerprint(row['smiles_to'])
        target_features.append(target_fp)
    
    source_features = torch.FloatTensor(np.array(source_features))
    edge_features = torch.FloatTensor(np.array(edge_features))
    target_features = torch.FloatTensor(np.array(target_features))
    
    return source_features, edge_features, target_features, property_stats


def split_data_by_molecules(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[List[int], List[int], List[int]]:
    """
    按分子分割数据，避免数据泄露

    Args:
        df: 数据DataFrame
        test_size: 测试集比例
        val_size: 验证集比例

    Returns:
        训练集、验证集、测试集索引
    """
    # 获取所有唯一的SMILES分子
    all_smiles = set(df['smiles_from'].tolist() + df['smiles_to'].tolist())
    all_smiles = list(all_smiles)

    # 随机打乱分子
    np.random.shuffle(all_smiles)

    # 计算分割点
    n_total = len(all_smiles)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    n_train = n_total - n_test - n_val

    # 分割分子
    train_molecules = set(all_smiles[:n_train])
    val_molecules = set(all_smiles[n_train:n_train + n_val])
    test_molecules = set(all_smiles[n_train + n_val:])

    # 根据分子分配数据索引
    train_idx, val_idx, test_idx = [], [], []

    for idx, row in df.iterrows():
        from_smiles = row['smiles_from']
        to_smiles = row['smiles_to']

        # 如果起始分子和目标分子都在训练集，则分配到训练集
        if from_smiles in train_molecules and to_smiles in train_molecules:
            train_idx.append(idx)
        # 如果起始分子和目标分子都在验证集，则分配到验证集
        elif from_smiles in val_molecules and to_smiles in val_molecules:
            val_idx.append(idx)
        # 如果起始分子和目标分子都在测试集，则分配到测试集
        elif from_smiles in test_molecules and to_smiles in test_molecules:
            test_idx.append(idx)
        # 否则分配到训练集（避免数据泄露）
        else:
            train_idx.append(idx)

    return train_idx, val_idx, test_idx


def train_model_enhanced(model: nn.Module, data: Data, target_changes: torch.Tensor,
                        train_idx: List[int], val_idx: List[int], test_idx: List[int] = None,
                        epochs: int = 200, lr: float = 0.001, weight_decay: float = 1e-5,
                        patience: int = 20, min_delta: float = 1e-4) -> Dict[str, List[float]]:
    """
    增强的训练函数

    Args:
        model: 模型
        data: 图数据
        target_changes: 目标属性变化
        train_idx: 训练集索引
        val_idx: 验证集索引
        test_idx: 测试集索引
        epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        patience: 早停耐心值
        min_delta: 最小改进阈值

    Returns:
        训练历史
    """
    # 优化器（带权重衰减）
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    # 损失函数（Huber损失，对异常值更鲁棒）
    criterion = nn.HuberLoss()

    train_idx_tensor = torch.LongTensor(train_idx)
    val_idx_tensor = torch.LongTensor(val_idx)
    test_idx_tensor = torch.LongTensor(test_idx) if test_idx is not None else None

    model.train()
    train_losses = []
    val_losses = []
    test_losses = [] if test_idx is not None else None
    learning_rates = []

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 前向传播
        predictions = model(data)

        # 对于当前模型结构，我们重复预测结果以匹配目标数量
        if predictions.shape[0] == 1:
            predictions = predictions.repeat(target_changes.shape[0], 1)

        # 计算训练损失
        train_loss = criterion(predictions[train_idx_tensor], target_changes[train_idx_tensor])

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 反向传播
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.item())

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_predictions = model(data)
            if val_predictions.shape[0] == 1:
                val_predictions = val_predictions.repeat(target_changes.shape[0], 1)

            val_loss = criterion(val_predictions[val_idx_tensor], target_changes[val_idx_tensor])
            val_losses.append(val_loss.item())

            # 测试阶段（如果提供）
            if test_idx is not None:
                test_predictions = model(data)
                if test_predictions.shape[0] == 1:
                    test_predictions = test_predictions.repeat(target_changes.shape[0], 1)
                test_loss = criterion(test_predictions[test_idx_tensor], target_changes[test_idx_tensor])
                test_losses.append(test_loss.item())

        model.train()

        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # 早停机制
        if val_loss.item() < best_val_loss - min_delta:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停: 验证损失在 {patience} 轮内未改善")
            break

        # 打印进度
        if (epoch + 1) % 10 == 0:
            test_info = f", Test Loss: {test_loss.item():.6f}" if test_idx is not None else ""
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.6f}, '
                  f'Val Loss: {val_loss.item():.6f}{test_info}, LR: {current_lr:.2e}')

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 返回训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates
    }

    if test_losses is not None:
        history['test_losses'] = test_losses

    return history


def train_transformer_model(model: nn.Module, source_features: torch.Tensor, 
                           edge_features: torch.Tensor, target_features: torch.Tensor,
                           epochs: int = 100, lr: float = 0.001, train_idx=None, val_idx=None):
    """
    训练转换器模型
    
    Args:
        model: 转换器模型
        source_features: 起始分子特征
        edge_features: 边特征
        target_features: 目标分子特征
        epochs: 训练轮数
        lr: 学习率
        train_idx: 训练集索引
        val_idx: 验证集索引
        
    Returns:
        损失历史
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 如果没有提供索引，则使用全部数据
    if train_idx is None:
        train_idx = list(range(source_features.shape[0]))
    
    train_idx_tensor = torch.LongTensor(train_idx)
    val_idx_tensor = torch.LongTensor(val_idx) if val_idx is not None else None
    
    model.train()
    train_losses = []
    val_losses = [] if val_idx is not None else None
    
    best_val_loss = float('inf') if val_idx is not None else None
    best_model_state = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向传播
        predictions = model(source_features[train_idx_tensor], edge_features[train_idx_tensor])
        
        # 计算训练损失
        train_loss = criterion(predictions, target_features[train_idx_tensor])
        
        # 反向传播
        train_loss.backward()
        optimizer.step()
        
        train_losses.append(train_loss.item())
        
        # 验证阶段
        if val_idx is not None:
            model.eval()
            with torch.no_grad():
                val_predictions = model(source_features[val_idx_tensor], edge_features[val_idx_tensor])
                val_loss = criterion(val_predictions, target_features[val_idx_tensor])
                val_losses.append(val_loss.item())
                
                # 保存最佳模型
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_model_state = model.state_dict().copy()
            
            model.train()
        
        if (epoch + 1) % 10 == 0:
            if val_idx is not None:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}')
            else:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss.item():.6f}')
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses


# 示例使用方法
def main():
    """
    主函数 - 演示如何使用分子进化GNN
    """
    print("分子进化图神经网络演示")
    
    # 注意：实际使用时需要提供真实的CSV文件路径
    # data, smiles_to_idx = build_molecule_graph('path/to/qm9-evo-pairs-step-1-with-properties.csv')
    
    # 创建模型
    model1 = MoleculeEvolutionPredictor()  # 使用QM9属性作为节点特征
    model2 = MoleculeEvolutionPredictorWithFingerprint()  # 使用指纹作为节点特征
    
    print(f"基于QM9属性的模型已创建: {model1}")
    print(f"基于指纹的模型已创建: {model2}")
    print("注意：要运行完整的训练示例，请提供有效的数据文件路径")


if __name__ == "__main__":
    main()