import torch
import torch.nn as nn
from torch_geometric.data import Data

from .molecule_feature_extractors import FragNetMoleculeFeatureExtractor
from .edge_feature_extractors import LinearEdgeFeatureExtractor
from .fusion_predictors import MLPFusionPredictor


class MoleculeEvolutionFragLinearPredictor(nn.Module):
    """
    基于FragNet的分子进化预测器 (Linear-Linear-Linear版本)
    
    使用线性组件组合：
    - molecule: FragNetMoleculeFeatureExtractor (FragNet特征提取器)
    - edge: LinearEdgeFeatureExtractor (线性边特征编码器)
    - fusion: MLPFusionPredictor (线性特征融合预测器)
    """
    
    def __init__(self, 
                 atom_feature_dim: int = 167,
                 frag_feature_dim: int = 167, 
                 edge_feature_dim: int = 16,
                 num_layers: int = 4,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 dropout_ratio: float = 0.15,
                 output_dim: int = 1):
        """
        初始化预测器

        Args:
            atom_feature_dim: 原子特征维度
            frag_feature_dim: 片段特征维度
            edge_feature_dim: 边特征维度 (操作信息)
            num_layers: FragNet层数
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            dropout_ratio: Dropout比例
            output_dim: 输出维度 (属性变化)
        """
        super(MoleculeEvolutionFragLinearPredictor, self).__init__()
        
        self.atom_feature_dim = atom_feature_dim
        self.frag_feature_dim = frag_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_ratio = dropout_ratio
        self.output_dim = output_dim
        
        # molecule组件: FragNet分子特征提取器
        self.molecule_extractor_from = FragNetMoleculeFeatureExtractor(
            atom_feature_dim=atom_feature_dim,
            frag_feature_dim=frag_feature_dim,
            edge_feature_dim=edge_feature_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_ratio=dropout_ratio
        )
        
        self.molecule_extractor_to = FragNetMoleculeFeatureExtractor(
            atom_feature_dim=atom_feature_dim,
            frag_feature_dim=frag_feature_dim,
            edge_feature_dim=edge_feature_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_ratio=dropout_ratio
        )
        
        # edge组件: 线性边特征编码器
        self.edge_encoder = LinearEdgeFeatureExtractor(edge_feature_dim, hidden_dim * 4)  # FragNet输出512维特征
        
        # fusion组件: 线性特征融合预测器
        self.fusion_predictor = MLPFusionPredictor(
            node_dim=hidden_dim * 4 * 2,  # FragNet输出是512维特征，from和to拼接
            edge_dim=hidden_dim * 4,      # FragNet输出512维特征
            hidden_dims=[512, 256, 128],
            output_dim=output_dim
        )
    
    def forward(self, from_data: Data, to_data: Data, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            from_data: 起始分子图数据
            to_data: 目标分子图数据
            edge_attr: 边特征 (操作信息)

        Returns:
            属性变化预测值
        """
        # 处理列表形式的输入（支持one-by-one训练）
        if isinstance(from_data, list) and isinstance(to_data, list) and isinstance(edge_attr, torch.Tensor):
            # 如果是列表形式，逐个处理
            results = []
            for i in range(len(from_data)):
                # 提取单个样本
                single_from = from_data[i]
                single_to = to_data[i]
                
                # molecule组件: 提取起始和目标分子特征
                from_features = self.molecule_extractor_from(single_from)
                to_features = self.molecule_extractor_to(single_to)
                
                # edge组件: 编码边特征
                single_edge_attr = edge_attr[i].unsqueeze(0)  # 保持维度
                edge_features = self.edge_encoder(single_edge_attr)
                
                # fusion组件: 融合特征并预测属性变化
                property_change = self.fusion_predictor(from_features, to_features, edge_features)
                results.append(property_change)
            
            # 将所有结果合并
            return torch.cat(results, dim=0)
        
        # 原有的单样本处理逻辑
        # molecule组件: 提取起始和目标分子特征
        from_features = self.molecule_extractor_from(from_data)
        to_features = self.molecule_extractor_to(to_data)
        
        # edge组件: 编码边特征
        edge_features = self.edge_encoder(edge_attr)
        
        # fusion组件: 融合特征并预测属性变化
        property_changes = self.fusion_predictor(from_features, to_features, edge_features)
        
        return property_changes