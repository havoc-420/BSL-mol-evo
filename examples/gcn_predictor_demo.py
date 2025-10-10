#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCN预测器演示脚本
展示如何为MoleculeEvolutionGCNPredictor创建训练数据
"""

import torch
import sys
import os
import pandas as pd

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from mol_evo.core.models.v0.gcn import MoleculeEvolutionGCNPredictor, MoleculeFeatureExtractor
from mol_evo.core.utils.molecule import smile_to_graph_xyz
from mol_evo.core.data.processing import prepare_edge_features
from torch_geometric.data import Data


def smiles_to_graph_data(smiles):
    """
    将SMILES字符串转换为图数据（使用项目中的实际函数）
    
    Args:
        smiles: SMILES字符串
    
    Returns:
        Data: 图数据对象
    """
    # 定义原子类型映射
    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    
    # 使用项目中的函数将SMILES转换为图结构
    x, z, pos, edge_index, edge_attr = smile_to_graph_xyz(smiles, types)
    
    # 检查转换是否成功
    if x is None:
        print(f"无法处理SMILES: {smiles}")
        return None
    
    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index)
    
    return data


def create_sample_edge_features():
    """
    创建示例边特征向量（直接使用processing.py中的prepare_edge_features函数）
    
    Returns:
        torch.Tensor: 边特征张量 [1, feature_dim]
    """
    # 创建一个模拟的pandas Series，模仿CSV中的一行数据
    row_data = pd.Series({
        'to_atom_symbol': 'C',
        'operation_type': 'replace'
    })
    
    # 直接调用processing.py中的函数
    edge_features_list = prepare_edge_features(row_data)
    
    # 转换为张量并增加批次维度
    edge_attr = torch.tensor([edge_features_list], dtype=torch.float)
    
    return edge_attr


def create_sample_evolution_data():
    """
    创建示例分子演化数据
    
    Returns:
        tuple: (from_data, to_data, edge_attr) 分别代表起始分子、目标分子和操作信息
    """
    # 示例分子对 (起始分子 -> 目标分子)
    from_smiles = "CCO"    # 乙醇
    to_smiles = "CC=O"     # 乙醛 (乙醇脱氢产物)
    
    # 起始分子数据
    from_data = smiles_to_graph_data(from_smiles)
    
    # 目标分子数据
    to_data = smiles_to_graph_data(to_smiles)
    
    # 边属性 (操作信息) - 表示从乙醇到乙醛的脱氢反应
    # 直接使用processing.py中的prepare_edge_features函数
    edge_attr = create_sample_edge_features()
    
    return from_data, to_data, edge_attr, from_smiles, to_smiles


def main():
    """主函数：演示MoleculeEvolutionGCNPredictor的使用"""
    print("MoleculeEvolutionGCNPredictor演示")
    print("=" * 50)
    
    # 创建示例分子演化数据
    print("\n创建示例分子演化数据...")
    from_data, to_data, edge_attr, from_smiles, to_smiles = create_sample_evolution_data()
    
    if from_data is None or to_data is None:
        print("无法创建分子数据，演示终止")
        return
    
    # 获取节点特征维度（从实际数据中）
    node_feature_dim = from_data.x.shape[1]
    edge_feature_dim = edge_attr.shape[1]
    hidden_dim = 64
    output_dim = 10
    
    # 创建模型
    model = MoleculeEvolutionGCNPredictor(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    
    print(f"模型创建成功:")
    print(f"  - 节点特征维度: {node_feature_dim}")
    print(f"  - 边特征维度: {edge_feature_dim}")
    print(f"  - 隐藏层维度: {hidden_dim}")
    print(f"  - 输出维度: {output_dim}")
    print(f"  - 总参数数量: {sum(p.numel() for p in model.parameters())}")
    
    print(f"  起始分子 ({from_smiles}): {from_data.x.shape[0]} 个原子, 节点特征维度 {from_data.x.shape[1]}")
    print(f"  目标分子 ({to_smiles}): {to_data.x.shape[0]} 个原子, 节点特征维度 {to_data.x.shape[1]}")
    print(f"  操作信息 (脱氢反应): 批量大小 {edge_attr.shape[0]}, 特征维度 {edge_attr.shape[1]}")
    
    # 解释三者之间的关系
    print("\n数据关系说明:")
    print("  一条完整的训练数据包含三个一一对应的组件:")
    print("    1. 起始分子 (from_data): 转换前的分子状态")
    print("    2. 目标分子 (to_data): 转换后的分子状态")
    print("    3. 操作信息 (edge_attr): 连接这两个状态的转换操作")
    print("  这三者共同构成一个分子演化样本，表示从乙醇到乙醛的脱氢过程")
    
    # 模型前向传播
    print("\n执行前向传播...")
    model.eval()
    
    with torch.no_grad():
        output = model(from_data, to_data, edge_attr)
    
    print(f"预测输出形状: {output.shape}")
    print(f"预测值 (前5维): {output.flatten()[:5]}")
    
    print("\n边特征构建说明:")
    print("  边特征构建说明:")
    print("  直接使用processing.py中的prepare_edge_features函数构建边特征:")
    print("    - 原子类型特征 (5维): 独热编码表示目标原子类型")
    print("    - 操作类型特征 (6维): 独热编码表示操作类型")
    print("    - 总计 11 维特征向量")
    
    print("\n批量处理说明:")
    print("  如果要处理多个分子对，可以:")
    print("    1. 分别处理每个分子对")
    print("    2. 或者扩展模型以支持批量处理")
    print("  当前模型每次处理一个分子对 (batch_size=1)")
    
    print("\n演示完成!")


if __name__ == "__main__":
    main()