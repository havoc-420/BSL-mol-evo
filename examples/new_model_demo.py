#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新模型演示脚本
展示如何使用新添加的图神经网络模型进行分子进化预测
"""

import torch
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from mol_evo.core.models import (
    MoleculeEvolutionGATv2Predictor,
    MoleculeEvolutionRGCNPredictor,
    MoleculeEvolutionTransformerPredictor
)
from torch_geometric.data import Data


def create_sample_data():
    """创建示例数据"""
    print("创建示例分子图数据...")
    
    # 节点数和边数
    num_nodes = 5
    num_edges = 6
    
    # 节点特征 (使用Morgan指纹维度2048)
    x = torch.randn(num_nodes, 2048)
    
    # 边索引 (简单分子结构)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ], dtype=torch.long)
    
    # 边特征 (11维特征)
    edge_attr = torch.randn(num_edges, 11)
    
    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # 对于RGCN模型，还需要添加edge_type
    data.edge_type = torch.randint(0, 6, (num_edges,), dtype=torch.long)
    
    print(f"  - 节点数: {num_nodes}")
    print(f"  - 边数: {num_edges}")
    print(f"  - 节点特征维度: {x.shape[1]}")
    print(f"  - 边特征维度: {edge_attr.shape[1]}")
    
    return data


def demo_gatv2_model():
    """演示GATv2模型"""
    print("\n" + "="*50)
    print("GATv2模型演示")
    print("="*50)
    
    # 创建模型
    model = MoleculeEvolutionGATv2Predictor(
        node_feature_dim=2048,
        edge_feature_dim=11,
        hidden_dim=128,
        output_dim=15,
        heads=4,
        num_layers=3
    )
    
    print(f"模型结构: {model}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建数据
    data = create_sample_data()
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(data)
    
    print(f"输出形状: {output.shape}")
    print("✓ GATv2模型演示完成")


def demo_rgcn_model():
    """演示RGCN模型"""
    print("\n" + "="*50)
    print("RGCN模型演示")
    print("="*50)
    
    # 创建模型
    model = MoleculeEvolutionRGCNPredictor(
        node_feature_dim=2048,
        edge_feature_dim=11,
        hidden_dim=128,
        output_dim=15,
        num_relations=6,
        num_layers=3
    )
    
    print(f"模型结构: {model}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建数据
    data = create_sample_data()
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(data)
    
    print(f"输出形状: {output.shape}")
    print("✓ RGCN模型演示完成")


def demo_transformer_model():
    """演示Transformer模型"""
    print("\n" + "="*50)
    print("Transformer模型演示")
    print("="*50)
    
    # 创建模型
    model = MoleculeEvolutionTransformerPredictor(
        node_feature_dim=2048,
        edge_feature_dim=11,
        hidden_dim=128,
        output_dim=15,
        heads=4,
        num_layers=3
    )
    
    print(f"模型结构: {model}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建数据
    data = create_sample_data()
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(data)
    
    print(f"输出形状: {output.shape}")
    print("✓ Transformer模型演示完成")


def main():
    """主函数"""
    print("分子进化新模型演示")
    print("该演示展示了如何使用新添加的图神经网络模型进行分子进化预测")
    
    # 演示各个模型
    demo_gatv2_model()
    demo_rgcn_model()
    demo_transformer_model()
    
    print("\n" + "="*50)
    print("所有模型演示完成!")
    print("="*50)
    print("\n这些新模型可以用于:")
    print("1. 分子属性变化预测")
    print("2. 分子演化路径建模")
    print("3. 分子生成指导")
    print("4. 药物发现中的分子优化")


if __name__ == "__main__":
    main()