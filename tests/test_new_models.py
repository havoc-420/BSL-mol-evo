#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新添加的模型
"""

import torch
from torch_geometric.data import Data
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from mol_evo.core.models import (
    MoleculeEvolutionGATv2Predictor,
    MoleculeEvolutionRGCNPredictor,
    MoleculeEvolutionTransformerPredictor
)


def create_test_data():
    """创建测试数据"""
    # 创建简单的测试图数据
    num_nodes = 4
    num_edges = 4
    node_feature_dim = 2048
    edge_feature_dim = 11
    
    # 节点特征 (Morgan指纹)
    x = torch.randn(num_nodes, node_feature_dim)
    
    # 边索引 (简单环状结构)
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 0]
    ], dtype=torch.long)
    
    # 边特征
    edge_attr = torch.randn(num_edges, edge_feature_dim)
    
    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # 对于RGCN模型，还需要edge_type
    edge_type = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    data.edge_type = edge_type
    
    return data


def test_gatv2_model():
    """测试GATv2模型"""
    print("测试 MoleculeEvolutionGATv2Predictor 模型...")
    
    # 创建模型
    model = MoleculeEvolutionGATv2Predictor(
        node_feature_dim=2048,
        edge_feature_dim=11,
        hidden_dim=64,
        output_dim=15,
        heads=4,
        num_layers=3
    )
    
    # 创建测试数据
    data = create_test_data()
    
    # 前向传播
    try:
        output = model(data)
        print(f"  输入节点数: {data.num_nodes}")
        print(f"  输入边数: {data.num_edges}")
        print(f"  输出形状: {output.shape}")
        print(f"  模型参数数量: {sum(p.numel() for p in model.parameters())}")
        print("  ✓ GATv2模型测试通过\n")
        return True
    except Exception as e:
        print(f"  ✗ GATv2模型测试失败: {e}\n")
        return False


def test_rgcn_model():
    """测试RGCN模型"""
    print("测试 MoleculeEvolutionRGCNPredictor 模型...")
    
    # 创建模型
    model = MoleculeEvolutionRGCNPredictor(
        node_feature_dim=2048,
        edge_feature_dim=11,
        hidden_dim=64,
        output_dim=15,
        num_relations=6,
        num_layers=3
    )
    
    # 创建测试数据
    data = create_test_data()
    
    # 前向传播
    try:
        output = model(data)
        print(f"  输入节点数: {data.num_nodes}")
        print(f"  输入边数: {data.num_edges}")
        print(f"  输出形状: {output.shape}")
        print(f"  模型参数数量: {sum(p.numel() for p in model.parameters())}")
        print("  ✓ RGCN模型测试通过\n")
        return True
    except Exception as e:
        print(f"  ✗ RGCN模型测试失败: {e}\n")
        return False


def test_transformer_model():
    """测试Transformer模型"""
    print("测试 MoleculeEvolutionTransformerPredictor 模型...")
    
    # 创建模型
    model = MoleculeEvolutionTransformerPredictor(
        node_feature_dim=2048,
        edge_feature_dim=11,
        hidden_dim=64,
        output_dim=15,
        heads=4,
        num_layers=3
    )
    
    # 创建测试数据
    data = create_test_data()
    
    # 前向传播
    try:
        output = model(data)
        print(f"  输入节点数: {data.num_nodes}")
        print(f"  输入边数: {data.num_edges}")
        print(f"  输出形状: {output.shape}")
        print(f"  模型参数数量: {sum(p.numel() for p in model.parameters())}")
        print("  ✓ Transformer模型测试通过\n")
        return True
    except Exception as e:
        print(f"  ✗ Transformer模型测试失败: {e}\n")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("新模型测试")
    print("=" * 60)
    
    # 测试所有新模型
    results = []
    results.append(test_gatv2_model())
    results.append(test_rgcn_model())
    results.append(test_transformer_model())
    
    # 汇总结果
    print("=" * 60)
    print("测试结果汇总:")
    passed = sum(results)
    total = len(results)
    print(f"  通过: {passed}/{total}")
    
    if passed == total:
        print("  ✓ 所有模型测试通过!")
    else:
        print("  ✗ 部分模型测试失败!")
    
    print("=" * 60)


if __name__ == "__main__":
    main()