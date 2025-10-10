#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试 MoleculeFeatureExtractor 的脚本
"""

import torch
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
# 添加项目根目录到Python路径
sys.path.insert(0, project_root)

# 直接导入需要的模块
try:
    from mol_evo.core.models.v0.gcn import MoleculeFeatureExtractor
    from mol_evo.core.utils.molecule import smile_to_graph_xyz
    from torch_geometric.data import Data
    print("成功导入模块")
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)

TARGET_SMILE = "[H]C([H])([H])[H]"


def main():
    """
    主函数：测试 MoleculeFeatureExtractor
    """
    print("测试 MoleculeFeatureExtractor")
    print("=" * 40)
    
    # 使用 smile_to_graph_xyz 获取真实分子数据
    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    x, z, pos, edge_index, edge_attr = smile_to_graph_xyz(TARGET_SMILE, types)  # 甲烷分子

    if x is None:
        print("无法生成分子图结构")
        return
    
    # 根据实际数据维度创建模型
    node_feature_dim = x.size(1)  # 使用实际的特征维度
    
    # 创建模型实例
    model = MoleculeFeatureExtractor(
        node_feature_dim=node_feature_dim,  # 使用实际维度
        hidden_dim=64,
        output_dim=32,
    )
    
    print("模型创建成功")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建PyG Data对象，包含edge_attr
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    print("\n测试数据:")
    print(f"  节点数: {graph_data.x.size(0)}")
    print(f"  节点特征维度: {graph_data.x.size(1)}")
    print(f"  边数: {graph_data.edge_index.size(1)}")
    if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
        print(f"  边特征维度: {graph_data.edge_attr.size(1)}")
    
    # 测试前向传播
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        features = model(graph_data)
    
    print("\n输出特征:")
    print(f"  特征维度: {features.size()}")
    print(f"  特征值 (前10维): {features.flatten()[:10]}")
    print("\n测试完成!")


if __name__ == "__main__":
    main()