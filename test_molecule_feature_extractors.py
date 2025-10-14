#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 molecule_feature_extractors 模块的脚本
"""

import sys
import os
import torch


# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

try:
    # 导入特征提取器
    from mol_evo.core.models.v0.molecule_feature_extractors.gcn import GCNMoleculeFeatureExtractor
    from mol_evo.core.models.v0.molecule_feature_extractors.visnet import ViSNet
    
    # 导入工具函数
    from mol_evo.utils.test.feature_extractor_utils import (
        select_feature_extractor_interactively, 
        get_test_smiles,
    )
    
    # 导入smiles_to_graph_data函数和MoleculeCache类
    from mol_evo.core.data.data_v0 import smiles_to_graph_data
    from mol_evo.core.utils.molecule import MoleculeCache
except ImportError as e:
    print(f"导入模块时出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def get_feature_extractor_info_custom(feature_extractor_class):
    """
    获取特征提取器信息（针对不同模型类型的自定义实现）
    
    Args:
        feature_extractor_class: 特征提取器类
        
    Returns:
        包含特征提取器信息的字典
    """
    try:
        # 根据不同模型类型使用不同的初始化方法
        if feature_extractor_class == ViSNet:
            # ViSNet需要不同的初始化参数
            dummy_model = feature_extractor_class(
                hidden_channels=128,
                num_layers=3,
                num_rbf=32,
                cutoff=5.0,
                max_z=100
            )
        else:
            # 其他特征提取器使用默认参数
            dummy_model = feature_extractor_class(node_feature_dim=10)
        
        return {
            "class_name": feature_extractor_class.__name__,
            "module": feature_extractor_class.__module__,
            "parameter_count": sum(p.numel() for p in dummy_model.parameters())
        }
    except Exception as e:
        print(f"获取特征提取器信息时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_molecule_feature_extractor():
    """
    测试分子特征提取器
    """
    print("测试 Molecule Feature Extractor")
    print("=" * 50)
    
    # 定义可用的特征提取器
    available_extractors = {
        "GCN": GCNMoleculeFeatureExtractor,
        "ViSNet": ViSNet,
    }
    
    # 选择特征提取器
    print("1. 选择特征提取器")
    feature_extractor_class = select_feature_extractor_interactively(available_extractors)
    if not feature_extractor_class:
        return False
    
    # 获取预定义的测试SMILES
    print("\n2. 获取测试SMILES")
    test_smiles = get_test_smiles()
    print(f"将测试 {len(test_smiles)} 个分子: {test_smiles}")
    
    print(f"\n3. 开始测试")
    print("=" * 50)
    
    # 获取特征提取器信息
    extractor_info = get_feature_extractor_info_custom(feature_extractor_class)
    if extractor_info:
        print(f"模型信息:")
        print(f"  模型类型: {extractor_info['class_name']}")
        print(f"  模型模块: {extractor_info['module']}")
        print(f"  模型参数数量: {extractor_info['parameter_count']}")
    
    # 创建分子缓存实例
    cache = MoleculeCache("test_feature_extractor")
    
    # 测试每个SMILES
    for i, smile in enumerate(test_smiles, 1):
        print(f"\n{'-' * 30}")
        print(f"测试分子 {i}/{len(test_smiles)}: {smile}")
        
        # 使用smiles_to_graph_data函数获取图数据
        graph_data = smiles_to_graph_data(smile, cache)
        
        if graph_data is None:
            print(f"错误: 无法生成分子 '{smile}' 的图结构")
            continue
        
        # 创建模型实例
        try:
            if feature_extractor_class == ViSNet:
                # ViSNet需要不同的初始化参数
                model = feature_extractor_class(
                    hidden_channels=128,
                    num_layers=3,  # 减少层数以加快测试
                    num_rbf=32,
                    cutoff=5.0,
                    max_z=100
                )
            else:
                # 其他特征提取器使用原始方式初始化
                node_feature_dim = graph_data.x.size(1)
                model = feature_extractor_class(node_feature_dim=node_feature_dim)
        except Exception as e:
            print(f"创建模型实例时出错: {e}")
            continue
        
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            try:
                if feature_extractor_class == ViSNet:
                    # ViSNet需要原子序数和位置信息
                    z = torch.zeros(graph_data.x.shape[0], dtype=torch.long)  # 简化的原子序数
                    pos = graph_data.pos if hasattr(graph_data, 'pos') else torch.zeros(graph_data.x.shape[0], 3)
                    batch = torch.zeros(graph_data.x.shape[0], dtype=torch.long)
                    features, graph_features = model(z, pos, batch)
                else:
                    features = model(graph_data)
            except Exception as e:
                print(f"模型前向传播时出错: {e}")
                continue
        
        # 输出结果
        print(f"  节点数: {graph_data.x.shape[0]}")
        print(f"  节点特征维度: {graph_data.x.shape[1]}")
        print(f"  边数: {graph_data.edge_index.shape[1]}")
        
        print(f"  输出特征:")
        if feature_extractor_class == ViSNet:
            print(f"    节点特征形状: {features.shape}")
            print(f"    图特征形状: {graph_features.shape}")
            print(f"    节点特征值 (前5维): {features.flatten()[:5].tolist()}")
            print(f"    图特征值 (前5维): {graph_features.flatten()[:5].tolist()}")
        else:
            print(f"    形状: {features.shape}")
            print(f"    特征值 (前5维): {features.flatten()[:5].tolist()}")
    
    print("\n" + "=" * 50)
    print("测试完成!")
    return True


if __name__ == "__main__":
    try:
        success = test_molecule_feature_extractor()
        if success:
            print("\n所有测试通过!")
        else:
            print("\n测试失败!")
            sys.exit(1)
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)