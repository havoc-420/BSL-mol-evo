#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 molecule_feature_extractors 模块的脚本
"""

import sys
import os

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

try:
    # 导入特征提取器
    from mol_evo.core.models.v0.molecule_feature_extractors.gcn import GCNMoleculeFeatureExtractor
    
    # 导入工具函数
    from mol_evo.utils.test.feature_extractor_utils import (
        select_feature_extractor_interactively, 
        get_test_smiles,
        test_single_molecule_feature_extractor,
        get_feature_extractor_info
    )
except ImportError as e:
    print(f"导入模块时出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def test_molecule_feature_extractor():
    """
    测试分子特征提取器
    """
    print("测试 Molecule Feature Extractor")
    print("=" * 50)
    
    # 定义可用的特征提取器
    available_extractors = {
        "GCN": GCNMoleculeFeatureExtractor,
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
    extractor_info = get_feature_extractor_info(feature_extractor_class)
    if extractor_info:
        print(f"模型信息:")
        print(f"  模型类型: {extractor_info['class_name']}")
        print(f"  模型模块: {extractor_info['module']}")
        print(f"  模型参数数量: {extractor_info['parameter_count']}")
    
    # 测试每个SMILES
    for i, smile in enumerate(test_smiles, 1):
        print(f"\n{'-' * 30}")
        print(f"测试分子 {i}/{len(test_smiles)}: {smile}")
        
        # 测试单个分子
        result = test_single_molecule_feature_extractor(feature_extractor_class, smile)
        
        if result is None:
            continue
        
        print(f"  节点数: {result['node_count']}")
        print(f"  节点特征维度: {result['node_feature_dim']}")
        print(f"  边数: {result['edge_count']}")
        
        print(f"  输出特征:")
        print(f"    形状: {result['features_shape']}")
        print(f"    特征值 (前5维): {result['features'].flatten()[:5]}")
    
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