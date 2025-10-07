#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN模型演示示例

这个示例展示了如何使用基于分子进化的GNN模型进行训练和预测。
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from core.gnn import (MoleculeGNN, MoleculeEvolutionPredictor, 
                      MoleculeGNNWithFingerprint, MoleculeEvolutionPredictorWithFingerprint,
                      build_molecule_graph_with_properties, build_molecule_graph_with_fingerprints)


def demo_gnn_model():
    """演示GNN模型的基本使用"""
    print("=" * 60)
    print("分子进化GNN模型演示")
    print("=" * 60)
    
    # 创建模型实例
    print("\n1. 创建GNN模型")
    gnn_model_prop = MoleculeGNN()
    gnn_model_fp = MoleculeGNNWithFingerprint()
    print(f"   基于QM9属性的GNN模型: {gnn_model_prop}")
    print(f"   基于指纹的GNN模型: {gnn_model_fp}")
    
    predictor_model_prop = MoleculeEvolutionPredictor()
    predictor_model_fp = MoleculeEvolutionPredictorWithFingerprint()
    print(f"   基于QM9属性的属性变化预测器: {predictor_model_prop}")
    print(f"   基于指纹的属性变化预测器: {predictor_model_fp}")
    
    print("\n2. 模型结构说明:")
    print("   - 节点表示:")
    print("     * QM9属性方法: 使用15个量子化学属性 (A, B, C, mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv)")
    print("     * 指纹方法: 使用2048维Morgan指纹")
    print("   - 边特征: 包含位置、原子类型、操作类型和属性变化统计信息 (19维)")
    print("   - 网络结构: 多层NNConv网络")
    print("   - 输出: 分子表示或属性变化预测")
    
    print("\n3. 使用方法:")
    print("   # 构建图数据")
    print("   # data_prop, smiles_to_idx = build_molecule_graph_with_properties('data.csv')")
    print("   # data_fp, smiles_to_idx = build_molecule_graph_with_fingerprints('data.csv')")
    print("   ")
    print("   # 创建模型")
    print("   # model_prop = MoleculeEvolutionPredictor()")
    print("   # model_fp = MoleculeEvolutionPredictorWithFingerprint()")
    print("   ")
    print("   # 训练模型")
    print("   # losses_prop = train_model(model_prop, data_prop, target_changes)")
    print("   # losses_fp = train_model(model_fp, data_fp, target_changes)")
    
    print("\n4. 两种方法的比较:")
    print("   - QM9属性方法:")
    print("     * 优点: 直接使用物理化学属性，与预测目标语义一致")
    print("     * 缺点: 可能缺少结构细节信息")
    print("   - Morgan指纹方法:")
    print("     * 优点: 结构信息丰富，成熟的技术")
    print("     * 缺点: 丢失具体属性数值信息")
    
    print("\n5. 模型特点:")
    print("   - 利用边特征中的量子化学属性变化信息")
    print("   - 支持多种分子进化操作类型")
    print("   - 可用于预测分子属性变化")
    print("   - 可用于学习分子表示")


def main():
    """主函数"""
    print("基于分子进化的图神经网络(GNN)演示")
    print("该演示展示了如何使用GNN模型处理分子进化数据")
    
    # 运行演示
    demo_gnn_model()
    
    print("\n" + "=" * 60)
    print("演示结束")
    print("=" * 60)


if __name__ == "__main__":
    main()