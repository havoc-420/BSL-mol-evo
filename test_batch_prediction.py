#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试批量预测功能
验证 evolution_optimizer.py 中的 predict_batch 方法是否正常工作
比较逐个预测和批量预测的用时
"""

import sys
import os
import time
from rdkit import Chem
from mol_evo.core.evolution_optimizer import EvolutionTreeOptimizer

# 设置项目根目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

def main():
    """测试批量预测功能"""
    # 使用用户提供的实际模型路径和参数
    model_path = "/home/rhj/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200/last.pth"
    model_dir = "/home/rhj/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200"
    config_file = "/home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml"
    
    # 创建优化器实例
    try:
        optimizer = EvolutionTreeOptimizer(
            model_path=model_path,
            model_dir=model_dir,
            config_file=config_file,
            target_property='lumo_change',
            optimization_mode='sub'
        )
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 准备测试数据（批量）
    # 示例：将甲烷（CH4）添加一个碳原子变成乙烷（C2H6）
    test_cases = [
        {
            "from_smiles": "C",  # 甲烷
            "to_smiles": "CC",   # 乙烷
            "atom_symbol": "C",
            "operation_type": "add_atom"
        },
        {
            "from_smiles": "CC",  # 乙烷
            "to_smiles": "CCC",   # 丙烷
            "atom_symbol": "C",
            "operation_type": "add_atom"
        },
        {
            "from_smiles": "CCC",  # 丙烷
            "to_smiles": "CCCC",   # 丁烷
            "atom_symbol": "C",
            "operation_type": "add_atom"
        }
    ]
    
    # 准备批量输入数据
    from_smiles_list = []
    to_smiles_list = []
    operation_details_list = []
    
    for case in test_cases:
        from_smiles_list.append(case["from_smiles"])
        to_smiles_list.append(case["to_smiles"])
        operation_details_list.append({
            "type": case["operation_type"],
            "params": {
                "atom_symbol": case["atom_symbol"]
            }
        })
    
    # 测试逐个预测的用时
    print("\n=== 逐个预测测试 ===")
    single_preds = []
    start_single = time.time()
    try:
        for case in test_cases:
            pred = optimizer.predict_property_change(
                smiles_from=case["from_smiles"],
                smiles_to=case["to_smiles"],
                operation_details={
                    "type": case["operation_type"],
                    "params": {
                        "atom_symbol": case["atom_symbol"]
                    }
                }
            )
            single_preds.append(pred)
        end_single = time.time()
        single_time = end_single - start_single
        
        print("逐个预测结果:")
        for i, (case, pred) in enumerate(zip(test_cases, single_preds)):
            print(f"测试用例 {i+1}:")
            print(f"  起始分子: {case['from_smiles']}")
            print(f"  目标分子: {case['to_smiles']}")
            print(f"  操作: {case['operation_type']} {case['atom_symbol']}")
            print(f"  预测结果: {pred:.6f}")
        print(f"\n逐个预测总用时: {single_time:.6f} 秒")
    except Exception as e:
        print(f"逐个预测失败: {e}")
        import traceback
        traceback.print_exc()

    # 测试批量预测的用时
    print("\n=== 批量预测测试 ===")
    start_batch = time.time()
    try:
        batch_preds = optimizer.predict_batch(from_smiles_list, to_smiles_list, operation_details_list)
        end_batch = time.time()
        batch_time = end_batch - start_batch
        
        print("批量预测结果:")
        for i, (case, pred) in enumerate(zip(test_cases, batch_preds)):
            print(f"测试用例 {i+1}:")
            print(f"  起始分子: {case['from_smiles']}")
            print(f"  目标分子: {case['to_smiles']}")
            print(f"  操作: {case['operation_type']} {case['atom_symbol']}")
            print(f"  预测结果: {pred[0]:.6f}")
        print(f"\n批量预测总用时: {batch_time:.6f} 秒")
    except Exception as e:
        print(f"批量预测失败: {e}")
        import traceback
        traceback.print_exc()

    # 比较结果和用时
    print("\n=== 比较结果 ===")
    if 'single_time' in locals() and 'batch_time' in locals():
        print(f"逐个预测总用时: {single_time:.6f} 秒")
        print(f"批量预测总用时: {batch_time:.6f} 秒")
        print(f"批量预测比逐个预测快 {single_time - batch_time:.6f} 秒")
        print(f"批量预测速度是逐个预测的 {single_time / batch_time:.2f} 倍")

    # 验证两种方法的预测结果是否一致
    if 'single_preds' in locals() and 'batch_preds' in locals():
        print("\n=== 结果一致性验证 ===")
        all_close = True
        for i, (single_pred, batch_pred) in enumerate(zip(single_preds, batch_preds)):
            diff = abs(single_pred - batch_pred[0])
            print(f"测试用例 {i+1} 结果差异: {diff:.6f}")
            if diff > 1e-6:  # 设置一个小的容差
                all_close = False
        if all_close:
            print("✅ 所有测试用例的预测结果一致")
        else:
            print("❌ 部分测试用例的预测结果不一致")

if __name__ == "__main__":
    main()
