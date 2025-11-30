#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子优化脚本
基于训练好的模型预测分子经过若干步操作后的属性变化，并筛选出满足条件的分子
"""

import sys
import os
import argparse
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from collections import deque
import itertools
import traceback
import json

# 设置项目根目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
sys.path.insert(0, project_root)

try:
    # 导入必要的模块
    from mol_evo.core.models.v0.visnet_linear_linear import MoleculeEvolutionVisnetLinearPredictor
    from mol_evo.core.data.data_v0 import smiles_to_graph_data
    from mol_evo.core.utils.molecule import MoleculeCache
    from mol_evo.utils.predict.model_utils import load_property_stats, load_training_params
    from mol_evo.utils.predict.data_utils import prepare_single_prediction_data
    from mol_evo.core.evolver import MoleculeEvolverAnalysis
except ImportError as e:
    print(f"无法导入必要的模块: {e}")
    traceback.print_exc()
    sys.exit(1)


class MoleculeTreeNode:
    """
    分子树节点，表示分子演化过程中的一个状态
    """
    def __init__(self, smiles, parent=None, operation=None, depth=0, predicted_change=0.0):
        self.smiles = smiles
        self.parent = parent
        self.operation = operation  # 到达该节点所执行的操作
        self.depth = depth         # 演化深度
        self.predicted_change = predicted_change  # 累计预测变化值
        self.children = []
        self.mol = Chem.MolFromSmiles(smiles) if smiles else None
        self.visited = False       # 是否已访问过该节点
        
    def add_child(self, child_node):
        """添加子节点"""
        self.children.append(child_node)
        
    def get_path(self):
        """获取从根节点到当前节点的路径"""
        path = []
        current = self
        while current.parent is not None:
            path.append(current.operation)
            current = current.parent
        return list(reversed(path))
        
    def get_path_smiles(self):
        """获取从根节点到当前节点的SMILES路径"""
        path = []
        current = self
        while current is not None:
            path.append(current.smiles)
            current = current.parent
        return list(reversed(path))


class MoleculeOptimizer:
    """
    分子优化器
    基于训练好的模型预测分子属性变化，并筛选出满足条件的分子
    """
    
    def __init__(self, model_path, model_dir):
        """
        初始化分子优化器
        
        Args:
            model_path: 模型文件路径
            model_dir: 模型目录路径
        """
        self.model_path = model_path
        self.model_dir = model_dir
        self.model = None
        self.property_stats = None
        self.target_property = None
        self.atom_types = ['C', 'N', 'O', 'F', 'P']  # 支持的原子类型
        self.operation_types = ['add_atom', 'replace_atom', 'remove_atom', 
                               'form_bond', 'form_double_bond', 'form_triple_bond', 
                               'form_ring', 'form_aromatic_ring']  # 支持的操作类型
        self._load_model()
        
    def _load_model(self):
        """加载训练好的模型"""
        # 获取目标属性
        training_params = load_training_params(self.model_dir)
        self.target_property = training_params.get('target_property', 'mu_change') if training_params else 'mu_change'
        
        # 加载属性统计信息
        self.property_stats = load_property_stats(self.model_dir, onfig_file_name="model_config.json")
        
        # 从训练数据中加载模型参数
        model_params = {}
        training_data_path = os.path.join(self.model_dir, "model_config.json")
        if os.path.exists(training_data_path):
            with open(training_data_path, 'r') as f:
                training_data = json.load(f)
                model_params = training_data.get("model_params", {})
        
        # 使用训练时的模型参数创建模型
        node_feature_dim = model_params.get("node_feature_dim", 11)
        edge_feature_dim = model_params.get("edge_feature_dim", 11)  # 从训练数据中获取
        output_dim = model_params.get("output_dim", 1)
        
        print(f"模型参数: node_feature_dim={node_feature_dim}, edge_feature_dim={edge_feature_dim}, output_dim={output_dim}")
        
        # 创建模型
        self.model = MoleculeEvolutionVisnetLinearPredictor(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,  # 使用训练时的参数
            hidden_dims=[128, 256, 256],
            output_dim=output_dim
        )
        
        # 加载模型权重
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.eval()
        
        print(f"模型加载成功: {self.model_path}")
        print(f"目标属性: {self.target_property}")
        
    def predict_single_step(self, smiles_from, smiles_to, to_atom_symbol, operation_type):
        """
        预测单步操作后的属性变化
        
        Args:
            smiles_from: 起始分子SMILES
            smiles_to: 目标分子SMILES
            to_atom_symbol: 变化涉及的原子类型
            operation_type: 操作类型
            
        Returns:
            预测的属性变化值
        """
        try:
            # 验证输入分子
            mol_from = Chem.MolFromSmiles(smiles_from)
            mol_to = Chem.MolFromSmiles(smiles_to)
            
            if not mol_from or not mol_to:
                print(f"无效的SMILES: from={smiles_from}, to={smiles_to}")
                return None
            
            # 准备数据
            from_data, to_data, edge_attr, _ = prepare_single_prediction_data(
                smiles_from, smiles_to, to_atom_symbol, operation_type, self.model_dir)
            
            # 检查数据有效性
            if from_data is None or to_data is None or edge_attr is None:
                print(f"数据准备失败: from={smiles_from}, to={smiles_to}")
                return None
                
            # 进行预测
            with torch.no_grad():
                predictions = self.model(from_data, to_data, edge_attr)
                
            predicted_change = predictions[0].numpy()[0]
            
            # 如果模型使用了标准化，则反标准化
            if self.property_stats and self.target_property in self.property_stats:
                mean, std = self.property_stats[self.target_property]
                original_change = predicted_change * std + mean
                return original_change
            else:
                return predicted_change
                
        except Exception as e:
            print(f"预测过程中发生错误: {e}")
            return None
            
    def generate_possible_operations(self, mol):
        """
        生成可能的操作
        
        Args:
            mol: RDKit分子对象
            
        Returns:
            可能的操作列表
        """
        operations = []
        
        # 添加原子操作
        for atom_type in self.atom_types:
            operations.append(('add_atom', atom_type))
            
        # 替换原子操作
        for atom in mol.GetAtoms():
            for atom_type in self.atom_types:
                if atom.GetSymbol() != atom_type:
                    operations.append(('replace_atom', atom_type, atom.GetIdx()))
                    
        # 删除原子操作
        for atom in mol.GetAtoms():
            # 避免删除所有原子
            if len(mol.GetAtoms()) > 1:
                operations.append(('remove_atom', atom.GetSymbol(), atom.GetIdx()))
                
        # 形成键操作
        atoms = list(mol.GetAtoms())
        for i in range(len(atoms)):
            for j in range(i+1, len(atoms)):
                atom1_idx = atoms[i].GetIdx()
                atom2_idx = atoms[j].GetIdx()
                
                # 检查是否已经存在键
                bond = mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
                if bond is None:
                    # 可以形成单键
                    operations.append(('form_bond', atom1_idx, atom2_idx))
                elif bond.GetBondType() == Chem.BondType.SINGLE:
                    # 可以形成双键
                    operations.append(('form_double_bond', atom1_idx, atom2_idx))
                elif bond.GetBondType() == Chem.BondType.DOUBLE:
                    # 可以形成三键
                    operations.append(('form_triple_bond', atom1_idx, atom2_idx))
                    
        # 成环操作 - 寻找可能形成环的原子对
        # 简化处理：寻找距离较远但在分子中可能形成环的原子对
        for i in range(len(atoms)):
            for j in range(i+1, len(atoms)):
                atom1_idx = atoms[i].GetIdx()
                atom2_idx = atoms[j].GetIdx()
                
                # 检查是否已经存在键
                bond = mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
                if bond is None:
                    # 如果两个原子之间没有键，可以尝试形成环
                    operations.append(('form_ring', atom1_idx, atom2_idx))
                    
        return operations
        
    def apply_operation(self, mol, operation):
        """
        应用操作到分子上
        
        Args:
            mol: RDKit分子对象
            operation: 操作元组
            
        Returns:
            新的分子SMILES或None（如果操作无效）
        """
        try:
            mol_copy = Chem.RWMol(mol)
            op_type = operation[0]
            
            if op_type == 'add_atom':
                # 添加原子
                atom_type = operation[1]
                atom = Chem.Atom(atom_type)
                idx = mol_copy.AddAtom(atom)
                # 添加一个键到随机现有原子
                if mol_copy.GetNumAtoms() > 1:
                    existing_atoms = list(range(mol_copy.GetNumAtoms()-1))
                    if existing_atoms:
                        neighbor_idx = existing_atoms[0]  # 简化处理，连接到第一个原子
                        # 检查键是否已存在
                        if not mol_copy.GetBondBetweenAtoms(neighbor_idx, idx):
                            mol_copy.AddBond(neighbor_idx, idx, Chem.BondType.SINGLE)
                        
            elif op_type == 'replace_atom':
                # 替换原子
                atom_type = operation[1]
                atom_idx = operation[2]
                # 确保索引有效
                if atom_idx < mol_copy.GetNumAtoms():
                    mol_copy.ReplaceAtom(atom_idx, Chem.Atom(atom_type))
                
            elif op_type == 'remove_atom':
                # 删除原子
                atom_idx = operation[2]
                # 确保索引有效且不会删除所有原子
                if atom_idx < mol_copy.GetNumAtoms() and mol_copy.GetNumAtoms() > 1:
                    mol_copy.RemoveAtom(atom_idx)
                
            elif op_type in ['form_bond', 'form_double_bond', 'form_triple_bond', 'form_ring', 'form_aromatic_ring']:
                # 形成键操作
                atom1_idx, atom2_idx = operation[1], operation[2]
                # 确保索引有效
                if (atom1_idx < mol_copy.GetNumAtoms() and 
                    atom2_idx < mol_copy.GetNumAtoms() and
                    atom1_idx != atom2_idx):
                    # 检查键是否已存在
                    if not mol_copy.GetBondBetweenAtoms(atom1_idx, atom2_idx):
                        # 根据操作类型设置键类型
                        if op_type == 'form_bond':
                            bond_type = Chem.BondType.SINGLE
                        elif op_type == 'form_double_bond':
                            bond_type = Chem.BondType.DOUBLE
                        elif op_type == 'form_triple_bond':
                            bond_type = Chem.BondType.TRIPLE
                        elif op_type == 'form_ring':
                            bond_type = Chem.BondType.SINGLE
                        elif op_type == 'form_aromatic_ring':
                            bond_type = Chem.BondType.AROMATIC
                        else:
                            bond_type = Chem.BondType.SINGLE
                            
                        mol_copy.AddBond(atom1_idx, atom2_idx, bond_type)
                
            # 检查分子是否有效
            new_mol = mol_copy.GetMol()
            if new_mol and Chem.SanitizeMol(new_mol, catchErrors=True) == 0:
                return Chem.MolToSmiles(new_mol)
                
        except Exception as e:
            # 操作导致无效分子
            # print(f"应用操作时出错: {e}")  # 可以取消注释用于调试
            pass
            
        return None
        
    def optimize_molecule_tree_search(self, start_smiles, max_depth=2, beam_width=10, desired_direction='increase'):
        """
        基于树形搜索的分子优化
        
        Args:
            start_smiles: 起始分子SMILES
            max_depth: 最大搜索深度
            beam_width: 每层保留的最优节点数
            desired_direction: 期望的变化方向 ('increase' 或 'decrease')
            
        Returns:
            优化结果列表
        """
        print(f"开始基于树形搜索的分子优化: {start_smiles}")
        print(f"最大搜索深度: {max_depth}")
        print(f"束宽度: {beam_width}")
        print(f"期望属性变化方向: {desired_direction}")
        
        # 验证起始分子
        start_mol = Chem.MolFromSmiles(start_smiles)
        if not start_mol:
            print(f"无效的起始分子SMILES: {start_smiles}")
            return []
            
        # 初始化根节点
        root_node = MoleculeTreeNode(start_smiles, depth=0, predicted_change=0.0)
        
        # 使用束搜索（beam search）进行优化
        current_layer = [root_node]
        all_nodes = [root_node]
        
        for depth in range(1, max_depth + 1):
            print(f"正在搜索第 {depth} 层...")
            next_layer = []
            
            for node in current_layer:
                if not node.mol:
                    continue
                    
                # 生成可能的操作
                operations = self.generate_possible_operations(node.mol)
                print(f"  节点 {node.smiles[:20]}... 生成了 {len(operations)} 个可能操作")
                
                # 对每个操作生成新分子并预测属性变化
                for op in operations:
                    new_smiles = self.apply_operation(node.mol, op)
                    if not new_smiles or new_smiles == node.smiles:
                        continue
                        
                    # 预测属性变化
                    # 简化处理：假设操作类型和原子类型基于操作元组
                    op_type = op[0]
                    atom_symbol = op[1] if len(op) > 1 else 'C'
                    
                    # 对于键操作，使用特殊处理
                    if op_type in ['form_bond', 'form_double_bond', 'form_triple_bond', 'form_ring', 'form_aromatic_ring']:
                        atom_symbol = 'C'  # 键操作不涉及特定原子类型
                        
                    predicted_change = self.predict_single_step(
                        node.smiles, new_smiles, atom_symbol, op_type)
                    
                    if predicted_change is not None:
                        # 计算累计变化
                        total_change = node.predicted_change + predicted_change
                        
                        # 创建新节点
                        new_node = MoleculeTreeNode(
                            smiles=new_smiles,
                            parent=node,
                            operation=(op, predicted_change),
                            depth=depth,
                            predicted_change=total_change
                        )
                        
                        node.add_child(new_node)
                        next_layer.append(new_node)
                        all_nodes.append(new_node)
                        
            # 根据束宽度筛选下一层节点
            if desired_direction == 'increase':
                next_layer.sort(key=lambda x: x.predicted_change, reverse=True)
            else:  # decrease
                next_layer.sort(key=lambda x: x.predicted_change, reverse=False)
                
            current_layer = next_layer[:beam_width]
            print(f"第 {depth} 层保留了 {len(current_layer)} 个节点")
            
        # 筛选满足条件的结果
        results = []
        for node in all_nodes:
            if node == root_node:  # 跳过根节点
                continue
                
            # 根据期望方向筛选
            if (desired_direction == 'increase' and node.predicted_change > root_node.predicted_change) or \
               (desired_direction == 'decrease' and node.predicted_change < root_node.predicted_change):
                results.append({
                    'smiles': node.smiles,
                    'predicted_change': node.predicted_change,
                    'depth': node.depth,
                    'path': node.get_path(),
                    'smiles_path': node.get_path_smiles()
                })
                
        # 按预测变化值排序
        if desired_direction == 'increase':
            results.sort(key=lambda x: x['predicted_change'], reverse=True)
        else:
            results.sort(key=lambda x: x['predicted_change'], reverse=False)
            
        return results
        
    def optimize_molecule(self, start_smiles, candidate_smiles_list, desired_direction='increase', max_steps=2):
        """
        优化分子，筛选出属性朝期望方向变化的分子
        
        Args:
            start_smiles: 起始分子SMILES
            candidate_smiles_list: 候选目标分子SMILES列表
            desired_direction: 期望的变化方向 ('increase' 或 'decrease')
            max_steps: 最大演化步数
            
        Returns:
            符合条件的分子列表及其预测属性变化值
        """
        print(f"开始优化分子: {start_smiles}")
        print(f"候选分子数量: {len(candidate_smiles_list)}")
        print(f"期望属性变化方向: {desired_direction}")
        print(f"最大演化步数: {max_steps}")
        
        # 验证起始分子
        start_mol = Chem.MolFromSmiles(start_smiles)
        if not start_mol:
            print(f"无效的起始分子SMILES: {start_smiles}")
            return []
            
        results = []
        
        for i, target_smiles in enumerate(candidate_smiles_list):
            target_mol = Chem.MolFromSmiles(target_smiles)
            if not target_mol:
                print(f"无效的目标分子SMILES: {target_smiles}")
                continue
                
            # 对于step=1的情况，直接预测
            if max_steps == 1:
                # 这里需要简化处理，因为我们不知道具体的操作类型和原子类型
                # 实际应用中应该提供这些信息
                pred_change = self._predict_step_by_step(start_smiles, target_smiles, max_steps)
                if pred_change is not None:
                    # 根据期望方向筛选
                    if (desired_direction == 'increase' and pred_change > 0) or \
                       (desired_direction == 'decrease' and pred_change < 0):
                        results.append({
                            'start_smiles': start_smiles,
                            'target_smiles': target_smiles,
                            'predicted_change': pred_change
                        })
                        
            # 对于step>1的情况，需要逐步预测
            else:
                pred_change = self._predict_step_by_step(start_smiles, target_smiles, max_steps)
                if pred_change is not None:
                    # 根据期望方向筛选
                    if (desired_direction == 'increase' and pred_change > 0) or \
                       (desired_direction == 'decrease' and pred_change < 0):
                        results.append({
                            'start_smiles': start_smiles,
                            'target_smiles': target_smiles,
                            'predicted_change': pred_change
                        })
            
            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/{len(candidate_smiles_list)} 个候选分子")
                
        return results
        
    def _predict_step_by_step(self, start_smiles, target_smiles, max_steps):
        """
        逐步预测多步演化后的属性变化
        
        Args:
            start_smiles: 起始分子SMILES
            target_smiles: 目标分子SMILES
            max_steps: 最大演化步数
            
        Returns:
            预测的累计属性变化值
        """
        # 注意：这是一个简化的实现
        # 在实际应用中，你需要提供具体的演化路径
        # 这里我们假设一步直接转化
        
        # 简化处理：假设是单步变化，需要用户提供操作信息
        # 在实际应用中，你应该有具体的演化路径生成逻辑
        
        # 作为示例，我们简单地假设操作类型为"replace"，变化原子为碳
        try:
            predicted_change = self.predict_single_step(
                start_smiles, target_smiles, 'C', 'replace')
            return predicted_change
        except Exception as e:
            print(f"预测 {start_smiles} -> {target_smiles} 时出错: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='基于模型的分子优化工具')
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型文件路径 (.pth文件)')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='模型目录路径')
    parser.add_argument('--start-smiles', type=str, required=True,
                        help='起始分子SMILES')
    parser.add_argument('--candidate-file', type=str,
                        help='候选分子SMILES文件路径 (CSV格式，包含smiles列)')
    parser.add_argument('--candidate-smiles', type=str, nargs='+',
                        help='候选分子SMILES列表')
    parser.add_argument('--direction', type=str, choices=['increase', 'decrease'], 
                        default='increase', help='期望的属性变化方向')
    parser.add_argument('--max-steps', type=int, default=1,
                        help='最大演化步数')
    parser.add_argument('--search-mode', type=str, choices=['candidate', 'tree'], 
                        default='candidate', help='搜索模式: candidate(基于候选列表) 或 tree(树形搜索)')
    parser.add_argument('--max-depth', type=int, default=2,
                        help='树形搜索的最大深度')
    parser.add_argument('--beam-width', type=int, default=10,
                        help='树形搜索的束宽度')
    
    args = parser.parse_args()
    
    # 创建优化器
    optimizer = MoleculeOptimizer(args.model_path, args.model_dir)
    
    if args.search_mode == 'tree':
        # 使用树形搜索模式
        results = optimizer.optimize_molecule_tree_search(
            args.start_smiles, args.max_depth, args.beam_width, args.direction)
        
        # 输出结果
        print(f"\n树形搜索结果 (共 {len(results)} 个符合条件的分子):")
        print("=" * 100)
        print(f"{'SMILES':<30} {'预测变化值':<12} {'深度':<6} {'路径':<30}")
        print("-" * 100)
        
        for result in results[:20]:  # 只显示前20个结果
            smiles_display = result['smiles'][:28] + "..." if len(result['smiles']) > 30 else result['smiles']
            path_display = str(result['path'])[:28] + "..." if len(str(result['path'])) > 30 else str(result['path'])
            print(f"{smiles_display:<30} {result['predicted_change']:<12.4f} {result['depth']:<6} {path_display:<30}")
            
        if results:
            # 保存结果到文件
            output_file = f"tree_search_results_depth{args.max_depth}.csv"
            results_df = pd.DataFrame(results)
            # 处理路径列，转换为字符串
            results_df['path'] = results_df['path'].apply(lambda x: str(x))
            results_df['smiles_path'] = results_df['smiles_path'].apply(lambda x: str(x))
            results_df.to_csv(output_file, index=False)
            print(f"\n结果已保存到: {output_file}")
            
    else:
        # 使用候选分子模式
        # 获取候选分子
        candidate_smiles_list = []
        if args.candidate_file:
            # 从文件读取候选分子
            df = pd.read_csv(args.candidate_file)
            if 'smiles' in df.columns:
                candidate_smiles_list = df['smiles'].tolist()
            else:
                print("CSV文件中未找到'smiles'列")
                return
        elif args.candidate_smiles:
            # 从命令行参数获取候选分子
            candidate_smiles_list = args.candidate_smiles
        else:
            print("请提供候选分子列表或文件")
            return
            
        # 执行优化
        results = optimizer.optimize_molecule(
            args.start_smiles, candidate_smiles_list, args.direction, args.max_steps)
        
        # 输出结果
        print(f"\n筛选结果 (共 {len(results)} 个符合条件的分子):")
        print("=" * 80)
        print(f"{'起始分子':<15} {'目标分子':<15} {'预测变化值':<12}")
        print("-" * 80)
        
        for result in results:
            start = result['start_smiles'][:12] + "..." if len(result['start_smiles']) > 15 else result['start_smiles']
            target = result['target_smiles'][:12] + "..." if len(result['target_smiles']) > 15 else result['target_smiles']
            print(f"{start:<15} {target:<15} {result['predicted_change']:<12.4f}")
        
        if results:
            # 保存结果到文件
            output_file = f"optimization_results_steps{args.max_steps}.csv"
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_file, index=False)
            print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()