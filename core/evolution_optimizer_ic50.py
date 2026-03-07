#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子进化树优化器
结合evolver.py中的进化树生成功能和molecule_optimizer.py中的优化功能，
实现完整的分子进化树优化系统
"""

import sys
import os
import argparse
import torch
import json
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import traceback
import time  # 添加时间统计功能
from datetime import datetime  # 用于生成默认文件名
import numpy as np  # 用于处理numpy数据类型
from torch_geometric.data import Data

# 禁用RDKit的警告信息
RDLogger.DisableLog('rdApp.*')

# 设置项目根目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
sys.path.insert(0, project_root)

try:
    # 导入必要的模块
    from mol_evo.core.molecular_evolution_expansion_ic50 import MolecularEvolutionExpansion
    from mol_evo.core.drp_ic50_p.schnet_gdscv2_predict_property_change import (
        PropertyChangePredictor,
        load_model as load_property_change_model,
        prepare_molecule_from_smiles,
        ALL_OPS,
        ATOM_TYPES,
        BOND_TYPES,
        MAX_ATOM_IDX,
        OP_DIM,
        OP_ADD_ATOM,
        OP_REMOVE_ATOM,
        OP_REPLACE_ATOM,
        OP_ADD_BOND,
        OP_REMOVE_BOND,
        OP_CHANGE_BOND
    )
except ImportError as e:
    print(f"无法导入必要的模块: {e}")
    traceback.print_exc()
    sys.exit(1)


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理numpy数据类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


class EvolutionTreeOptimizer:
    """
    分子进化树优化器
    结合进化树生成和属性预测模型，实现完整的分子优化流程
    """
    
    def __init__(self, model_path, model_dir, config_file=None, initial_smiles_csv=None, 
                 target_property=None, initial_property_value=None, optimization_mode='pct', batch_size=128):
        """
        初始化分子进化树优化器
        
        Args:
            model_path: 模型文件路径
            model_dir: 模型目录路径
            config_file: 配置文件路径
            initial_smiles_csv: 包含起始分子属性的CSV文件路径
            target_property: 目标属性名称
            initial_property_value: 初始分子的目标属性值
            optimization_mode: 优化模式 ('sub' 或 'pct')
            batch_size: 批处理大小，用于控制预测时的批量大小
        """
        self.model_path = model_path
        self.model_dir = model_dir
        self.config_file = config_file
        self.initial_smiles_csv = initial_smiles_csv
        self.target_property = target_property
        self.initial_property_value = initial_property_value
        self.optimization_mode = optimization_mode  # 'sub' 或 'pct'
        self.batch_size = batch_size  # 批处理大小
        self.model = None
        self.initial_properties = {}  # 存储起始分子的属性
        self.generation_time = 0  # 添加生成时间统计
        self.attempt_count = 0  # 添加尝试次数统计
        
        # 检查GPU是否可用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_model()
        if self.initial_smiles_csv:
            self._load_initial_properties()
        
    def _load_model(self):
        """加载训练好的 PropertyChangePredictor 模型"""
        # 使用新的 PropertyChangePredictor 模型加载函数
        self.model = load_property_change_model(self.model_path, device=self.device)
        
        print(f"模型加载成功: {self.model_path}")
        print(f"目标属性: {self.target_property}")
        print(f"优化模式: {self.optimization_mode}")
        print(f"模型类型: PropertyChangePredictor (SchNet-based)")
        
    def _load_initial_properties(self):
        """从CSV文件加载起始分子的属性"""
        if not os.path.exists(self.initial_smiles_csv):
            print(f"警告: CSV文件不存在: {self.initial_smiles_csv}")
            return
            
        try:
            df = pd.read_csv(self.initial_smiles_csv)
            # 建立SMILES到属性的映射
            for _, row in df.iterrows():
                smiles = row['smiles']
                # 根据目标属性名称确定实际的属性列名
                if self.target_property.endswith('_change_pct') or self.target_property.endswith('_change_sub'):
                    # 提取属性名前缀 (例如: homo_change_pct -> homo)
                    property_name = self.target_property.replace('_change_pct', '').replace('_change_sub', '')
                    if property_name in row:
                        self.initial_properties[smiles] = row[property_name]
                elif self.target_property in row:
                    self.initial_properties[smiles] = row[self.target_property]
                else:
                    print(f"警告: 未找到属性 {self.target_property} 在CSV中")
                    
            print(f"成功加载 {len(self.initial_properties)} 个分子的初始属性")
        except Exception as e:
            print(f"加载初始属性时出错: {e}")
    
    def predict_batch(self, valid_operations, current_smiles):
        """
        批量预测分子属性变化（使用 PropertyChangePredictor 模型）
        
        Args:
            valid_operations: 有效操作列表，每个元素是 (operation, new_smiles) 元组
                            operation: 操作详情字典，包含操作类型和相关参数
                            new_smiles: 目标分子SMILES（当前版本未使用，保留以保持接口兼容）
            current_smiles: 当前分子SMILES（用于准备初始分子图数据）
            
        Returns:
            批量预测的属性变化值列表（遇到错误时对应位置为 None）
        """
        if not valid_operations:
            return []
            
        # 准备批量数据
        ops_tensors = []
        valid_indices = []
        
        for idx, (operation, new_smiles) in enumerate(valid_operations):
            try:
                # 将操作详情转换为操作张量
                ops_tensor = self._encode_operation_to_tensor(operation)
                ops_tensors.append(ops_tensor)
                valid_indices.append(idx)
                
            except Exception as e:
                print(f"处理操作时出错: idx={idx}, operation={operation}, error={e}")
                continue
        
        # 如果没有有效的数据，返回全 None 列表
        if not ops_tensors:
            return [None] * len(valid_operations)
        
        # 准备初始分子图数据（所有操作基于同一个初始分子）
        try:
            initial_graph = prepare_molecule_from_smiles(current_smiles, device=self.device)
        except ValueError as e:
            if "Bad Conformer Id" in str(e):
                print(f"准备分子图数据时出错 (Bad Conformer Id): current_smiles={current_smiles}, error={e}")
                # 为所有操作返回 None
                return [None] * len(valid_operations)
            else:
                raise
        
        # 分批处理操作
        all_predictions = []
        num_ops = len(ops_tensors)
        
        for i in range(0, num_ops, self.batch_size):
            batch_end = min(i + self.batch_size, num_ops)
            batch_ops_tensors = ops_tensors[i:batch_end]
            batch_size = len(batch_ops_tensors)
            
            # 批量处理操作张量
            ops_batch = torch.cat(batch_ops_tensors, dim=0).to(self.device)
            
            # 复制初始分子图以匹配操作批量大小
            z_batch = initial_graph.z.repeat(batch_size)
            pos_batch = initial_graph.pos.repeat(batch_size, 1)
            batch_indices = torch.arange(batch_size, device=self.device).repeat_interleave(len(initial_graph.z))
            
            # 创建批量图数据
            batched_graph = Data(z=z_batch, pos=pos_batch, batch=batch_indices)
            
            # 执行批量预测
            # 模型现在支持真正的批量处理，可以一次性预测所有操作
            try:
                with torch.no_grad():
                    predicted_changes = self.model(batched_graph, ops_batch)
                    predictions = predicted_changes.cpu().numpy().flatten().tolist()
                    all_predictions.extend(predictions)
            except Exception as e:
                print(f"批量预测时出错: error={e}")
                # 为该批次的所有操作返回 None
                all_predictions.extend([None] * batch_size)
        
        # 构建完整的预测结果列表，确保长度与 valid_operations 一致
        result = [None] * len(valid_operations)
        for idx, prediction in zip(valid_indices, all_predictions):
            result[idx] = prediction
        
        return result
    
    def _encode_operation_to_tensor(self, operation_details):
        """
        将操作详情编码为 one-hot 张量
        
        Args:
            operation_details: 操作详情字典，包含操作类型和相关参数
            
        Returns:
            one-hot 编码的操作张量，形状为 [1, 1, OP_DIM]
        """
        op_type = operation_details.get("type", "unknown")
        op_params = operation_details.get("params", {})
        
        # 创建 one-hot 向量
        op_vec = torch.zeros(OP_DIM, dtype=torch.float32)
        
        # 编码操作类型
        if op_type in ALL_OPS:
            op_idx = ALL_OPS.index(op_type)
            op_vec[op_idx] = 1.0
        
        # 编码原子类型
        atom_symbol = op_params.get("atom_symbol", "")
        if atom_symbol in ATOM_TYPES:
            atom_idx = ATOM_TYPES.index(atom_symbol)
            op_vec[len(ALL_OPS) + atom_idx] = 1.0
        
        # 编码键类型
        bond_type = op_params.get("bond_type", 1.0)
        if bond_type in BOND_TYPES:
            bond_idx = BOND_TYPES.index(bond_type)
            op_vec[len(ALL_OPS) + len(ATOM_TYPES) + bond_idx] = 1.0
        
        # 编码原子索引
        atom_idx = op_params.get("atom_idx", -1)
        if atom_idx >= 0 and atom_idx < MAX_ATOM_IDX:
            op_vec[len(ALL_OPS) + len(ATOM_TYPES) + len(BOND_TYPES) + atom_idx] = 1.0
        
        # 编码目标原子索引（用于 ADD_BOND, REMOVE_BOND, CHANGE_BOND）
        target_atom_idx = op_params.get("atom2_idx", op_params.get("target_atom_idx", -1))
        if target_atom_idx >= 0 and target_atom_idx < MAX_ATOM_IDX:
            op_vec[len(ALL_OPS) + len(ATOM_TYPES) + len(BOND_TYPES) + MAX_ATOM_IDX + target_atom_idx] = 1.0
        
        # 返回形状为 [1, 1, OP_DIM] 的张量
        return op_vec.unsqueeze(0).unsqueeze(0)
        
    def predict_property_change(self, smiles_from, smiles_to, operation_details):
        """
        预测分子属性变化（使用 PropertyChangePredictor 模型）
        
        Args:
            smiles_from: 起始分子SMILES
            smiles_to: 目标分子SMILES（当前版本未使用，保留以保持接口兼容）
            operation_details: 操作详情字典，包含操作类型和相关参数
            
        Returns:
            预测的属性变化值
        """
        try:
            # 准备初始分子图数据
            initial_graph = prepare_molecule_from_smiles(smiles_from, device=self.device)
            
            # 将操作详情转换为操作张量
            ops_tensor = self._encode_operation_to_tensor(operation_details)
            
            # 执行预测
            with torch.no_grad():
                predicted_change = self.model(initial_graph, ops_tensor)
            
            return predicted_change.item()
            
        except Exception as e:
            print(f"预测属性变化时出错: from={smiles_from}, error={e}")
            # traceback.print_exc()
            return None
            
    def _should_prune_node(self, parent_node, child_node, optimization_direction, patience):
        """
        判断是否应该剪枝某个节点
        
        Args:
            parent_node: 父节点
            child_node: 子节点
            optimization_direction: 优化方向 ('increase' 或 'decrease')
            patience: 耐心值，连续多少代没有改善就剪枝
            
        Returns:
            bool: 是否应该剪枝
        """
        # 如果耐心值为0，则不进行剪枝
        if patience <= 0:
            return False
            
        # 获取当前节点的累计变化值
        child_cumulative = child_node.get("cumulative_change", 0.0) or 0.0
        parent_cumulative = parent_node.get("cumulative_change", 0.0) or 0.0
        
        # 检查当前节点是否改善了属性值
        if optimization_direction == 'increase':
            improved = child_cumulative > parent_cumulative
        else:  # decrease
            improved = child_cumulative < parent_cumulative
            
        # 如果有改善，则不剪枝
        if improved:
            return False
            
        # 检查从祖先节点开始是否连续多代没有改善
        # 获取从根节点到当前节点的路径
        ancestor_id = child_node.get("parent_id")
        no_improvement_count = 1  # 当前节点已经没有改善
        
        nodes = getattr(self, 'nodes_cache', {})  # 获取节点缓存
        
        while ancestor_id is not None and ancestor_id != "0" and no_improvement_count < patience:
            if ancestor_id not in nodes:
                break
                
            ancestor_node = nodes[ancestor_id]
            ancestor_cumulative = ancestor_node.get("cumulative_change", 0.0) or 0.0
            
            # 获取父节点
            ancestor_parent_id = ancestor_node.get("parent_id")
            if ancestor_parent_id is None or ancestor_parent_id not in nodes:
                break
                
            ancestor_parent_node = nodes[ancestor_parent_id]
            ancestor_parent_cumulative = ancestor_parent_node.get("cumulative_change", 0.0) or 0.0
            
            # 检查祖先节点是否改善
            if optimization_direction == 'increase':
                ancestor_improved = ancestor_cumulative > ancestor_parent_cumulative
            else:  # decrease
                ancestor_improved = ancestor_cumulative < ancestor_parent_cumulative
                
            if ancestor_improved:
                break  # 如果祖先节点有改善，则停止检查
                
            no_improvement_count += 1
            ancestor_id = ancestor_parent_id
            
        # 如果连续多代没有改善，则剪枝
        return no_improvement_count >= patience

    def _prune_subtree(self, nodes, edges, children_map, node_id):
        """
        剪除以指定节点为根的整个子树
        
        Args:
            nodes: 节点字典
            edges: 边列表
            children_map: 子节点映射
            node_id: 要剪除的节点ID
            
        Returns:
            int: 剪除的节点数量
        """
        if node_id not in nodes:
            return 0
            
        pruned_count = 1  # 包括当前节点
        
        # 递归剪除所有子节点
        children = children_map.get(node_id, [])
        for child_id in children[:]:  # 使用[:]创建副本以避免在迭代时修改列表
            pruned_count += self._prune_subtree(nodes, edges, children_map, child_id)
            
        # 从数据结构中移除节点
        del nodes[node_id]
        if node_id in children_map:
            del children_map[node_id]
            
        # 从边列表中移除相关的边
        edges[:] = [edge for edge in edges if edge["from"] != node_id and edge["to"] != node_id]
        
        return pruned_count

    def _prune_tree(self, evolution_tree, optimization_direction, patience):
        """
        对进化树进行剪枝
        
        Args:
            evolution_tree: 进化树数据
            optimization_direction: 优化方向 ('increase' 或 'decrease')
            patience: 耐心值，连续多少代没有改善就剪枝
        """
        print(f"开始剪枝，优化方向: {optimization_direction}, 耐心值: {patience}")
        
        nodes = evolution_tree["nodes"]
        edges = evolution_tree["edges"]
        
        # 构建父子关系映射
        children_map = {}
        for edge in edges:
            parent_id = edge["from"] if edge["from"] is not None else "0"
            child_id = edge["to"]
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(child_id)
            
        # 缓存节点以提高访问速度
        self.nodes_cache = nodes
        
        # 从根节点开始，逐层检查并剪枝
        queue = ["0"]
        pruned_count = 0
        
        while queue:
            node_id = queue.pop(0)
            if node_id not in nodes:
                continue
                
            node = nodes[node_id]
            children = children_map.get(node_id, [])
            
            # 检查每个子节点
            for child_id in children[:]:  # 使用[:]创建副本以避免在迭代时修改列表
                if child_id not in nodes:
                    continue
                    
                child_node = nodes[child_id]
                
                # 检查是否需要剪枝
                should_prune = self._should_prune_node(node, child_node, optimization_direction, patience)
                
                if should_prune:
                    # 执行剪枝
                    pruned_count += self._prune_subtree(nodes, edges, children_map, child_id)
                    # 从children列表中移除已剪枝的子节点
                    if child_id in children:
                        children.remove(child_id)
                else:
                    # 不剪枝，继续向下探索
                    queue.append(child_id)
        
        print(f"剪枝完成，共剪除 {pruned_count} 个节点")

    def _calculate_cumulative_changes(self, evolution_tree):
        """
        计算累计预测变化值
        
        Args:
            evolution_tree: 进化树数据
        """
        nodes = evolution_tree["nodes"]
        
        # 使用BFS方式逐层计算累计变化值
        queue = [("0", 0.0)]  # (node_id, cumulative_change)
        
        while queue:
            node_id, parent_cumulative = queue.pop(0)
            if node_id not in nodes:
                continue
                
            node = nodes[node_id]
            # 累计值 = 父节点累计值 + 当前节点变化值
            current_change = node.get("predicted_change", 0.0) or 0.0
            node["cumulative_change"] = parent_cumulative + current_change
            
            # 将所有子节点加入队列
            for edge in evolution_tree["edges"]:
                if edge["from"] == node_id:
                    queue.append((edge["to"], node["cumulative_change"]))
        
    def optimize_evolution_tree(self, initial_smiles, max_depth=2, max_branching=3, 
                               optimization_direction='increase', pruning_patience=3, 
                               logp_range=(0, 5), logp_patience=3):
        """
        优化分子进化树
        
        Args:
            initial_smiles: 初始分子SMILES
            max_depth: 最大演化深度
            max_branching: 最大分支数
            optimization_direction: 优化方向 ('increase' 或 'decrease')
            pruning_patience: 剪枝耐心值，连续多少代没有改善就剪枝
            logp_range: logP值的有效范围，默认(0, 5)
            logp_patience: logP剪枝耐心值，连续多少代logP超出范围就剪枝
            
        Returns:
            带有预测属性变化值的进化树
        """
        print(f"开始优化分子进化树: {initial_smiles}")
        print(f"最大演化深度: {max_depth}")
        print(f"最大分支数: {max_branching}")
        print(f"优化方向: {optimization_direction}")
        print(f"剪枝耐心值: {pruning_patience}")
        print(f"logP有效范围: {logp_range}")
        print(f"logP剪枝耐心值: {logp_patience}")
        
        # 获取初始分子的属性值
        initial_property_value = self.initial_property_value
        if initial_property_value is None and self.initial_smiles_csv:  # UPDATE 根据默认文件中的属性值
            if initial_smiles in self.initial_properties:
                initial_property_value = self.initial_properties[initial_smiles]
                print(f"初始分子 {initial_smiles} 的属性值: {initial_property_value}")
            else:
                print(f"警告: 未找到初始分子 {initial_smiles} 的属性值")
        elif initial_property_value is not None:
            print(f"使用指定的初始属性值: {initial_property_value}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 生成进化树，同时进行预测和剪枝
        evolver = MolecularEvolutionExpansion(initial_smiles, config_file=self.config_file)
        
        # 添加中断标志到evolver
        evolver.interrupted = hasattr(self, 'interrupted') and self.interrupted
        
        # TAG 传入预测器和相关参数，实现生成过程中的预测和剪枝
        evolution_tree = evolver.generate_expansion_tree(
            max_depth=max_depth, 
            max_branching=max_branching,
            predictor=self, # INFO 关键预测器
            optimization_direction=optimization_direction,
            pruning_patience=pruning_patience,
            initial_property_value=initial_property_value,
            optimization_mode=self.optimization_mode,
            logp_range=logp_range,
            logp_patience=logp_patience
        )
        
        # 更新尝试次数（这里简单地使用节点数量作为尝试次数）
        self.attempt_count = len(evolution_tree.get("nodes", {}))
        
        # 记录结束时间
        end_time = time.time()
        self.generation_time = end_time - start_time
        
        return evolution_tree
        
    # TAG dev 打印优化后的进化树
    def print_optimized_tree(self, evolution_tree, format='text'):
        """
        打印优化后的进化树
        
        Args:
            evolution_tree: 带有预测值的进化树
            format: 输出格式 ('text' 或 'json')
        """
        if format == 'json':
            print(json.dumps(evolution_tree, ensure_ascii=False, indent=2, cls=NumpyEncoder))
            return
            
        # 以树形结构打印
        nodes = evolution_tree["nodes"]
        initial_smiles = evolution_tree["initial_smiles"]
        
        print(f"初始分子: {initial_smiles}")
        print(f"目标属性: {self.target_property}")
        print(f"优化方向: {'最大化' if getattr(self, 'optimization_direction', 'increase') == 'increase' else '最小化'}")
        print(f"优化模式: {'百分比变化' if self.optimization_mode == 'pct' else '绝对差值'}")
        print(f"生成耗时: {self.generation_time:.2f} 秒")
        print(f"尝试次数: {self.attempt_count}")
        print("\n进化树结构:")
        
        # 打印根节点
        root_node = nodes["0"]
        # cumulative_change = root_node.get('cumulative_change', 0.0) or 0.0
        property_value = root_node.get('property_value')
        property_info = f", 属性值: {property_value:.6f}" if property_value is not None else ""
        print(f"└── {root_node['smiles']} (深度: {root_node['depth']}{property_info})")
        
        # 构建父子关系映射
        children_map = {}
        for edge in evolution_tree["edges"]:
            parent_id = edge["from"] if edge["from"] is not None else "0"
            child_id = edge["to"]
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(child_id)
        
        # 递归打印树形结构
        self._print_tree_recursive(nodes, children_map, "0", "    ", True)
        
    def _print_tree_recursive(self, nodes, children_map, node_id, prefix, is_last):
        """
        递归打印带预测值的树形结构
        """
        # 获取当前节点的子节点
        children = children_map.get(node_id, [])
        
        # 打印子节点
        for i, child_id in enumerate(children):
            if child_id not in nodes:
                continue
                
            child_node = nodes[child_id]
            is_last_child = (i == len(children) - 1)
            
            # 构造操作信息和预测值
            operation_info = f" [{child_node['operation']}]" if child_node['operation'] else ""
            predicted_change = child_node.get('accumulated_change', 0.0) or 0.0     # UPDATE
            # cumulative_change = child_node.get('cumulative_change', 0.0) or 0.0
            property_value = child_node.get('property_value')
            property_info = f", 属性值: {property_value:.6f}" if property_value is not None else ""
            predicted_info = f" (变化: {predicted_change:.4f})"
            depth_info = f" (深度: {child_node['depth']})"
            
            # 打印当前子节点
            if is_last_child:
                print(f"{prefix}└── {child_node['smiles']}{operation_info}{predicted_info}{property_info}{depth_info}")
                new_prefix = f"{prefix}    "
            else:
                print(f"{prefix}├── {child_node['smiles']}{operation_info}{predicted_info}{property_info}{depth_info}")
                new_prefix = f"{prefix}│   "
            
            # 递归打印孙节点
            self._print_tree_recursive(nodes, children_map, child_id, new_prefix, is_last_child)

    def save_optimized_tree(self, evolution_tree, output_file=None, output_dir=None):
        """
        保存优化后的进化树到文件
        
        Args:
            evolution_tree: 带有预测值的进化树
            output_file: 输出文件路径，如果为None则使用默认路径
            output_dir: 输出目录路径，如果为None则使用默认路径
        """
        # 添加统计信息到进化树
        evolution_tree["statistics"] = {
            "generation_time": self.generation_time,
            "attempt_count": self.attempt_count
        }
        
        # 如果没有指定输出文件，则使用默认路径和文件名
        if output_file is None:
            # 创建输出目录
            if output_dir is None:
                default_output_dir = os.path.join(project_root, "mol_evo", "output", "ic50")
            else:
                default_output_dir = output_dir
            os.makedirs(default_output_dir, exist_ok=True)
            
            # 生成默认文件名
            initial_smiles = evolution_tree.get("initial_smiles", "unknown")
            # 简化文件名，只取SMILES的前20个字符
            smiles_part = initial_smiles[:20] if initial_smiles else "unknown"
            output_file = os.path.join(default_output_dir, f"evolution_tree_{smiles_part}.json")
        else:
            # 检查并创建输出目录
            dir_path = os.path.dirname(output_file)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evolution_tree, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        print(f"进化树已保存到: {output_file}")
        return output_file
        
    def get_topK_results(self, evolution_tree, topK=5):
        """
        获取进化树中效果最好的K个节点结果
        
        Args:
            evolution_tree: 带有预测值的进化树
            topK: 要保留的结果数量
            
        Returns:
            排序后的topK个结果列表，每个结果包含SMILES和属性值
        """
        nodes = evolution_tree.get("nodes", {})
        initial_smiles = evolution_tree.get("initial_smiles")
        
        # 获取根节点信息（初始分子）
        root_node = nodes.get("0")
        initial_property_value = root_node.get("property_value") if root_node else None
        
        # 收集所有叶节点（深度最大的节点）
        leaf_nodes = []
        for node_id, node in nodes.items():
            # 跳过根节点，只考虑非根节点
            if node_id != "0":
                property_value = node.get("property_value")
                if property_value is not None:
                    leaf_nodes.append({
                        "smiles": node["smiles"],
                        "property_value": property_value,
                        "depth": node.get("depth", 0)
                    })
        
        # 根据优化方向排序
        optimization_direction = getattr(self, 'optimization_direction', 'increase')
        if optimization_direction == 'increase':
            # 最大化目标，降序排序
            sorted_nodes = sorted(leaf_nodes, key=lambda x: x["property_value"], reverse=True)
        else:
            # 最小化目标，升序排序
            sorted_nodes = sorted(leaf_nodes, key=lambda x: x["property_value"])
        
        # 返回topK个结果
        topK_results = sorted_nodes[:topK]
        
        # 添加初始分子信息
        result = {
            "initial_smiles": initial_smiles,
            "initial_property_value": initial_property_value,
            "topK_results": topK_results
        }
        
        return result
        
    def save_topK_results_to_csv(self, topK_results, csv_output_file=None, output_dir=None):
        """
        将topK结果保存为CSV文件，格式为：mol_start, value_start, mol_1, value_1, mol_2, value_2, ...
        
        Args:
            topK_results: 包含初始分子和topK结果的字典
            csv_output_file: 输出CSV文件路径，如果为None则使用默认路径
            output_dir: 输出目录路径，如果为None则使用默认路径
            
        Returns:
            输出CSV文件路径
        """
        initial_smiles = topK_results.get("initial_smiles", "")
        initial_property_value = topK_results.get("initial_property_value", 0.0)
        results = topK_results.get("topK_results", [])
        
        # 如果没有指定输出文件，则使用默认路径和文件名
        if csv_output_file is None:
            # 创建输出目录
            # 如果指定了输出目录，使用它；否则使用默认目录
            if output_dir is None:
                default_output_dir = os.path.join(project_root, "mol_evo", "output", "ic50")
            else:
                default_output_dir = output_dir
            os.makedirs(default_output_dir, exist_ok=True)
            
            # 生成默认文件名
            # 简化文件名，只取SMILES的前20个字符
            smiles_part = initial_smiles[:20] if initial_smiles else "unknown"
            csv_output_file = os.path.join(default_output_dir, f"topK_results_{smiles_part}.csv")
        else:
            # 检查并创建输出目录
            dir_path = os.path.dirname(csv_output_file)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # 构建CSV数据
        csv_data = {}
        # 添加初始分子信息
        csv_data["mol_start"] = initial_smiles
        csv_data["value_start"] = initial_property_value
        
        # 添加topK结果
        for i, result in enumerate(results):
            csv_data[f"mol_{i+1}"] = result["smiles"]
            csv_data[f"value_{i+1}"] = result["property_value"]
        
        # 创建DataFrame并保存为CSV
        df = pd.DataFrame([csv_data])
        df.to_csv(csv_output_file, index=False, encoding='utf-8')
        
        print(f"TopK结果已保存到CSV文件: {csv_output_file}")
        return csv_output_file


def run():
    parser = argparse.ArgumentParser(description='分子进化树优化器')
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型文件路径 (.pth文件)')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='模型目录路径')
    parser.add_argument('--config-file', type=str, 
                        help='配置文件路径 (YAML格式)')
    parser.add_argument('--initial-smiles-csv', type=str,
                        help='包含初始分子属性的CSV文件路径')
    parser.add_argument('--initial-smiles', type=str, required=True,
                        help='初始分子SMILES')
    parser.add_argument('--initial-property-value', type=float,
                        help='初始分子的目标属性值')
    parser.add_argument('--target-property', type=str,
                        help='目标属性名称')
    parser.add_argument('--optimization-mode', type=str, choices=['sub', 'pct'], 
                        default='pct', help='优化模式 (sub: 绝对差值, pct: 百分比变化)')
    parser.add_argument('--max-depth', type=int, default=2,
                        help='最大演化深度')
    parser.add_argument('--max-branching', type=int, default=3,
                        help='最大分支数')
    parser.add_argument('--direction', type=str, choices=['increase', 'decrease'], 
                        default='increase', help='优化方向')
    parser.add_argument('--pruning-patience', type=int, default=3,
                        help='剪枝耐心值，连续多少代没有改善就剪枝')
    parser.add_argument('--format', type=str, choices=['text', 'json'], 
                        default='text', help='输出格式')
    parser.add_argument('--output-file', type=str,
                        help='输出文件路径 (默认: mol_evo/output/ic50/{timestamp}/evolution_tree_{smiles}.json)')
    parser.add_argument('--output-dir', type=str,
                        help='输出目录路径 (默认: mol_evo/output/ic50/{timestamp}/)')
    parser.add_argument('--topK', type=int, default=5,
                        help='保留效果最好的K个结果并输出到CSV文件')
    
    args = parser.parse_args()
    
    # 创建优化器
    optimizer = EvolutionTreeOptimizer(
        args.model_path, 
        args.model_dir, 
        args.config_file, 
        args.initial_smiles_csv,
        args.target_property,
        args.initial_property_value,
        args.optimization_mode
    )
    optimizer.optimization_direction = args.direction
    
    # 优化进化树
    optimized_tree = optimizer.optimize_evolution_tree(
        args.initial_smiles, 
        args.max_depth, 
        args.max_branching, 
        args.direction,
        args.pruning_patience
    )
    
    # 保存到文件（如果指定了输出文件）
    output_file = optimizer.save_optimized_tree(optimized_tree, args.output_file, args.output_dir)
    
    # 处理topK结果并保存为CSV
    topK_results = optimizer.get_topK_results(optimized_tree, args.topK)
    # 生成CSV文件路径（如果有指定的JSON输出文件，则基于它生成CSV文件名）
    csv_output_file = None
    if args.output_file:
        csv_output_file = os.path.splitext(args.output_file)[0] + "_topK.csv"
    optimizer.save_topK_results_to_csv(topK_results, csv_output_file, args.output_dir)
    
    # 输出结果
    optimizer.print_optimized_tree(optimized_tree, args.format)


def run4debug():
    """
    使用硬编码参数运行优化器，方便调试
    ic50 任务配置
    """
    
    # 硬编码的参数
    model_path = "/home/rhj/projects/mol_opt/mol-ofo/mol_evo/core/drp_ic50_p/SchNet_909729_checkpoints/0.9_N4_20260106_191804/best_model.pth"
    model_dir = "/home/rhj/projects/mol_opt/mol-ofo/mol_evo/core/drp_ic50_p/SchNet_909729_checkpoints/0.9_N4_20260106_191804"
    config_file = "/home/rhj/projects/mol_opt/mol-ofo/mol_evo/core/drp_ic50_p/ic50.yaml"
    initial_smiles = "C1=C(C(=O)NC(=O)N1)F"
    initial_property_value = 0.6027509202620565  # 将由模型计算
    target_property = "ic50"
    optimization_mode = "sub"
    # max_depth = 3
    # max_branching = 8
    max_depth = 2   # TEST
    max_branching = 4
    direction = "decrease"  # 对应于 optimization-mode=sub 的情况
    pruning_patience = 3
    format_type = "text"
    output_file = None  # 使用默认输出路径
    
    # 创建优化器
    optimizer = EvolutionTreeOptimizer(
        model_path, 
        model_dir, 
        config_file, 
        None,  # initial_smiles_csv
        target_property,
        initial_property_value,
        optimization_mode
    )
    optimizer.optimization_direction = direction
    
    # 优化进化树
    optimized_tree = optimizer.optimize_evolution_tree(
        initial_smiles, 
        max_depth, 
        max_branching, 
        direction,
        pruning_patience
    )
    
    # 保存到文件（如果指定了输出文件）
    output_file = optimizer.save_optimized_tree(optimized_tree, output_file, output_dir=None)
    
    # 处理topK结果并保存为CSV（调试模式下使用默认值5）
    topK_results = optimizer.get_topK_results(optimized_tree, topK=5)
    # 生成CSV文件路径（如果有指定的JSON输出文件，则基于它生成CSV文件名）
    csv_output_file = None
    if output_file:
        csv_output_file = os.path.splitext(output_file)[0] + "_topK.csv"
    optimizer.save_topK_results_to_csv(topK_results, csv_output_file, output_dir=None)
    
    # 输出结果
    optimizer.print_optimized_tree(optimized_tree, format_type)


if __name__ == "__main__":
    # 检查是否有命令行参数传入
    if len(sys.argv) > 1:
        # 如果有命令行参数，则调用run函数处理
        run()
    else:
        # 如果没有命令行参数，则调用run4debug函数进行调试
        run4debug()
