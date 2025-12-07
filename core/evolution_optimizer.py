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
from torch_geometric.data import Batch

# 禁用RDKit的警告信息
RDLogger.DisableLog('rdApp.*')

# 设置项目根目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
sys.path.insert(0, project_root)

try:
    # 导入必要的模块
    from mol_evo.core.models.v0.visnet_linear_linear import MoleculeEvolutionVisnetLinearPredictor
    from mol_evo.core.evolver import MolecularEvolutionExpansion
    from mol_evo.core.data.processing import load_operation_config, get_atom_types, get_operation_types
    from mol_evo.utils.predict.model_utils import load_property_stats, load_training_params
    from mol_evo.utils.predict.data_utils import prepare_single_prediction_data
    from mol_evo.core.data.data_v0 import smiles_to_graph_data
    from mol_evo.core.utils.molecule import MoleculeCache
    from mol_evo.core.data.processing import prepare_edge_features
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
                 target_property=None, initial_property_value=None, optimization_mode='pct'):
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
        """
        self.model_path = model_path
        self.model_dir = model_dir
        self.config_file = config_file
        self.initial_smiles_csv = initial_smiles_csv
        self.target_property = target_property
        self.initial_property_value = initial_property_value
        self.optimization_mode = optimization_mode  # 'sub' 或 'pct'
        self.model = None
        self.property_stats = None
        self.molecule_cache = MoleculeCache("prediction_dataset")
        self.initial_properties = {}  # 存储起始分子的属性
        self.generation_time = 0  # 添加生成时间统计
        self.attempt_count = 0  # 添加尝试次数统计
        self._load_model()
        if self.initial_smiles_csv:
            self._load_initial_properties()
        
    def _load_model(self):
        """加载训练好的模型"""
        # 获取目标属性
        if not self.target_property:
            training_params = load_training_params(self.model_dir, config_file_name="model_config.json")
            self.target_property = training_params.get('target_property', 'homo_change_pct') if training_params else 'homo_change_pct'
        
        # 加载属性统计信息
        self.property_stats = load_property_stats(self.model_dir, config_file_name="training_process.json")
        
        # 从训练数据中加载模型参数
        model_params = {}
        training_data_path = os.path.join(self.model_dir, "model_config.json")
        if os.path.exists(training_data_path):
            with open(training_data_path, 'r') as f:
                training_data = json.load(f)
                model_params = training_data.get("model_params", {})
                print(f"从训练数据 [{training_data_path}] 中加载模型参数", model_params)
        
        # 使用训练时的模型参数创建模型
        node_feature_dim = model_params.get("node_feature_dim", 11)  # 默认值
        edge_feature_dim = model_params.get("edge_feature_dim", 11)  # 使用训练时的edge_feature_dim
        output_dim = model_params.get("output_dim", 1)
        
        print(f"模型参数: node_feature_dim={node_feature_dim}, edge_feature_dim={edge_feature_dim}, output_dim={output_dim}")
        
        # 加载操作配置
        if self.config_file and os.path.exists(self.config_file):
            load_operation_config(config_path=self.config_file)
        else:
            # 根据数据集路径加载配置文件
            dataset_config_path = os.path.join(self.model_dir, "data_config.json")
            load_operation_config(dataset_path=dataset_config_path)
            
        # 计算正确的边缘特征维度
        atom_types = get_atom_types()
        operation_types = get_operation_types()
        calculated_edge_feature_dim = len(atom_types) + len(operation_types)
        print(f"计算得到的边缘特征维度: {calculated_edge_feature_dim} (原子类型数: {len(atom_types)}, 操作类型数: {len(operation_types)})")
        
        # 如果计算得到的维度与模型参数中的维度不同，使用计算得到的维度
        if calculated_edge_feature_dim != edge_feature_dim:
            print(f"警告: 模型参数中的边缘特征维度 ({edge_feature_dim}) 与计算得到的维度 ({calculated_edge_feature_dim}) 不一致，使用计算得到的维度")
            edge_feature_dim = calculated_edge_feature_dim
        
        # 创建模型
        self.model = MoleculeEvolutionVisnetLinearPredictor(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dims=[128, 256, 256],
            output_dim=output_dim
        )
        
        # 加载模型权重
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.eval()
        
        print(f"模型加载成功: {self.model_path}")
        print(f"目标属性: {self.target_property}")
        print(f"优化模式: {self.optimization_mode}")
        
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
            
    def predict_batch(self, from_smiles_list, to_smiles_list, operation_details_list):
        """
        批量预测分子属性变化
        
        Args:
            from_smiles_list: 起始分子SMILES列表
            to_smiles_list: 目标分子SMILES列表
            operation_details_list: 操作详情字典列表，每个字典包含操作类型和相关参数
            
        Returns:
            批量预测的属性变化值
        """
        if not from_smiles_list or not to_smiles_list or not operation_details_list:
            return []
            
        # 确保三个列表长度相同
        assert len(from_smiles_list) == len(to_smiles_list) == len(operation_details_list), "输入列表长度必须相同"
        
        # 准备批量数据
        from_data_list = []
        to_data_list = []
        edge_attr_list = []
        
        for i in range(len(from_smiles_list)):
            smiles_from = from_smiles_list[i]
            smiles_to = to_smiles_list[i]
            operation_details = operation_details_list[i]
            
            try:
                # 验证输入分子
                mol_from = Chem.MolFromSmiles(smiles_from)
                mol_to = Chem.MolFromSmiles(smiles_to)
                
                if not mol_from or not mol_to:
                    print(f"无效的SMILES: from={smiles_from}, to={smiles_to}")
                    continue
                    
                # 从操作详情中提取必要信息
                operation_type = operation_details.get("type", "unknown")
                operation_params = operation_details.get("params", {})
                atom = operation_params.get("atom_symbol", "")
                position = operation_params.get("atom_idx", "")
                
                # 准备分子图数据
                from_data = smiles_to_graph_data(smiles_from, self.molecule_cache)
                to_data = smiles_to_graph_data(smiles_to, self.molecule_cache)
                
                if from_data is None or to_data is None:
                    print(f"无法将分子转换为图数据: from={smiles_from}, to={smiles_to}")
                    continue
                    
                # 添加batch信息
                from_data = self._add_batch_info(from_data)
                to_data = self._add_batch_info(to_data)
                
                # 构建边特征（不含属性变化）
                data_dict = {
                    'smiles_from': smiles_from,
                    'smiles_to': smiles_to,
                    'operations': [{
                        "atom": atom,
                        "operation": operation_type,
                        "position": str(position)
                    }]
                }
                
                # 构建边特征，不包含属性变化和位置编码
                edge_feat = prepare_edge_features(data_dict, self.property_stats, 
                                                include_property_changes=False, 
                                                include_position_encoding=False)
                
                edge_attr = torch.FloatTensor(np.array(edge_feat))
                
                # 添加到批量列表
                from_data_list.append(from_data)
                to_data_list.append(to_data)
                edge_attr_list.append(edge_attr)
                
            except Exception as e:
                print(f"处理SMILES对时出错: from={smiles_from}, to={smiles_to}, error={e}")
                continue
        
        # 如果没有有效的数据，返回空列表
        if not from_data_list or not to_data_list or not edge_attr_list:
            return []
        
        # 批量处理图数据
        from_batch = Batch.from_data_list(from_data_list)
        to_batch = Batch.from_data_list(to_data_list)
        
        # 批量处理边特征
        edge_attr_batch = torch.stack(edge_attr_list)
        
        # 执行批量预测
        with torch.no_grad():
            predictions = self.model(from_batch, to_batch, edge_attr_batch)
        
        # 反标准化处理
        if self.property_stats and self.target_property in self.property_stats:
            mean, std = self.property_stats[self.target_property]
            predictions = predictions * std + mean
        
        # 确保返回的是一维浮点数列表，即使模型返回的是二维张量
        if len(predictions.shape) > 1:
            predictions = predictions.squeeze()
            
        return predictions.cpu().numpy().tolist()
            
    def _add_batch_info(self, data):
        """
        为图数据添加batch信息
        
        Args:
            data: 图数据对象
            
        Returns:
            添加了batch信息的图数据对象
        """
        if data is None:
            return None
            
        # 获取节点数量
        num_nodes = data.x.size(0) if hasattr(data, 'x') and data.x is not None else 0
        
        # 创建batch属性，所有节点都属于同一个图（批次）
        data.batch = torch.zeros(num_nodes, dtype=torch.long)
        
        return data
        
    def predict_property_change(self, smiles_from, smiles_to, operation_details):
        """
        预测分子属性变化
        
        Args:
            smiles_from: 起始分子SMILES
            smiles_to: 目标分子SMILES
            operation_details: 操作详情字典，包含操作类型和相关参数
            
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
                
            # 从操作详情中提取必要信息
            operation_type = operation_details.get("type", "unknown")
            operation_params = operation_details.get("params", {})
            atom = operation_params.get("atom_symbol", "")
            position = operation_params.get("atom_idx", "")
            
            # 准备数据
            from_data = smiles_to_graph_data(smiles_from, self.molecule_cache)
            to_data = smiles_to_graph_data(smiles_to, self.molecule_cache)
            
            if from_data is None:
                print(f"无法将起始分子转换为图数据: {smiles_from}")
                return None
                
            if to_data is None:
                print(f"[WARNING] 无法将目标分子转换为图数据: {smiles_to}")
                return None
                
            # 添加batch信息
            from_data = self._add_batch_info(from_data)
            to_data = self._add_batch_info(to_data)
            
            # 构建边特征（不含属性变化）
            data_dict = {
                'smiles_from': smiles_from,
                'smiles_to': smiles_to,
                'operations': [{
                    "atom": atom,
                    "operation": operation_type,
                    "position": str(position)
                }]
            }
            
            # 构建边特征，不包含属性变化和位置编码
            edge_feat = prepare_edge_features(data_dict, self.property_stats, 
                                            include_property_changes=False, 
                                            include_position_encoding=False)
            edge_attr = torch.FloatTensor(np.array([edge_feat]))
            
            # 检查数据有效性
            if from_data is None or to_data is None or edge_attr is None:
                print(f"数据准备失败: from={smiles_from}, to={smiles_to}")
                return None
                
            # 添加batch信息
            from_data = self._add_batch_info(from_data)
            to_data = self._add_batch_info(to_data)
                
            # 检查数据完整性
            required_attrs = ['z', 'pos', 'batch']
            for attr in required_attrs:
                if (not hasattr(from_data, attr) or not hasattr(to_data, attr) or
                    getattr(from_data, attr) is None or getattr(to_data, attr) is None):
                    print(f"图数据缺少必要属性 {attr}: from={smiles_from}, to={smiles_to}")
                    return None
            
            # 特别检查pos属性的维度
            if from_data.pos is not None and len(from_data.pos.shape) < 2:
                print(f"图数据pos属性维度不正确: from={smiles_from}")
                return None
                
            if to_data.pos is not None and len(to_data.pos.shape) < 2:
                print(f"图数据pos属性维度不正确: to={smiles_to}")
                return None
            
            # INFO 进行预测
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
            traceback.print_exc()
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
                               optimization_direction='increase', pruning_patience=3):
        """
        优化分子进化树
        
        Args:
            initial_smiles: 初始分子SMILES
            max_depth: 最大演化深度
            max_branching: 最大分支数
            optimization_direction: 优化方向 ('increase' 或 'decrease')
            pruning_patience: 剪枝耐心值，连续多少代没有改善就剪枝
            
        Returns:
            带有预测属性变化值的进化树
        """
        print(f"开始优化分子进化树: {initial_smiles}")
        print(f"最大演化深度: {max_depth}")
        print(f"最大分支数: {max_branching}")
        print(f"优化方向: {optimization_direction}")
        print(f"剪枝耐心值: {pruning_patience}")
        
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
        
        # TAG 传入预测器和相关参数，实现生成过程中的预测和剪枝
        evolution_tree = evolver.generate_expansion_tree(
            max_depth=max_depth, 
            max_branching=max_branching,
            predictor=self, # INFO 关键预测器
            optimization_direction=optimization_direction,
            pruning_patience=pruning_patience,
            initial_property_value=initial_property_value,
            optimization_mode=self.optimization_mode
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
            # 创建带时间戳的输出目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if output_dir is None:
                default_output_dir = os.path.join(project_root, "mol_evo", "output", "evo-mo", timestamp)
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
            # 创建带时间戳的输出目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 如果指定了输出目录，使用它；否则使用默认目录
            if output_dir is None:
                default_output_dir = os.path.join(project_root, "mol_evo", "output", "evo-mo", timestamp)
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
                        help='输出文件路径 (默认: mol_evo/output/evo-mo/{timestamp}/evolution_tree_{smiles}.json)')
    parser.add_argument('--output-dir', type=str,
                        help='输出目录路径 (默认: mol_evo/output/evo-mo/{timestamp}/)')
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
    示例用法，参数来自:
    python -m mol_evo.core.evolution_optimizer \
      --model-path /home/data2/rhj/project/mol_editor/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200/last.pth \
      --model-dir /home/data2/rhj/project/mol_editor/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200 \
      --config-file /home/data2/rhj/project/mol_editor/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml \
      --initial-smiles CC \
      --initial-property-value 2.83270525932312 \
      --target-property lumo \
      --optimization-mode sub \
      --max-depth 2 \
      --max-branching 4 \
      --format text
    """
    
    # 硬编码的参数
    model_path = "/home/data2/rhj/project/mol_editor/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200/last.pth"
    model_dir = "/home/data2/rhj/project/mol_editor/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200"
    config_file = "/home/data2/rhj/project/mol_editor/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml"
    initial_smiles = "CC"
    initial_property_value = 2.83270525932312
    target_property = "lumo"
    optimization_mode = "sub"
    max_depth = 2
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
