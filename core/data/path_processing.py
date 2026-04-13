#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0.3 长链路路径数据构建模块

负责：
- 读取长链路 JSON 路径文件
- 节点合法性检查与容错（非法中间节点 fallback，非法首尾丢弃）
- 非法步骤显式标记：生成 step_invalid_types 记录每步的非法原因
- 路径级 & 步骤级标签提取
- 节点有效性掩码 & 步骤有效性掩码生成
- 与现有 MoleculeCache / prepare_edge_features / smiles_to_graph_data 的复用
"""

import json
import sys
import os
import multiprocessing
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, CancelledError
from threading import Lock

from .data_v0 import smiles_to_graph_data
from .processing import prepare_edge_features, load_operation_config, get_atom_types, get_operation_types
from ..utils.molecule import MoleculeCache

# 全局中断标志（复用现有模式）
interrupted = False


def _check_smiles_valid(smiles: str) -> bool:
    """检查 SMILES 字符串是否能被 RDKit 合法解析。"""
    try:
        from rdkit import Chem
        if not smiles or not isinstance(smiles, str):
            return False
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def load_path_json(data_file: str) -> List[Dict]:
    """
    读取长链路路径 JSON 文件。
    
    预期格式（列表中每个元素为一条路径）：
    {
        "path_id": "...",
        "node_smiles_list": ["s0", "s1", "s2", ..., "sT"],
        "operations": [op0, op1, ..., op(T-1)],
        "start_smiles": "s0",
        "end_smiles": "sT",
        "target_property": "lumo_change",
        "path_target": 0.5,                 # 整条路径总变化
        "step_targets": [0.1, 0.2, 0.2],    # 可选，每步变化
    }
    
    也兼容退化的单步 pair 格式（只有 smiles_from / smiles_to）。
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    if not isinstance(raw, list):
        raise ValueError(f"路径 JSON 文件应为列表格式，但得到 {type(raw)}")
    
    paths = []
    for item in raw:
        # 兼容旧的 pair 格式
        if 'node_smiles_list' not in item and 'smiles_from' in item and 'smiles_to' in item:
            item = _convert_pair_to_path(item)
        paths.append(item)
    
    return paths


def _convert_pair_to_path(pair: Dict) -> Dict:
    """将旧单步 pair 格式转换为路径格式（长度为 1 的路径）。"""
    ops = pair.get('operations', [])
    if isinstance(ops, list) and len(ops) > 0 and isinstance(ops[0], dict):
        operations = ops
    else:
        operations = [{
            'atom': pair.get('to_atom_symbol', ''),
            'operation': pair.get('operation_type', ''),
            'position': str(pair.get('atom_idx', '')),
        }]
    
    return {
        'path_id': pair.get('path_id', f"{pair['smiles_from']}->{pair['smiles_to']}"),
        'node_smiles_list': [pair['smiles_from'], pair['smiles_to']],
        'operations': operations,
        'start_smiles': pair['smiles_from'],
        'end_smiles': pair['smiles_to'],
        'target_property': pair.get('target_property', ''),
        'path_target': pair.get('path_target', None),
        'step_targets': pair.get('step_targets', None),
    }


def build_molecule_path_dataset_v0_3(
    data_file: str,
    max_paths: int = None,
    target_property: str = 'lumo_change',
    logger=None,
    max_path_length: int = 20,
    keep_invalid_middle: bool = True,
) -> Tuple[List[Dict], Dict[str, Tuple[float, float]], Dict[str, int]]:
    """
    构建 v0.3 长链路路径数据集。
    
    Args:
        data_file: 路径 JSON 文件路径
        max_paths: 最大路径数（调试用）
        target_property: 目标属性名称
        logger: 日志记录器
        max_path_length: 最大路径步数（超出则截断）
        keep_invalid_middle: 是否保留含非法中间节点的路径
        
    Returns:
        processed_paths: 处理后的路径样本列表
        property_stats: 属性统计信息 {name: (mean, std)}
        build_stats: 构建统计 {total, kept, dropped_invalid_endpoint, dropped_all_invalid, ...}
    """
    global interrupted
    
    # 加载操作配置
    try:
        load_operation_config(dataset_path=data_file)
    except Exception:
        try:
            load_operation_config()
        except Exception as e:
            if logger:
                logger.warning(f"无法加载操作配置: {e}")
    
    # 读取路径数据
    if logger:
        logger.info(f"正在读取路径数据: {data_file}")
    
    raw_paths = load_path_json(data_file)
    
    if max_paths:
        raw_paths = raw_paths[:max_paths]
    
    if logger:
        logger.info(f"共读取 {len(raw_paths)} 条路径")
    
    # 创建分子缓存
    cache = MoleculeCache(csv_file=data_file, logger=logger)
    atom_types = get_atom_types()
    types = {atom: i for i, atom in enumerate(atom_types)}
    
    # 统计信息
    stats = {
        'total': len(raw_paths),
        'kept': 0,
        'dropped_invalid_endpoint': 0,
        'dropped_all_invalid': 0,
        'dropped_too_short': 0,
        'total_nodes': 0,
        'valid_nodes': 0,
        'invalid_middle_nodes': 0,
        'total_steps': 0,
        'valid_steps': 0,
        'avg_path_length': 0.0,
    }
    
    # 先收集所有 path_targets 来计算 property_stats
    all_path_targets = []
    all_step_targets = []
    for p in raw_paths:
        pt = p.get('path_target')
        if pt is not None and not (isinstance(pt, float) and (np.isnan(pt) or np.isinf(pt))):
            all_path_targets.append(pt)
        sts = p.get('step_targets')
        if sts:
            for st in sts:
                if st is not None and not (isinstance(st, float) and (np.isnan(st) or np.isinf(st))):
                    all_step_targets.append(st)
    
    if all_path_targets:
        pt_arr = np.array(all_path_targets, dtype=np.float64)
        property_stats = {target_property: (float(pt_arr.mean()), float(pt_arr.std()))}
    else:
        property_stats = {target_property: (0.0, 1.0)}
    
    if all_step_targets:
        st_arr = np.array(all_step_targets, dtype=np.float64)
        step_property_key = f"{target_property}_step"
        property_stats[step_property_key] = (float(st_arr.mean()), float(st_arr.std()))
    
    if logger:
        logger.info(f"属性统计: {property_stats}")
    
    # 处理每条路径
    processed_paths = []
    
    for path_idx, path in enumerate(tqdm(raw_paths, desc="构建路径数据[v0.3]")):
        if interrupted:
            break
        
        result = _process_single_path(
            path_idx, path, cache, types, property_stats,
            target_property, max_path_length, keep_invalid_middle, logger
        )
        
        if result is None:
            continue
        
        status = result.get('_drop_reason')
        if status == 'invalid_endpoint':
            stats['dropped_invalid_endpoint'] += 1
            continue
        elif status == 'all_invalid':
            stats['dropped_all_invalid'] += 1
            continue
        elif status == 'too_short':
            stats['dropped_too_short'] += 1
            continue
        
        # 更新统计
        n_nodes = len(result['node_smiles_list'])
        n_valid = sum(result['node_valid_mask'])
        n_steps = result['num_steps']
        n_valid_steps = sum(result['step_valid_mask'])
        
        stats['total_nodes'] += n_nodes
        stats['valid_nodes'] += n_valid
        stats['invalid_middle_nodes'] += (n_nodes - n_valid)
        stats['total_steps'] += n_steps
        stats['valid_steps'] += n_valid_steps
        stats['kept'] += 1
        
        processed_paths.append(result)
    
    if stats['kept'] > 0:
        stats['avg_path_length'] = stats['total_steps'] / stats['kept']
    
    # 保存缓存
    if hasattr(cache, '_save_cache'):
        cache._save_cache()
    
    if logger:
        logger.info(f"路径数据构建完成: {stats}")
    
    return processed_paths, property_stats, stats


def _process_single_path(
    path_idx: int,
    path: Dict,
    cache: MoleculeCache,
    types: Dict[str, int],
    property_stats: Dict,
    target_property: str,
    max_path_length: int,
    keep_invalid_middle: bool,
    logger=None,
) -> Optional[Dict]:
    """
    处理单条路径，返回处理后的样本 dict 或 None（需丢弃）。
    
    返回的 dict 包含：
    - path_id
    - node_smiles_list: List[str]
    - node_graph_data_list: List[Data or None]  (None 表示非法节点)
    - node_valid_mask: List[bool]
    - operations: List[Dict]
    - edge_features: List[List[float]]
    - step_valid_mask: List[bool]
    - step_invalid_types: List[str]  (每步非法原因: 'valid'/'from_invalid'/'to_invalid'/'both_invalid')
    - num_steps: int
    - path_target: float (标准化后)
    - step_targets: List[float] or None (标准化后)
    - path_target_raw: float or None
    - step_targets_raw: List[float] or None
    """
    node_smiles_list = path.get('node_smiles_list', [])
    operations = path.get('operations', [])
    
    # 最低要求：至少 2 个节点、1 个操作
    if len(node_smiles_list) < 2:
        return {'_drop_reason': 'too_short'}
    
    # 截断到 max_path_length + 1 个节点
    if len(node_smiles_list) > max_path_length + 1:
        node_smiles_list = node_smiles_list[:max_path_length + 1]
        operations = operations[:max_path_length]
    
    num_steps = len(node_smiles_list) - 1
    
    # 确保 operations 长度与步数一致
    while len(operations) < num_steps:
        operations.append({'atom': '', 'operation': 'unknown', 'position': ''})
    operations = operations[:num_steps]
    
    # ====== 节点合法性检查 & 图构建 ======
    node_graph_data_list = []
    node_valid_mask = []
    
    for i, smi in enumerate(node_smiles_list):
        graph_data = smiles_to_graph_data(smi, cache)
        if graph_data is not None:
            node_graph_data_list.append(graph_data)
            node_valid_mask.append(True)
        else:
            node_graph_data_list.append(None)
            node_valid_mask.append(False)
    
    # 首尾节点非法 → 直接丢弃
    if not node_valid_mask[0] or not node_valid_mask[-1]:
        return {'_drop_reason': 'invalid_endpoint'}
    
    # 如果不保留非法中间节点，且存在非法中间节点 → 丢弃
    if not keep_invalid_middle:
        if not all(node_valid_mask):
            return {'_drop_reason': 'all_invalid'}
    
    # ====== 步骤有效性 & 非法类型标记 & 边特征 ======
    step_valid_mask = []
    step_invalid_types = []
    edge_features_list = []
    
    for step_i in range(num_steps):
        from_valid = node_valid_mask[step_i]
        to_valid = node_valid_mask[step_i + 1]
        step_valid = from_valid and to_valid
        step_valid_mask.append(step_valid)
        
        # 记录非法原因
        if step_valid:
            step_invalid_types.append('valid')
        elif not from_valid and not to_valid:
            step_invalid_types.append('both_invalid')
        elif not from_valid:
            step_invalid_types.append('from_invalid')
        else:
            step_invalid_types.append('to_invalid')
        
        # 构建边特征（即使步骤无效也构建，用零填充）
        op = operations[step_i]
        if isinstance(op, dict):
            data_dict = {
                'smiles_from': node_smiles_list[step_i],
                'smiles_to': node_smiles_list[step_i + 1],
                'operations': [op] if 'operation' in op else [{'atom': op.get('atom', ''),
                                                                 'operation': op.get('type', op.get('operation', '')),
                                                                 'position': str(op.get('position', op.get('atom_idx', '')))}],
            }
        else:
            data_dict = {
                'smiles_from': node_smiles_list[step_i],
                'smiles_to': node_smiles_list[step_i + 1],
                'operations': [{'atom': '', 'operation': 'unknown', 'position': ''}],
            }
        
        try:
            edge_feat = prepare_edge_features(
                data_dict, property_stats,
                include_property_changes=False,
                include_position_encoding=False
            )
        except Exception:
            # fallback: 全零边特征
            atom_types = get_atom_types()
            operation_types = get_operation_types()
            edge_feat = [0.0] * (len(atom_types) + len(operation_types))
        
        edge_features_list.append(edge_feat)
    
    # ====== 标签处理 ======
    mean, std = property_stats.get(target_property, (0.0, 1.0))
    
    # 路径级标签
    path_target_raw = path.get('path_target', None)
    if path_target_raw is not None:
        path_target = (path_target_raw - mean) / std if std > 0 else 0.0
    else:
        path_target = 0.0
    
    # 步骤级标签
    step_targets_raw = path.get('step_targets', None)
    step_targets = None
    if step_targets_raw is not None and len(step_targets_raw) >= num_steps:
        step_targets_raw = step_targets_raw[:num_steps]
        # 步骤标签使用同样的统计量标准化（也可以用 step 专属统计）
        step_key = f"{target_property}_step"
        s_mean, s_std = property_stats.get(step_key, property_stats.get(target_property, (0.0, 1.0)))
        step_targets = []
        for st in step_targets_raw:
            if st is not None and s_std > 0:
                step_targets.append((st - s_mean) / s_std)
            else:
                step_targets.append(0.0)
    
    return {
        'path_id': path.get('path_id', str(path_idx)),
        'node_smiles_list': node_smiles_list,
        'node_graph_data_list': node_graph_data_list,
        'node_valid_mask': node_valid_mask,
        'operations': operations,
        'edge_features': edge_features_list,
        'step_valid_mask': step_valid_mask,
        'step_invalid_types': step_invalid_types,
        'num_steps': num_steps,
        'path_target': path_target,
        'step_targets': step_targets,
        'path_target_raw': path_target_raw,
        'step_targets_raw': step_targets_raw,
    }
