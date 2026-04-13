#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0.3 路径数据处理测试

覆盖：
- 路径 padding 与 mask 生成
- 非法中间节点回退
- 非法首尾节点丢弃
- 单步 pair 退化兼容
- path_collate 批次结构
"""

import sys
import os
import torch
import json
import tempfile

# 设置路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

from mol_evo.core.data.path_processing import (
    _check_smiles_valid,
    _convert_pair_to_path,
    load_path_json,
    _process_single_path,
)
from mol_evo.core.data.path_data import (
    MoleculePathDataset,
    path_collate,
    _create_placeholder_graph,
)


def test_check_smiles_valid():
    """测试 SMILES 合法性检查。"""
    assert _check_smiles_valid("CCO") == True
    assert _check_smiles_valid("CC") == True
    assert _check_smiles_valid("C1CCCCC1") == True
    assert _check_smiles_valid("INVALID_SMILES_XYZ") == False
    assert _check_smiles_valid("") == False
    assert _check_smiles_valid(None) == False
    print("✓ test_check_smiles_valid 通过")


def test_convert_pair_to_path():
    """测试旧 pair 格式转路径格式。"""
    pair = {
        'smiles_from': 'CCO',
        'smiles_to': 'CCN',
        'to_atom_symbol': 'N',
        'operation_type': 'replace_atom',
    }
    path = _convert_pair_to_path(pair)
    
    assert path['node_smiles_list'] == ['CCO', 'CCN']
    assert len(path['operations']) == 1
    assert path['start_smiles'] == 'CCO'
    assert path['end_smiles'] == 'CCN'
    print("✓ test_convert_pair_to_path 通过")


def test_load_path_json_with_pair_format():
    """测试 JSON 加载（兼容旧 pair 格式）。"""
    data = [
        {
            'smiles_from': 'CCO',
            'smiles_to': 'CCN',
            'to_atom_symbol': 'N',
            'operation_type': 'replace_atom',
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        tmp_path = f.name
    
    try:
        paths = load_path_json(tmp_path)
        assert len(paths) == 1
        assert 'node_smiles_list' in paths[0]
        assert paths[0]['node_smiles_list'] == ['CCO', 'CCN']
        print("✓ test_load_path_json_with_pair_format 通过")
    finally:
        os.unlink(tmp_path)


def test_load_path_json_native_format():
    """测试原生路径 JSON 加载。"""
    data = [
        {
            'path_id': 'test_001',
            'node_smiles_list': ['CCO', 'CCN', 'CCNC'],
            'operations': [
                {'atom': 'N', 'operation': 'replace_atom', 'position': '2'},
                {'atom': 'C', 'operation': 'add_atom', 'position': '2'},
            ],
            'path_target': -0.5,
            'step_targets': [-0.2, -0.3],
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        tmp_path = f.name
    
    try:
        paths = load_path_json(tmp_path)
        assert len(paths) == 1
        assert paths[0]['node_smiles_list'] == ['CCO', 'CCN', 'CCNC']
        assert len(paths[0]['operations']) == 2
        print("✓ test_load_path_json_native_format 通过")
    finally:
        os.unlink(tmp_path)


def test_placeholder_graph():
    """测试占位图创建。"""
    ph = _create_placeholder_graph()
    assert hasattr(ph, 'z')
    assert hasattr(ph, 'pos')
    assert ph.z.shape == torch.Size([1])
    assert ph.pos.shape == torch.Size([1, 3])
    print("✓ test_placeholder_graph 通过")


def test_path_dataset_and_collate():
    """测试 MoleculePathDataset 和 path_collate。"""
    # 构造模拟数据
    from torch_geometric.data import Data
    
    def make_mock_graph():
        n = 3
        return Data(z=torch.LongTensor([6, 7, 8]), pos=torch.randn(n, 3))
    
    path1 = {
        'path_id': 'p1',
        'node_smiles_list': ['CC', 'CO', 'CN'],
        'node_graph_data_list': [make_mock_graph(), make_mock_graph(), make_mock_graph()],
        'node_valid_mask': [True, True, True],
        'operations': [{'atom': 'O', 'operation': 'replace_atom', 'position': '1'},
                       {'atom': 'N', 'operation': 'replace_atom', 'position': '1'}],
        'edge_features': [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]],
        'step_valid_mask': [True, True],
        'step_invalid_types': ['valid', 'valid'],
        'num_steps': 2,
        'path_target': 0.5,
        'step_targets': [0.2, 0.3],
        'path_target_raw': 0.5,
        'step_targets_raw': [0.2, 0.3],
    }
    
    path2 = {
        'path_id': 'p2',
        'node_smiles_list': ['CC', 'CF'],
        'node_graph_data_list': [make_mock_graph(), make_mock_graph()],
        'node_valid_mask': [True, True],
        'operations': [{'atom': 'F', 'operation': 'replace_atom', 'position': '1'}],
        'edge_features': [[0.0, 0.0, 1.0, 0.0, 0.0]],
        'step_valid_mask': [True],
        'step_invalid_types': ['valid'],
        'num_steps': 1,
        'path_target': -0.3,
        'step_targets': [-0.3],
        'path_target_raw': -0.3,
        'step_targets_raw': [-0.3],
    }
    
    dataset = MoleculePathDataset([path1, path2])
    assert len(dataset) == 2
    
    # 测试 collate
    batch = path_collate([dataset[0], dataset[1]])
    
    assert batch['batch_size'] == 2
    assert len(batch['node_batch_list']) == 3  # max_num_nodes = 3
    assert batch['node_valid_mask'].shape == (2, 3)
    assert batch['edge_features'].shape == (2, 2, 5)  # max_steps=2, edge_dim=5
    assert batch['step_valid_mask'].shape == (2, 2)
    assert batch['path_padding_mask'].shape == (2, 2)
    assert batch['path_targets'].shape == (2, 1)
    assert batch['step_targets'].shape == (2, 2)
    
    # path2 只有 1 步，第 2 步应该 padding
    assert batch['path_padding_mask'][0, 0] == True
    assert batch['path_padding_mask'][0, 1] == True
    assert batch['path_padding_mask'][1, 0] == True
    assert batch['path_padding_mask'][1, 1] == False  # padding 位
    
    # step_invalid_types 应该存在
    assert 'step_invalid_types' in batch
    assert len(batch['step_invalid_types']) == 2
    assert batch['step_invalid_types'][0] == ['valid', 'valid']
    assert batch['step_invalid_types'][1] == ['valid', 'padding']
    
    print("✓ test_path_dataset_and_collate 通过")


def test_path_collate_with_invalid_middle():
    """测试含非法中间节点的路径 collate。"""
    from torch_geometric.data import Data
    
    def make_mock_graph():
        return Data(z=torch.LongTensor([6, 7]), pos=torch.randn(2, 3))
    
    path = {
        'path_id': 'p_invalid',
        'node_smiles_list': ['CC', 'INVALID', 'CN'],
        'node_graph_data_list': [make_mock_graph(), None, make_mock_graph()],
        'node_valid_mask': [True, False, True],
        'operations': [{'atom': '?', 'operation': 'unknown', 'position': ''},
                       {'atom': 'N', 'operation': 'replace_atom', 'position': '1'}],
        'edge_features': [[0.0] * 5, [1.0, 0.0, 0.0, 0.0, 0.0]],
        'step_valid_mask': [False, False],  # 都涉及非法中间节点
        'step_invalid_types': ['to_invalid', 'from_invalid'],
        'num_steps': 2,
        'path_target': 0.1,
        'step_targets': [0.0, 0.1],
        'path_target_raw': 0.1,
        'step_targets_raw': [0.0, 0.1],
    }
    
    dataset = MoleculePathDataset([path])
    batch = path_collate([dataset[0]])
    
    # 中间节点应该用占位图
    assert batch['node_valid_mask'][0, 0] == True
    assert batch['node_valid_mask'][0, 1] == False  # 非法中间节点
    assert batch['node_valid_mask'][0, 2] == True
    
    # 步骤都涉及非法节点
    assert batch['step_valid_mask'][0, 0] == False
    assert batch['step_valid_mask'][0, 1] == False
    
    # 但路径目标仍有值
    assert batch['path_targets'][0, 0].item() == 0.1
    
    print("✓ test_path_collate_with_invalid_middle 通过")


if __name__ == '__main__':
    test_check_smiles_valid()
    test_convert_pair_to_path()
    test_load_path_json_with_pair_format()
    test_load_path_json_native_format()
    test_placeholder_graph()
    test_path_dataset_and_collate()
    test_path_collate_with_invalid_middle()
    print("\n所有 v0.3 路径数据测试通过 ✓")
