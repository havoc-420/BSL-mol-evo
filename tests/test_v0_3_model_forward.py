#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0.3 模型前向传播测试

覆盖：
- 前向输出形状
- padding mask 生效
- 步骤头输出
- 单步退化兼容（长度为 1 的路径）
"""

import sys
import os
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

from torch_geometric.data import Data, Batch
from mol_evo.core.data.path_data import path_collate, _create_placeholder_graph


def make_mock_graph(n_atoms=3):
    """创建模拟分子图。"""
    return Data(
        z=torch.LongTensor([6] * n_atoms),
        pos=torch.randn(n_atoms, 3),
    )


def make_mock_path(num_steps=3, edge_dim=15, with_step_targets=True, has_invalid=False):
    """创建模拟路径样本。"""
    num_nodes = num_steps + 1
    graphs = []
    valid_mask = []
    
    for i in range(num_nodes):
        if has_invalid and i == 1:
            graphs.append(None)
            valid_mask.append(False)
        else:
            graphs.append(make_mock_graph())
            valid_mask.append(True)
    
    step_valid = []
    for s in range(num_steps):
        step_valid.append(valid_mask[s] and valid_mask[s + 1])
    
    edge_feats = [list(torch.randn(edge_dim).numpy()) for _ in range(num_steps)]
    
    step_targets = None
    if with_step_targets:
        step_targets = [float(torch.randn(1).item()) for _ in range(num_steps)]
    
    return {
        'path_id': f'mock_{num_steps}',
        'node_smiles_list': [f'smi_{i}' for i in range(num_nodes)],
        'node_graph_data_list': graphs,
        'node_valid_mask': valid_mask,
        'operations': [{'atom': 'C', 'operation': 'add_atom', 'position': str(i)} for i in range(num_steps)],
        'edge_features': edge_feats,
        'step_valid_mask': step_valid,
        'num_steps': num_steps,
        'path_target': float(torch.randn(1).item()),
        'step_targets': step_targets,
        'path_target_raw': None,
        'step_targets_raw': None,
    }


def test_model_forward_shape():
    """测试模型前向输出形状。"""
    from mol_evo.core.models.v0_3.visnet_path_v0_3 import MoleculeEvolutionVisnetPathPredictorV03
    
    edge_dim = 15
    model = MoleculeEvolutionVisnetPathPredictorV03(
        node_feature_dim=11,
        edge_feature_dim=edge_dim,
        hidden_dims=[128, 256, 256],
        output_dim=1,
        path_nhead=4,
        path_num_layers=2,
        step_hidden_dim=256,
        enable_step_head=True,
    )
    model.eval()
    
    # 构造 batch
    paths = [
        make_mock_path(num_steps=3, edge_dim=edge_dim),
        make_mock_path(num_steps=2, edge_dim=edge_dim),
    ]
    batch = path_collate(paths)
    
    with torch.no_grad():
        output = model(batch)
    
    assert 'path_pred' in output
    assert 'step_pred' in output
    
    # batch_size=2, output_dim=1
    assert output['path_pred'].shape == (2, 1), f"Expected (2,1), got {output['path_pred'].shape}"
    
    # max_steps=3
    assert output['step_pred'].shape == (2, 3, 1), f"Expected (2,3,1), got {output['step_pred'].shape}"
    
    print("✓ test_model_forward_shape 通过")


def test_single_step_degradation():
    """测试单步退化（长度为 1 的路径 = 旧 pair 模式）。"""
    from mol_evo.core.models.v0_3.visnet_path_v0_3 import MoleculeEvolutionVisnetPathPredictorV03
    
    edge_dim = 15
    model = MoleculeEvolutionVisnetPathPredictorV03(
        edge_feature_dim=edge_dim,
        path_num_layers=1,
        enable_step_head=True,
    )
    model.eval()
    
    # 单步路径
    path = make_mock_path(num_steps=1, edge_dim=edge_dim)
    batch = path_collate([path])
    
    with torch.no_grad():
        output = model(batch)
    
    assert output['path_pred'].shape == (1, 1)
    assert output['step_pred'].shape == (1, 1, 1)
    
    print("✓ test_single_step_degradation 通过")


def test_invalid_middle_node():
    """测试含非法中间节点的路径前向。"""
    from mol_evo.core.models.v0_3.visnet_path_v0_3 import MoleculeEvolutionVisnetPathPredictorV03
    
    edge_dim = 15
    model = MoleculeEvolutionVisnetPathPredictorV03(
        edge_feature_dim=edge_dim,
        path_num_layers=1,
        enable_step_head=True,
    )
    model.eval()
    
    # 含非法中间节点
    path = make_mock_path(num_steps=3, edge_dim=edge_dim, has_invalid=True)
    batch = path_collate([path])
    
    with torch.no_grad():
        output = model(batch)
    
    # 仍然能正常输出
    assert output['path_pred'].shape == (1, 1)
    assert not torch.isnan(output['path_pred']).any(), "路径预测包含 NaN"
    
    print("✓ test_invalid_middle_node 通过")


def test_step_head_disabled():
    """测试禁用步骤头。"""
    from mol_evo.core.models.v0_3.visnet_path_v0_3 import MoleculeEvolutionVisnetPathPredictorV03
    
    edge_dim = 15
    model = MoleculeEvolutionVisnetPathPredictorV03(
        edge_feature_dim=edge_dim,
        path_num_layers=1,
        enable_step_head=False,
    )
    model.eval()
    
    path = make_mock_path(num_steps=2, edge_dim=edge_dim)
    batch = path_collate([path])
    
    with torch.no_grad():
        output = model(batch)
    
    assert output['path_pred'].shape == (1, 1)
    assert output['step_pred'] is None
    
    print("✓ test_step_head_disabled 通过")


if __name__ == '__main__':
    test_model_forward_shape()
    test_single_step_degradation()
    test_invalid_middle_node()
    test_step_head_disabled()
    print("\n所有 v0.3 模型前向测试通过 ✓")
