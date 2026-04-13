#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0.3 模型前向传播测试

覆盖：
- 前向输出形状（默认无 step_head、无 validity_head）
- padding mask 生效
- step_head 显式启用
- validity_head 显式启用
- 非法中间节点前向
- 单步退化兼容（长度为 1 的路径）
- collate 中 step_invalid_types 字段
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
    step_invalid_types = []
    for s in range(num_steps):
        from_v = valid_mask[s]
        to_v = valid_mask[s + 1]
        step_valid.append(from_v and to_v)
        if from_v and to_v:
            step_invalid_types.append('valid')
        elif not from_v and not to_v:
            step_invalid_types.append('both_invalid')
        elif not from_v:
            step_invalid_types.append('from_invalid')
        else:
            step_invalid_types.append('to_invalid')
    
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
        'step_invalid_types': step_invalid_types,
        'num_steps': num_steps,
        'path_target': float(torch.randn(1).item()),
        'step_targets': step_targets,
        'path_target_raw': None,
        'step_targets_raw': None,
    }


def test_model_forward_default():
    """测试模型默认前向（无 step_head、无 validity_head）。"""
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
        # 默认: enable_step_head=False, enable_validity_head=False
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
    assert output['path_pred'].shape == (2, 1), f"Expected (2,1), got {output['path_pred'].shape}"
    assert output['step_pred'] is None, "默认 step_pred 应为 None"
    assert output['validity_pred'] is None, "默认 validity_pred 应为 None"
    
    print("✓ test_model_forward_default 通过")


def test_model_with_step_head():
    """测试显式启用 step_head。"""
    from mol_evo.core.models.v0_3.visnet_path_v0_3 import MoleculeEvolutionVisnetPathPredictorV03
    
    edge_dim = 15
    model = MoleculeEvolutionVisnetPathPredictorV03(
        edge_feature_dim=edge_dim,
        path_num_layers=2,
        enable_step_head=True,
    )
    model.eval()
    
    paths = [
        make_mock_path(num_steps=3, edge_dim=edge_dim),
        make_mock_path(num_steps=2, edge_dim=edge_dim),
    ]
    batch = path_collate(paths)
    
    with torch.no_grad():
        output = model(batch)
    
    assert output['path_pred'].shape == (2, 1)
    assert output['step_pred'] is not None
    assert output['step_pred'].shape == (2, 3, 1), f"Expected (2,3,1), got {output['step_pred'].shape}"
    assert output['validity_pred'] is None
    
    print("✓ test_model_with_step_head 通过")


def test_model_with_validity_head():
    """测试显式启用 validity_head。"""
    from mol_evo.core.models.v0_3.visnet_path_v0_3 import MoleculeEvolutionVisnetPathPredictorV03
    
    edge_dim = 15
    model = MoleculeEvolutionVisnetPathPredictorV03(
        edge_feature_dim=edge_dim,
        path_num_layers=1,
        enable_validity_head=True,
    )
    model.eval()
    
    paths = [
        make_mock_path(num_steps=3, edge_dim=edge_dim, has_invalid=True),
    ]
    batch = path_collate(paths)
    
    with torch.no_grad():
        output = model(batch)
    
    assert output['path_pred'].shape == (1, 1)
    assert output['validity_pred'] is not None
    assert output['validity_pred'].shape == (1, 3, 1), f"Expected (1,3,1), got {output['validity_pred'].shape}"
    assert not torch.isnan(output['validity_pred']).any(), "validity_pred 包含 NaN"
    
    print("✓ test_model_with_validity_head 通过")


def test_single_step_degradation():
    """测试单步退化（长度为 1 的路径 = 旧 pair 模式）。"""
    from mol_evo.core.models.v0_3.visnet_path_v0_3 import MoleculeEvolutionVisnetPathPredictorV03
    
    edge_dim = 15
    model = MoleculeEvolutionVisnetPathPredictorV03(
        edge_feature_dim=edge_dim,
        path_num_layers=1,
    )
    model.eval()
    
    # 单步路径
    path = make_mock_path(num_steps=1, edge_dim=edge_dim)
    batch = path_collate([path])
    
    with torch.no_grad():
        output = model(batch)
    
    assert output['path_pred'].shape == (1, 1)
    assert output['step_pred'] is None  # 默认关闭
    
    print("✓ test_single_step_degradation 通过")


def test_invalid_middle_node():
    """测试含非法中间节点的路径前向。"""
    from mol_evo.core.models.v0_3.visnet_path_v0_3 import MoleculeEvolutionVisnetPathPredictorV03
    
    edge_dim = 15
    model = MoleculeEvolutionVisnetPathPredictorV03(
        edge_feature_dim=edge_dim,
        path_num_layers=1,
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


def test_collate_step_invalid_types():
    """测试 collate 输出包含 step_invalid_types。"""
    edge_dim = 15
    path_valid = make_mock_path(num_steps=2, edge_dim=edge_dim, has_invalid=False)
    path_invalid = make_mock_path(num_steps=3, edge_dim=edge_dim, has_invalid=True)
    
    batch = path_collate([path_valid, path_invalid])
    
    assert 'step_invalid_types' in batch
    assert len(batch['step_invalid_types']) == 2  # batch_size = 2
    
    # path_valid: 2 步，都是 valid，padding 到 3 步
    assert batch['step_invalid_types'][0] == ['valid', 'valid', 'padding']
    
    # path_invalid: 3 步，第 0 步 to_invalid (节点 1 非法)，第 1 步 from_invalid (节点 1 非法)
    assert batch['step_invalid_types'][1][0] == 'to_invalid'
    assert batch['step_invalid_types'][1][1] == 'from_invalid'
    assert batch['step_invalid_types'][1][2] == 'valid'
    
    print("✓ test_collate_step_invalid_types 通过")


def test_model_config_includes_new_fields():
    """测试 get_model_config 包含新字段。"""
    from mol_evo.core.models.v0_3.visnet_path_v0_3 import MoleculeEvolutionVisnetPathPredictorV03
    
    model = MoleculeEvolutionVisnetPathPredictorV03(
        enable_step_head=True,
        enable_validity_head=True,
    )
    config = model.get_model_config()
    
    assert 'enable_step_head' in config
    assert config['enable_step_head'] == True
    assert 'enable_validity_head' in config
    assert config['enable_validity_head'] == True
    
    print("✓ test_model_config_includes_new_fields 通过")


if __name__ == '__main__':
    test_model_forward_default()
    test_model_with_step_head()
    test_model_with_validity_head()
    test_single_step_degradation()
    test_invalid_middle_node()
    test_collate_step_invalid_types()
    test_model_config_includes_new_fields()
    print("\n所有 v0.3 模型前向测试通过 ✓")
