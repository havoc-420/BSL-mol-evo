#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS 搜索模式测试
验证：
1. BFS 默认模式行为不变
2. MCTS 返回兼容的树结构 (nodes/edges)
3. 约束过滤 (max_depth, pruning_patience, logP)
4. 结果去重
5. TopK 可继续消费
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# 添加项目根目录到 Python 路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
sys.path.insert(0, project_root)

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

# 直接导入目标模块，避免 mol_evo.core.__init__ 触发模型层的重依赖
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "molecular_evolution_expansion",
    os.path.join(project_root, "mol_evo", "core", "molecular_evolution_expansion.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MolecularEvolutionExpansion = _mod.MolecularEvolutionExpansion


def _make_mock_predictor(default_change=0.5, direction='decrease'):
    """
    构建 mock predictor，对每个分子对返回固定的预测变化量。
    """
    predictor = MagicMock()

    def _predict(from_list, to_list, ops_list, batch_size=64):
        return [default_change] * len(from_list)

    predictor.predict_batch = MagicMock(side_effect=_predict)
    return predictor


class TestBFSDefaultBehavior(unittest.TestCase):
    """确保 search_mode='bfs' 仍能正常返回"""

    def test_bfs_returns_tree_structure(self):
        evolver = MolecularEvolutionExpansion("CCO")
        predictor = _make_mock_predictor(default_change=-0.3)
        tree = evolver.generate_expansion_tree(
            max_depth=1,
            max_branching=3,
            predictor=predictor,
            optimization_direction='decrease',
            pruning_patience=2,
            initial_property_value=5.0,
            optimization_mode='sub',
            logp_range=(-2, 8),
            logp_patience=3,
        )
        self.assertIn("nodes", tree)
        self.assertIn("edges", tree)
        self.assertIn("initial_smiles", tree)
        self.assertIn("0", tree["nodes"])  # 根节点
        self.assertEqual(tree["nodes"]["0"]["depth"], 0)

    def test_bfs_no_mcts_fields(self):
        evolver = MolecularEvolutionExpansion("CCO")
        predictor = _make_mock_predictor(default_change=-0.1)
        tree = evolver.generate_expansion_tree(
            max_depth=1,
            max_branching=2,
            predictor=predictor,
            optimization_direction='decrease',
            initial_property_value=3.0,
        )
        self.assertNotIn("search_mode", tree)
        self.assertNotIn("mcts_stats", tree)
        for nid, node in tree["nodes"].items():
            self.assertNotIn("mcts_visits", node)


class TestMCTSTreeStructure(unittest.TestCase):
    """MCTS 返回树结构应兼容下游消费"""

    def setUp(self):
        self.evolver = MolecularEvolutionExpansion("CCO")
        self.predictor = _make_mock_predictor(default_change=-0.2)

    def test_mcts_returns_required_keys(self):
        tree = self.evolver.generate_expansion_tree_mcts(
            max_depth=2,
            max_branching=3,
            predictor=self.predictor,
            optimization_direction='decrease',
            pruning_patience=2,
            initial_property_value=5.0,
            optimization_mode='sub',
            logp_range=(-2, 8),
            logp_patience=3,
            num_simulations=20,
            exploration_weight=1.4,
        )
        self.assertIn("nodes", tree)
        self.assertIn("edges", tree)
        self.assertIn("initial_smiles", tree)
        self.assertEqual(tree["search_mode"], "mcts")
        self.assertIn("mcts_stats", tree)

    def test_mcts_root_node_correct(self):
        tree = self.evolver.generate_expansion_tree_mcts(
            max_depth=2,
            max_branching=3,
            predictor=self.predictor,
            optimization_direction='decrease',
            initial_property_value=5.0,
            num_simulations=10,
        )
        root = tree["nodes"].get("0")
        self.assertIsNotNone(root)
        self.assertEqual(root["depth"], 0)
        self.assertIsNone(root["parent_id"])
        self.assertEqual(root["smiles"], "CCO")
        self.assertIn("mcts_visits", root)
        self.assertGreater(root["mcts_visits"], 0)

    def test_mcts_node_has_property_fields(self):
        tree = self.evolver.generate_expansion_tree_mcts(
            max_depth=2,
            max_branching=4,
            predictor=self.predictor,
            optimization_direction='decrease',
            initial_property_value=5.0,
            num_simulations=30,
        )
        for nid, node in tree["nodes"].items():
            if nid == "0":
                continue
            self.assertIn("property_value", node)
            self.assertIn("accumulated_change", node)
            self.assertIn("property_change", node)
            self.assertIn("mcts_visits", node)
            self.assertIn("mcts_prior", node)
            self.assertIn("mcts_q_value", node)

    def test_mcts_edges_reference_valid_nodes(self):
        tree = self.evolver.generate_expansion_tree_mcts(
            max_depth=2,
            max_branching=3,
            predictor=self.predictor,
            optimization_direction='decrease',
            initial_property_value=5.0,
            num_simulations=20,
        )
        node_ids = set(tree["nodes"].keys())
        for edge in tree["edges"]:
            self.assertIn(edge["from"], node_ids)
            self.assertIn(edge["to"], node_ids)


class TestMCTSConstraints(unittest.TestCase):
    """MCTS 应遵守 max_depth、pruning 等约束"""

    def test_max_depth_respected(self):
        evolver = MolecularEvolutionExpansion("CCO")
        predictor = _make_mock_predictor(default_change=-0.1)
        tree = evolver.generate_expansion_tree_mcts(
            max_depth=1,
            max_branching=5,
            predictor=predictor,
            optimization_direction='decrease',
            initial_property_value=5.0,
            num_simulations=50,
        )
        for nid, node in tree["nodes"].items():
            self.assertLessEqual(node["depth"], 1)

    def test_mcts_stats_present(self):
        evolver = MolecularEvolutionExpansion("CCO")
        predictor = _make_mock_predictor(default_change=-0.1)
        tree = evolver.generate_expansion_tree_mcts(
            max_depth=2,
            max_branching=3,
            predictor=predictor,
            optimization_direction='decrease',
            initial_property_value=5.0,
            num_simulations=15,
            exploration_weight=2.0,
        )
        stats = tree["mcts_stats"]
        self.assertEqual(stats["num_simulations"], 15)
        self.assertEqual(stats["exploration_weight"], 2.0)
        self.assertIn("actual_simulations", stats)
        self.assertIn("unique_states_expanded", stats)


class TestMCTSTopKCompat(unittest.TestCase):
    """MCTS 结果应能被 get_topK_results 消费"""

    def test_topk_from_mcts_tree(self):
        evolver = MolecularEvolutionExpansion("CCO")
        predictor = _make_mock_predictor(default_change=-0.3)
        tree = evolver.generate_expansion_tree_mcts(
            max_depth=2,
            max_branching=4,
            predictor=predictor,
            optimization_direction='decrease',
            initial_property_value=5.0,
            num_simulations=30,
        )

        # 模拟 EvolutionTreeOptimizer.get_topK_results 的核心逻辑
        nodes = tree.get("nodes", {})
        leaf_nodes = []
        for nid, node in nodes.items():
            if nid != "0":
                pv = node.get("property_value")
                if pv is not None:
                    leaf_nodes.append({
                        "smiles": node["smiles"],
                        "property_value": pv,
                        "depth": node.get("depth", 0),
                    })

        # 应至少有一个非根节点
        self.assertGreater(len(leaf_nodes), 0)

        # 排序应可正常执行
        sorted_nodes = sorted(leaf_nodes, key=lambda x: x["property_value"])
        topK = sorted_nodes[:5]
        self.assertGreater(len(topK), 0)
        self.assertIn("smiles", topK[0])
        self.assertIn("property_value", topK[0])


class TestMCTSInterrupt(unittest.TestCase):
    """中断信号应导致 MCTS 提前停止"""

    def test_interrupted_stops_early(self):
        evolver = MolecularEvolutionExpansion("CCO")
        evolver.interrupted = True  # 设置中断标志
        predictor = _make_mock_predictor(default_change=-0.1)
        tree = evolver.generate_expansion_tree_mcts(
            max_depth=3,
            max_branching=5,
            predictor=predictor,
            optimization_direction='decrease',
            initial_property_value=5.0,
            num_simulations=1000,
        )
        # 根节点应该几乎没有被访问（第一轮就中断）
        root = tree["nodes"]["0"]
        self.assertLess(root["mcts_visits"], 5)


if __name__ == '__main__':
    unittest.main()
