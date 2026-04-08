#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 astar_demo 搜索模式的基本功能：
  - generate_expansion_tree_astar_demo() 输出结构
  - astar_stats 字段完整性
  - 节点 A* 额外字段（g_score, h_score, f_score, policy_score）
  - PolicyNet / ValueNet 集成（使用随机初始化权重）

运行方式：
    python -m pytest mol_evo/tests/test_astar_demo_search_mode.py -v
    # 或直接运行
    python mol_evo/tests/test_astar_demo_search_mode.py
"""

import os
import sys
import json
import types
import unittest

# 项目根路径
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..', '..')
sys.path.insert(0, _root)


# ---------------------------------------------------------------------------
# 轻量 mock predictor（避免依赖真实 OFO 模型）
# ---------------------------------------------------------------------------

class MockPredictor:
    """模拟 predict_batch()，返回随机小值列表。"""

    def predict_batch(self, from_smiles_list, to_smiles_list, operations_list):
        import random
        return [random.uniform(-0.5, 0.5) for _ in from_smiles_list]


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------

class TestAstarDemoSearchMode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """导入依赖模块并初始化测试用对象。"""
        try:
            from mol_evo.core.molecular_evolution_expansion import MolecularEvolutionExpansion
            from mol_evo.core.models.astar_rl import PolicyNet, ValueNet
            from mol_evo.core.data.rl_demo_processing import STATE_DIM, ACTION_DIM
            import torch

            cls.MolecularEvolutionExpansion = MolecularEvolutionExpansion
            cls.PolicyNet = PolicyNet
            cls.ValueNet = ValueNet
            cls.STATE_DIM = STATE_DIM
            cls.ACTION_DIM = ACTION_DIM
            cls.torch = torch
            cls._imports_ok = True
        except ImportError as e:
            cls._imports_ok = False
            cls._import_error = str(e)

        cls.test_smiles = "c1ccccc1"  # 苯
        cls.predictor = MockPredictor()

    def _require_imports(self):
        if not self._imports_ok:
            self.skipTest(f"导入失败，跳过测试: {self._import_error}")

    # ------------------------------------------------------------------
    # 测试：无 policy/value 时纯 A* 搜索可以运行并返回正确结构
    # ------------------------------------------------------------------

    def test_basic_tree_structure(self):
        """astar_demo 搜索返回标准树结构（nodes / edges / initial_smiles）。"""
        self._require_imports()
        evolver = self.MolecularEvolutionExpansion(self.test_smiles)
        tree = evolver.generate_expansion_tree_astar_demo(
            max_depth=2,
            max_branching=3,
            predictor=self.predictor,
            optimization_direction="decrease",
            open_set_budget=10,
        )

        self.assertIn("nodes", tree)
        self.assertIn("edges", tree)
        self.assertIn("initial_smiles", tree)
        self.assertEqual(tree["initial_smiles"], self.test_smiles)
        self.assertGreater(len(tree["nodes"]), 0)  # 至少有根节点

    def test_astar_stats_present(self):
        """astar_stats 字段应包含所有预期子键。"""
        self._require_imports()
        evolver = self.MolecularEvolutionExpansion(self.test_smiles)
        tree = evolver.generate_expansion_tree_astar_demo(
            max_depth=2,
            max_branching=3,
            predictor=self.predictor,
            optimization_direction="decrease",
            open_set_budget=10,
        )

        self.assertIn("astar_stats", tree)
        stats = tree["astar_stats"]
        for key in ("expanded_nodes", "open_set_peak", "policy_prefilter_size",
                    "ofo_scored_candidates", "actual_expansions"):
            self.assertIn(key, stats, f"astar_stats 缺少字段: {key}")

    def test_node_astar_fields(self):
        """非根节点应包含 A* 额外字段。"""
        self._require_imports()
        evolver = self.MolecularEvolutionExpansion(self.test_smiles)
        tree = evolver.generate_expansion_tree_astar_demo(
            max_depth=2,
            max_branching=3,
            predictor=self.predictor,
            optimization_direction="decrease",
            open_set_budget=15,
        )

        non_root_nodes = [
            n for nid, n in tree["nodes"].items()
            if n.get("depth", 0) > 0
        ]
        if not non_root_nodes:
            self.skipTest("搜索未展开非根节点，跳过字段检查")

        for node in non_root_nodes:
            for key in ("g_score", "h_score", "f_score", "policy_score"):
                self.assertIn(key, node, f"节点缺少字段: {key}")

    def test_with_policy_and_value_net(self):
        """带 PolicyNet / ValueNet 时搜索可以正常运行。"""
        self._require_imports()
        policy_net = self.PolicyNet(
            state_dim=self.STATE_DIM, action_dim=self.ACTION_DIM
        )
        value_net = self.ValueNet(state_dim=self.STATE_DIM)
        policy_net.eval()
        value_net.eval()

        evolver = self.MolecularEvolutionExpansion(self.test_smiles)
        tree = evolver.generate_expansion_tree_astar_demo(
            max_depth=2,
            max_branching=4,
            predictor=self.predictor,
            optimization_direction="decrease",
            policy_net=policy_net,
            value_net=value_net,
            top_n_prefilter=5,
            open_set_budget=15,
        )

        self.assertIn("nodes", tree)
        self.assertIn("astar_stats", tree)
        stats = tree["astar_stats"]
        # policy 预筛应有记录
        self.assertGreaterEqual(stats["policy_prefilter_size"], 0)

    def test_edge_references_valid_nodes(self):
        """所有边的 from/to 都指向有效节点。"""
        self._require_imports()
        evolver = self.MolecularEvolutionExpansion(self.test_smiles)
        tree = evolver.generate_expansion_tree_astar_demo(
            max_depth=2,
            max_branching=3,
            predictor=self.predictor,
            optimization_direction="decrease",
            open_set_budget=15,
        )

        node_ids = set(tree["nodes"].keys())
        for edge in tree["edges"]:
            src = edge.get("from")
            tgt = edge.get("to")
            self.assertIn(src, node_ids, f"边的 from={src} 不在节点集中")
            self.assertIn(tgt, node_ids, f"边的 to={tgt} 不在节点集中")

    def test_open_set_budget_limits_expansions(self):
        """open_set_budget 应限制展开次数不超过预算。"""
        self._require_imports()
        budget = 5
        evolver = self.MolecularEvolutionExpansion(self.test_smiles)
        tree = evolver.generate_expansion_tree_astar_demo(
            max_depth=4,
            max_branching=8,
            predictor=self.predictor,
            optimization_direction="decrease",
            open_set_budget=budget,
        )
        stats = tree["astar_stats"]
        self.assertLessEqual(
            stats["expanded_nodes"], budget,
            f"expanded_nodes={stats['expanded_nodes']} 超过 budget={budget}"
        )


# ---------------------------------------------------------------------------
# 直接运行
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
