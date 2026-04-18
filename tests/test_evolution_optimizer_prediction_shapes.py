#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回归测试：确保批量预测输出在 batch_size=1 时不会退化为 float。"""

import os
import sys
import unittest

import torch


_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..', '..')
sys.path.insert(0, _root)


class TestEvolutionOptimizerPredictionShapes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from mol_evo.core.evolution_optimizer import EvolutionTreeOptimizer
            cls.EvolutionTreeOptimizer = EvolutionTreeOptimizer
            cls._imports_ok = True
        except Exception as exc:
            cls._imports_ok = False
            cls._import_error = str(exc)

    def _require_imports(self):
        if not self._imports_ok:
            self.skipTest(f"导入失败，跳过测试: {self._import_error}")

    def test_singleton_column_vector_returns_list(self):
        self._require_imports()
        predictions = torch.tensor([[1.25]], dtype=torch.float32)
        result = self.EvolutionTreeOptimizer._prediction_tensor_to_list(predictions)
        self.assertEqual(result, [1.25])

    def test_scalar_tensor_returns_single_item_list(self):
        self._require_imports()
        predictions = torch.tensor(2.5, dtype=torch.float32)
        result = self.EvolutionTreeOptimizer._prediction_tensor_to_list(predictions)
        self.assertEqual(result, [2.5])

    def test_batch_column_vector_preserves_batch_dimension(self):
        self._require_imports()
        predictions = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
        result = self.EvolutionTreeOptimizer._prediction_tensor_to_list(predictions)
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_unsupported_multi_output_shape_raises(self):
        self._require_imports()
        predictions = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        with self.assertRaises(ValueError):
            self.EvolutionTreeOptimizer._prediction_tensor_to_list(predictions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
