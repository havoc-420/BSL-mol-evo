#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型模块 v0 版本初始化文件
"""

from .gcn_linear import MoleculeEvolutionGCNPredictor
from .gcn_tf import MoleculeEvolutionGCNTransformerPredictor

__all__ = [
    "MoleculeEvolutionGCNPredictor",
    "MoleculeEvolutionGCNTransformerPredictor"
]