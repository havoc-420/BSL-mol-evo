#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型模块初始化文件
"""

# 从 v1 目录导入以保持向后兼容性
from .v1.base import BaseModel, BaseGNNModel
from .v1.gnn import MoleculeGNN, EnhancedMoleculeGNN, MoleculeGNNWithFingerprint
from .v1.simple import MoleculeEvolutionPredictorWithFingerprint, MoleculeEvolutionTransformer
from .v1.nnconv import MoleculeEvolutionNNConvPredictor
from .v1.rgatconv import MoleculeEvolutionGATv2Predictor
from .v1.rgcnconv import MoleculeEvolutionRGCNPredictor
from .v1.transformerconv import MoleculeEvolutionTransformerPredictor

# 从 v0 目录导入新实现的GCN模型
from .v0 import MoleculeEvolutionGCNPredictor

__all__ = [
    'BaseModel',
    'BaseGNNModel',
    'MoleculeGNN',
    'EnhancedMoleculeGNN',
    'MoleculeGNNWithFingerprint',
    'MoleculeEvolutionPredictorWithFingerprint',
    'MoleculeEvolutionTransformer',
    'MoleculeEvolutionNNConvPredictor',
    'MoleculeEvolutionGATv2Predictor',
    'MoleculeEvolutionRGCNPredictor',
    'MoleculeEvolutionTransformerPredictor',
    'MoleculeEvolutionGCNPredictor'
]