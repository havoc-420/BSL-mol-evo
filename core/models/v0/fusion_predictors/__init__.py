#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征融合预测器模块
"""

from .transformer import TransformerFusionPredictor
from .mlp import MLPFusionPredictor

__all__ = [
    'TransformerFusionPredictor',
    'MLPFusionPredictor'
]