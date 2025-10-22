#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FragNet 数据处理工具函数
"""

from rdkit import Chem


def remove_bond(rwmol, idx1, idx2):
    """
    从分子中移除指定的键
    
    Args:
        rwmol: 可编辑的RDKit分子对象
        idx1: 第一个原子的索引
        idx2: 第二个原子的索引
    """
    rwmol.RemoveBond(idx1, idx2)
    for idx in [idx1, idx2]:
        atom = rwmol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "N" and atom.GetIsAromatic() is True:
            atom.SetNumExplicitHs(1)