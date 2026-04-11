#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL Demo 数据处理模块（A* RL Demo）

负责：
  - 把 BFS/MCTS 树导出的离线样本整理成 BC 训练可消费的数据结构
  - 状态编码（encode_state）：分子指纹 + 累计变化 + 剩余深度等
  - 动作编码（encode_action）：操作类型 one-hot + OFO 预测变化等
  - value 标签（future_best_gain）：从节点回溯叶子的最优累计收益
  - Dataset / DataLoader 封装，兼容 train_v0_3_path.py 的批量组织方式

状态向量（state_dim=64）默认布局：
  [morgan_fp_50d | accumulated_change | remaining_depth_norm |
   logp_value_norm | logp_in_range | stagnation_count_norm |
   direction_flag | target_property_norm | padding...]
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

STATE_DIM = 64      # 状态向量维度（encode_state 输出维度）
ACTION_DIM = 16     # 动作向量维度（encode_action 输出维度）

# Morgan 指纹维度（取前 N 位，嵌入到 state 的前缀）
_FP_BITS = 50
# 操作类型最大数量（one-hot）
_MAX_OP_TYPES = 10


# ---------------------------------------------------------------------------
# 状态 / 动作编码工具
# ---------------------------------------------------------------------------

def encode_state(
    smiles: str,
    accumulated_change: float,
    remaining_depth: int,
    max_depth: int,
    logp_value: float = 0.0,
    logp_in_range: bool = True,
    stagnation_count: int = 0,
    direction: str = "decrease",
    target_property_value: float = 0.0,
    target_property_scale: float = 10.0,
) -> torch.Tensor:
    """将当前状态编码为固定维度向量。

    Parameters
    ----------
    smiles:
        当前分子 SMILES。
    accumulated_change:
        当前节点的累计属性变化量。
    remaining_depth:
        剩余可扩展深度 = max_depth - current_depth。
    max_depth:
        搜索最大深度（用于归一化）。
    logp_value:
        当前分子的 logP 值（用于 logP 约束感知）。
    logp_in_range:
        logP 是否在约束范围内。
    stagnation_count:
        连续停滞步数。
    direction:
        优化方向，'increase' 或 'decrease'。
    target_property_value:
        初始分子的属性值（上下文信息）。
    target_property_scale:
        属性值归一化尺度（粗略量级，避免尺度差异影响训练）。

    Returns
    -------
    torch.Tensor
        形状 ``(STATE_DIM,)`` 的 float32 向量。
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fp_bits = np.zeros(_FP_BITS, dtype=np.float32)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            arr = np.zeros(2048, dtype=np.float32)
            from rdkit.DataStructs import ConvertToNumpyArray
            ConvertToNumpyArray(fp, arr)
            # 取前 _FP_BITS 位（简化表示，BC 阶段足够）
            fp_bits = arr[:_FP_BITS]
    except Exception:
        fp_bits = np.zeros(_FP_BITS, dtype=np.float32)

    # 标量特征
    scalar_features = np.array([
        float(accumulated_change),                             # 累计属性变化
        float(remaining_depth) / max(max_depth, 1),           # 剩余深度归一化
        float(logp_value) / 10.0,                             # logP 归一化
        1.0 if logp_in_range else 0.0,                        # logP 约束满足标志
        min(float(stagnation_count), 5.0) / 5.0,              # 停滞计数归一化
        1.0 if direction == "decrease" else 0.0,              # 方向标志
        float(target_property_value) / max(abs(target_property_scale), 1e-6),  # 初始属性归一化
    ], dtype=np.float32)

    # 拼接 → padding 到 STATE_DIM
    combined = np.concatenate([fp_bits, scalar_features])
    if len(combined) < STATE_DIM:
        combined = np.pad(combined, (0, STATE_DIM - len(combined)))
    else:
        combined = combined[:STATE_DIM]

    return torch.tensor(combined, dtype=torch.float32)


def encode_action(
    operation: Any,
    ofo_predicted_change: float = 0.0,
    op_type_list: Optional[List[str]] = None,
) -> torch.Tensor:
    """将一个操作动作编码为固定维度向量。

    Parameters
    ----------
    operation:
        操作信息，既兼容 ``{"type": ..., "params": ...}`` 字典，
        也兼容搜索树中直接保存为字符串的操作类型。
    ofo_predicted_change:
        OFO 对此操作的单步属性变化预测值（如可用）。
    op_type_list:
        操作类型枚举列表，用于 one-hot 编码。若为 None 则使用 hash 映射。

    Returns
    -------
    torch.Tensor
        形状 ``(ACTION_DIM,)`` 的 float32 向量。
    """
    if isinstance(operation, dict):
        op_type = operation.get("type", "unknown")
        params = operation.get("params") or operation.get("details", {}) or {}
    elif isinstance(operation, str):
        op_type = operation or "unknown"
        params = {}
    else:
        op_type = "unknown"
        params = {}

    # 操作类型 one-hot（_MAX_OP_TYPES 维）
    if op_type_list:
        type_idx = op_type_list.index(op_type) if op_type in op_type_list else 0
        type_idx = min(type_idx, _MAX_OP_TYPES - 1)
    else:
        # fallback: hash 映射到 [0, _MAX_OP_TYPES)
        type_idx = abs(hash(op_type)) % _MAX_OP_TYPES
    op_onehot = np.zeros(_MAX_OP_TYPES, dtype=np.float32)
    op_onehot[type_idx] = 1.0

    # 参数特征
    if "position" in params:
        raw_position = params.get("position", 0)
    elif "atom_idx" in params and "atom2_idx" in params:
        raw_position = (float(params.get("atom_idx", 0)) + float(params.get("atom2_idx", 0))) / 2.0
    elif "atom_idx" in params:
        raw_position = params.get("atom_idx", 0)
    elif "bond_idx" in params:
        raw_position = params.get("bond_idx", 0)
    else:
        raw_position = 0

    if "fragment_size" in params:
        raw_fragment_size = params.get("fragment_size", 1)
    elif "atom2_idx" in params:
        raw_fragment_size = 2
    elif params:
        raw_fragment_size = 1
    else:
        raw_fragment_size = 0

    position = float(raw_position) / 100.0
    fragment_size = float(raw_fragment_size) / 20.0
    predicted_change = float(ofo_predicted_change)

    scalar = np.array([position, fragment_size, predicted_change], dtype=np.float32)

    combined = np.concatenate([op_onehot, scalar])  # 13 dims
    if len(combined) < ACTION_DIM:
        combined = np.pad(combined, (0, ACTION_DIM - len(combined)))
    else:
        combined = combined[:ACTION_DIM]

    return torch.tensor(combined, dtype=torch.float32)


# ---------------------------------------------------------------------------
# BC Dataset
# ---------------------------------------------------------------------------

class BCDataset(Dataset):
    """Behavioral Cloning 预训练数据集。

    每个样本对应一条 BFS/MCTS 树中的 (parent → child) 边，
    来源于 export_rl_demo_transitions.py 的输出。

    样本字段（参考 plan Key Code Structures）：
      smiles_from, smiles_to, operation, property_change,
      accumulated_from, reward, future_best_gain, remaining_depth
    """

    def __init__(
        self,
        samples: List[Dict],
        max_depth: int = 4,
        direction: str = "decrease",
    ) -> None:
        self.samples = samples
        self.max_depth = max_depth
        self.direction = direction

    @classmethod
    def from_json(cls, path: str, **kwargs) -> "BCDataset":
        """从 export_rl_demo_transitions 输出的 JSON 文件加载。"""
        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        logger.info(f"[BCDataset] 加载 {len(samples)} 条样本 ← {path}")
        return cls(samples, **kwargs)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        state = encode_state(
            smiles=s["smiles_from"],
            accumulated_change=s.get("accumulated_from", 0.0),
            remaining_depth=s.get("remaining_depth", self.max_depth),
            max_depth=self.max_depth,
            direction=self.direction,
        )

        next_state = encode_state(
            smiles=s["smiles_to"],
            accumulated_change=s.get("accumulated_from", 0.0) + s.get("property_change", 0.0),
            remaining_depth=max(s.get("remaining_depth", self.max_depth) - 1, 0),
            max_depth=self.max_depth,
            direction=self.direction,
        )

        action = encode_action(
            operation=s.get("operation", {}),
            ofo_predicted_change=s.get("property_change", 0.0),
        )

        return {
            "state": state,                                                    # (STATE_DIM,)
            "action": action,                                                  # (ACTION_DIM,)
            "next_state": next_state,                                          # (STATE_DIM,)
            "reward": torch.tensor(s.get("reward", 0.0), dtype=torch.float32),
            "future_best_gain": torch.tensor(
                s.get("future_best_gain", 0.0), dtype=torch.float32
            ),
            "property_change": torch.tensor(
                s.get("property_change", 0.0), dtype=torch.float32
            ),
        }


def build_bc_dataloader(
    samples: List[Dict],
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    max_depth: int = 4,
    direction: str = "decrease",
) -> DataLoader:
    """构建 BC 预训练 DataLoader。"""
    dataset = BCDataset(samples, max_depth=max_depth, direction=direction)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Tanimoto novelty helper
# ---------------------------------------------------------------------------

def compute_novelty_score(smiles: str, visited_smiles_set: set) -> float:
    """计算当前分子相对于已访问集合的新颖度（Tanimoto 距离最小值）。

    简化版：直接返回 0/1 二值（是否为新分子），BC 阶段足够用。
    如需精细版本可替换为 Tanimoto 距离。
    """
    return 0.0 if smiles in visited_smiles_set else 1.0
