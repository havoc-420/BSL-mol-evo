#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一奖励函数模块（A* RL Demo）

实现:
  - RewardConfig: 超参 dataclass，避免魔法数字散落在搜索代码里
  - compute_step_reward(): 单步奖励（属性改善 + logP 惩罚 + 停滞惩罚 + 多样性 bonus）
  - compute_episode_reward(): 整条轨迹的全局奖励，用作 REINFORCE baseline 参考

所有搜索侧/训练侧代码均应从此模块导入，不要在别处重新定义打分逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class RewardConfig:
    """奖励函数超参，字段与 plan 中 Key Code Structures 保持一致。

    Attributes
    ----------
    property_weight:
        属性改善奖励权重。正向优化时 property_change > 0 即为正奖励。
    logp_penalty:
        logP 超出约束范围时的惩罚值（固定值，不与 violation 量成比例）。
    stagnation_penalty:
        连续无属性改善时的停滞惩罚值。
    stagnation_patience:
        允许连续停滞的步数，超过此数才开始施加 stagnation_penalty。
        建议与现有 BFS pruning_patience 对齐（默认 2）。
    diversity_weight:
        多样性 bonus 权重。novelty_score 由调用方计算（如与已访问集合的
        最小 Tanimoto 距离），设为 0 即关闭。
    episode_weight:
        整条轨迹全局奖励权重，用于 compute_episode_reward()。
    direction:
        优化方向，'increase' 或 'decrease'。influence property_change 的符号约定。
    """

    property_weight: float = 1.0
    logp_penalty: float = 0.5
    stagnation_penalty: float = 0.3
    stagnation_patience: int = 2
    diversity_weight: float = 0.1
    episode_weight: float = 0.5
    direction: str = "decrease"  # 'increase' | 'decrease'

    def __post_init__(self) -> None:
        assert self.direction in ("increase", "decrease"), (
            f"direction must be 'increase' or 'decrease', got '{self.direction}'"
        )


# ---------------------------------------------------------------------------
# Step-level reward
# ---------------------------------------------------------------------------

def compute_step_reward(
    property_change: float,
    logp_in_range: bool,
    stagnation_count: int,
    novelty_score: float = 0.0,
    config: Optional[RewardConfig] = None,
) -> float:
    """计算单步奖励。

    Parameters
    ----------
    property_change:
        本步的属性变化量。方向由 ``config.direction`` 决定：
        * ``'decrease'`` → property_change 越负越好，故奖励 = -property_change * weight
        * ``'increase'`` → property_change 越正越好，故奖励 = +property_change * weight
    logp_in_range:
        本步生成分子的 logP 是否在允许范围内。
    stagnation_count:
        当前连续无改善步数（由搜索侧维护）。
    novelty_score:
        本步分子相对于已访问集合的新颖度（0-1），由搜索侧计算。
    config:
        奖励超参，默认使用 ``RewardConfig()``。

    Returns
    -------
    float
        本步标量奖励。
    """
    if config is None:
        config = RewardConfig()

    r = 0.0

    # 1. 属性改善奖励（核心信号）
    if config.direction == "decrease":
        # 属性越小越好：property_change 为负时代表改善
        r += config.property_weight * (-property_change)
    else:
        # 属性越大越好：property_change 为正时代表改善
        r += config.property_weight * property_change

    # 2. logP 约束惩罚
    if not logp_in_range:
        r -= config.logp_penalty

    # 3. 停滞惩罚
    if stagnation_count > config.stagnation_patience:
        r -= config.stagnation_penalty

    # 4. 多样性 bonus
    r += config.diversity_weight * novelty_score

    return r


# ---------------------------------------------------------------------------
# Episode-level reward
# ---------------------------------------------------------------------------

def compute_episode_reward(
    step_rewards: Sequence[float],
    best_property_improvement: float,
    config: Optional[RewardConfig] = None,
) -> float:
    """计算整条轨迹的全局奖励。

    主要用于 REINFORCE 中作为 baseline 参考值，或替代 per-step return。

    Parameters
    ----------
    step_rewards:
        本 episode 每步的单步奖励序列（由 compute_step_reward 产出）。
    best_property_improvement:
        本 episode 内所有节点中，属性改善量的最大值（已按 direction 转换为正值）。
    config:
        奖励超参，默认使用 ``RewardConfig()``。

    Returns
    -------
    float
        episode 级标量奖励。
    """
    if config is None:
        config = RewardConfig()

    episode_reward = (
        sum(step_rewards) + config.episode_weight * best_property_improvement
    )
    return episode_reward


# ---------------------------------------------------------------------------
# Discount / return helpers
# ---------------------------------------------------------------------------

def compute_discounted_returns(
    rewards: List[float],
    gamma: float = 0.99,
) -> List[float]:
    """计算折扣累计回报（从后往前）。

    Parameters
    ----------
    rewards:
        每步奖励列表（时序顺序）。
    gamma:
        折扣因子。

    Returns
    -------
    List[float]
        每步的折扣累计回报 G_t。
    """
    returns: List[float] = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    return returns


def normalize_returns(returns: List[float], eps: float = 1e-8) -> List[float]:
    """对回报序列做零均值单位方差标准化，用于减小 REINFORCE 梯度方差。"""
    if len(returns) == 0:
        return returns
    mean = sum(returns) / len(returns)
    var = sum((x - mean) ** 2 for x in returns) / max(len(returns), 1)
    std = var ** 0.5
    return [(x - mean) / (std + eps) for x in returns]
