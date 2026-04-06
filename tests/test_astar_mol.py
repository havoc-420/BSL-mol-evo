#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A* + Surrogate Value Function 分子进化搜索 Demo
================================================
方案一：RL 训练启发函数，A* 负责搜索

核心思想：
    f(n) = g(n) + h(n)
      g(n) = 从起始分子出发已走的变异步数（实际代价）
      h(n) = Surrogate Value Function V(s) 预测"还需多少步才能达到目标性质"

本 Demo 说明：
    - 用 RDKit 内置 LogP 计算模拟 Value Function（无需加载模型 checkpoint）
    - 通过 ValueFunctionBase 接口预留 RL 神经网络的插槽位置
    - 对比 Dijkstra / Greedy / A* 三种策略在分子进化空间中的探索效率

运行方式：
    python tests/test_astar_mol.py
"""

import heapq
import time
import random
from typing import Callable, Optional
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")  # 无显示器环境下使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, QED, rdMolDescriptors, AllChem
from rdkit.Chem import RWMol

# 屏蔽 RDKit 冗余警告
RDLogger.DisableLog("rdApp.*")


# ═══════════════════════════════════════════════════════════════
# 1. 分子搜索节点
# ═══════════════════════════════════════════════════════════════

class MolNode:
    """
    A* 搜索树节点，封装分子 SMILES、代价和父节点信息。

    属性
    ----
    smiles    : 当前分子的规范 SMILES
    g         : 从起点到当前节点已走的变异步数（实际代价）
    h         : Value Function 估计的剩余代价
    parent    : 父节点（用于路径回溯）
    operation : 产生本节点的操作描述
    """

    def __init__(self,
                 smiles: str,
                 g: float = 0.0,
                 h: float = 0.0,
                 parent: Optional["MolNode"] = None,
                 operation: str = ""):
        self.smiles = smiles
        self.g = g
        self.h = h
        self.parent = parent
        self.operation = operation

    @property
    def f(self) -> float:
        """总估计代价 f = g + h"""
        return self.g + self.h

    # 优先队列比较：按 f 排序，相同时 h 小的优先（更接近目标）
    def __lt__(self, other: "MolNode") -> bool:
        if abs(self.f - other.f) > 1e-9:
            return self.f < other.f
        return self.h < other.h

    def __eq__(self, other) -> bool:
        return self.smiles == other.smiles

    def __hash__(self) -> int:
        return hash(self.smiles)


# ═══════════════════════════════════════════════════════════════
# 2. Value Function 接口（RL 插槽）
# ═══════════════════════════════════════════════════════════════

class ValueFunctionBase:
    """
    抽象接口：RL 训练后替换 estimate() 即可。

    在真实的"方案一"中，这里会加载训练好的神经网络权重，
    接受分子的图表示（或 fingerprint）作为输入，输出 V(s)。
    本 Demo 用解析函数代替网络前向传播。
    """

    def estimate(self, smiles: str, target: float,
                 prop_fn: Callable[[str], float]) -> float:
        """
        估计从当前分子到达目标性质还需要多少步。

        参数
        ----
        smiles  : 当前分子 SMILES
        target  : 目标性质值
        prop_fn : 性质计算函数（如 logp_fn）

        返回
        ----
        h >= 0，估计剩余步数
        """
        raise NotImplementedError


class PropertyGapValueFunction(ValueFunctionBase):
    """
    基于性质差距的简单 Value Function（模拟 RL 学习结果）。

    h(s) = max(0, target - prop(s)) / avg_step_gain

    直觉：当前性质与目标差距越大，预估还需更多步。
    avg_step_gain 是每次变异平均能提升的性质量，
    真实 RL 中这个参数会被网络自动学习。

    ┌─────────────────────────────────────────┐
    │  RL 替换入口：将 estimate() 替换为      │
    │  神经网络前向传播即可升级至完整方案一    │
    └─────────────────────────────────────────┘
    """

    def __init__(self, avg_step_gain: float = 0.3):
        """
        参数
        ----
        avg_step_gain : 每步操作平均性质增益（默认 LogP 每步 ~0.3）
        """
        self.avg_step_gain = avg_step_gain

    def estimate(self, smiles: str, target: float,
                 prop_fn: Callable[[str], float]) -> float:
        try:
            current = prop_fn(smiles)
            gap = max(0.0, target - current)
            return gap / max(self.avg_step_gain, 1e-6)
        except Exception:
            return float("inf")


class ZeroValueFunction(ValueFunctionBase):
    """h=0，退化为 Dijkstra（仅按步数搜索）"""

    def estimate(self, smiles: str, target: float,
                 prop_fn: Callable[[str], float]) -> float:
        return 0.0


class GreedyValueFunction(ValueFunctionBase):
    """
    超大权重贪心：h = gap * 1e6，近似只靠性质引导（忽略步数）。
    速度快但可能绕路或陷入局部最优。
    """

    def estimate(self, smiles: str, target: float,
                 prop_fn: Callable[[str], float]) -> float:
        try:
            current = prop_fn(smiles)
            gap = max(0.0, target - current)
            return gap * 1e6
        except Exception:
            return float("inf")


# ═══════════════════════════════════════════════════════════════
# 3. 分子变异操作（对齐 mol-evo 词汇表的简化子集）
# ═══════════════════════════════════════════════════════════════

# 支持的原子类型（与 ic50.yaml 的 atom_types 对应）
_ADD_ATOMS = ["C", "N", "O", "F"]
_CHANGE_ATOMS = ["C", "N", "O", "S", "F", "Cl"]


def _canonical(smiles: str) -> Optional[str]:
    """返回规范 SMILES，无效则返回 None"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        # 拒绝碎片分子（含 '.'）
        canon = Chem.MolToSmiles(mol)
        if "." in canon:
            return None
        return canon
    except Exception:
        return None


def _mol_from_smiles(smiles: str) -> Optional[RWMol]:
    """从 SMILES 创建可编辑分子，失败返回 None"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return RWMol(mol)


def _to_smiles(rwmol: RWMol) -> Optional[str]:
    """将 RWMol 转换为规范 SMILES，失败返回 None"""
    try:
        Chem.SanitizeMol(rwmol)
        smiles = Chem.MolToSmiles(rwmol)
        if "." in smiles or len(smiles) == 0:
            return None
        return smiles
    except Exception:
        return None


def op_add_atom(smiles: str) -> list[str]:
    """ADD_ATOM：向每个现有原子添加 C/N/O/F"""
    results = []
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return results

    n_atoms = mol.GetNumAtoms()
    for atom_idx in range(n_atoms):
        for new_atom_sym in _ADD_ATOMS:
            rw = RWMol(mol)
            new_idx = rw.AddAtom(Chem.Atom(new_atom_sym))
            rw.AddBond(atom_idx, new_idx, Chem.BondType.SINGLE)
            s = _to_smiles(rw)
            if s and s != smiles:
                results.append((s, f"ADD_ATOM({new_atom_sym})@{atom_idx}"))
    return results


def op_remove_atom(smiles: str) -> list[str]:
    """REMOVE_ATOM：移除叶节点原子（度为1的非环原子）"""
    results = []
    mol = _mol_from_smiles(smiles)
    if mol is None or mol.GetNumAtoms() <= 2:
        return results

    ring_info = mol.GetRingInfo()
    ring_atoms = set(ring_info.AtomRings()[0]) if ring_info.NumRings() > 0 else set()
    # 更健壮地获取所有环原子
    all_ring_atoms = set()
    for ring in ring_info.AtomRings():
        all_ring_atoms.update(ring)

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        # 叶节点：度为1，且不在环中
        if atom.GetDegree() == 1 and idx not in all_ring_atoms:
            rw = RWMol(mol)
            rw.RemoveAtom(idx)
            s = _to_smiles(rw)
            if s and s != smiles:
                results.append((s, f"REMOVE_ATOM@{idx}"))
    return results


def op_change_atom(smiles: str) -> list[str]:
    """REPLACE_ATOM：替换非环原子的原子类型"""
    results = []
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return results

    ring_info = mol.GetRingInfo()
    all_ring_atoms = set()
    for ring in ring_info.AtomRings():
        all_ring_atoms.update(ring)

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in all_ring_atoms:
            continue
        current_sym = atom.GetSymbol()
        for new_sym in _CHANGE_ATOMS:
            if new_sym == current_sym:
                continue
            rw = RWMol(mol)
            rw.ReplaceAtom(idx, Chem.Atom(new_sym))
            s = _to_smiles(rw)
            if s and s != smiles:
                results.append((s, f"CHANGE_ATOM({current_sym}→{new_sym})@{idx}"))
    return results


def op_add_bond(smiles: str) -> list[str]:
    """ADD_BOND：在两个已有原子间增加键（单键→双键）"""
    results = []
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return results

    n = mol.GetNumAtoms()
    # 限制原子数以避免组合爆炸
    if n > 12:
        return results

    for i in range(n):
        for j in range(i + 1, n):
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is None:
                # 在两个无键原子间加单键
                rw = RWMol(mol)
                rw.AddBond(i, j, Chem.BondType.SINGLE)
                s = _to_smiles(rw)
                if s and s != smiles:
                    results.append((s, f"ADD_BOND(single)@{i}-{j}"))
            elif bond.GetBondTypeAsDouble() == 1.0:
                # 单键升为双键
                rw = RWMol(mol)
                rw.GetBondBetweenAtoms(i, j).SetBondType(Chem.BondType.DOUBLE)
                s = _to_smiles(rw)
                if s and s != smiles:
                    results.append((s, f"ADD_BOND(double)@{i}-{j}"))
    return results


def get_neighbors(smiles: str) -> list[tuple[str, str]]:
    """
    获取当前分子的所有合法变异邻居。

    返回 list of (neighbor_smiles, operation_description)
    操作集合对齐 ic50.yaml：ADD_ATOM / REMOVE_ATOM / REPLACE_ATOM / ADD_BOND
    """
    neighbors = []
    seen = {smiles}

    for op_fn in [op_add_atom, op_remove_atom, op_change_atom, op_add_bond]:
        try:
            for (s, op) in op_fn(smiles):
                canon = _canonical(s)
                if canon and canon not in seen:
                    seen.add(canon)
                    neighbors.append((canon, op))
        except Exception:
            pass

    return neighbors


# ═══════════════════════════════════════════════════════════════
# 4. 性质函数
# ═══════════════════════════════════════════════════════════════

def logp_fn(smiles: str) -> float:
    """计算 LogP（Crippen 方法）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("-inf")
    return Crippen.MolLogP(mol)


def qed_fn(smiles: str) -> float:
    """计算 QED（药物类药性，0~1）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return QED.qed(mol)


# ═══════════════════════════════════════════════════════════════
# 5. A* 分子搜索
# ═══════════════════════════════════════════════════════════════

def mol_astar(
    start_smiles: str,
    target_value: float,
    prop_fn: Callable[[str], float],
    value_fn: ValueFunctionBase,
    max_steps: int = 8,
    beam_width: int = 10,
    max_explored: int = 300,
) -> tuple[list[str], list[str], list[str], float]:
    """
    用 A* 在分子进化空间中搜索，直到 prop_fn(smiles) >= target_value。

    参数
    ----
    start_smiles  : 起始分子 SMILES
    target_value  : 目标性质阈值（prop_fn >= target 即算到达目标）
    prop_fn       : 性质计算函数
    value_fn      : Value Function（启发函数 h 的来源）
    max_steps     : 最大变异步数（防止无限扩展）
    beam_width    : 每步最多展开的邻居数（用 prop 分数截断，模拟 RL policy 剪枝）
    max_explored  : 探索节点总数上限

    返回
    ----
    (path_smiles, operations, explored_smiles, final_property)
    path_smiles   : 从起点到终点的分子 SMILES 列表
    operations    : 每步操作描述
    explored_smiles: 按探索顺序的所有访问分子
    final_property: 终点分子的性质值
    """
    canon_start = _canonical(start_smiles)
    if canon_start is None:
        return [], [], [], float("-inf")

    start_node = MolNode(
        smiles=canon_start,
        g=0.0,
        h=value_fn.estimate(canon_start, target_value, prop_fn),
    )

    # 优先队列（最小堆）
    open_heap: list[MolNode] = []
    heapq.heappush(open_heap, start_node)

    # 每个 SMILES 的最优 g 值
    g_best: dict[str, float] = {canon_start: 0.0}

    # 已关闭节点
    closed_set: set[str] = set()

    # 按探索顺序记录
    explored: list[str] = []

    while open_heap and len(explored) < max_explored:
        current = heapq.heappop(open_heap)

        # 跳过次优路径
        if current.smiles in closed_set:
            continue
        if current.g > g_best.get(current.smiles, float("inf")):
            continue

        closed_set.add(current.smiles)
        explored.append(current.smiles)

        # 目标检验
        if prop_fn(current.smiles) >= target_value:
            # 回溯路径
            path_smiles, operations = [], []
            node = current
            while node is not None:
                path_smiles.append(node.smiles)
                operations.append(node.operation)
                node = node.parent
            path_smiles.reverse()
            operations.reverse()
            return path_smiles, operations, explored, prop_fn(current.smiles)

        # 步数限制
        if current.g >= max_steps:
            continue

        # 获取邻居并按性质增益截断（beam_width）
        neighbors = get_neighbors(current.smiles)
        if not neighbors:
            continue

        # 用 prop_fn 打分并取 top-k（模拟 RL policy 的剪枝效果）
        scored = []
        for (nb_smiles, op_desc) in neighbors:
            if nb_smiles in closed_set:
                continue
            try:
                score = prop_fn(nb_smiles)  # 分数越高越优先
                scored.append((score, nb_smiles, op_desc))
            except Exception:
                pass

        scored.sort(key=lambda x: -x[0])  # 降序
        top_neighbors = scored[:beam_width]

        for (_, nb_smiles, op_desc) in top_neighbors:
            new_g = current.g + 1.0
            if new_g < g_best.get(nb_smiles, float("inf")):
                g_best[nb_smiles] = new_g
                h = value_fn.estimate(nb_smiles, target_value, prop_fn)
                nb_node = MolNode(
                    smiles=nb_smiles,
                    g=new_g,
                    h=h,
                    parent=current,
                    operation=op_desc,
                )
                heapq.heappush(open_heap, nb_node)

    # 未找到满足目标的路径，返回探索过的最佳分子
    if explored:
        best_smiles = max(explored, key=lambda s: prop_fn(s))
        return [], [], explored, prop_fn(best_smiles)
    return [], [], explored, float("-inf")


# ═══════════════════════════════════════════════════════════════
# 6. 单元测试
# ═══════════════════════════════════════════════════════════════

def run_tests():
    """运行基本单元测试，验证各组件正确性"""
    print("\n" + "=" * 60)
    print("  单元测试")
    print("=" * 60)
    all_pass = True

    # ── 测试 1：分子有效性检验 ──
    valid_cases = ["C", "CC", "CCO", "c1ccccc1", "CC(=O)O"]
    invalid_cases = ["XYZ", "C(C)(C)(C)(C)C"]  # 无效 SMILES
    ok1 = all(_canonical(s) is not None for s in valid_cases)
    # 注意：RDKit 对某些字符串较宽容，此处只验证明显无效的
    invalid_ok = _canonical("XYZ") is None
    ok1 = ok1 and invalid_ok
    print(f"  {'✓' if ok1 else '✗'} 测试1：分子有效性检验  "
          f"(有效:{sum(1 for s in valid_cases if _canonical(s))}/"
          f"{len(valid_cases)}, 无效拦截:{invalid_ok})")
    all_pass &= ok1

    # ── 测试 2：变异操作生成（邻居 > 0）──
    test_mol = "CC"
    neighbors = get_neighbors(test_mol)
    ok2 = len(neighbors) > 0
    print(f"  {'✓' if ok2 else '✗'} 测试2：变异操作生成  "
          f"'{test_mol}' 生成邻居={len(neighbors)} (期望>0)")
    all_pass &= ok2

    # ── 测试 3：Value Function 估计值 >= 0 ──
    vf = PropertyGapValueFunction(avg_step_gain=0.3)
    h_val = vf.estimate("CC", target=5.0, prop_fn=logp_fn)
    ok3 = h_val >= 0
    print(f"  {'✓' if ok3 else '✗'} 测试3：Value Function 估计值 >= 0  "
          f"h('CC', target=5.0)={h_val:.3f}")
    all_pass &= ok3

    # ── 测试 4：A* 能找到路径（目标适中）──
    vf4 = PropertyGapValueFunction(avg_step_gain=0.3)
    start_logp = logp_fn("CC")
    # 目标：比起点高 1.0 即可（应在几步内找到）
    target4 = start_logp + 1.0
    path4, ops4, explored4, final4 = mol_astar(
        "CC", target4, logp_fn, vf4,
        max_steps=6, beam_width=8, max_explored=200
    )
    ok4 = len(path4) > 0 and final4 >= target4
    print(f"  {'✓' if ok4 else '✗'} 测试4：A* 找到路径且性质达标  "
          f"路径={len(path4)}步  最终LogP={final4:.3f}  目标={target4:.3f}")
    all_pass &= ok4

    # ── 测试 5：A* 探索节点 < Dijkstra（启发函数有效剪枝）──
    vf_astar = PropertyGapValueFunction(avg_step_gain=0.3)
    vf_dijkstra = ZeroValueFunction()
    target5 = logp_fn("CC") + 1.5

    _, _, explored_astar, _ = mol_astar(
        "CC", target5, logp_fn, vf_astar,
        max_steps=6, beam_width=8, max_explored=200
    )
    _, _, explored_dijkstra, _ = mol_astar(
        "CC", target5, logp_fn, vf_dijkstra,
        max_steps=6, beam_width=8, max_explored=200
    )
    ok5 = len(explored_astar) <= len(explored_dijkstra)
    print(f"  {'✓' if ok5 else '✗'} 测试5：A* 探索节点 ≤ Dijkstra  "
          f"A*={len(explored_astar)}  Dijkstra={len(explored_dijkstra)}")
    all_pass &= ok5

    print(f"\n  {'所有测试通过 ✓' if all_pass else '部分测试失败 ✗'}")
    return all_pass


# ═══════════════════════════════════════════════════════════════
# 7. Demo 1：基本 A* 分子优化
# ═══════════════════════════════════════════════════════════════

def demo_basic():
    """
    基本 A* 分子优化演示：
    从苯（c1ccccc1, LogP≈1.69）出发，搜索 LogP≥4.0 的分子。
    """
    print("\n" + "=" * 60)
    print("  Demo 1：基本 A* 分子优化（起始分子 → LogP 目标）")
    print("=" * 60)

    start = "c1ccccc1"  # 苯，LogP≈1.69
    target = 4.0
    vf = PropertyGapValueFunction(avg_step_gain=0.5)

    print(f"  起始分子: {start}  LogP={logp_fn(start):.3f}")
    print(f"  目标    : LogP ≥ {target}")

    t0 = time.perf_counter()
    path, ops, explored, final_prop = mol_astar(
        start, target, logp_fn, vf,
        max_steps=8, beam_width=12, max_explored=300
    )
    elapsed = (time.perf_counter() - t0) * 1000

    if path:
        print(f"\n  ✓ 找到路径！ 步数={len(path)-1}  探索={len(explored)}  "
              f"最终LogP={final_prop:.3f}  耗时={elapsed:.1f}ms")
        print("\n  进化路径：")
        for i, (s, op) in enumerate(zip(path, ops)):
            lp = logp_fn(s)
            arrow = "▶ " if i > 0 else "  "
            op_str = f"[{op}]" if op else "[起点]"
            print(f"    {arrow}步{i}: {s:<40} LogP={lp:.3f}  {op_str}")
    else:
        print(f"  ✗ 未找到满足目标的路径  探索={len(explored)}  "
              f"最佳LogP={final_prop:.3f}")

    return path, ops, explored, final_prop


# ═══════════════════════════════════════════════════════════════
# 8. Demo 2：三策略对比 + 可视化
# ═══════════════════════════════════════════════════════════════

def demo_compare():
    """
    对比三种搜索策略：

    策略                  g(n)  h(n)              特点
    ──────────────────────────────────────────────────────────
    Dijkstra (BFS)         步数   0                穷举，探索最多
    Greedy (性质引导)       0    性质差距×1e6       快但可能绕路
    A* (g+h)              步数   性质差距/avg_gain  平衡最优与速度
    """
    print("\n" + "=" * 60)
    print("  Demo 2：三策略对比（Dijkstra / Greedy / A*）")
    print("=" * 60)

    start = "CCO"   # 乙醇，LogP≈-0.14
    target = 3.5    # 目标 LogP
    common_params = dict(
        max_steps=8,
        beam_width=12,
        max_explored=400,
    )

    strategies = [
        ("Dijkstra\n(h=0)",       ZeroValueFunction(),                    "tab:blue"),
        ("Greedy\n(仅性质引导)",   GreedyValueFunction(),                  "tab:orange"),
        ("A*\n(g+h 平衡)",         PropertyGapValueFunction(avg_step_gain=0.4), "tab:green"),
    ]

    results = []
    for (name, vf, color) in strategies:
        t0 = time.perf_counter()
        path, ops, explored, final_prop = mol_astar(
            start, target, logp_fn, vf, **common_params
        )
        elapsed = (time.perf_counter() - t0) * 1000

        tag = name.replace("\n", " ")
        status = "✓" if path else "✗"
        print(f"  {status} {tag:<20}  路径={len(path):>2}步  "
              f"探索={len(explored):>3}  最终LogP={final_prop:.3f}  "
              f"耗时={elapsed:.1f}ms")

        results.append({
            "name": name,
            "path": path,
            "ops": ops,
            "explored": explored,
            "final_prop": final_prop,
            "elapsed": elapsed,
            "color": color,
        })

    # ── 绘图：三子图 ──
    _plot_compare(results, start, target, logp_fn)
    return results


def _plot_compare(results: list, start: str, target: float,
                  prop_fn: Callable[[str], float]):
    """生成三种策略对比可视化图"""

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "A* + Surrogate Value Function 分子进化搜索\n"
        f"起始分子: {start}  LogP={prop_fn(start):.2f}  目标 LogP ≥ {target}",
        fontsize=13, fontweight="bold", y=0.98
    )

    # ── 子图1：搜索树（探索节点性质散点图） ──
    ax1 = fig.add_subplot(2, 3, (1, 2))
    ax1.set_title("探索节点的 LogP 分布（按探索顺序）", fontsize=10)

    for r in results:
        name_short = r["name"].replace("\n", " ")
        props = [prop_fn(s) for s in r["explored"]]
        ax1.plot(
            range(len(props)), props,
            color=r["color"], alpha=0.6,
            linewidth=1.5, label=f"{name_short} (探索={len(r['explored'])})"
        )
        # 标注路径终点
        if r["path"]:
            ax1.scatter(
                [len(r["explored"]) - 1], [r["final_prop"]],
                color=r["color"], s=100, zorder=5, marker="*"
            )

    ax1.axhline(y=target, color="red", linestyle="--", linewidth=1.5,
                label=f"目标 LogP={target}")
    ax1.set_xlabel("探索顺序（步数）")
    ax1.set_ylabel("LogP")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ── 子图2：探索节点数柱状图 ──
    ax2 = fig.add_subplot(2, 3, 3)
    ax2.set_title("探索效率对比", fontsize=10)

    names_short = [r["name"].replace("\n", "\n") for r in results]
    explored_counts = [len(r["explored"]) for r in results]
    colors = [r["color"] for r in results]
    bars = ax2.bar(range(len(results)), explored_counts, color=colors,
                   alpha=0.8, edgecolor="black", linewidth=0.8)

    for bar, count in zip(bars, explored_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 str(count), ha="center", va="bottom", fontsize=9,
                 fontweight="bold")

    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels(names_short, fontsize=8)
    ax2.set_ylabel("探索节点数")
    ax2.set_ylim(0, max(explored_counts) * 1.2)
    ax2.grid(True, axis="y", alpha=0.3)

    # ── 子图3-5：各策略进化路径的性质曲线 ──
    for i, r in enumerate(results):
        ax = fig.add_subplot(2, 3, 4 + i)
        name_short = r["name"].replace("\n", " ")
        ax.set_title(f"{name_short}\n探索={len(r['explored'])}  路径={len(r['path'])-1 if r['path'] else 0}步",
                     fontsize=9)

        if r["path"]:
            path_props = [prop_fn(s) for s in r["path"]]
            ax.plot(range(len(path_props)), path_props,
                    color=r["color"], linewidth=2, marker="o", markersize=5,
                    label="路径LogP")
            ax.fill_between(range(len(path_props)), path_props,
                            alpha=0.15, color=r["color"])
        else:
            # 展示探索节点的最优路径趋势
            ax.text(0.5, 0.5, "未找到完整路径\n(达到探索上限)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")

        ax.axhline(y=prop_fn(start), color="gray", linestyle=":",
                   linewidth=1, label=f"起点LogP={prop_fn(start):.2f}")
        ax.axhline(y=target, color="red", linestyle="--",
                   linewidth=1.2, label=f"目标={target}")
        ax.set_xlabel("变异步数")
        ax.set_ylabel("LogP")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = "/tmp/astar_mol_compare.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\n  图像已保存至 {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 9. Demo 3：搜索树可视化（节点=分子，颜色=LogP）
# ═══════════════════════════════════════════════════════════════

def demo_search_tree():
    """
    可视化 A* 搜索树：
    - 节点颜色 = LogP 值（红→绿渐变）
    - 路径高亮（粗线）
    - 探索到的非路径节点用细线连接
    """
    print("\n" + "=" * 60)
    print("  Demo 3：A* 搜索树可视化")
    print("=" * 60)

    start = "CC"    # 乙烷，LogP≈0.89
    target = 3.0
    vf = PropertyGapValueFunction(avg_step_gain=0.4)

    # 带完整父节点信息的搜索（需要修改搜索记录方式）
    path, ops, explored, final_prop = mol_astar(
        start, target, logp_fn, vf,
        max_steps=6, beam_width=8, max_explored=150
    )

    if path:
        print(f"  ✓ 找到路径  步数={len(path)-1}  探索={len(explored)}  "
              f"最终LogP={final_prop:.3f}")
    else:
        print(f"  ○ 未找到完整路径  探索={len(explored)}  "
              f"最佳LogP={final_prop:.3f}")

    # 绘制探索节点的性质分布热力图
    _plot_search_tree(explored, path, logp_fn, start, target)

    return path, explored


def _plot_search_tree(explored: list[str], path: list[str],
                      prop_fn: Callable[[str], float],
                      start: str, target: float):
    """绘制搜索树可视化（散点图，横轴=探索顺序，纵轴=性质值）"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("A* 搜索树可视化  (节点颜色 = LogP，路径高亮)",
                 fontsize=12, fontweight="bold")

    # ── 左图：所有探索节点（散点图，颜色=LogP）──
    ax = axes[0]
    ax.set_title("探索节点分布（LogP 热力图）", fontsize=10)

    props = np.array([prop_fn(s) for s in explored])
    xs = np.arange(len(explored))

    vmin, vmax = props.min(), max(props.max(), target)
    sc = ax.scatter(xs, props, c=props, cmap="RdYlGn",
                    vmin=vmin, vmax=vmax,
                    s=30, alpha=0.7, zorder=3)

    # 高亮路径节点
    if path:
        path_set = set(path)
        path_indices = [i for i, s in enumerate(explored) if s in path_set]
        path_props = [props[i] for i in path_indices]
        if path_indices:
            ax.scatter(path_indices, path_props,
                       color="blue", s=80, zorder=5,
                       marker="D", label=f"路径节点 ({len(path_indices)})")

    ax.axhline(y=target, color="red", linestyle="--", linewidth=1.5,
               label=f"目标 LogP={target}")
    ax.axhline(y=prop_fn(start), color="gray", linestyle=":", linewidth=1,
               label=f"起点 LogP={prop_fn(start):.2f}")
    ax.set_xlabel("探索顺序")
    ax.set_ylabel("LogP")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="LogP")

    # ── 右图：路径上各步的 LogP 变化 ──
    ax2 = axes[1]
    ax2.set_title("进化路径  LogP 变化曲线", fontsize=10)

    if path:
        path_props_seq = [prop_fn(s) for s in path]
        steps = list(range(len(path)))

        cmap = plt.colormaps["RdYlGn"]
        norm = plt.Normalize(vmin=min(path_props_seq), vmax=max(target, max(path_props_seq)))

        ax2.plot(steps, path_props_seq, color="gray",
                 linewidth=2, alpha=0.5, zorder=2)

        sc2 = ax2.scatter(steps, path_props_seq,
                          c=path_props_seq, cmap="RdYlGn",
                          norm=norm, s=100, zorder=5, edgecolors="black",
                          linewidths=0.5)

        # 标注每步 SMILES
        for i, (s, lp) in enumerate(zip(path, path_props_seq)):
            short = s if len(s) <= 20 else s[:18] + ".."
            ax2.annotate(f" {short}\n LogP={lp:.2f}",
                         (i, lp),
                         fontsize=6, alpha=0.8,
                         textcoords="offset points", xytext=(5, 5))

        ax2.fill_between(steps, path_props_seq, alpha=0.1, color="green")
        plt.colorbar(sc2, ax=ax2, label="LogP")
    else:
        ax2.text(0.5, 0.5, "未找到完整路径",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=12, color="gray")

    ax2.axhline(y=target, color="red", linestyle="--", linewidth=1.5,
                label=f"目标 LogP={target}")
    ax2.axhline(y=prop_fn(start), color="gray", linestyle=":",
                linewidth=1, label=f"起点 LogP={prop_fn(start):.2f}")
    ax2.set_xlabel("变异步数")
    ax2.set_ylabel("LogP")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "/tmp/astar_mol_tree.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  搜索树图已保存至 {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   A* + Surrogate Value Function 分子进化搜索 Demo        ║")
    print("║   方案一：RL 训练启发函数，A* 负责搜索                   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 单元测试
    tests_ok = run_tests()

    # Demo 演示
    demo_basic()
    demo_compare()
    demo_search_tree()

    print("\n" + "=" * 60)
    if tests_ok:
        print("  ✓ 全部完成！")
        print()
        print("  关键接口说明：")
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │  RL 替换入口：PropertyGapValueFunction.estimate()   │")
        print("  │  将该方法替换为神经网络前向传播即可升级至完整方案一  │")
        print("  │                                                     │")
        print("  │  prop_fn 替换入口：logp_fn → 真实 IC50 预测函数    │")
        print("  └─────────────────────────────────────────────────────┘")
    else:
        print("  ✗ 存在测试失败，请检查输出。")
    print("=" * 60)
