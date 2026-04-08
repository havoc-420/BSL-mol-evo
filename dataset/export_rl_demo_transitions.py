#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线样本导出脚本（A* RL Demo）

从现有 BFS/MCTS 搜索树 JSON 或 batch 结果目录中提取
(smiles_from, smiles_to, operation, property_change, accumulated_from,
 reward, future_best_gain, remaining_depth) 格式的训练样本，
用于 BC 预训练（train_bc_pretrain.py 的第一阶段）。

使用方式：
  python -m mol_evo.dataset.export_rl_demo_transitions \\
    --input-dir  mol_evo/output/evo-mo/batch_optimization_XXX \\
    --output-json mol_evo/dataset/rl_demo/bc_transitions.json \\
    --target-property lumo \\
    --direction decrease \\
    --max-depth 4 \\
    --logp-min 0.0 \\
    --logp-max 5.0

树 JSON 结构假设（与 evolution_optimizer.save_optimized_tree 一致）：
  {
    "initial_smiles": "...",
    "nodes": {
      "node_id": {
        "smiles": "...",
        "depth": int,
        "property_value": float,
        "predicted_change": float,
        "accumulated_change": float,
        "operation": { "type": "...", "params": {...} },
        "parent_id": "...",
        ...
      }
    },
    "edges": [{ "source": "...", "target": "..." }, ...]
  }
"""

from __future__ import annotations

import os
import sys
import json
import glob
import argparse
import logging
from typing import Dict, List, Optional, Tuple

# 项目根路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
sys.path.insert(0, project_root)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

from mol_evo.core.models.astar_rl.reward import RewardConfig, compute_step_reward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 树遍历工具
# ---------------------------------------------------------------------------

def _get_logp(smiles: str) -> float:
    """计算分子 logP，失败时返回 0.0。"""
    if not RDKIT_OK:
        return 0.0
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        return Descriptors.MolLogP(mol)
    except Exception:
        return 0.0


def _build_children_map(nodes: Dict, edges: List[Dict]) -> Dict[str, List[str]]:
    """构建 parent → children 的映射（用于 future_best_gain 回溯）。"""
    children: Dict[str, List[str]] = {nid: [] for nid in nodes}
    for edge in edges:
        src = edge.get("source") or edge.get("from")
        tgt = edge.get("target") or edge.get("to")
        if src in children:
            children[src].append(tgt)
    return children


def _compute_future_best_gain(
    node_id: str,
    nodes: Dict,
    children_map: Dict[str, List[str]],
    direction: str,
) -> float:
    """从某节点出发，递归找到后代中最优的累计属性改善量。

    "最优"定义：按 direction 选择 accumulated_change 最极端的叶子。
    返回值已转换为正值（越大越好）。
    """
    children = children_map.get(node_id, [])
    current_acc = nodes[node_id].get("accumulated_change", 0.0)

    if not children:
        # 叶子节点：future_best_gain = 0（从此节点出发没有更多改善）
        return 0.0

    # 递归计算所有子节点的 future_best_gain
    child_gains = []
    for child_id in children:
        child_node = nodes.get(child_id)
        if child_node is None:
            continue
        child_acc = child_node.get("accumulated_change", 0.0)
        step_gain = child_acc - current_acc  # 这一步的属性变化
        # 方向转换：decrease 时 step_gain 为负表示改善
        if direction == "decrease":
            step_gain = -step_gain
        subtree_gain = _compute_future_best_gain(child_id, nodes, children_map, direction)
        child_gains.append(step_gain + subtree_gain)

    return max(child_gains) if child_gains else 0.0


# ---------------------------------------------------------------------------
# 单树提取
# ---------------------------------------------------------------------------

def extract_transitions_from_tree(
    tree_data: Dict,
    direction: str = "decrease",
    max_depth: int = 4,
    logp_min: float = 0.0,
    logp_max: float = 5.0,
    reward_config: Optional[RewardConfig] = None,
) -> List[Dict]:
    """从单棵树中提取所有 (parent → child) 边作为训练样本。

    Parameters
    ----------
    tree_data:
        搜索树 JSON（evolution_optimizer 的输出）。
    direction:
        属性优化方向。
    max_depth:
        搜索最大深度（用于 remaining_depth 计算）。
    logp_min / logp_max:
        logP 约束范围。
    reward_config:
        奖励超参，默认使用 RewardConfig()。

    Returns
    -------
    List[Dict]
        每个元素为一条 transition 样本。
    """
    if reward_config is None:
        reward_config = RewardConfig(direction=direction)

    nodes = tree_data.get("nodes", {})
    edges = tree_data.get("edges", [])

    if not nodes or not edges:
        return []

    children_map = _build_children_map(nodes, edges)
    transitions = []

    # 预计算每个节点的 future_best_gain
    future_gains: Dict[str, float] = {}
    for nid in nodes:
        future_gains[nid] = _compute_future_best_gain(nid, nodes, children_map, direction)

    # 遍历所有边
    for edge in edges:
        src_id = edge.get("source") or edge.get("from")
        tgt_id = edge.get("target") or edge.get("to")

        src_node = nodes.get(src_id)
        tgt_node = nodes.get(tgt_id)
        if src_node is None or tgt_node is None:
            continue

        src_smiles = src_node.get("smiles", "")
        tgt_smiles = tgt_node.get("smiles", "")
        if not src_smiles or not tgt_smiles:
            continue

        operation = tgt_node.get("operation", {})
        property_change = tgt_node.get("predicted_change", 0.0)
        accumulated_from = src_node.get("accumulated_change", 0.0)

        # logP 约束判断
        tgt_logp = _get_logp(tgt_smiles)
        logp_in_range = logp_min <= tgt_logp <= logp_max

        # 计算 stagnation_count（简化：看父节点有无改善记录）
        stagnation_count = 0  # 离线数据无运行时状态，保守置 0

        # 奖励计算
        reward = compute_step_reward(
            property_change=property_change,
            logp_in_range=logp_in_range,
            stagnation_count=stagnation_count,
            config=reward_config,
        )

        src_depth = src_node.get("depth", 0)
        remaining_depth = max(max_depth - src_depth, 0)

        transitions.append({
            "smiles_from": src_smiles,
            "smiles_to": tgt_smiles,
            "operation": operation,
            "property_change": float(property_change),
            "accumulated_from": float(accumulated_from),
            "reward": float(reward),
            "future_best_gain": float(future_gains.get(tgt_id, 0.0)),
            "remaining_depth": int(remaining_depth),
            "logp_in_range": bool(logp_in_range),
            "direction": direction,
        })

    return transitions


# ---------------------------------------------------------------------------
# 批量处理
# ---------------------------------------------------------------------------

def collect_tree_jsons(input_dir: str) -> List[str]:
    """在目录中递归收集所有搜索树 JSON 文件（排除 _topK.csv 等）。"""
    patterns = [
        os.path.join(input_dir, "**", "*.json"),
        os.path.join(input_dir, "*.json"),
    ]
    files = set()
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):
            basename = os.path.basename(f)
            # 排除 batch_results.json 和 topK 文件
            if "topK" not in basename and "batch_results" not in basename:
                files.add(f)
    return sorted(files)


def export_transitions(
    input_dir: str,
    output_json: str,
    direction: str = "decrease",
    max_depth: int = 4,
    logp_min: float = 0.0,
    logp_max: float = 5.0,
    max_trees: Optional[int] = None,
) -> int:
    """从 BFS/MCTS 输出目录导出 BC 训练样本。

    Parameters
    ----------
    input_dir:
        包含搜索树 JSON 的目录（batch_optimizer 输出目录）。
    output_json:
        导出文件路径。
    max_trees:
        最多处理的树数量，None 表示全部处理。

    Returns
    -------
    int
        导出的样本总数。
    """
    tree_files = collect_tree_jsons(input_dir)
    logger.info(f"找到 {len(tree_files)} 个树文件 in {input_dir}")

    if max_trees is not None:
        tree_files = tree_files[:max_trees]
        logger.info(f"截断至 {len(tree_files)} 个")

    reward_config = RewardConfig(direction=direction)
    all_transitions: List[Dict] = []
    skipped = 0

    for i, fpath in enumerate(tree_files):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                tree_data = json.load(f)

            transitions = extract_transitions_from_tree(
                tree_data,
                direction=direction,
                max_depth=max_depth,
                logp_min=logp_min,
                logp_max=logp_max,
                reward_config=reward_config,
            )
            all_transitions.extend(transitions)

            if (i + 1) % 20 == 0:
                logger.info(f"进度 {i+1}/{len(tree_files)}，已提取 {len(all_transitions)} 条样本")

        except Exception as e:
            logger.warning(f"跳过文件 {fpath}：{e}")
            skipped += 1

    logger.info(f"共提取 {len(all_transitions)} 条样本，跳过 {skipped} 个文件")

    # 保存
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_transitions, f, indent=2, ensure_ascii=False)
    logger.info(f"样本已保存 → {output_json}")

    return len(all_transitions)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 BFS/MCTS 搜索树导出 RL Demo BC 预训练样本"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="包含搜索树 JSON 的目录（batch_optimizer 输出目录）",
    )
    parser.add_argument(
        "--output-json", type=str,
        default="mol_evo/dataset/rl_demo/bc_transitions.json",
        help="导出样本 JSON 文件路径",
    )
    parser.add_argument("--direction", type=str, choices=["increase", "decrease"],
                        default="decrease", help="属性优化方向")
    parser.add_argument("--max-depth", type=int, default=4, help="搜索最大深度")
    parser.add_argument("--logp-min", type=float, default=0.0)
    parser.add_argument("--logp-max", type=float, default=5.0)
    parser.add_argument("--max-trees", type=int, default=None,
                        help="最多处理的树数量，不指定则全部")
    parser.add_argument("--target-property", type=str, default="lumo",
                        help="目标属性（仅用于日志信息）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n = export_transitions(
        input_dir=args.input_dir,
        output_json=args.output_json,
        direction=args.direction,
        max_depth=args.max_depth,
        logp_min=args.logp_min,
        logp_max=args.logp_max,
        max_trees=args.max_trees,
    )
    print(f"导出完成：{n} 条样本 → {args.output_json}")


if __name__ == "__main__":
    main()
