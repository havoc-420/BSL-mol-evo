#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 MCTS ablation 运行结果（预测口径）。

从 `run_summary.tsv` 或 ablation 输出根目录中读取 `batch_results.json`，
计算每个 run unit 的预测口径指标：
- avg_improvement
- success_rate
- avg_runtime
- avg_expanded_nodes
- avg_best_property

说明：
- 这里使用的是搜索输出中的 **模型预测值**，不是真值评估结果；
- 适合 smoke / pilot 阶段快速确认趋势与链路是否正常；
- 正式论文结果仍建议补齐真值评估链路。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="汇总 MCTS ablation 运行结果（预测口径）")
    p.add_argument("--run-root", type=str, required=True, help="ablation 输出根目录，内部应包含 run_summary.tsv")
    p.add_argument("--output", type=str, default=None, help="输出 TSV 路径；默认写到 run-root/predicted_summary.tsv")
    return p.parse_args()


def safe_mean(values: List[float]) -> Optional[float]:
    return mean(values) if values else None


def safe_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return pstdev(values)


def parse_run_summary(run_root: Path) -> List[Dict[str, str]]:
    summary_path = run_root / "run_summary.tsv"
    if not summary_path.exists():
        raise FileNotFoundError(f"未找到 run_summary.tsv: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def choose_best_property(initial_value: float, topk_results: List[dict], direction: str) -> Optional[float]:
    values = [item.get("property_value") for item in topk_results if item.get("property_value") is not None]
    if not values:
        return None
    if direction == "increase":
        return max(values)
    return min(values)


def extract_metrics_from_batch_json(batch_json: Path, direction: str) -> Dict[str, Optional[float]]:
    with batch_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    improvements: List[float] = []
    runtimes: List[float] = []
    expanded_nodes: List[float] = []
    best_properties: List[float] = []

    for _, item in payload.items():
        result = item.get("optimization_result", {})
        if result.get("status") != "success":
            continue

        initial_value = result.get("initial_property")
        topk = result.get("topk_results", {}).get("topK_results", [])
        best_property = choose_best_property(initial_value, topk, direction) if initial_value is not None else None
        if initial_value is not None and best_property is not None:
            if direction == "increase":
                improvements.append(best_property - initial_value)
            else:
                improvements.append(initial_value - best_property)
            best_properties.append(best_property)

        runtime = result.get("runtime")
        if runtime is not None:
            runtimes.append(float(runtime))

        mcts_stats = result.get("optimized_result", {}).get("mcts_stats", {})
        unique_states = mcts_stats.get("unique_states_expanded")
        if unique_states is not None:
            expanded_nodes.append(float(unique_states))

    success_rate = None
    if improvements:
        success_rate = sum(1 for x in improvements if x > 0) / len(improvements)

    return {
        "molecule_count": len(improvements),
        "avg_improvement": safe_mean(improvements),
        "success_rate": success_rate,
        "avg_runtime": safe_mean(runtimes),
        "avg_expanded_nodes": safe_mean(expanded_nodes),
        "avg_best_property": safe_mean(best_properties),
    }


def aggregate_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["task"]), str(row["variant"]))
        grouped.setdefault(key, []).append(row)

    agg_rows: List[Dict[str, object]] = []
    for (task, variant), items in sorted(grouped.items()):
        improvements = [float(x["avg_improvement"]) for x in items if x.get("avg_improvement") is not None]
        success_rates = [float(x["success_rate"]) for x in items if x.get("success_rate") is not None]
        runtimes = [float(x["avg_runtime"]) for x in items if x.get("avg_runtime") is not None]
        expanded = [float(x["avg_expanded_nodes"]) for x in items if x.get("avg_expanded_nodes") is not None]

        agg_rows.append(
            {
                "task": task,
                "variant": variant,
                "seed_count": len(items),
                "avg_improvement_mean": safe_mean(improvements),
                "avg_improvement_std": safe_std(improvements),
                "success_rate_mean": safe_mean(success_rates),
                "success_rate_std": safe_std(success_rates),
                "avg_runtime_mean": safe_mean(runtimes),
                "avg_runtime_std": safe_std(runtimes),
                "avg_expanded_nodes_mean": safe_mean(expanded),
                "avg_expanded_nodes_std": safe_std(expanded),
            }
        )
    return agg_rows


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(v) for k, v in row.items()})


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    output_path = Path(args.output).resolve() if args.output else run_root / "predicted_summary.tsv"
    agg_output_path = output_path.with_name(output_path.stem + "_aggregated.tsv")

    summary_rows = parse_run_summary(run_root)
    run_rows: List[Dict[str, object]] = []

    for row in summary_rows:
        batch_json = row.get("batch_json", "")
        status = row.get("status", "")
        if status != "success" or not batch_json:
            continue
        batch_json_path = Path(batch_json)
        if not batch_json_path.exists():
            continue

        metrics = extract_metrics_from_batch_json(batch_json_path, direction="increase" if row["task"].endswith("_up") else "decrease")
        run_rows.append(
            {
                "task": row["task"],
                "variant": row["variant"],
                "seed": row["seed"],
                "molecule_count": metrics["molecule_count"],
                "avg_improvement": metrics["avg_improvement"],
                "success_rate": metrics["success_rate"],
                "avg_runtime": metrics["avg_runtime"],
                "avg_expanded_nodes": metrics["avg_expanded_nodes"],
                "avg_best_property": metrics["avg_best_property"],
                "search_dir": row["search_dir"],
            }
        )

    aggregated_rows = aggregate_rows(run_rows)
    write_tsv(output_path, run_rows)
    write_tsv(agg_output_path, aggregated_rows)

    print(f"已写出逐 run 汇总: {output_path}")
    print(f"已写出聚合汇总: {agg_output_path}")


if __name__ == "__main__":
    main()
