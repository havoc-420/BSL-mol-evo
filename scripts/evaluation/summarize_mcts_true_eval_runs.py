#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 MCTS ablation 的真值评估结果（论文口径）。

读取 `eval_summary.tsv`，自动汇总每个 run unit 的：
- paper_avg_improvement（将 evaluate_batch/evaluate_csv 的负号口径翻回正向改善）
- paper_success_rate（best-of-topK 是否改善）
- morgan_similarity_best_mean
- intdiv_best（优先读统计文件，否则基于 best_results 重新计算）
- avg_runtime / avg_expanded_nodes（从搜索产物补齐）

并输出：
1. `true_eval_summary.tsv`：逐 run 结果
2. `true_eval_summary_aggregated.tsv`：按 task × variant 聚合后的 mean/std
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
HAMDIV_DIR = WORKSPACE_ROOT / "utils" / "HamDiv"
if HAMDIV_DIR.exists():
    sys.path.append(str(HAMDIV_DIR))

try:
    from diversity import diversity_all  # type: ignore
except Exception:  # pragma: no cover - 缺依赖时退化为不计算 best diversity
    diversity_all = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 MCTS ablation 的真值评估结果（论文口径）")
    parser.add_argument("--run-root", type=str, required=True, help="ablation 输出根目录，内部应包含 eval_summary.tsv")
    parser.add_argument("--eval-summary", type=str, default=None, help="显式指定 eval_summary.tsv 路径")
    parser.add_argument("--output", type=str, default=None, help="输出 TSV 路径；默认写到 run-root/true_eval_summary.tsv")
    return parser.parse_args()


def safe_mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return mean(values) if values else None


def safe_std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) <= 1:
        return 0.0
    return pstdev(values)


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


def parse_eval_summary(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"未找到 eval_summary.tsv: {path}")
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def latest_matching_file(base_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(base_dir.glob(pattern))
    return matches[-1] if matches else None


def resolve_best_results_csv(row: Dict[str, str]) -> Optional[Path]:
    csv_eval_dir = Path(row.get("csv_eval_dir", "")) if row.get("csv_eval_dir") else None
    if csv_eval_dir and csv_eval_dir.exists():
        best_csv = latest_matching_file(csv_eval_dir, "best_results_*.csv")
        if best_csv:
            return best_csv

    batch_eval_csv = Path(row.get("batch_eval_csv", "")) if row.get("batch_eval_csv") else None
    if batch_eval_csv and batch_eval_csv.exists():
        parent = batch_eval_csv.parent
        eval_dirs = sorted(parent.glob("eval_*"))
        for eval_dir in reversed(eval_dirs):
            best_csv = latest_matching_file(eval_dir, "best_results_*.csv")
            if best_csv:
                return best_csv
    return None


def resolve_stats_json(row: Dict[str, str]) -> Optional[Path]:
    csv_eval_dir = Path(row.get("csv_eval_dir", "")) if row.get("csv_eval_dir") else None
    if csv_eval_dir and csv_eval_dir.exists():
        stats_json = latest_matching_file(csv_eval_dir, "statistics_summary_*.json")
        if stats_json:
            return stats_json

    batch_eval_csv = Path(row.get("batch_eval_csv", "")) if row.get("batch_eval_csv") else None
    if batch_eval_csv and batch_eval_csv.exists():
        parent = batch_eval_csv.parent
        eval_dirs = sorted(parent.glob("eval_*"))
        for eval_dir in reversed(eval_dirs):
            stats_json = latest_matching_file(eval_dir, "statistics_summary_*.json")
            if stats_json:
                return stats_json
    return None


def compute_best_diversity(best_smiles: List[str]) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {
        "richness_best": None,
        "intdiv_best": None,
        "circles_best": None,
        "hamdiv_best": None,
    }
    if not best_smiles or diversity_all is None:
        return result

    metric_map = {
        "richness_best": "Richness",
        "intdiv_best": "IntDiv",
        "circles_best": "NCircles-0.7",
        "hamdiv_best": "HamDiv",
    }
    for key, mode in metric_map.items():
        try:
            value = diversity_all(smiles=best_smiles, mode=mode)
            result[key] = float(value) if value is not None else None
        except Exception:
            result[key] = None
    return result


def extract_search_metrics(search_dir: Path) -> Dict[str, Optional[float]]:
    batch_json = search_dir / "batch_results.json"
    if not batch_json.exists():
        return {
            "search_start_count": None,
            "avg_runtime": None,
            "avg_expanded_nodes": None,
            "search_success_rate": None,
        }

    with batch_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    runtimes: List[float] = []
    expanded_nodes: List[float] = []
    success_count = 0
    total_count = 0

    for item in payload.values():
        result = item.get("optimization_result", {})
        total_count += 1
        if result.get("status") != "success":
            continue
        success_count += 1

        runtime = result.get("runtime")
        if runtime is not None:
            runtimes.append(float(runtime))

        mcts_stats = result.get("optimized_result", {}).get("mcts_stats", {})
        expanded = mcts_stats.get("unique_states_expanded")
        if expanded is not None:
            expanded_nodes.append(float(expanded))

    return {
        "search_start_count": total_count,
        "avg_runtime": safe_mean(runtimes),
        "avg_expanded_nodes": safe_mean(expanded_nodes),
        "search_success_rate": (success_count / total_count) if total_count else None,
    }


def extract_true_eval_metrics(best_csv: Path, stats_json: Optional[Path]) -> Dict[str, Optional[float]]:
    with best_csv.open("r", encoding="utf-8") as f:
        best_rows = list(csv.DictReader(f))

    paper_improvements: List[float] = []
    morgan_sims: List[float] = []
    ged_values: List[float] = []
    ged_sims: List[float] = []
    best_smiles: List[str] = []

    for row in best_rows:
        improvement = row.get("improvement", "")
        if improvement not in (None, ""):
            raw_improvement = float(improvement)
            # evaluate_batch_mo / evaluate_csv_results 当前约定：改善为负；论文口径统一翻为正。
            paper_improvements.append(-raw_improvement)

        morgan = row.get("morgan_similarity", "")
        if morgan not in (None, ""):
            morgan_sims.append(float(morgan))

        ged = row.get("ged", "")
        if ged not in (None, ""):
            ged_values.append(float(ged))

        ged_similarity = row.get("ged_similarity", "")
        if ged_similarity not in (None, ""):
            ged_sims.append(float(ged_similarity))

        smi = row.get("best_opt_smiles", "")
        if smi:
            best_smiles.append(smi)

    stats: Dict[str, object] = {}
    if stats_json and stats_json.exists():
        with stats_json.open("r", encoding="utf-8") as f:
            stats = json.load(f)

    best_diversity = compute_best_diversity(best_smiles)

    return {
        "start_count": len(best_rows),
        "total_optimized_molecules": stats.get("total_optimized_molecules"),
        "paper_avg_improvement": safe_mean(paper_improvements),
        "paper_success_rate": (sum(1 for x in paper_improvements if x > 0) / len(paper_improvements)) if paper_improvements else None,
        "morgan_similarity_best_mean": safe_mean(morgan_sims),
        "ged_best_mean": safe_mean(ged_values),
        "ged_similarity_best_mean": safe_mean(ged_sims),
        "average_start_property": stats.get("average_start_lumo", stats.get("average_start_homo")),
        "average_optimized_property": stats.get("average_optimized_lumo", stats.get("average_optimized_homo")),
        "intdiv_avg": stats.get("intdiv_avg"),
        "richness_best": stats.get("richness_best", best_diversity["richness_best"]),
        "intdiv_best": stats.get("intdiv_best", best_diversity["intdiv_best"]),
        "circles_best": stats.get("circles_best", best_diversity["circles_best"]),
        "hamdiv_best": stats.get("hamdiv_best", best_diversity["hamdiv_best"]),
    }


def aggregate_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["task"]), str(row["variant"]))
        grouped.setdefault(key, []).append(row)

    numeric_fields = [
        "start_count",
        "paper_avg_improvement",
        "paper_success_rate",
        "morgan_similarity_best_mean",
        "ged_best_mean",
        "ged_similarity_best_mean",
        "intdiv_best",
        "avg_runtime",
        "avg_expanded_nodes",
    ]

    agg_rows: List[Dict[str, object]] = []
    for (task, variant), items in sorted(grouped.items()):
        agg_row: Dict[str, object] = {
            "task": task,
            "variant": variant,
            "seed_count": len(items),
        }
        for field in numeric_fields:
            values = [float(x[field]) for x in items if x.get(field) not in (None, "")]
            agg_row[f"{field}_mean"] = safe_mean(values)
            agg_row[f"{field}_std"] = safe_std(values)
        agg_rows.append(agg_row)
    return agg_rows


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    eval_summary_path = Path(args.eval_summary).resolve() if args.eval_summary else run_root / "eval_summary.tsv"
    output_path = Path(args.output).resolve() if args.output else run_root / "true_eval_summary.tsv"
    agg_output_path = output_path.with_name(output_path.stem + "_aggregated.tsv")

    summary_rows = parse_eval_summary(eval_summary_path)
    run_rows: List[Dict[str, object]] = []

    for row in summary_rows:
        if row.get("status") != "success":
            continue
        search_dir = Path(row.get("search_dir", "")) if row.get("search_dir") else None
        if not search_dir or not search_dir.exists():
            continue

        best_csv = resolve_best_results_csv(row)
        if not best_csv or not best_csv.exists():
            continue

        stats_json = resolve_stats_json(row)
        true_eval_metrics = extract_true_eval_metrics(best_csv, stats_json)
        search_metrics = extract_search_metrics(search_dir)

        run_rows.append(
            {
                "task": row["task"],
                "variant": row["variant"],
                "seed": row["seed"],
                "profile": row.get("profile", ""),
                "item_size": row.get("item_size", ""),
                "start_count": true_eval_metrics["start_count"],
                "total_optimized_molecules": true_eval_metrics["total_optimized_molecules"],
                "paper_avg_improvement": true_eval_metrics["paper_avg_improvement"],
                "paper_success_rate": true_eval_metrics["paper_success_rate"],
                "morgan_similarity_best_mean": true_eval_metrics["morgan_similarity_best_mean"],
                "ged_best_mean": true_eval_metrics["ged_best_mean"],
                "ged_similarity_best_mean": true_eval_metrics["ged_similarity_best_mean"],
                "intdiv_avg": true_eval_metrics["intdiv_avg"],
                "intdiv_best": true_eval_metrics["intdiv_best"],
                "richness_best": true_eval_metrics["richness_best"],
                "circles_best": true_eval_metrics["circles_best"],
                "hamdiv_best": true_eval_metrics["hamdiv_best"],
                "avg_runtime": search_metrics["avg_runtime"],
                "avg_expanded_nodes": search_metrics["avg_expanded_nodes"],
                "search_success_rate": search_metrics["search_success_rate"],
                "search_dir": str(search_dir),
                "csv_eval_dir": row.get("csv_eval_dir", ""),
                "best_results_csv": str(best_csv),
                "stats_json": str(stats_json) if stats_json else "",
            }
        )

    aggregated_rows = aggregate_rows(run_rows)
    write_tsv(output_path, run_rows)
    write_tsv(agg_output_path, aggregated_rows)

    print(f"已写出逐 run 真值汇总: {output_path}")
    print(f"已写出聚合真值汇总: {agg_output_path}")


if __name__ == "__main__":
    main()
