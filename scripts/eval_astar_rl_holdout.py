#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行或汇总 A* RL holdout 评估结果。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行或汇总 astar_demo 的 holdout 评估")
    parser.add_argument("--eval-json", type=str, default=None, help="已有评估 JSON；提供时可只做汇总")
    parser.add_argument("--input-csv", type=str, default=None, help="holdout CSV；在需要实际运行评估时必填")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--label", type=str, default="holdout_eval", help="本次汇总的标签")
    parser.add_argument("--reference-json", type=str, default=None, help="参考评估 JSON，用于计算配对 win-rate / delta")
    parser.add_argument("--trim-count", type=int, default=3, help="trimmed mean 两端裁掉的样本数")
    parser.add_argument("--target-property", type=str, default="lumo")
    parser.add_argument("--optimization-mode", type=str, default="sub", choices=["sub", "pct"])
    parser.add_argument("--direction", type=str, default="decrease", choices=["increase", "decrease"])
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-branching", type=int, default=8)
    parser.add_argument("--open-set-budget", type=int, default=50)
    parser.add_argument("--top-n-prefilter", type=int, default=50)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--config-file", type=str, default=None)
    parser.add_argument("--policy-path", type=str, default=None)
    parser.add_argument("--value-path", type=str, default=None)
    parser.add_argument("--topk", type=int, default=20)
    return parser.parse_args()


def compute_improvement(item: Dict, direction: str) -> Tuple[float, int, float, float]:
    opt = (item or {}).get("optimization_result") or {}
    topk_block = opt.get("topk_results") or {}
    topk = topk_block.get("topK_results") or []
    init = topk_block.get("initial_property_value")

    improvement = 0.0
    nonempty = 0
    if topk and init is not None:
        top1_value = float(topk[0]["property_value"])
        init_value = float(init)
        improvement = init_value - top1_value if direction == "decrease" else top1_value - init_value
        nonempty = 1

    stats = (opt.get("optimized_result") or {}).get("astar_stats") or {}
    actual_expansions = float(stats.get("actual_expansions", 0.0))
    ofo_calls = float(stats.get("ofo_scored_candidates", 0.0))
    return improvement, nonempty, actual_expansions, ofo_calls


def safe_mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def safe_median(values: List[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def trimmed_mean(values: List[float], trim_count: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if trim_count <= 0 or len(sorted_values) <= 2 * trim_count:
        return safe_mean(sorted_values)
    trimmed = sorted_values[trim_count:-trim_count]
    return safe_mean(trimmed)


def summarize_eval_json(eval_json: Path, direction: str, trim_count: int) -> Dict:
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    rows = []
    per_smiles = {}
    for smiles, item in data.items():
        improvement, nonempty, actual_expansions, ofo_calls = compute_improvement(item, direction)
        row = {
            "smiles": smiles,
            "improvement": improvement,
            "nonempty_topk": nonempty,
            "actual_expansions": actual_expansions,
            "ofo_calls": ofo_calls,
        }
        rows.append(row)
        per_smiles[smiles] = row

    improvements = [r["improvement"] for r in rows]
    actual_expansions = [r["actual_expansions"] for r in rows]
    ofo_calls = [r["ofo_calls"] for r in rows]

    return {
        "mols": len(rows),
        "nonempty_topk": int(sum(r["nonempty_topk"] for r in rows)),
        "top1_mean": safe_mean(improvements),
        "top1_median": safe_median(improvements),
        "trimmed_mean": trimmed_mean(improvements, trim_count),
        "actual_expansions_mean": safe_mean(actual_expansions),
        "ofo_calls_mean": safe_mean(ofo_calls),
        "per_smiles": per_smiles,
    }


def summarize_against_reference(summary: Dict, reference_summary: Dict, trim_count: int) -> Dict:
    shared_smiles = sorted(set(summary["per_smiles"]) & set(reference_summary["per_smiles"]))
    deltas = []
    wins = 0
    losses = 0
    ties = 0

    for smiles in shared_smiles:
        delta = (
            summary["per_smiles"][smiles]["improvement"]
            - reference_summary["per_smiles"][smiles]["improvement"]
        )
        deltas.append(delta)
        if delta > 1e-12:
            wins += 1
        elif delta < -1e-12:
            losses += 1
        else:
            ties += 1

    return {
        "shared_mols": len(shared_smiles),
        "win_count": wins,
        "loss_count": losses,
        "tie_count": ties,
        "win_rate": (wins / len(shared_smiles)) if shared_smiles else 0.0,
        "delta_mean": safe_mean(deltas),
        "delta_median": safe_median(deltas),
        "trimmed_delta_mean": trimmed_mean(deltas, trim_count),
    }


def maybe_run_batch_optimizer(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.eval_json:
        return Path(args.eval_json).resolve()

    required = {
        "input_csv": args.input_csv,
        "model_path": args.model_path,
        "model_dir": args.model_dir,
        "config_file": args.config_file,
        "policy_path": args.policy_path,
        "value_path": args.value_path,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"缺少运行评估所需参数: {', '.join(missing)}")

    project_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_json = output_dir / f"{args.label}_{timestamp}.json"
    cmd = [
        sys.executable,
        "-m",
        "mol_evo.scripts.batch_optimizer",
        "--input-csv",
        str(Path(args.input_csv).resolve()),
        "--output-json",
        str(eval_json),
        "--model-path",
        str(Path(args.model_path).resolve()),
        "--model-dir",
        str(Path(args.model_dir).resolve()),
        "--config-file",
        str(Path(args.config_file).resolve()),
        "--target-property",
        args.target_property,
        "--optimization-mode",
        args.optimization_mode,
        "--search-mode",
        "astar_demo",
        "--rl-eval",
        "--policy-path",
        str(Path(args.policy_path).resolve()),
        "--value-path",
        str(Path(args.value_path).resolve()),
        "--direction",
        args.direction,
        "--max-depth",
        str(args.max_depth),
        "--max-branching",
        str(args.max_branching),
        "--open-set-budget",
        str(args.open_set_budget),
        "--top-n-prefilter",
        str(args.top_n_prefilter),
        "--topK",
        str(args.topk),
    ]

    command_txt = output_dir / f"{args.label}_command.txt"
    command_txt.write_text(" ".join(cmd), encoding="utf-8")
    subprocess.run(cmd, cwd=str(project_root), check=True)
    return eval_json


def write_outputs(output_dir: Path, label: str, summary_payload: Dict) -> None:
    summary_json = output_dir / f"{label}_summary.json"
    summary_tsv = output_dir / f"{label}_summary.tsv"
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    headers = [
        "label",
        "mols",
        "nonempty_topk",
        "top1_mean",
        "top1_median",
        "trimmed_mean",
        "actual_expansions_mean",
        "ofo_calls_mean",
        "shared_mols",
        "win_count",
        "loss_count",
        "tie_count",
        "win_rate",
        "delta_mean",
        "delta_median",
        "trimmed_delta_mean",
        "eval_json",
        "reference_json",
    ]

    comparison = summary_payload.get("comparison_vs_reference") or {}
    values = {
        "label": label,
        "mols": summary_payload["summary"]["mols"],
        "nonempty_topk": summary_payload["summary"]["nonempty_topk"],
        "top1_mean": f"{summary_payload['summary']['top1_mean']:.6f}",
        "top1_median": f"{summary_payload['summary']['top1_median']:.6f}",
        "trimmed_mean": f"{summary_payload['summary']['trimmed_mean']:.6f}",
        "actual_expansions_mean": f"{summary_payload['summary']['actual_expansions_mean']:.6f}",
        "ofo_calls_mean": f"{summary_payload['summary']['ofo_calls_mean']:.6f}",
        "shared_mols": comparison.get("shared_mols", ""),
        "win_count": comparison.get("win_count", ""),
        "loss_count": comparison.get("loss_count", ""),
        "tie_count": comparison.get("tie_count", ""),
        "win_rate": f"{comparison['win_rate']:.6f}" if "win_rate" in comparison else "",
        "delta_mean": f"{comparison['delta_mean']:.6f}" if "delta_mean" in comparison else "",
        "delta_median": f"{comparison['delta_median']:.6f}" if "delta_median" in comparison else "",
        "trimmed_delta_mean": f"{comparison['trimmed_delta_mean']:.6f}" if "trimmed_delta_mean" in comparison else "",
        "eval_json": summary_payload["eval_json"],
        "reference_json": summary_payload.get("reference_json", ""),
    }
    summary_tsv.write_text(
        "\t".join(headers) + "\n" + "\t".join(str(values[h]) for h in headers) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_json = maybe_run_batch_optimizer(args, output_dir)
    summary = summarize_eval_json(eval_json, args.direction, args.trim_count)
    comparison = None
    reference_json = None

    if args.reference_json:
        reference_json = str(Path(args.reference_json).resolve())
        reference_summary = summarize_eval_json(Path(reference_json), args.direction, args.trim_count)
        comparison = summarize_against_reference(summary, reference_summary, args.trim_count)

    payload = {
        "label": args.label,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_json": str(Path(eval_json).resolve()),
        "reference_json": reference_json,
        "trim_count": args.trim_count,
        "summary": {k: v for k, v in summary.items() if k != "per_smiles"},
        "comparison_vs_reference": comparison,
    }
    write_outputs(output_dir, args.label, payload)

    print(f"label={args.label}")
    print(f"eval_json={Path(eval_json).resolve()}")
    print(f"summary_json={output_dir / f'{args.label}_summary.json'}")
    print(f"summary_tsv={output_dir / f'{args.label}_summary.tsv'}")
    print(f"mols={payload['summary']['mols']}")
    print(f"nonempty_topk={payload['summary']['nonempty_topk']}")
    print(f"top1_mean={payload['summary']['top1_mean']:.6f}")
    print(f"top1_median={payload['summary']['top1_median']:.6f}")
    print(f"trimmed_mean={payload['summary']['trimmed_mean']:.6f}")
    if comparison is not None:
        print(f"win_rate={comparison['win_rate']:.6f}")
        print(f"delta_median={comparison['delta_median']:.6f}")
        print(f"trimmed_delta_mean={comparison['trimmed_delta_mean']:.6f}")


if __name__ == "__main__":
    main()
