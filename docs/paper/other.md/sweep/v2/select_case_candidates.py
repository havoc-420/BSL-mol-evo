from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import dataclass
from typing import Any

BASE_DIR = "/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/paper/ablations/20260413_1536_pilot_mcts"
TASKS = ("homo_down", "lumo_up")
COMPARE_VARIANT = "random_topb"
TARGET_VARIANT = "full"


@dataclass
class EvalRow:
    initial_smiles: str
    group_id: str
    start_value: float
    best_opt_value: float
    gain: float
    best_opt_smiles: str
    morgan_similarity: float
    ged: float
    ged_similarity: float


@dataclass
class PathStats:
    json_path: str
    depth: int
    pred_gain: float
    nonstereo_steps: int
    stereo_steps: int
    unique_smiles: int
    repeated_smiles: int
    monotonicity: float
    operations: list[str]


@dataclass
class Candidate:
    task: str
    initial_smiles: str
    full: EvalRow
    baseline: EvalRow
    path: PathStats
    score: float
    advantage_vs_baseline: float
    readability: float


def iter_best_result_csvs(task: str, variant: str) -> list[str]:
    pattern = os.path.join(
        BASE_DIR,
        task,
        variant,
        "seed42",
        "search",
        ".evaluation_results_*",
        "*",
        "eval_*",
        "best_results_*.csv",
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No best_results CSV found for {task}/{variant}: {pattern}")
    return paths


def load_eval_rows(task: str, variant: str) -> dict[str, EvalRow]:
    path = iter_best_result_csvs(task, variant)[0]
    rows: dict[str, EvalRow] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            group_id = row["group_id"]
            initial_smiles = group_id.rsplit("_", 2)[0]
            start_value = float(row["start_value"])
            best_opt_value = float(row["best_opt_value"])
            gain = (best_opt_value - start_value) if task == "lumo_up" else (start_value - best_opt_value)
            rows[initial_smiles] = EvalRow(
                initial_smiles=initial_smiles,
                group_id=group_id,
                start_value=start_value,
                best_opt_value=best_opt_value,
                gain=gain,
                best_opt_smiles=row["best_opt_smiles"],
                morgan_similarity=float(row["morgan_similarity"]),
                ged=float(row["ged"]),
                ged_similarity=float(row["ged_similarity"]),
            )
    return rows


def load_json_paths(task: str, variant: str) -> dict[str, str]:
    out: dict[str, str] = {}
    pattern = os.path.join(BASE_DIR, task, variant, "seed42", "search", "*.json")
    for json_path in sorted(glob.glob(pattern)):
        name = os.path.basename(json_path)
        if name in {"batch_results.json", "manifest.json"}:
            continue
        with open(json_path, encoding="utf-8") as handle:
            data = json.load(handle)
        out[data["initial_smiles"]] = json_path
    return out


def extract_main_chain(data: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = data["nodes"]
    expected_ids = [str(i) for i in range(len(nodes))]
    if all(node_id in nodes for node_id in expected_ids):
        return [nodes[node_id] for node_id in expected_ids]

    tip = max(nodes.values(), key=lambda node: int(node["depth"]))
    reverse_chain = []
    current = tip
    while current is not None:
        reverse_chain.append(current)
        parent_id = current.get("parent_id")
        current = nodes.get(parent_id) if parent_id is not None else None
    return list(reversed(reverse_chain))


def compute_monotonicity(task: str, deltas: list[float]) -> float:
    if task == "lumo_up":
        aligned = [delta for delta in deltas if delta > 1e-9]
    else:
        aligned = [delta for delta in deltas if delta < -1e-9]
    nonzero = [delta for delta in deltas if abs(delta) > 1e-9]
    return len(aligned) / len(nonzero) if nonzero else 0.0


def analyze_path(task: str, json_path: str) -> PathStats:
    with open(json_path, encoding="utf-8") as handle:
        data = json.load(handle)

    chain = extract_main_chain(data)
    smiles_list = [node["smiles"] for node in chain]
    operations = [node.get("operation") or "start" for node in chain[1:]]
    deltas = [float(node.get("property_change", 0.0)) for node in chain[1:]]

    return PathStats(
        json_path=json_path,
        depth=len(chain) - 1,
        pred_gain=float(chain[-1]["accumulated_change"]),
        nonstereo_steps=sum(1 for op in operations if op != "add_stereo"),
        stereo_steps=sum(1 for op in operations if op == "add_stereo"),
        unique_smiles=len(dict.fromkeys(smiles_list)),
        repeated_smiles=sum(1 for left, right in zip(smiles_list, smiles_list[1:]) if left == right),
        monotonicity=compute_monotonicity(task, deltas),
        operations=operations,
    )


def compute_readability(path: PathStats) -> float:
    return (
        1.5 * path.nonstereo_steps
        + 0.6 * path.unique_smiles
        + 2.0 * path.monotonicity
        - 1.2 * path.stereo_steps
        - 1.0 * path.repeated_smiles
    )


def collect_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for task in TASKS:
        full_rows = load_eval_rows(task, TARGET_VARIANT)
        baseline_rows = load_eval_rows(task, COMPARE_VARIANT)
        full_json_paths = load_json_paths(task, TARGET_VARIANT)

        shared_initials = sorted(set(full_rows) & set(baseline_rows) & set(full_json_paths))
        for initial_smiles in shared_initials:
            full = full_rows[initial_smiles]
            baseline = baseline_rows[initial_smiles]
            path = analyze_path(task, full_json_paths[initial_smiles])
            readability = compute_readability(path)
            advantage = full.gain - baseline.gain
            score = 2.5 * advantage + 1.0 * full.gain + 0.4 * readability
            candidates.append(
                Candidate(
                    task=task,
                    initial_smiles=initial_smiles,
                    full=full,
                    baseline=baseline,
                    path=path,
                    score=score,
                    advantage_vs_baseline=advantage,
                    readability=readability,
                )
            )
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def print_report(candidates: list[Candidate], top_k: int = 8) -> None:
    by_task: dict[str, list[Candidate]] = {task: [] for task in TASKS}
    for candidate in candidates:
        by_task[candidate.task].append(candidate)

    for task in TASKS:
        print(f"\n=== {task} top {top_k} candidates ===")
        for candidate in by_task[task][:top_k]:
            print(
                json.dumps(
                    {
                        "initial_smiles": candidate.initial_smiles,
                        "score": round(candidate.score, 3),
                        "full_true_gain": round(candidate.full.gain, 3),
                        "random_true_gain": round(candidate.baseline.gain, 3),
                        "advantage_vs_random": round(candidate.advantage_vs_baseline, 3),
                        "pred_gain": round(candidate.path.pred_gain, 3),
                        "readability": round(candidate.readability, 3),
                        "nonstereo_steps": candidate.path.nonstereo_steps,
                        "stereo_steps": candidate.path.stereo_steps,
                        "best_opt_smiles": candidate.full.best_opt_smiles,
                        "operations": candidate.path.operations,
                        "json_path": candidate.path.json_path,
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    print_report(collect_candidates())
