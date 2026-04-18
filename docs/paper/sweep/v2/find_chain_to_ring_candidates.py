from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

ABLATIONS_DIR = "/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/paper/ablations"
FIGURES_ROOT = "/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/paper/figures/case_paths"
METADATA_DIR = os.path.join(FIGURES_ROOT, "metadata")
OUTPUT_JSON = os.path.join(METADATA_DIR, "chain_to_ring_candidates.json")
OUTPUT_TSV = os.path.join(METADATA_DIR, "chain_to_ring_candidates.tsv")
TASKS = ("lumo_up", "homo_down")


@dataclass
class EvalRow:
    gain: float
    best_opt_smiles: str


@dataclass
class Candidate:
    run: str
    task: str
    initial_smiles: str
    best_opt_smiles: str
    full_gain: float
    random_gain: float | None
    advantage_vs_random: float | None
    path_len: int
    first_ring_step: int
    last_chain_step: int
    stable_after_first_ring: bool
    pre_ring_nonstereo: int
    stereo_steps: int
    form_ring_steps: int
    open_ring_after_first: int
    final_pred_gain: float
    delayed_ring_score: float
    final_only_display_score: float
    selected_nodes: list[int]
    path_node_ids: list[int]
    operations: list[str]
    path_smiles: list[str]
    json_path: str


def has_ring(smiles: str) -> bool:
    return any(ch.isdigit() for ch in smiles)


def load_eval_rows(run_dir: str, task: str, variant: str) -> dict[str, EvalRow]:
    base = os.path.join(run_dir, task, variant, "seed42", "search")
    pattern = os.path.join(base, ".evaluation_results_*", "**", "best_results_*.csv")
    csv_paths = sorted(glob.glob(pattern, recursive=True))
    if not csv_paths:
        return {}

    rows: dict[str, EvalRow] = {}
    with open(csv_paths[0], newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            initial_smiles = row["group_id"].rsplit("_", 2)[0]
            start_value = float(row["start_value"])
            best_value = float(row["best_opt_value"])
            gain = (best_value - start_value) if task == "lumo_up" else (start_value - best_value)
            rows[initial_smiles] = EvalRow(gain=gain, best_opt_smiles=row["best_opt_smiles"])
    return rows


def extract_best_path(data: dict[str, Any], target_smiles: str, task: str) -> list[tuple[int, dict[str, Any]]]:
    nodes = data["nodes"]
    matches: list[tuple[int, dict[str, Any]]] = []
    for node_id_str, node in nodes.items():
        if node["smiles"] == target_smiles:
            matches.append((int(node_id_str), node))

    if matches:
        key_fn = lambda pair: float(pair[1].get("accumulated_change", 0.0))
        target_id, target_node = max(matches, key=key_fn) if task == "lumo_up" else min(matches, key=key_fn)
    else:
        all_nodes = [(int(node_id_str), node) for node_id_str, node in nodes.items()]
        key_fn = lambda pair: float(pair[1].get("accumulated_change", 0.0))
        target_id, target_node = max(all_nodes, key=key_fn) if task == "lumo_up" else min(all_nodes, key=key_fn)

    reverse_path: list[tuple[int, dict[str, Any]]] = [(target_id, target_node)]
    current = target_node
    while current.get("parent_id") is not None:
        parent_id = int(current["parent_id"])
        parent = nodes[str(parent_id)]
        reverse_path.append((parent_id, parent))
        current = parent
    return list(reversed(reverse_path))


def pick_selected_nodes(path_node_ids: list[int], ring_flags: list[bool]) -> list[int]:
    last_chain_index = max(index for index, flag in enumerate(ring_flags) if not flag)
    selected_indexes: list[int] = [0]

    if last_chain_index >= 1:
        early_indexes = [1, 2]
        for index in early_indexes:
            if index <= last_chain_index:
                selected_indexes.append(index)

        if last_chain_index >= 4:
            mid_index = last_chain_index // 2
            selected_indexes.append(mid_index)

        selected_indexes.append(last_chain_index)

    selected_indexes.append(len(path_node_ids) - 1)
    selected_indexes = sorted(dict.fromkeys(selected_indexes))
    return [path_node_ids[index] for index in selected_indexes]


def delayed_ring_score(
    first_ring_step: int,
    stable_after_first_ring: bool,
    pre_ring_nonstereo: int,
    advantage_vs_random: float | None,
    full_gain: float,
    stereo_steps: int,
    final_pred_gain: float,
) -> float:
    pred_penalty = 1.2 if abs(final_pred_gain) > 50 else 0.0
    advantage = advantage_vs_random or 0.0
    return (
        1.8 * first_ring_step
        + 1.2 * pre_ring_nonstereo
        + 1.0 * full_gain
        + 0.8 * advantage
        + (2.0 if stable_after_first_ring else -1.5)
        - 0.5 * stereo_steps
        - pred_penalty
    )


def final_only_display_score(
    last_chain_step: int,
    advantage_vs_random: float | None,
    full_gain: float,
    stereo_steps: int,
    open_ring_after_first: int,
) -> float:
    advantage = advantage_vs_random or 0.0
    return (
        2.0 * last_chain_step
        + 1.2 * advantage
        + 0.8 * full_gain
        - 0.4 * stereo_steps
        - 0.8 * open_ring_after_first
    )


def collect_candidates() -> list[Candidate]:
    out: list[Candidate] = []
    run_dirs = sorted(glob.glob(os.path.join(ABLATIONS_DIR, "20*")))

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        for task in TASKS:
            full_rows = load_eval_rows(run_dir, task, "full")
            random_rows = load_eval_rows(run_dir, task, "random_topb")
            search_dir = os.path.join(run_dir, task, "full", "seed42", "search")
            if not os.path.isdir(search_dir):
                continue

            for json_path in sorted(glob.glob(os.path.join(search_dir, "*.json"))):
                name = os.path.basename(json_path)
                if name in {"batch_results.json", "manifest.json"}:
                    continue

                with open(json_path, encoding="utf-8") as handle:
                    data = json.load(handle)

                initial_smiles = data["initial_smiles"]
                full_row = full_rows.get(initial_smiles)
                if full_row is None:
                    continue

                if has_ring(initial_smiles) or not has_ring(full_row.best_opt_smiles):
                    continue

                best_path = extract_best_path(data, full_row.best_opt_smiles, task)
                path_node_ids = [node_id for node_id, _ in best_path]
                path_nodes = [node for _, node in best_path]
                path_smiles = [node["smiles"] for node in path_nodes]
                ring_flags = [has_ring(smiles) for smiles in path_smiles]
                if not any(ring_flags[1:]):
                    continue

                first_ring_step = next(index for index, flag in enumerate(ring_flags[1:], start=1) if flag)
                last_chain_step = max(index for index, flag in enumerate(ring_flags) if not flag)
                operations = [node.get("operation") or "start" for node in path_nodes[1:]]
                open_ring_after_first = sum(
                    1
                    for index, operation in enumerate(operations, start=1)
                    if operation == "remove_form_ring" and index >= first_ring_step
                )
                stable_after_first_ring = (
                    open_ring_after_first == 0
                    and all(ring_flags[index] for index in range(first_ring_step, len(ring_flags)))
                )
                pre_ring_nonstereo = sum(
                    1 for operation in operations[: first_ring_step - 1] if operation != "add_stereo"
                )
                stereo_steps = sum(1 for operation in operations if operation == "add_stereo")
                form_ring_steps = sum(1 for operation in operations if operation == "form_ring")
                final_pred_gain = float(path_nodes[-1].get("accumulated_change", 0.0))
                random_gain = random_rows.get(initial_smiles).gain if initial_smiles in random_rows else None
                advantage_vs_random = (
                    full_row.gain - random_gain if random_gain is not None else None
                )

                out.append(
                    Candidate(
                        run=run_name,
                        task=task,
                        initial_smiles=initial_smiles,
                        best_opt_smiles=full_row.best_opt_smiles,
                        full_gain=full_row.gain,
                        random_gain=random_gain,
                        advantage_vs_random=advantage_vs_random,
                        path_len=len(path_nodes) - 1,
                        first_ring_step=first_ring_step,
                        last_chain_step=last_chain_step,
                        stable_after_first_ring=stable_after_first_ring,
                        pre_ring_nonstereo=pre_ring_nonstereo,
                        stereo_steps=stereo_steps,
                        form_ring_steps=form_ring_steps,
                        open_ring_after_first=open_ring_after_first,
                        final_pred_gain=final_pred_gain,
                        delayed_ring_score=delayed_ring_score(
                            first_ring_step=first_ring_step,
                            stable_after_first_ring=stable_after_first_ring,
                            pre_ring_nonstereo=pre_ring_nonstereo,
                            advantage_vs_random=advantage_vs_random,
                            full_gain=full_row.gain,
                            stereo_steps=stereo_steps,
                            final_pred_gain=final_pred_gain,
                        ),
                        final_only_display_score=final_only_display_score(
                            last_chain_step=last_chain_step,
                            advantage_vs_random=advantage_vs_random,
                            full_gain=full_row.gain,
                            stereo_steps=stereo_steps,
                            open_ring_after_first=open_ring_after_first,
                        ),
                        selected_nodes=pick_selected_nodes(path_node_ids, ring_flags),
                        path_node_ids=path_node_ids,
                        operations=operations,
                        path_smiles=path_smiles,
                        json_path=json_path,
                    )
                )

    return out


def write_outputs(candidates: list[Candidate]) -> None:
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    ranked = sorted(candidates, key=lambda item: item.final_only_display_score, reverse=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump([asdict(candidate) for candidate in ranked], handle, indent=2, ensure_ascii=False)

    with open(OUTPUT_TSV, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "rank",
                "run",
                "task",
                "initial_smiles",
                "best_opt_smiles",
                "full_gain",
                "random_gain",
                "advantage_vs_random",
                "path_len",
                "first_ring_step",
                "last_chain_step",
                "stable_after_first_ring",
                "pre_ring_nonstereo",
                "stereo_steps",
                "form_ring_steps",
                "open_ring_after_first",
                "final_pred_gain",
                "delayed_ring_score",
                "final_only_display_score",
                "selected_nodes",
                "json_path",
            ]
        )
        for rank, candidate in enumerate(ranked, start=1):
            writer.writerow(
                [
                    rank,
                    candidate.run,
                    candidate.task,
                    candidate.initial_smiles,
                    candidate.best_opt_smiles,
                    f"{candidate.full_gain:.6f}",
                    "" if candidate.random_gain is None else f"{candidate.random_gain:.6f}",
                    "" if candidate.advantage_vs_random is None else f"{candidate.advantage_vs_random:.6f}",
                    candidate.path_len,
                    candidate.first_ring_step,
                    candidate.last_chain_step,
                    int(candidate.stable_after_first_ring),
                    candidate.pre_ring_nonstereo,
                    candidate.stereo_steps,
                    candidate.form_ring_steps,
                    candidate.open_ring_after_first,
                    f"{candidate.final_pred_gain:.6f}",
                    f"{candidate.delayed_ring_score:.6f}",
                    f"{candidate.final_only_display_score:.6f}",
                    json.dumps(candidate.selected_nodes, ensure_ascii=False),
                    candidate.json_path,
                ]
            )


def print_shortlists(candidates: list[Candidate]) -> None:
    clean = [c for c in candidates if c.advantage_vs_random is None or c.advantage_vs_random > -0.2]
    terminal = sorted(clean, key=lambda item: item.final_only_display_score, reverse=True)
    delayed = sorted(clean, key=lambda item: item.delayed_ring_score, reverse=True)

    print("=== shortlist: best final-only display options ===")
    for candidate in terminal[:12]:
        print(
            json.dumps(
                {
                    "run": candidate.run,
                    "task": candidate.task,
                    "initial": candidate.initial_smiles,
                    "best": candidate.best_opt_smiles,
                    "full_gain": round(candidate.full_gain, 3),
                    "random_gain": None if candidate.random_gain is None else round(candidate.random_gain, 3),
                    "adv": None if candidate.advantage_vs_random is None else round(candidate.advantage_vs_random, 3),
                    "first_ring": candidate.first_ring_step,
                    "last_chain": candidate.last_chain_step,
                    "stable": candidate.stable_after_first_ring,
                    "selected_nodes": candidate.selected_nodes,
                    "ops": candidate.operations,
                    "json_path": candidate.json_path,
                },
                ensure_ascii=False,
            )
        )

    print("=== shortlist: genuinely delayed-ring paths ===")
    for candidate in delayed[:12]:
        print(
            json.dumps(
                {
                    "run": candidate.run,
                    "task": candidate.task,
                    "initial": candidate.initial_smiles,
                    "best": candidate.best_opt_smiles,
                    "full_gain": round(candidate.full_gain, 3),
                    "random_gain": None if candidate.random_gain is None else round(candidate.random_gain, 3),
                    "adv": None if candidate.advantage_vs_random is None else round(candidate.advantage_vs_random, 3),
                    "first_ring": candidate.first_ring_step,
                    "last_chain": candidate.last_chain_step,
                    "stable": candidate.stable_after_first_ring,
                    "pre_ring_nonstereo": candidate.pre_ring_nonstereo,
                    "selected_nodes": candidate.selected_nodes,
                    "ops": candidate.operations,
                    "json_path": candidate.json_path,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    candidates = collect_candidates()
    write_outputs(candidates)
    print_shortlists(candidates)
