#!/usr/bin/env python3
"""
build_fragment_op_dataset.py
============================
Data construction pipeline for Phase B of mol-ofo.

Reads existing pair / path data and outputs fragment_op training datasets:

    fragment_pairs_train.jsonl
    fragment_pairs_valid.jsonl
    fragment_pairs_test.jsonl
    fragment_paths_train.jsonl
    fragment_dataset_stats.json
    fragment_action_config.yaml

Usage
-----
    python build_fragment_op_dataset.py \\
        --pairs_input  data/pairs/qm9_pairs.jsonl \\
        --paths_input  data/paths/qm9_paths.json \\
        --output_dir   data/fragment_dataset/ \\
        --split        0.8 0.1 0.1 \\
        --max_workers  4

The script can be run in --pairs_only or --paths_only mode.

Input formats
-------------
Pairs file  (JSONL, one record per line):
    {"smiles_from": "...", "smiles_to": "...", "target_property": "...",
     "step_delta": 0.3, ...}

Paths file  (JSON list or JSONL):
    {"path_id": "...", "node_smiles_list": ["s0","s1",...], "path_target": 0.5,
     "step_targets": [0.1, 0.2, ...], "target_property": "...", ...}

Output: single-step pair record
    {"smiles_from": "...", "smiles_to": "...", "fragment_op": {...},
     "target_property": "...", "step_target": 0.3, "meta": {...}}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import random
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml

# Project path setup
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from dataset.fragment_alignment_utils import (
    interpret_pair,
    validate_fragment_op,
    ACTIONLIB_VERSION,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def iter_jsonl(path: str) -> Iterator[dict]:
    """Yield dicts from a JSONL file (one JSON object per line)."""
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d in %s: %s", lineno, path, exc)


def load_json_or_jsonl(path: str) -> List[dict]:
    """Load either a JSON array file or a JSONL file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as fh:
        first_char = fh.read(1)

    if first_char == "[":
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    else:
        return list(iter_jsonl(path))


def write_jsonl(records: List[dict], path: str) -> None:
    """Write a list of dicts to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records -> %s", len(records), path)


# ---------------------------------------------------------------------------
# Worker: interpret one pair
# ---------------------------------------------------------------------------

def _interpret_one_pair(record: dict) -> dict:
    """
    Worker function: interpret a single pair record.
    Returns a result dict with all needed fields for downstream categorisation.

    Input record expected keys: smiles_from, smiles_to, [target_property],
    [step_delta / step_target], [meta / ...]
    """
    smiles_from = record.get("smiles_from") or record.get("mol_from") or ""
    smiles_to = record.get("smiles_to") or record.get("mol_to") or ""
    target_property = record.get("target_property", "unknown")
    step_target = record.get("step_target", record.get("step_delta", None))

    result = interpret_pair(smiles_from, smiles_to)

    output = {
        "smiles_from": smiles_from,
        "smiles_to": smiles_to,
        "fragment_op": result["fragment_op"],
        "target_property": target_property,
        "step_target": step_target,
        "meta": {
            "diff_type": result["diff_type"],
            "confidence": result["confidence"],
            "status": result["status"],
            "error": result["error"],
            "actionlib_version": ACTIONLIB_VERSION,
        },
    }

    # Copy extra fields from input (e.g. mol_id, dataset_split)
    for k in ("mol_id", "pair_id", "source_dataset"):
        if k in record:
            output["meta"][k] = record[k]

    return output


# ---------------------------------------------------------------------------
# Pair processing pipeline
# ---------------------------------------------------------------------------

def process_pairs(
    input_path: str,
    max_workers: int = 4,
    max_records: Optional[int] = None,
) -> Tuple[List[dict], dict]:
    """
    Process a pairs file and return:
        (processed_records, stats)

    Stats keys: total, ok, approximate, complex, invalid, by_op_type, by_diff_type
    """
    records = load_json_or_jsonl(input_path)
    if max_records:
        records = records[:max_records]

    logger.info("Processing %d pair records from %s ...", len(records), input_path)

    processed = []
    stats: dict = {
        "total": len(records),
        "ok": 0,
        "approximate": 0,
        "complex": 0,
        "invalid": 0,
        "by_op_type": defaultdict(int),
        "by_diff_type": defaultdict(int),
        "validation_fail": 0,
    }

    t0 = time.time()

    if max_workers <= 1:
        for rec in records:
            out = _interpret_one_pair(rec)
            _tally(out, stats, processed)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_interpret_one_pair, rec): rec for rec in records}
            for future in as_completed(futures):
                try:
                    out = future.result()
                    _tally(out, stats, processed)
                except Exception as exc:
                    logger.warning("Worker error: %s", exc)
                    stats["invalid"] += 1

    elapsed = time.time() - t0
    stats["elapsed_sec"] = round(elapsed, 2)
    stats["by_op_type"] = dict(stats["by_op_type"])
    stats["by_diff_type"] = dict(stats["by_diff_type"])
    logger.info(
        "Done: %d ok, %d approximate, %d complex, %d invalid in %.1fs",
        stats["ok"], stats["approximate"], stats["complex"], stats["invalid"], elapsed,
    )
    return processed, stats


def _tally(out: dict, stats: dict, processed: list) -> None:
    """Accumulate statistics and append valid records."""
    status = out["meta"]["status"]
    diff_type = out["meta"]["diff_type"]

    stats[status] = stats.get(status, 0) + 1
    stats["by_diff_type"][diff_type] = stats["by_diff_type"].get(diff_type, 0) + 1

    fop = out.get("fragment_op")
    if fop is not None:
        op_type = fop.get("op_type", "unknown")
        stats["by_op_type"][op_type] = stats["by_op_type"].get(op_type, 0) + 1

        # Validate
        is_valid, errs = validate_fragment_op(fop)
        if not is_valid:
            logger.debug("Validation failed for record: %s", errs)
            stats["validation_fail"] += 1
            out["meta"]["validation_errors"] = errs

    if status in ("ok", "approximate"):
        processed.append(out)


# ---------------------------------------------------------------------------
# Path processing pipeline
# ---------------------------------------------------------------------------

def _path_record_to_pair_records(path_rec: dict) -> List[dict]:
    """
    Convert a path record (with node_smiles_list) to a list of single-step
    pair records, one per consecutive pair (s_i, s_{i+1}).
    """
    node_smiles = path_rec.get("node_smiles_list", [])
    step_targets = path_rec.get("step_targets", [])
    target_property = path_rec.get("target_property", "unknown")
    path_id = path_rec.get("path_id", "")
    path_target = path_rec.get("path_target", None)

    if len(node_smiles) < 2:
        return []

    pairs = []
    for i in range(len(node_smiles) - 1):
        st = step_targets[i] if i < len(step_targets) else None
        pairs.append({
            "smiles_from": node_smiles[i],
            "smiles_to": node_smiles[i + 1],
            "target_property": target_property,
            "step_target": st,
            "meta_path_id": path_id,
            "meta_step_idx": i,
            "meta_path_length": len(node_smiles) - 1,
            "meta_path_target": path_target,
        })
    return pairs


def process_paths(
    input_path: str,
    max_workers: int = 4,
    max_records: Optional[int] = None,
) -> Tuple[List[dict], dict]:
    """
    Process a paths file.
    Returns (path_records_with_fragment_ops, stats).

    Each output record has shape:
    {
        "path_id": ...,
        "start_smiles": ...,
        "end_smiles": ...,
        "path_target": ...,
        "target_property": ...,
        "steps": [
            {"smiles_from": ..., "smiles_to": ..., "fragment_op": ...,
             "step_target": ..., "meta": ...},
            ...
        ],
        "path_length": int,
        "meta": {...},
    }
    """
    raw_paths = load_json_or_jsonl(input_path)
    if max_records:
        raw_paths = raw_paths[:max_records]

    logger.info("Processing %d path records from %s ...", len(raw_paths), input_path)

    output_paths = []
    stats: dict = {
        "total_paths": len(raw_paths),
        "total_steps": 0,
        "ok_steps": 0,
        "complex_steps": 0,
        "invalid_steps": 0,
        "by_op_type": defaultdict(int),
    }

    for path_rec in raw_paths:
        pair_records = _path_record_to_pair_records(path_rec)
        if not pair_records:
            continue

        steps = []
        for pr in pair_records:
            stats["total_steps"] += 1
            out = _interpret_one_pair(pr)
            status = out["meta"]["status"]
            if status in ("ok", "approximate"):
                stats["ok_steps"] += 1
                fop = out.get("fragment_op")
                if fop:
                    stats["by_op_type"][fop.get("op_type", "unknown")] += 1
            elif status == "complex":
                stats["complex_steps"] += 1
            else:
                stats["invalid_steps"] += 1

            step_entry = {
                "smiles_from": out["smiles_from"],
                "smiles_to": out["smiles_to"],
                "fragment_op": out["fragment_op"],
                "step_target": pr.get("step_target"),
                "step_idx": pr.get("meta_step_idx"),
                "meta": out["meta"],
            }
            steps.append(step_entry)

        node_smiles = path_rec.get("node_smiles_list", [])
        output_paths.append({
            "path_id": path_rec.get("path_id", ""),
            "start_smiles": node_smiles[0] if node_smiles else "",
            "end_smiles": node_smiles[-1] if node_smiles else "",
            "path_target": path_rec.get("path_target"),
            "target_property": path_rec.get("target_property", "unknown"),
            "steps": steps,
            "path_length": len(steps),
            "meta": {
                "actionlib_version": ACTIONLIB_VERSION,
                "n_ok_steps": sum(1 for s in steps if s["meta"]["status"] in ("ok", "approximate")),
            },
        })

    stats["by_op_type"] = dict(stats["by_op_type"])
    logger.info(
        "Paths: %d records, %d steps total, %d ok, %d complex, %d invalid",
        len(output_paths), stats["total_steps"], stats["ok_steps"],
        stats["complex_steps"], stats["invalid_steps"],
    )
    return output_paths, stats


# ---------------------------------------------------------------------------
# Train/valid/test split
# ---------------------------------------------------------------------------

def split_records(
    records: List[dict],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Randomly split records into train/valid/test."""
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    train = shuffled[:n_train]
    valid = shuffled[n_train: n_train + n_valid]
    test = shuffled[n_train + n_valid:]
    return train, valid, test


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def compute_delta_stats(records: List[dict]) -> dict:
    """Compute size/ring/charge delta distributions from processed pair records."""
    try:
        from rdkit import Chem

        size_deltas = []
        ring_count_deltas = []
        charge_deltas = []

        for rec in records:
            fop = rec.get("fragment_op") or {}
            constraints = fop.get("constraints") or {}
            smf = rec.get("smiles_from", "")
            smt = rec.get("smiles_to", "")
            mol_f = Chem.MolFromSmiles(smf) if smf else None
            mol_t = Chem.MolFromSmiles(smt) if smt else None
            if mol_f and mol_t:
                size_deltas.append(mol_t.GetNumHeavyAtoms() - mol_f.GetNumHeavyAtoms())
                ri_f = mol_f.GetRingInfo().NumRings()
                ri_t = mol_t.GetRingInfo().NumRings()
                ring_count_deltas.append(ri_t - ri_f)
                c_f = sum(a.GetFormalCharge() for a in mol_f.GetAtoms())
                c_t = sum(a.GetFormalCharge() for a in mol_t.GetAtoms())
                charge_deltas.append(c_t - c_f)

        def _hist(vals: list) -> dict:
            from collections import Counter
            if not vals:
                return {}
            return {str(k): v for k, v in sorted(Counter(vals).items())}

        return {
            "size_delta_histogram": _hist(size_deltas),
            "ring_count_delta_histogram": _hist(ring_count_deltas),
            "charge_delta_histogram": _hist(charge_deltas),
            "n_samples": len(size_deltas),
        }
    except Exception as exc:
        logger.warning("compute_delta_stats failed: %s", exc)
        return {}


def build_fragment_dataset_stats(
    pair_stats: dict,
    path_stats: dict,
    train_records: List[dict],
    valid_records: List[dict],
    test_records: List[dict],
) -> dict:
    """Aggregate all statistics into a single fragment_dataset_stats.json payload."""
    delta_stats = compute_delta_stats(train_records)

    scaffold_preserving_count = sum(
        1 for r in train_records
        if (r.get("fragment_op") or {}).get("constraints", {}).get("scaffold_preserving", False)
    )

    return {
        "actionlib_version": ACTIONLIB_VERSION,
        "pair_processing": pair_stats,
        "path_processing": path_stats,
        "split_sizes": {
            "train": len(train_records),
            "valid": len(valid_records),
            "test": len(test_records),
        },
        "scaffold_preserving_ratio": round(
            scaffold_preserving_count / max(1, len(train_records)), 4
        ),
        "delta_stats": delta_stats,
    }


# ---------------------------------------------------------------------------
# Action config YAML
# ---------------------------------------------------------------------------

def build_action_config() -> dict:
    """Return the fragment_action_config dict to be saved as YAML."""
    return {
        "actionlib_version": ACTIONLIB_VERSION,
        "op_types": [
            "attach_fragment",
            "replace_substituent",
            "grow_r_group",
            "bioisostere_swap",
            "delete_fragment",
        ],
        "scaffold_preserving_rule": "murcko_v1",
        "scaffold_preserving_description": (
            "v1: True if canonical Murcko scaffold SMILES of mol_from equals "
            "canonical Murcko scaffold SMILES of mol_to."
        ),
        "mcs_params": {
            "timeout_sec": 5,
            "complete_rings_only": True,
            "bond_compare": "CompareOrderExact",
            "atom_compare": "CompareElements",
            "min_coverage": 0.5,
        },
        "status_labels": {
            "ok": "Clean single-step fragment action, high MCS coverage.",
            "approximate": "Fragment action extracted but may cover multi-change.",
            "complex": "MCS coverage below threshold; action not extracted.",
            "invalid": "Invalid SMILES or identical molecules.",
        },
        "data_sources": {
            "data_mcs": "Extracted from (smiles_from, smiles_to) pair via MCS alignment.",
            "rule_brics": "Generated by BRICS fragmentation rules.",
            "rule_rgroup": "Generated by R-group enumeration rules.",
            "template": "Instantiated from a pre-defined template action.",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    pair_stats: dict = {}
    path_stats: dict = {}
    processed_pairs: List[dict] = []
    processed_paths: List[dict] = []

    # ---- Pair processing ----
    if args.pairs_input and not args.paths_only:
        if not os.path.isfile(args.pairs_input):
            logger.error("Pairs input not found: %s", args.pairs_input)
            sys.exit(1)
        processed_pairs, pair_stats = process_pairs(
            args.pairs_input,
            max_workers=args.max_workers,
            max_records=args.max_records,
        )
        train, valid, test = split_records(
            processed_pairs,
            ratios=tuple(args.split),
            seed=args.seed,
        )
        write_jsonl(train, os.path.join(args.output_dir, "fragment_pairs_train.jsonl"))
        write_jsonl(valid, os.path.join(args.output_dir, "fragment_pairs_valid.jsonl"))
        write_jsonl(test, os.path.join(args.output_dir, "fragment_pairs_test.jsonl"))
    else:
        train, valid, test = [], [], []

    # ---- Path processing ----
    if args.paths_input and not args.pairs_only:
        if not os.path.isfile(args.paths_input):
            logger.error("Paths input not found: %s", args.paths_input)
            sys.exit(1)
        processed_paths, path_stats = process_paths(
            args.paths_input,
            max_workers=args.max_workers,
            max_records=args.max_records,
        )
        write_jsonl(
            processed_paths,
            os.path.join(args.output_dir, "fragment_paths_train.jsonl"),
        )

    # ---- Statistics ----
    stats = build_fragment_dataset_stats(pair_stats, path_stats, train, valid, test)
    stats_path = os.path.join(args.output_dir, "fragment_dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    logger.info("Stats -> %s", stats_path)

    # ---- Action config ----
    cfg = build_action_config()
    cfg_path = os.path.join(args.output_dir, "fragment_action_config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info("Action config -> %s", cfg_path)

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("Phase B dataset build complete.")
    logger.info("Output dir : %s", args.output_dir)
    if pair_stats:
        logger.info("Pairs      : total=%d ok=%d approx=%d complex=%d invalid=%d",
                    pair_stats.get("total", 0), pair_stats.get("ok", 0),
                    pair_stats.get("approximate", 0), pair_stats.get("complex", 0),
                    pair_stats.get("invalid", 0))
    if path_stats:
        logger.info("Paths      : total=%d steps=%d ok_steps=%d",
                    path_stats.get("total_paths", 0), path_stats.get("total_steps", 0),
                    path_stats.get("ok_steps", 0))
    logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build fragment_op dataset from existing pair/path data (Phase B)."
    )
    p.add_argument(
        "--pairs_input", type=str, default=None,
        help="Path to input pairs file (JSONL or JSON array). "
             "Each record needs at least smiles_from, smiles_to.",
    )
    p.add_argument(
        "--paths_input", type=str, default=None,
        help="Path to input paths file (JSONL or JSON array). "
             "Each record needs node_smiles_list.",
    )
    p.add_argument(
        "--output_dir", type=str, default="data/fragment_dataset",
        help="Directory to write output files.",
    )
    p.add_argument(
        "--split", type=float, nargs=3, default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VALID", "TEST"),
        help="Train/valid/test split ratios (must sum to 1.0). Default: 0.8 0.1 0.1",
    )
    p.add_argument(
        "--max_workers", type=int, default=4,
        help="Number of parallel workers for pair processing. Default: 4",
    )
    p.add_argument(
        "--max_records", type=int, default=None,
        help="Limit processing to the first N records (useful for quick tests).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/valid/test split.",
    )
    p.add_argument(
        "--pairs_only", action="store_true",
        help="Skip path processing even if --paths_input is given.",
    )
    p.add_argument(
        "--paths_only", action="store_true",
        help="Skip pair processing even if --pairs_input is given.",
    )
    p.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    total = sum(args.split)
    if abs(total - 1.0) > 1e-6:
        logger.error("--split ratios must sum to 1.0, got %.4f", total)
        sys.exit(1)

    main(args)
