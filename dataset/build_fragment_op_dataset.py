#!/usr/bin/env python3
"""
build_fragment_op_dataset.py
============================
Data construction pipeline for Phase B / Phase C bootstrap of mol-ofo.

Reads existing pair / path data and exports dual-view datasets:

    semantic_pairs_train.jsonl
    semantic_pairs_valid.jsonl
    semantic_pairs_test.jsonl
    fragment_pairs_train.jsonl
    fragment_pairs_valid.jsonl
    fragment_pairs_test.jsonl
    semantic_paths_train.jsonl
    semantic_paths_valid.jsonl
    semantic_paths_test.jsonl
    fragment_dataset_stats.json
    fragment_action_config.yaml

The script keeps the existing fragment_op extraction logic, but upgrades the
output protocol to include `semantic_step`, `semantic_level`,
`annotation_status`, and preserved primitive traces whenever available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterator, List, Optional, Tuple

import yaml

# Project path setup
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from dataset.fragment_alignment_utils import (  # noqa: E402
    ACTIONLIB_VERSION,
    build_semantic_step,
    interpret_pair,
    validate_fragment_op,
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
    return list(iter_jsonl(path))



def write_jsonl(records: List[dict], path: str) -> None:
    """Write a list of dicts to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records -> %s", len(records), path)


# ---------------------------------------------------------------------------
# Primitive-op normalisation helpers
# ---------------------------------------------------------------------------

def _normalize_primitive_op(op: dict) -> dict:
    if not isinstance(op, dict):
        return {"operation": "unknown", "position": "", "atom": ""}

    normalized = dict(op)
    normalized["operation"] = op.get("operation", op.get("type", "unknown"))
    normalized["position"] = str(op.get("position", op.get("atom_idx", "")))
    normalized.setdefault("atom", op.get("to_atom_symbol", ""))
    return normalized



def _extract_primitive_ops(record: dict) -> List[dict]:
    if isinstance(record.get("primitive_ops"), list):
        return [_normalize_primitive_op(op) for op in record.get("primitive_ops", [])]

    if isinstance(record.get("operations"), list) and record.get("operations"):
        return [_normalize_primitive_op(op) for op in record.get("operations", [])]

    fallback_fields = (
        record.get("operation_type")
        or record.get("to_atom_symbol")
        or record.get("atom_idx")
        or record.get("position")
    )
    if fallback_fields:
        return [{
            "operation": record.get("operation_type", "unknown"),
            "position": str(record.get("position", record.get("atom_idx", ""))),
            "atom": record.get("to_atom_symbol", record.get("atom", "")),
        }]

    return []



def _infer_primitive_span(record: dict, primitive_ops: List[dict]) -> Optional[List[int]]:
    primitive_span = record.get("primitive_span")
    if isinstance(primitive_span, list) and len(primitive_span) == 2:
        try:
            return [int(primitive_span[0]), int(primitive_span[1])]
        except (TypeError, ValueError):
            pass

    if record.get("meta_step_idx") is not None:
        try:
            idx = int(record["meta_step_idx"])
            return [idx, idx]
        except (TypeError, ValueError):
            return None

    if primitive_ops:
        return [0, max(0, len(primitive_ops) - 1)]

    return None



def _make_semantic_step_id(record: dict, smiles_from: str, smiles_to: str) -> str:
    if record.get("pair_id"):
        return f"pair:{record['pair_id']}"
    if record.get("meta_path_id") and record.get("meta_step_idx") is not None:
        return f"path:{record['meta_path_id']}:step:{record['meta_step_idx']}"
    raw = f"{smiles_from}>>{smiles_to}"
    return f"pair:auto:{hashlib.sha1(raw.encode()).hexdigest()[:12]}"



def _source_record_id(record: dict, semantic_step_id: str) -> str:
    return (
        str(record.get("pair_id") or "")
        or str(record.get("meta_path_id") or "")
        or semantic_step_id
    )


# ---------------------------------------------------------------------------
# Worker: interpret one pair
# ---------------------------------------------------------------------------

def _interpret_one_pair(record: dict) -> dict:
    """
    Interpret a single pair record and upgrade it to the semantic-step protocol.
    """
    smiles_from = record.get("smiles_from") or record.get("mol_from") or ""
    smiles_to = record.get("smiles_to") or record.get("mol_to") or ""
    target_property = record.get("target_property", "unknown")
    step_target = record.get("step_target", record.get("step_delta", None))

    primitive_ops = _extract_primitive_ops(record)
    primitive_span = _infer_primitive_span(record, primitive_ops)
    semantic_step_id = _make_semantic_step_id(record, smiles_from, smiles_to)

    result = interpret_pair(smiles_from, smiles_to)
    semantic_step = build_semantic_step(
        result,
        semantic_step_id=semantic_step_id,
        primitive_ops=primitive_ops,
        primitive_span=primitive_span,
        provenance_extra={
            "source_record_id": _source_record_id(record, semantic_step_id),
            "notes": result.get("diff_type"),
        },
    )

    output = {
        "smiles_from": smiles_from,
        "smiles_to": smiles_to,
        "primitive_ops": primitive_ops,
        "semantic_step": semantic_step,
        "semantic_level": semantic_step["semantic_level"],
        "fragment_op": semantic_step["fragment_op"],
        "annotation_status": semantic_step["annotation_status"],
        "annotation_confidence": semantic_step["annotation_confidence"],
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

    for k in (
        "mol_id",
        "pair_id",
        "source_dataset",
        "split_key",
        "dataset_split",
        "path_id",
        "meta_path_id",
        "meta_step_idx",
        "meta_path_length",
        "meta_path_target",
    ):
        if k in record:
            output["meta"][k] = record[k]

    if "property_changes" in record:
        output["property_changes"] = record["property_changes"]

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
    Process a pairs file and return semantic pair records.

    Returned records include fragment-resolved steps and atomic fallbacks.
    Fatal invalid pairs are excluded from the exported semantic dataset.
    """
    records = load_json_or_jsonl(input_path)
    if max_records:
        records = records[:max_records]

    logger.info("Processing %d pair records from %s ...", len(records), input_path)

    processed: List[dict] = []
    stats: dict = {
        "total": len(records),
        "ok": 0,
        "approximate": 0,
        "complex": 0,
        "invalid": 0,
        "by_op_type": defaultdict(int),
        "by_diff_type": defaultdict(int),
        "by_semantic_level": defaultdict(int),
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
    stats["by_semantic_level"] = dict(stats["by_semantic_level"])
    logger.info(
        "Done: %d semantic records exported (%d ok, %d approximate, %d complex, %d invalid) in %.1fs",
        len(processed), stats["ok"], stats["approximate"], stats["complex"], stats["invalid"], elapsed,
    )
    return processed, stats



def _tally(out: dict, stats: dict, processed: List[dict]) -> None:
    status = out["meta"]["status"]
    diff_type = out["meta"]["diff_type"]
    semantic_level = out.get("semantic_level", "unknown")

    stats[status] = stats.get(status, 0) + 1
    stats["by_diff_type"][diff_type] = stats["by_diff_type"].get(diff_type, 0) + 1
    stats["by_semantic_level"][semantic_level] = stats["by_semantic_level"].get(semantic_level, 0) + 1

    fop = out.get("fragment_op")
    if fop is not None:
        op_type = fop.get("op_type", "unknown")
        stats["by_op_type"][op_type] = stats["by_op_type"].get(op_type, 0) + 1

        is_valid, errs = validate_fragment_op(fop)
        if not is_valid:
            logger.debug("Validation failed for record: %s", errs)
            stats["validation_fail"] += 1
            out["meta"]["validation_errors"] = errs

    if status != "invalid":
        processed.append(out)


# ---------------------------------------------------------------------------
# Path processing pipeline
# ---------------------------------------------------------------------------

def _path_record_to_pair_records(path_rec: dict) -> List[dict]:
    """Expand a path record to consecutive pair records."""
    node_smiles = path_rec.get("node_smiles_list", [])
    step_targets = path_rec.get("step_targets", [])
    operations = path_rec.get("operations", [])
    target_property = path_rec.get("target_property", "unknown")
    path_id = path_rec.get("path_id", "")
    path_target = path_rec.get("path_target", None)

    if len(node_smiles) < 2:
        return []

    pairs = []
    for i in range(len(node_smiles) - 1):
        st = step_targets[i] if i < len(step_targets) else None
        primitive_op = operations[i] if i < len(operations) else None
        pairs.append({
            "smiles_from": node_smiles[i],
            "smiles_to": node_smiles[i + 1],
            "target_property": target_property,
            "step_target": st,
            "primitive_ops": [primitive_op] if isinstance(primitive_op, dict) else [],
            "primitive_span": [i, i],
            "meta_path_id": path_id,
            "meta_step_idx": i,
            "meta_path_length": len(node_smiles) - 1,
            "meta_path_target": path_target,
        })
    return pairs



def process_paths(
    input_path: str,
    max_records: Optional[int] = None,
) -> Tuple[List[dict], dict]:
    """
    Process a paths file and return semantic path records.
    """
    raw_paths = load_json_or_jsonl(input_path)
    if max_records:
        raw_paths = raw_paths[:max_records]

    logger.info("Processing %d path records from %s ...", len(raw_paths), input_path)

    output_paths = []
    stats: dict = {
        "total_paths": len(raw_paths),
        "kept_paths": 0,
        "total_steps": 0,
        "ok_steps": 0,
        "approximate_steps": 0,
        "complex_steps": 0,
        "invalid_steps": 0,
        "by_op_type": defaultdict(int),
        "by_semantic_level": defaultdict(int),
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
            semantic_level = out["semantic_level"]
            stats["by_semantic_level"][semantic_level] += 1

            if status == "ok":
                stats["ok_steps"] += 1
            elif status == "approximate":
                stats["approximate_steps"] += 1
            elif status == "complex":
                stats["complex_steps"] += 1
            else:
                stats["invalid_steps"] += 1

            fop = out.get("fragment_op")
            if fop:
                stats["by_op_type"][fop.get("op_type", "unknown")] += 1

            step_entry = {
                "smiles_from": out["smiles_from"],
                "smiles_to": out["smiles_to"],
                "primitive_ops": out["primitive_ops"],
                "semantic_step": out["semantic_step"],
                "semantic_level": out["semantic_level"],
                "fragment_op": out["fragment_op"],
                "annotation_status": out["annotation_status"],
                "annotation_confidence": out["annotation_confidence"],
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
            "node_smiles_list": node_smiles,
            "path_target": path_rec.get("path_target"),
            "step_targets": path_rec.get("step_targets"),
            "target_property": path_rec.get("target_property", "unknown"),
            "semantic_steps": [step["semantic_step"] for step in steps],
            "steps": steps,
            "path_length": len(steps),
            "meta": {
                "actionlib_version": ACTIONLIB_VERSION,
                "n_fragment_steps": sum(1 for s in steps if s["semantic_level"] == "fragment"),
                "n_unresolved_steps": sum(1 for s in steps if s["annotation_status"] == "unresolved"),
            },
        })
        stats["kept_paths"] += 1

    stats["by_op_type"] = dict(stats["by_op_type"])
    stats["by_semantic_level"] = dict(stats["by_semantic_level"])
    logger.info(
        "Paths: %d kept, %d steps total, %d ok, %d approximate, %d complex, %d invalid",
        len(output_paths),
        stats["total_steps"],
        stats["ok_steps"],
        stats["approximate_steps"],
        stats["complex_steps"],
        stats["invalid_steps"],
    )
    return output_paths, stats


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def split_records(
    records: List[dict],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
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



def filter_fragment_records(records: List[dict], approximate_min_conf: float) -> List[dict]:
    """Return the high-confidence fragment subset from semantic pair records."""
    filtered = []
    for rec in records:
        if rec.get("semantic_level") != "fragment":
            continue
        status = rec.get("annotation_status")
        confidence = rec.get("annotation_confidence") or 0.0
        if status == "resolved":
            filtered.append(rec)
        elif status == "approximate" and confidence >= approximate_min_conf:
            filtered.append(rec)
    return filtered


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
            smf = rec.get("smiles_from", "")
            smt = rec.get("smiles_to", "")
            mol_f = Chem.MolFromSmiles(smf) if smf else None
            mol_t = Chem.MolFromSmiles(smt) if smt else None
            if mol_f and mol_t:
                size_deltas.append(mol_t.GetNumHeavyAtoms() - mol_f.GetNumHeavyAtoms())
                ring_count_deltas.append(mol_t.GetRingInfo().NumRings() - mol_f.GetRingInfo().NumRings())
                charge_deltas.append(
                    sum(a.GetFormalCharge() for a in mol_t.GetAtoms())
                    - sum(a.GetFormalCharge() for a in mol_f.GetAtoms())
                )

        def _hist(vals: List[int]) -> dict:
            if not vals:
                return {}
            from collections import Counter
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
    semantic_pair_splits: Tuple[List[dict], List[dict], List[dict]],
    fragment_pair_splits: Tuple[List[dict], List[dict], List[dict]],
    semantic_path_splits: Tuple[List[dict], List[dict], List[dict]],
) -> dict:
    semantic_train, semantic_valid, semantic_test = semantic_pair_splits
    fragment_train, fragment_valid, fragment_test = fragment_pair_splits
    semantic_paths_train, semantic_paths_valid, semantic_paths_test = semantic_path_splits

    scaffold_preserving_count = sum(
        1
        for r in fragment_train
        if (r.get("fragment_op") or {}).get("constraints", {}).get("scaffold_preserving", False)
    )

    return {
        "actionlib_version": ACTIONLIB_VERSION,
        "pair_processing": pair_stats,
        "path_processing": path_stats,
        "semantic_pair_split_sizes": {
            "train": len(semantic_train),
            "valid": len(semantic_valid),
            "test": len(semantic_test),
        },
        "fragment_pair_split_sizes": {
            "train": len(fragment_train),
            "valid": len(fragment_valid),
            "test": len(fragment_test),
        },
        "semantic_path_split_sizes": {
            "train": len(semantic_paths_train),
            "valid": len(semantic_paths_valid),
            "test": len(semantic_paths_test),
        },
        "scaffold_preserving_ratio": round(
            scaffold_preserving_count / max(1, len(fragment_train)), 4
        ),
        "semantic_delta_stats": compute_delta_stats(semantic_train),
        "fragment_delta_stats": compute_delta_stats(fragment_train),
    }


# ---------------------------------------------------------------------------
# Action config YAML
# ---------------------------------------------------------------------------

def build_action_config() -> dict:
    return {
        "actionlib_version": ACTIONLIB_VERSION,
        "semantic_levels": ["fragment", "atomic_fallback"],
        "annotation_status": ["resolved", "approximate", "unresolved"],
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
            "complex": "MCS coverage below threshold; action falls back to semantic atomic/unresolved view.",
            "invalid": "Invalid SMILES or identical molecules.",
        },
        "semantic_dataset_views": {
            "semantic_pairs": "Wide single-step view containing fragment and atomic_fallback steps.",
            "fragment_pairs": "High-confidence subset filtered from semantic_pairs.",
            "semantic_paths": "Path view with semantic_steps attached to each step.",
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

    semantic_pair_splits = ([], [], [])
    fragment_pair_splits = ([], [], [])
    semantic_path_splits = ([], [], [])

    if args.pairs_input and not args.paths_only:
        if not os.path.isfile(args.pairs_input):
            logger.error("Pairs input not found: %s", args.pairs_input)
            sys.exit(1)

        processed_pairs, pair_stats = process_pairs(
            args.pairs_input,
            max_workers=args.max_workers,
            max_records=args.max_records,
        )
        semantic_pair_splits = split_records(
            processed_pairs,
            ratios=tuple(args.split),
            seed=args.seed,
        )
        semantic_train, semantic_valid, semantic_test = semantic_pair_splits
        fragment_pair_splits = (
            filter_fragment_records(semantic_train, args.approximate_min_conf),
            filter_fragment_records(semantic_valid, args.approximate_min_conf),
            filter_fragment_records(semantic_test, args.approximate_min_conf),
        )
        fragment_train, fragment_valid, fragment_test = fragment_pair_splits

        write_jsonl(semantic_train, os.path.join(args.output_dir, "semantic_pairs_train.jsonl"))
        write_jsonl(semantic_valid, os.path.join(args.output_dir, "semantic_pairs_valid.jsonl"))
        write_jsonl(semantic_test, os.path.join(args.output_dir, "semantic_pairs_test.jsonl"))
        write_jsonl(fragment_train, os.path.join(args.output_dir, "fragment_pairs_train.jsonl"))
        write_jsonl(fragment_valid, os.path.join(args.output_dir, "fragment_pairs_valid.jsonl"))
        write_jsonl(fragment_test, os.path.join(args.output_dir, "fragment_pairs_test.jsonl"))

    if args.paths_input and not args.pairs_only:
        if not os.path.isfile(args.paths_input):
            logger.error("Paths input not found: %s", args.paths_input)
            sys.exit(1)

        processed_paths, path_stats = process_paths(
            args.paths_input,
            max_records=args.max_records,
        )
        semantic_path_splits = split_records(
            processed_paths,
            ratios=tuple(args.split),
            seed=args.seed,
        )
        semantic_paths_train, semantic_paths_valid, semantic_paths_test = semantic_path_splits
        write_jsonl(semantic_paths_train, os.path.join(args.output_dir, "semantic_paths_train.jsonl"))
        write_jsonl(semantic_paths_valid, os.path.join(args.output_dir, "semantic_paths_valid.jsonl"))
        write_jsonl(semantic_paths_test, os.path.join(args.output_dir, "semantic_paths_test.jsonl"))

    stats = build_fragment_dataset_stats(
        pair_stats,
        path_stats,
        semantic_pair_splits,
        fragment_pair_splits,
        semantic_path_splits,
    )
    stats_path = os.path.join(args.output_dir, "fragment_dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    logger.info("Stats -> %s", stats_path)

    cfg = build_action_config()
    cfg_path = os.path.join(args.output_dir, "fragment_action_config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info("Action config -> %s", cfg_path)

    logger.info("=" * 60)
    logger.info("Semantic/fragment dataset build complete.")
    logger.info("Output dir : %s", args.output_dir)
    if pair_stats:
        logger.info(
            "Pairs      : total=%d ok=%d approx=%d complex=%d invalid=%d semantic_train=%d fragment_train=%d",
            pair_stats.get("total", 0),
            pair_stats.get("ok", 0),
            pair_stats.get("approximate", 0),
            pair_stats.get("complex", 0),
            pair_stats.get("invalid", 0),
            len(semantic_pair_splits[0]),
            len(fragment_pair_splits[0]),
        )
    if path_stats:
        logger.info(
            "Paths      : total=%d kept=%d steps=%d semantic_train=%d",
            path_stats.get("total_paths", 0),
            path_stats.get("kept_paths", 0),
            path_stats.get("total_steps", 0),
            len(semantic_path_splits[0]),
        )
    logger.info("=" * 60)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build semantic_step / fragment_op dataset views from existing pair/path data."
    )
    p.add_argument(
        "--pairs_input", type=str, default=None,
        help="Path to input pairs file (JSONL or JSON array). Each record needs at least smiles_from, smiles_to.",
    )
    p.add_argument(
        "--paths_input", type=str, default=None,
        help="Path to input paths file (JSONL or JSON array). Each record needs node_smiles_list.",
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
        "--approximate_min_conf", type=float, default=0.8,
        help="Minimum confidence required for an approximate semantic step to enter fragment_pairs. Default: 0.8",
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
