#!/usr/bin/env python3
"""
export_training_views.py
========================
Export train/valid/test semantic and fragment views from annotated semantic
records.

This script is the export-stage entrypoint corresponding to the
`03-semantic-data-and-step-views-plan.md` workflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Project path setup
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from dataset.build_fragment_op_dataset import (  # noqa: E402
    filter_fragment_records,
    load_json_or_jsonl,
    split_records,
    write_jsonl,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    stats = {
        "approximate_min_conf": args.approximate_min_conf,
    }

    if args.annotated_pairs_input:
        semantic_pairs = load_json_or_jsonl(args.annotated_pairs_input)
        semantic_pair_splits = split_records(semantic_pairs, tuple(args.split), args.seed)
        fragment_pair_splits = tuple(
            filter_fragment_records(split, args.approximate_min_conf)
            for split in semantic_pair_splits
        )

        split_names = ["train", "valid", "test"]
        for split_name, semantic_split, fragment_split in zip(split_names, semantic_pair_splits, fragment_pair_splits):
            write_jsonl(semantic_split, os.path.join(args.output_dir, f"semantic_pairs_{split_name}.jsonl"))
            write_jsonl(fragment_split, os.path.join(args.output_dir, f"fragment_pairs_{split_name}.jsonl"))

        stats["semantic_pair_split_sizes"] = {
            "train": len(semantic_pair_splits[0]),
            "valid": len(semantic_pair_splits[1]),
            "test": len(semantic_pair_splits[2]),
        }
        stats["fragment_pair_split_sizes"] = {
            "train": len(fragment_pair_splits[0]),
            "valid": len(fragment_pair_splits[1]),
            "test": len(fragment_pair_splits[2]),
        }

    if args.annotated_paths_input:
        semantic_paths = load_json_or_jsonl(args.annotated_paths_input)
        semantic_path_splits = split_records(semantic_paths, tuple(args.split), args.seed)
        split_names = ["train", "valid", "test"]
        for split_name, semantic_split in zip(split_names, semantic_path_splits):
            write_jsonl(semantic_split, os.path.join(args.output_dir, f"semantic_paths_{split_name}.jsonl"))

        stats["semantic_path_split_sizes"] = {
            "train": len(semantic_path_splits[0]),
            "valid": len(semantic_path_splits[1]),
            "test": len(semantic_path_splits[2]),
        }

    stats_path = os.path.join(args.output_dir, "training_view_stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    logger.info("Training view stats -> %s", stats_path)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export semantic_pairs / fragment_pairs / semantic_paths from annotated inputs.")
    p.add_argument("--annotated_pairs_input", type=str, default=None, help="Annotated semantic pair file (JSON or JSONL).")
    p.add_argument("--annotated_paths_input", type=str, default=None, help="Annotated semantic path file (JSON or JSONL).")
    p.add_argument("--output_dir", type=str, required=True, help="Directory for exported training views.")
    p.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VALID", "TEST"),
        help="Train/valid/test split ratios. Default: 0.8 0.1 0.1",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    p.add_argument(
        "--approximate_min_conf",
        type=float,
        default=0.8,
        help="Minimum confidence required for an approximate semantic step to enter fragment_pairs. Default: 0.8",
    )
    p.add_argument(
        "--log_level",
        type=str,
        default="INFO",
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
    if not args.annotated_pairs_input and not args.annotated_paths_input:
        logger.error("At least one of --annotated_pairs_input or --annotated_paths_input must be provided.")
        sys.exit(1)

    main(args)
