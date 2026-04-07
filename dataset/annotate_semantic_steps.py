#!/usr/bin/env python3
"""
annotate_semantic_steps.py
==========================
Annotate existing pair/path records with the semantic-step protocol.

This script is the annotation-stage entrypoint corresponding to the
`03-semantic-data-and-step-views-plan.md` workflow. It does not split data or
export train/valid/test views; it only upgrades raw pair/path records into
annotated semantic records.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

# Project path setup
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from dataset.build_fragment_op_dataset import (  # noqa: E402
    process_pairs,
    process_paths,
    write_jsonl,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



def annotate_pairs(input_path: str, output_path: str, max_workers: int, max_records: Optional[int]) -> dict:
    records, stats = process_pairs(input_path, max_workers=max_workers, max_records=max_records)
    write_jsonl(records, output_path)
    return stats



def annotate_paths(input_path: str, output_path: str, max_records: Optional[int]) -> dict:
    records, stats = process_paths(input_path, max_records=max_records)
    write_jsonl(records, output_path)
    return stats



def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    summary = {}

    if args.pairs_input:
        if not os.path.isfile(args.pairs_input):
            raise FileNotFoundError(f"Pairs input not found: {args.pairs_input}")
        pairs_out = os.path.join(args.output_dir, "semantic_pairs_annotated.jsonl")
        summary["pairs"] = annotate_pairs(
            args.pairs_input,
            pairs_out,
            max_workers=args.max_workers,
            max_records=args.max_records,
        )
        logger.info("Annotated pairs -> %s", pairs_out)

    if args.paths_input:
        if not os.path.isfile(args.paths_input):
            raise FileNotFoundError(f"Paths input not found: {args.paths_input}")
        paths_out = os.path.join(args.output_dir, "semantic_paths_annotated.jsonl")
        summary["paths"] = annotate_paths(
            args.paths_input,
            paths_out,
            max_records=args.max_records,
        )
        logger.info("Annotated paths -> %s", paths_out)

    summary_path = os.path.join(args.output_dir, "semantic_annotation_stats.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info("Annotation stats -> %s", summary_path)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate pair/path records with semantic_step metadata.")
    p.add_argument("--pairs_input", type=str, default=None, help="Input pairs file (JSON or JSONL).")
    p.add_argument("--paths_input", type=str, default=None, help="Input paths file (JSON or JSONL).")
    p.add_argument("--output_dir", type=str, required=True, help="Directory for annotated outputs.")
    p.add_argument("--max_workers", type=int, default=4, help="Worker count for pair annotation.")
    p.add_argument("--max_records", type=int, default=None, help="Optional cap for quick smoke runs.")
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
    if not args.pairs_input and not args.paths_input:
        logger.error("At least one of --pairs_input or --paths_input must be provided.")
        sys.exit(1)
    main(args)
