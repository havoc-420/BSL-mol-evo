#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""准备方案 A 的固定 holdout / train pool 切分。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备方案 A 的固定 holdout 与训练池 CSV")
    parser.add_argument("--input-csv", required=True, help="源 CSV 路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--holdout-size", type=int, default=50, help="holdout 样本数")
    parser.add_argument("--start-index", type=int, default=0, help="holdout 起始索引（含）")
    parser.add_argument("--prefix", type=str, default="lumo_plan_a", help="输出文件名前缀")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    total_rows = len(df)
    start = args.start_index
    end = start + args.holdout_size

    if start < 0:
        raise ValueError("start-index 不能小于 0")
    if args.holdout_size <= 0:
        raise ValueError("holdout-size 必须大于 0")
    if end > total_rows:
        raise ValueError(
            f"holdout 区间越界：start={start}, size={args.holdout_size}, total_rows={total_rows}"
        )

    holdout_df = df.iloc[start:end].reset_index(drop=True)
    train_df = pd.concat([df.iloc[:start], df.iloc[end:]], axis=0).reset_index(drop=True)

    holdout_csv = output_dir / f"{args.prefix}_holdout{args.holdout_size}_start{start}.csv"
    train_csv = output_dir / f"{args.prefix}_train_pool_excluding_holdout.csv"
    metadata_json = output_dir / f"{args.prefix}_split_metadata.json"

    holdout_df.to_csv(holdout_csv, index=False)
    train_df.to_csv(train_csv, index=False)

    metadata = {
        "source_csv": str(input_csv),
        "total_rows": total_rows,
        "holdout_size": args.holdout_size,
        "holdout_start_index": start,
        "holdout_end_index_exclusive": end,
        "holdout_csv": str(holdout_csv),
        "train_pool_csv": str(train_csv),
        "columns": list(df.columns),
        "holdout_smiles_preview": holdout_df["smiles"].head(10).tolist() if "smiles" in holdout_df.columns else [],
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"source_csv={input_csv}")
    print(f"total_rows={total_rows}")
    print(f"holdout_rows={len(holdout_df)}")
    print(f"train_pool_rows={len(train_df)}")
    print(f"holdout_csv={holdout_csv}")
    print(f"train_pool_csv={train_csv}")
    print(f"metadata_json={metadata_json}")


if __name__ == "__main__":
    main()
