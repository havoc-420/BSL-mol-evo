#!/usr/bin/env python3
"""
build_molecule_manifest.py
=========================

将“普通分子 item 表”规范化为统一的 `molecule_manifest.jsonl`。

支持输入：
- CSV
- JSONL
- JSON 数组

目标：
- 为 QM9 / ZINC / 其他外部分子表提供统一事实源入口
- 规范化 `mol_id`、`smiles`、`scaffold_key`
- 为后续 primitive pair/path 构造提供稳定 item 清单
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

LOGGER = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".jsonl":
        rows: List[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL 第 {line_no} 行解析失败: {exc}") from exc
        return pd.DataFrame(rows)

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("JSON 输入必须是对象数组")
        return pd.DataFrame(data)

    raise ValueError(f"不支持的输入格式: {path.suffix}")


def canonicalize_smiles(smiles: Any) -> Optional[tuple[str, Chem.Mol]]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol), mol


def get_scaffold_key(mol: Chem.Mol) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or ""
    except Exception:
        return ""


def sanitize_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def infer_property_columns(df: pd.DataFrame, reserved: Iterable[str]) -> List[str]:
    reserved_set = set(reserved)
    property_columns: List[str] = []
    for column in df.columns:
        if column in reserved_set:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            property_columns.append(column)
    return property_columns


def build_manifest_records(
    df: pd.DataFrame,
    source_dataset: str,
    smiles_column: str,
    id_column: Optional[str],
    property_columns: List[str],
    max_records: Optional[int],
) -> tuple[List[dict], dict]:
    if max_records is not None:
        df = df.head(max_records)

    records: List[dict] = []
    stats = {
        "input_rows": int(len(df)),
        "valid_rows": 0,
        "invalid_smiles": 0,
        "duplicate_smiles": 0,
        "property_columns": property_columns,
        "source_dataset": source_dataset,
    }
    seen_smiles = set()

    for row_id, row in df.iterrows():
        canonical = canonicalize_smiles(row.get(smiles_column))
        if canonical is None:
            stats["invalid_smiles"] += 1
            continue

        smiles, mol = canonical
        if smiles in seen_smiles:
            stats["duplicate_smiles"] += 1
            continue
        seen_smiles.add(smiles)

        raw_id = row.get(id_column, row_id) if id_column else row_id
        source_row_id = sanitize_scalar(raw_id)
        if source_row_id is None or source_row_id == "":
            source_row_id = int(row_id)

        mol_id = f"{source_dataset}_{source_row_id}"
        scaffold_key = get_scaffold_key(mol)
        num_heavy_atoms = int(mol.GetNumHeavyAtoms())

        record = {
            "mol_id": mol_id,
            "source_dataset": source_dataset,
            "source_row_id": source_row_id,
            "smiles": smiles,
            "num_atoms": num_heavy_atoms,
            "num_heavy_atoms": num_heavy_atoms,
            "scaffold_key": scaffold_key,
            "split_key": scaffold_key or mol_id,
        }

        for column in property_columns:
            if column in row:
                record[column] = sanitize_scalar(row[column])

        records.append(record)
        stats["valid_rows"] += 1

    return records, stats


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_stats(path: Path, input_file: Path, args: argparse.Namespace, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_file": str(input_file),
        "output_file": str(args.output_file),
        "source_dataset": args.source_dataset,
        "smiles_column": args.smiles_column,
        "id_column": args.id_column,
        "max_records": args.max_records,
        "stats": stats,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将普通 item 表规范化为 molecule manifest JSONL")
    parser.add_argument("--input-file", type=Path, required=True, help="输入 item 表，支持 CSV / JSONL / JSON")
    parser.add_argument("--output-file", type=Path, required=True, help="输出 manifest JSONL 路径")
    parser.add_argument("--stats-file", type=Path, default=None, help="输出统计 JSON 路径，默认与 output 同目录")
    parser.add_argument("--source-dataset", type=str, default="unknown", help="数据源名称，例如 qm9 / zinc")
    parser.add_argument("--smiles-column", type=str, default="smiles", help="SMILES 列名")
    parser.add_argument("--id-column", type=str, default="index", help="样本 ID 列名；不存在时回退到行号")
    parser.add_argument(
        "--property-columns",
        type=str,
        nargs="*",
        default=None,
        help="显式指定要保留到 manifest 的属性列；默认自动保留数值列",
    )
    parser.add_argument("--max-records", type=int, default=None, help="最多处理多少条 item，用于 smoke")
    parser.add_argument("--log-level", type=str, default="INFO", help="日志级别")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    if not args.input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input_file}")

    df = load_table(args.input_file)
    if args.smiles_column not in df.columns:
        raise ValueError(f"输入表缺少 SMILES 列: {args.smiles_column}")

    property_columns = args.property_columns
    if property_columns is None:
        property_columns = infer_property_columns(
            df,
            reserved={args.smiles_column, args.id_column, "mol_id", "source_dataset", "source_row_id", "split_key", "scaffold_key"},
        )

    records, stats = build_manifest_records(
        df=df,
        source_dataset=args.source_dataset,
        smiles_column=args.smiles_column,
        id_column=args.id_column if args.id_column in df.columns else None,
        property_columns=property_columns,
        max_records=args.max_records,
    )

    write_jsonl(args.output_file, records)
    stats_file = args.stats_file or args.output_file.with_suffix(".stats.json")
    save_stats(stats_file, args.input_file, args, stats)

    LOGGER.info("molecule manifest 构造完成：%d 条有效记录", stats["valid_rows"])
    LOGGER.info("输出文件：%s", args.output_file)
    LOGGER.info("统计文件：%s", stats_file)


if __name__ == "__main__":
    main()
