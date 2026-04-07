#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 QM9 构造适合 `OFO-frag` 上游 bootstrap / 调试使用的 canonical local pair 数据。

主线：
    QM9 CSV -> heavy-atom 局部候选 -> fingerprint 预筛 ->
    MoleculeEvolverAnalysis -> canonical primitive operations -> JSONL 导出

注意：QM9 在这里仅作为小分子 bootstrap 与管线验证源，
不应被视为最终的 fragment 主训练源。

这一步只负责生成 `canonical_pairs_raw.jsonl`，不负责：
- 属性变化标注
- `fragment_op` 标注
- train/valid/test split

这些步骤会由后续脚本继续处理。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mol_evo.core.evolver import MoleculeEvolverAnalysis


LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT = PROJECT_ROOT / "mol_evo" / "dataset" / "data" / "qm9_smiles_all_atoms.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "mol_evo" / "dataset" / "data" / "canonical_pairs_raw.jsonl"
DEFAULT_STATS = PROJECT_ROOT / "mol_evo" / "dataset" / "data" / "canonical_pairs_raw.stats.json"


@dataclass
class MoleculeRecord:
    """缓存单个 QM9 分子的轻量信息。"""

    record_id: int
    qm9_index: int
    smiles: str
    heavy_atoms: int
    scaffold_key: str
    fingerprint: DataStructs.ExplicitBitVect


@dataclass
class CandidatePair:
    """候选 pair 的轻量描述。"""

    target_record_id: int
    similarity: float
    same_scaffold: bool
    heavy_atom_delta: int
    candidate_rank: int


@dataclass
class BuildStats:
    """运行期统计信息。"""

    total_rows: int = 0
    valid_rows: int = 0
    source_rows: int = 0
    candidate_pairs_scored: int = 0
    candidate_pairs_passing_similarity: int = 0
    pairs_with_operations: int = 0
    pairs_filtered_by_step: int = 0
    pairs_written: int = 0
    operation_failures: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "source_rows": self.source_rows,
            "candidate_pairs_scored": self.candidate_pairs_scored,
            "candidate_pairs_passing_similarity": self.candidate_pairs_passing_similarity,
            "pairs_with_operations": self.pairs_with_operations,
            "pairs_filtered_by_step": self.pairs_filtered_by_step,
            "pairs_written": self.pairs_written,
            "operation_failures": self.operation_failures,
        }


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def build_fingerprint(mol: Chem.Mol, radius: int = 2, fp_size: int = 1024):
    """为分子构建 Morgan 指纹。"""
    try:
        generator = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size)
        return generator.GetFingerprint(mol)
    except AttributeError:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size)


def canonicalize_smiles(smiles: str) -> Optional[Tuple[str, Chem.Mol]]:
    """规范化 SMILES，并返回分子对象。"""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol), mol


def get_scaffold_key(mol: Chem.Mol) -> str:
    """计算 Murcko scaffold key。"""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or ""
    except Exception:
        return ""


# 与 `extract_evolution_pairs_v1.py` 保持同语义的轻量差分逻辑。
def analyze_evolution_operation_dict(path1_dict: Sequence[dict], path2_dict: Sequence[dict]) -> List[dict]:
    """基于两条演化路径的差异构造 primitive operations。"""

    def op_to_str(op: dict) -> str:
        return f"{op.get('operation', '')}@{op.get('position', '')}@{op.get('atom', '')}"

    set1_str = {op_to_str(op) for op in path1_dict}
    set2_str = {op_to_str(op) for op in path2_dict}

    removed_ops_str = set1_str - set2_str
    added_ops_str = set2_str - set1_str

    removed_ops = [op for op in path1_dict if op_to_str(op) in removed_ops_str]
    added_ops = [op for op in path2_dict if op_to_str(op) in added_ops_str]

    replace_ops: List[dict] = []
    remaining_removed: List[dict] = []
    remaining_added = added_ops.copy()

    for removed_op in removed_ops:
        found_replace = False
        for added_op in added_ops:
            if (
                removed_op.get("operation") == added_op.get("operation")
                and removed_op.get("position") == added_op.get("position")
                and removed_op.get("atom") != added_op.get("atom")
            ):
                replace_ops.append(
                    {
                        "position": added_op.get("position"),
                        "atom": added_op.get("atom"),
                        "operation": "replace_atom",
                        "from_atom": removed_op.get("atom"),
                    }
                )
                found_replace = True
                if added_op in remaining_added:
                    remaining_added.remove(added_op)
                break

        if not found_replace:
            remaining_removed.append(removed_op)

    marked_removed_ops = []
    for op in remaining_removed:
        new_op = op.copy()
        new_op["operation"] = "remove_" + op.get("operation", "")
        marked_removed_ops.append(new_op)

    return marked_removed_ops + replace_ops + remaining_added


@lru_cache(maxsize=10000)
def get_path_dict(smiles: str) -> Tuple[dict, ...]:
    """缓存 `MoleculeEvolverAnalysis` 结果，减少重复构造。"""
    analyzer = MoleculeEvolverAnalysis(smiles)
    path_dict = analyzer.get_full_path_dict()
    return tuple(path_dict)


def load_qm9_records(
    csv_path: Path,
    smiles_column: str,
    index_column: str,
    source_start: int,
    source_end: Optional[int],
    max_sources: Optional[int],
    heavy_atom_min: Optional[int],
    heavy_atom_max: Optional[int],
    deduplicate_smiles: bool,
) -> Tuple[List[MoleculeRecord], List[int], BuildStats]:
    """加载并预处理 QM9 CSV。"""
    df = pd.read_csv(csv_path)
    stats = BuildStats(total_rows=len(df))

    records: List[MoleculeRecord] = []
    seen_smiles = set()

    for row_id, row in tqdm(df.iterrows(), total=len(df), desc="加载 QM9 分子", unit="mol"):
        canonical = canonicalize_smiles(row.get(smiles_column, ""))
        if canonical is None:
            continue

        smiles, mol = canonical
        if deduplicate_smiles and smiles in seen_smiles:
            continue
        seen_smiles.add(smiles)

        heavy_atoms = int(mol.GetNumHeavyAtoms())
        if heavy_atom_min is not None and heavy_atoms < heavy_atom_min:
            continue
        if heavy_atom_max is not None and heavy_atoms > heavy_atom_max:
            continue

        raw_index = row.get(index_column, row_id)
        try:
            qm9_index = int(raw_index)
        except Exception:
            qm9_index = int(row_id)

        records.append(
            MoleculeRecord(
                record_id=len(records),
                qm9_index=qm9_index,
                smiles=smiles,
                heavy_atoms=heavy_atoms,
                scaffold_key=get_scaffold_key(mol),
                fingerprint=build_fingerprint(mol),
            )
        )

    stats.valid_rows = len(records)

    if source_end is None:
        source_end = len(records)
    else:
        source_end = min(source_end, len(records))

    source_ids = list(range(max(source_start, 0), source_end))
    if max_sources is not None:
        source_ids = source_ids[: max_sources]
    stats.source_rows = len(source_ids)

    return records, source_ids, stats


def build_bucket_index(records: Sequence[MoleculeRecord]) -> Dict[int, List[int]]:
    """按重原子数建立 bucket 索引。"""
    buckets: Dict[int, List[int]] = defaultdict(list)
    for record in records:
        buckets[record.heavy_atoms].append(record.record_id)
    return dict(buckets)


def iter_local_candidate_ids(
    source: MoleculeRecord,
    buckets: Dict[int, List[int]],
    heavy_atom_window: int,
) -> Iterable[int]:
    """枚举 source 附近 heavy-atom bucket 的候选。"""
    for bucket in range(source.heavy_atoms - heavy_atom_window, source.heavy_atoms + heavy_atom_window + 1):
        for target_id in buckets.get(bucket, []):
            if target_id != source.record_id:
                yield target_id


def select_candidate_pairs(
    source: MoleculeRecord,
    records: Sequence[MoleculeRecord],
    buckets: Dict[int, List[int]],
    heavy_atom_window: int,
    min_similarity: float,
    max_similarity: Optional[float],
    top_k_targets: int,
    prefer_same_scaffold: bool,
    require_same_scaffold: bool,
    stats: BuildStats,
) -> List[CandidatePair]:
    """为单个 source 选择局部高相似候选。"""
    target_ids: List[int] = []
    target_fps = []
    scaffold_flags: List[bool] = []
    heavy_atom_deltas: List[int] = []

    for target_id in iter_local_candidate_ids(source, buckets, heavy_atom_window):
        target = records[target_id]
        if target.smiles == source.smiles:
            continue
        same_scaffold = bool(source.scaffold_key and source.scaffold_key == target.scaffold_key)
        if require_same_scaffold and not same_scaffold:
            continue
        target_ids.append(target_id)
        target_fps.append(target.fingerprint)
        scaffold_flags.append(same_scaffold)
        heavy_atom_deltas.append(target.heavy_atoms - source.heavy_atoms)

    if not target_fps:
        return []

    similarities = DataStructs.BulkTanimotoSimilarity(source.fingerprint, target_fps)
    stats.candidate_pairs_scored += len(similarities)

    candidates: List[CandidatePair] = []
    for target_id, similarity, same_scaffold, heavy_atom_delta in zip(
        target_ids, similarities, scaffold_flags, heavy_atom_deltas
    ):
        if similarity < min_similarity:
            continue
        if max_similarity is not None and similarity > max_similarity:
            continue
        candidates.append(
            CandidatePair(
                target_record_id=target_id,
                similarity=float(similarity),
                same_scaffold=same_scaffold,
                heavy_atom_delta=int(heavy_atom_delta),
                candidate_rank=-1,
            )
        )

    stats.candidate_pairs_passing_similarity += len(candidates)

    candidates.sort(
        key=lambda item: (
            0 if (prefer_same_scaffold and item.same_scaffold) else 1,
            -item.similarity,
            abs(item.heavy_atom_delta),
            item.target_record_id,
        )
    )

    selected = candidates[:top_k_targets]
    for rank, item in enumerate(selected, start=1):
        item.candidate_rank = rank
    return selected


def build_pair_record(
    source: MoleculeRecord,
    target: MoleculeRecord,
    candidate: CandidatePair,
    min_steps: int,
    max_steps: int,
    generation_version: str,
    stats: BuildStats,
) -> Optional[dict]:
    """为单个 source/target 构造 canonical pair 记录。"""
    try:
        path_from = list(get_path_dict(source.smiles))
        path_to = list(get_path_dict(target.smiles))
        operations = analyze_evolution_operation_dict(path_from, path_to)
    except Exception as exc:
        stats.operation_failures += 1
        LOGGER.debug("pair 构造失败 %s -> %s: %s", source.smiles, target.smiles, exc)
        return None

    if not operations:
        return None

    stats.pairs_with_operations += 1
    num_steps = len(operations)
    if num_steps < min_steps or num_steps > max_steps:
        stats.pairs_filtered_by_step += 1
        return None

    pair_record = {
        "pair_id": f"qm9_{source.qm9_index}_{target.qm9_index}",
        "from_index": int(source.qm9_index),
        "to_index": int(target.qm9_index),
        "smiles_from": source.smiles,
        "smiles_to": target.smiles,
        "operations": operations,
        "num_steps": int(num_steps),
        "similarity": round(candidate.similarity, 6),
        "from_num_atoms": int(source.heavy_atoms),
        "to_num_atoms": int(target.heavy_atoms),
        "scaffold_preserving": bool(candidate.same_scaffold),
        "candidate_meta": {
            "candidate_rank": int(candidate.candidate_rank),
            "heavy_atom_delta": int(candidate.heavy_atom_delta),
            "selection_strategy": "local_bucket_similarity",
            "source_dataset": "qm9",
            "generation_version": generation_version,
        },
    }
    return pair_record


def write_jsonl(path: Path, rows: Iterable[dict], append: bool) -> int:
    """按 JSONL 格式落盘。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def build_canonical_pairs(args: argparse.Namespace) -> BuildStats:
    """主执行逻辑。"""
    records, source_ids, stats = load_qm9_records(
        csv_path=args.input_csv,
        smiles_column=args.smiles_column,
        index_column=args.index_column,
        source_start=args.source_start,
        source_end=args.source_end,
        max_sources=args.max_sources,
        heavy_atom_min=args.heavy_atom_min,
        heavy_atom_max=args.heavy_atom_max,
        deduplicate_smiles=args.deduplicate_smiles,
    )
    buckets = build_bucket_index(records)

    LOGGER.info("加载完成：总行数=%s，有效分子=%s，source 数量=%s", stats.total_rows, stats.valid_rows, stats.source_rows)
    LOGGER.info(
        "候选筛选配置：heavy_atom_window=%s, top_k_targets=%s, min_similarity=%.3f, min_steps=%s, max_steps=%s",
        args.heavy_atom_window,
        args.top_k_targets,
        args.min_similarity,
        args.min_steps,
        args.max_steps,
    )

    if args.output_file.exists() and not args.append:
        LOGGER.info("输出文件已存在，将覆盖：%s", args.output_file)

    buffer: List[dict] = []
    first_write = not args.append

    for source_id in tqdm(source_ids, desc="构造 canonical pairs", unit="source"):
        source = records[source_id]
        candidates = select_candidate_pairs(
            source=source,
            records=records,
            buckets=buckets,
            heavy_atom_window=args.heavy_atom_window,
            min_similarity=args.min_similarity,
            max_similarity=args.max_similarity,
            top_k_targets=args.top_k_targets,
            prefer_same_scaffold=not args.no_prefer_same_scaffold,
            require_same_scaffold=args.require_same_scaffold,
            stats=stats,
        )

        for candidate in candidates:
            target = records[candidate.target_record_id]
            pair_record = build_pair_record(
                source=source,
                target=target,
                candidate=candidate,
                min_steps=args.min_steps,
                max_steps=args.max_steps,
                generation_version=args.generation_version,
                stats=stats,
            )
            if pair_record is None:
                continue

            buffer.append(pair_record)
            stats.pairs_written += 1

            if len(buffer) >= args.flush_every:
                write_jsonl(args.output_file, buffer, append=not first_write)
                buffer.clear()
                first_write = False

            if args.max_pairs is not None and stats.pairs_written >= args.max_pairs:
                LOGGER.info("达到 max_pairs=%s，提前停止。", args.max_pairs)
                if buffer:
                    write_jsonl(args.output_file, buffer, append=not first_write)
                    buffer.clear()
                return stats

    if buffer:
        write_jsonl(args.output_file, buffer, append=not first_write)

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从 QM9 构造 canonical local pair 数据集")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT, help="QM9 输入 CSV 路径")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT, help="输出 JSONL 路径")
    parser.add_argument("--stats-file", type=Path, default=DEFAULT_STATS, help="统计信息 JSON 路径")
    parser.add_argument("--smiles-column", type=str, default="smiles", help="SMILES 列名")
    parser.add_argument("--index-column", type=str, default="index", help="QM9 index 列名，不存在时退回到行号")
    parser.add_argument("--source-start", type=int, default=0, help="source 起始下标（基于过滤后的有效分子）")
    parser.add_argument("--source-end", type=int, default=None, help="source 结束下标（开区间）")
    parser.add_argument("--max-sources", type=int, default=None, help="最多处理多少个 source 分子")
    parser.add_argument("--max-pairs", type=int, default=None, help="最多输出多少条 pair")
    parser.add_argument("--top-k-targets", type=int, default=24, help="每个 source 保留多少个局部高相似 target")
    parser.add_argument("--heavy-atom-window", type=int, default=1, help="候选 target 的 heavy-atom bucket 半径")
    parser.add_argument("--heavy-atom-min", type=int, default=None, help="过滤最小重原子数")
    parser.add_argument("--heavy-atom-max", type=int, default=None, help="过滤最大重原子数")
    parser.add_argument("--min-similarity", type=float, default=0.55, help="候选 pair 的最小 Tanimoto 相似度")
    parser.add_argument("--max-similarity", type=float, default=None, help="候选 pair 的最大 Tanimoto 相似度")
    parser.add_argument("--min-steps", type=int, default=1, help="保留 pair 的最小 primitive step 数")
    parser.add_argument("--max-steps", type=int, default=3, help="保留 pair 的最大 primitive step 数")
    parser.add_argument("--require-same-scaffold", action="store_true", help="仅保留相同 Murcko scaffold 的候选")
    parser.add_argument("--no-prefer-same-scaffold", action="store_true", help="候选排序时不优先相同 scaffold")
    parser.add_argument("--deduplicate-smiles", action="store_true", help="加载阶段按 canonical SMILES 去重")
    parser.add_argument("--append", action="store_true", help="以 append 模式写入输出文件，便于分块生成")
    parser.add_argument("--flush-every", type=int, default=200, help="每积累多少条 pair flush 一次")
    parser.add_argument("--generation-version", type=str, default="canonical_pairs_v0", help="写入 `candidate_meta` 的生成版本")
    parser.add_argument("--log-level", type=str, default="INFO", help="日志级别")
    return parser


def save_stats(stats_file: Path, args: argparse.Namespace, stats: BuildStats) -> None:
    """保存统计摘要。"""
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_csv": str(args.input_csv),
        "output_file": str(args.output_file),
        "generation_version": args.generation_version,
        "filters": {
            "source_start": args.source_start,
            "source_end": args.source_end,
            "max_sources": args.max_sources,
            "max_pairs": args.max_pairs,
            "top_k_targets": args.top_k_targets,
            "heavy_atom_window": args.heavy_atom_window,
            "heavy_atom_min": args.heavy_atom_min,
            "heavy_atom_max": args.heavy_atom_max,
            "min_similarity": args.min_similarity,
            "max_similarity": args.max_similarity,
            "min_steps": args.min_steps,
            "max_steps": args.max_steps,
            "require_same_scaffold": args.require_same_scaffold,
            "prefer_same_scaffold": not args.no_prefer_same_scaffold,
            "deduplicate_smiles": args.deduplicate_smiles,
            "append": args.append,
        },
        "stats": stats.to_dict(),
    }
    with stats_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)

    if not args.input_csv.exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input_csv}")

    if args.min_steps < 1:
        raise ValueError("--min-steps 必须 >= 1")
    if args.max_steps < args.min_steps:
        raise ValueError("--max-steps 不能小于 --min-steps")
    if args.top_k_targets < 1:
        raise ValueError("--top-k-targets 必须 >= 1")
    if args.heavy_atom_window < 0:
        raise ValueError("--heavy-atom-window 不能为负数")
    if args.flush_every < 1:
        raise ValueError("--flush-every 必须 >= 1")

    stats = build_canonical_pairs(args)
    save_stats(args.stats_file, args, stats)

    LOGGER.info("canonical pair 构造完成，写出 %s 条记录。", stats.pairs_written)
    LOGGER.info("统计信息已保存到: %s", args.stats_file)


if __name__ == "__main__":
    main()
