from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


THIS_DIR = Path(__file__).resolve().parent
MOL_EVO_ROOT = THIS_DIR.parents[3]
SOURCE_CSV = MOL_EVO_ROOT / "output" / "evo-mo" / "超参数实验" / "all_hparam_summary.csv"

METRIC_COLUMNS = [
    "average_improvement",
    "improved_percentage",
    "average_drug_likeness",
    "intdiv_avg",
    "morgan_similarity_avg",
]

METRIC_PRECISION = {
    "average_improvement": 4,
    "improved_percentage": 2,
    "average_drug_likeness": 4,
    "intdiv_avg": 4,
    "morgan_similarity_avg": 4,
}

TABLE_SPECS: List[Dict[str, object]] = [
    {
        "table_id": "appendix_3_1_lumo_up_num_simulations",
        "markdown_file": "mcts_hparam_sensitivity_cross_task_appendix.md",
        "section": "3.1 搜索预算 / LUMO(U)",
        "csv_file": "table_3_1_lumo_up_num_simulations.csv",
        "param_key": "mcts_simulations",
        "param_name": "num_simulations",
        "target_prop": "lumo",
        "direction": "increase",
        "rows": [
            {"batch_dir": "batch_optimization_20260413_134936", "param_value": "200"},
            {"batch_dir": "batch_optimization_20260413_135641", "param_value": "400"},
            {"batch_dir": "batch_optimization_20260413_140631", "param_value": "800"},
        ],
    },
    {
        "table_id": "appendix_3_1_homo_down_num_simulations",
        "markdown_file": "mcts_hparam_sensitivity_cross_task_appendix.md",
        "section": "3.1 搜索预算 / HOMO(D)",
        "csv_file": "table_3_1_homo_down_num_simulations.csv",
        "param_key": "mcts_simulations",
        "param_name": "num_simulations",
        "target_prop": "homo",
        "direction": "decrease",
        "rows": [
            {"batch_dir": "batch_optimization_20260413_162511", "param_value": "200"},
            {"batch_dir": "batch_optimization_20260413_162650", "param_value": "400"},
            {"batch_dir": "batch_optimization_20260413_162834", "param_value": "800"},
        ],
    },
    {
        "table_id": "appendix_3_2_lumo_up_exploration_weight",
        "markdown_file": "mcts_hparam_sensitivity_cross_task_appendix.md",
        "section": "3.2 探索系数 / LUMO(U)",
        "csv_file": "table_3_2_lumo_up_exploration_weight.csv",
        "param_key": "mcts_exploration_coef",
        "param_name": "exploration_weight",
        "target_prop": "lumo",
        "direction": "increase",
        "rows": [
            {"batch_dir": "batch_optimization_20260413_142447", "param_value": "1.0"},
            {"batch_dir": "batch_optimization_20260413_143231", "param_value": "1.4"},
            {"batch_dir": "batch_optimization_20260413_140631", "param_value": "2.0"},
        ],
    },
    {
        "table_id": "appendix_3_2_homo_down_exploration_weight",
        "markdown_file": "mcts_hparam_sensitivity_cross_task_appendix.md",
        "section": "3.2 探索系数 / HOMO(D)",
        "csv_file": "table_3_2_homo_down_exploration_weight.csv",
        "param_key": "mcts_exploration_coef",
        "param_name": "exploration_weight",
        "target_prop": "homo",
        "direction": "decrease",
        "rows": [
            {"batch_dir": "batch_optimization_20260413_163029", "param_value": "1.0"},
            {"batch_dir": "batch_optimization_20260413_163208", "param_value": "1.4"},
            {"batch_dir": "batch_optimization_20260413_162834", "param_value": "2.0"},
        ],
    },
    {
        "table_id": "appendix_3_3_lumo_up_max_branching",
        "markdown_file": "mcts_hparam_sensitivity_cross_task_appendix.md",
        "section": "3.3 分支宽度 / LUMO(U)",
        "csv_file": "table_3_3_lumo_up_max_branching.csv",
        "param_key": "max_branches",
        "param_name": "max_branching",
        "target_prop": "lumo",
        "direction": "increase",
        "rows": [
            {"batch_dir": "batch_optimization_20260413_150054", "param_value": "8"},
            {"batch_dir": "batch_optimization_20260413_140631", "param_value": "20"},
        ],
    },
    {
        "table_id": "appendix_3_3_homo_down_max_branching",
        "markdown_file": "mcts_hparam_sensitivity_cross_task_appendix.md",
        "section": "3.3 分支宽度 / HOMO(D)",
        "csv_file": "table_3_3_homo_down_max_branching.csv",
        "param_key": "max_branches",
        "param_name": "max_branching",
        "target_prop": "homo",
        "direction": "decrease",
        "rows": [
            {"batch_dir": "batch_optimization_20260413_163545", "param_value": "8"},
            {"batch_dir": "batch_optimization_20260413_162834", "param_value": "20"},
        ],
    },
    {
        "table_id": "appendix_4_1_lumo_down_exploration_weight",
        "markdown_file": "mcts_hparam_sensitivity_cross_task_appendix.md",
        "section": "4.1 LUMO(D) 探索系数",
        "csv_file": "table_4_1_lumo_down_exploration_weight.csv",
        "param_key": "mcts_exploration_coef",
        "param_name": "exploration_weight",
        "target_prop": "lumo",
        "direction": "decrease",
        "rows": [
            {"batch_dir": "batch_optimization_20260412_232639", "param_value": "0.5"},
            {"batch_dir": "batch_optimization_20260412_222529", "param_value": "1.0"},
            {"batch_dir": "batch_optimization_20260412_222658", "param_value": "2.0"},
            {"batch_dir": "batch_optimization_20260412_222632", "param_value": "3.0"},
        ],
    },
    {
        "table_id": "appendix_4_2_lumo_down_max_depth",
        "markdown_file": "mcts_hparam_sensitivity_cross_task_appendix.md",
        "section": "4.2 LUMO(D) 搜索深度",
        "csv_file": "table_4_2_lumo_down_max_depth.csv",
        "param_key": "max_depth",
        "param_name": "max_depth",
        "target_prop": "lumo",
        "direction": "decrease",
        "rows": [
            {"batch_dir": "batch_optimization_20260413_004229", "param_value": "2"},
            {"batch_dir": "batch_optimization_20260413_004357", "param_value": "4"},
            {"batch_dir": "batch_optimization_20260413_004614", "param_value": "6"},
            {"batch_dir": "batch_optimization_20260413_004852", "param_value": "8"},
            {"batch_dir": "batch_optimization_20260413_005150", "param_value": "10"},
            {"batch_dir": "batch_optimization_20260412_222529", "param_value": "12"},
        ],
    },
    {
        "table_id": "appendix_4_3_lumo_down_pruning_patience",
        "markdown_file": "mcts_hparam_sensitivity_cross_task_appendix.md",
        "section": "4.3 LUMO(D) 剪枝耐心值",
        "csv_file": "table_4_3_lumo_down_pruning_patience.csv",
        "param_key": "prune_patience",
        "param_name": "pruning_patience",
        "target_prop": "lumo",
        "direction": "decrease",
        "rows": [
            {"batch_dir": "batch_optimization_20260413_005802", "param_value": "1"},
            {"batch_dir": "batch_optimization_20260413_010020", "param_value": "2"},
            {"batch_dir": "batch_optimization_20260412_222529", "param_value": "3"},
            {"batch_dir": "batch_optimization_20260413_010615", "param_value": "4"},
        ],
    },
    {
        "table_id": "appendix_4_4_lumo_down_max_branching",
        "markdown_file": "mcts_hparam_sensitivity_cross_task_appendix.md",
        "section": "4.4 LUMO(D) 分支宽度补充",
        "csv_file": "table_4_4_lumo_down_max_branching.csv",
        "param_key": "max_branches",
        "param_name": "max_branching",
        "target_prop": "lumo",
        "direction": "decrease",
        "rows": [
            {"batch_dir": "batch_optimization_20260412_222529", "param_value": "20"},
            {"batch_dir": "batch_optimization_20260413_003920", "param_value": "30"},
        ],
    },
]

MAIN_SUMMARY_TABLE: Dict[str, object] = {
    "table_id": "main_3_key_evidence",
    "markdown_file": "mcts_hparam_sensitivity_cross_task_main.md",
    "section": "3. 正文可保留的关键证据",
    "csv_file": "table_main_key_evidence.csv",
    "rows": [
        {
            "dimension": "num_simulations",
            "cross_task_observation": "在 LUMO(U) 上，200 -> 800 使平均改善值由 1.1933 升至 1.3681；在 HOMO(D) 上，200 -> 800 的平均改善从 -1.5795 变化到 -1.5211，说明收益已接近饱和。",
            "paper_interpretation": "预算增加的收益与任务难度相关：对更难的上升任务更有帮助，对另一类任务则更早进入平台区。",
            "recommended_writing": "不写成“预算越大越好”，而写成“中高预算通常足够，额外预算收益具有任务依赖性”。",
            "source_table_ids": "appendix_3_1_lumo_up_num_simulations|appendix_3_1_homo_down_num_simulations",
        },
        {
            "dimension": "exploration_weight",
            "cross_task_observation": "LUMO(U) 的平均改善在 2.0 最强（1.3681），但成功率在 1.4 最好（88.64%）；HOMO(D) 则在 1.0 上取得最强改善（-1.5899）和最高成功率（94.14%）。",
            "paper_interpretation": "探索系数主要控制搜索激进程度与稳定性，其最优点不是全任务统一的。",
            "recommended_writing": "正文宜写成“中等探索最稳健，过弱或过强都可能偏离任务最优工作点”。",
            "source_table_ids": "appendix_3_2_lumo_up_exploration_weight|appendix_3_2_homo_down_exploration_weight",
        },
        {
            "dimension": "max_branching",
            "cross_task_observation": "LUMO(U) 上 8 分支取得更强改善（1.5173），但成功率明显低于 20（72.21% vs 83.13%）；HOMO(D) 上 20 在改善值和成功率上都优于 8。",
            "paper_interpretation": "分支宽度控制的是激进搜索 vs 稳定覆盖；过窄宽度可能更冒进，但不一定是更稳健的默认设置。",
            "recommended_writing": "正文更适合把 20 写成安全默认值，而把更小宽度写成“在部分任务上可换取更强改善的激进选项”。",
            "source_table_ids": "appendix_3_3_lumo_up_max_branching|appendix_3_3_homo_down_max_branching",
        },
    ],
}


def load_rows() -> Dict[str, Dict[str, str]]:
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    by_batch = {row["batch_dir"]: row for row in rows}
    if len(by_batch) != len(rows):
        raise ValueError("batch_dir is not unique in source CSV.")
    return by_batch


def format_metric(value: str, key: str) -> str:
    return f"{float(value):.{METRIC_PRECISION[key]}f}"


def validate_row(row: Dict[str, str], spec: Dict[str, object], selected: Dict[str, str]) -> None:
    if row["target_prop"] != spec["target_prop"]:
        raise ValueError(
            f"Unexpected target_prop for {selected['batch_dir']}: {row['target_prop']} != {spec['target_prop']}"
        )
    if row["direction"] != spec["direction"]:
        raise ValueError(
            f"Unexpected direction for {selected['batch_dir']}: {row['direction']} != {spec['direction']}"
        )
    param_key = str(spec["param_key"])
    if row[param_key] != selected["param_value"]:
        raise ValueError(
            f"Unexpected {param_key} for {selected['batch_dir']}: {row[param_key]} != {selected['param_value']}"
        )


def build_output_row(row: Dict[str, str], spec: Dict[str, object], selected: Dict[str, str], order: int) -> Dict[str, str]:
    out = {
        "table_id": str(spec["table_id"]),
        "markdown_file": str(spec["markdown_file"]),
        "section": str(spec["section"]),
        "row_order": str(order),
        str(spec["param_name"]): selected["param_value"],
        "average_improvement": format_metric(row["average_improvement"], "average_improvement"),
        "improved_percentage": format_metric(row["improved_percentage"], "improved_percentage"),
        "average_drug_likeness": format_metric(row["average_drug_likeness"], "average_drug_likeness"),
        "intdiv_avg": format_metric(row["intdiv_avg"], "intdiv_avg"),
        "morgan_similarity_avg": format_metric(row["morgan_similarity_avg"], "morgan_similarity_avg"),
        "batch_dir": row["batch_dir"],
        "target_prop": row["target_prop"],
        "direction": row["direction"],
        "source": row.get("source", ""),
    }
    return out


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    by_batch = load_rows()
    manifest_rows: List[Dict[str, str]] = []
    long_rows: List[Dict[str, str]] = []

    for spec in TABLE_SPECS:
        param_name = str(spec["param_name"])
        table_rows: List[Dict[str, str]] = []
        batch_dirs: List[str] = []

        for order, selected in enumerate(spec["rows"], start=1):
            batch_dir = selected["batch_dir"]
            if batch_dir not in by_batch:
                raise FileNotFoundError(f"batch_dir not found in source CSV: {batch_dir}")
            source_row = by_batch[batch_dir]
            validate_row(source_row, spec, selected)
            out_row = build_output_row(source_row, spec, selected, order)
            table_rows.append(out_row)
            long_rows.append(out_row.copy())
            batch_dirs.append(batch_dir)

        csv_file = str(spec["csv_file"])
        csv_path = THIS_DIR / csv_file
        fieldnames = [
            param_name,
            "average_improvement",
            "improved_percentage",
            "average_drug_likeness",
            "intdiv_avg",
            "morgan_similarity_avg",
            "batch_dir",
            "target_prop",
            "direction",
            "source",
        ]
        slim_rows = [
            {key: row[key] for key in fieldnames}
            for row in table_rows
        ]
        write_csv(csv_path, fieldnames, slim_rows)

        manifest_rows.append(
            {
                "table_id": str(spec["table_id"]),
                "markdown_file": str(spec["markdown_file"]),
                "section": str(spec["section"]),
                "csv_file": csv_file,
                "target_prop": str(spec["target_prop"]),
                "direction": str(spec["direction"]),
                "sweep_parameter": param_name,
                "row_count": str(len(table_rows)),
                "batch_dirs": "|".join(batch_dirs),
                "source_reference": str(SOURCE_CSV.relative_to(MOL_EVO_ROOT)),
            }
        )

    main_csv_file = str(MAIN_SUMMARY_TABLE["csv_file"])
    main_rows = list(MAIN_SUMMARY_TABLE["rows"])
    write_csv(
        THIS_DIR / main_csv_file,
        [
            "dimension",
            "cross_task_observation",
            "paper_interpretation",
            "recommended_writing",
            "source_table_ids",
        ],
        main_rows,
    )
    manifest_rows.append(
        {
            "table_id": str(MAIN_SUMMARY_TABLE["table_id"]),
            "markdown_file": str(MAIN_SUMMARY_TABLE["markdown_file"]),
            "section": str(MAIN_SUMMARY_TABLE["section"]),
            "csv_file": main_csv_file,
            "target_prop": "cross_task",
            "direction": "mixed",
            "sweep_parameter": "summary",
            "row_count": str(len(main_rows)),
            "batch_dirs": "",
            "source_reference": "|".join(row["source_table_ids"] for row in main_rows),
        }
    )

    write_csv(
        THIS_DIR / "table_manifest.csv",
        [
            "table_id",
            "markdown_file",
            "section",
            "csv_file",
            "target_prop",
            "direction",
            "sweep_parameter",
            "row_count",
            "batch_dirs",
            "source_reference",
        ],
        manifest_rows,
    )

    write_csv(
        THIS_DIR / "table_data_long.csv",
        [
            "table_id",
            "markdown_file",
            "section",
            "row_order",
            "batch_dir",
            "target_prop",
            "direction",
            "source",
            "average_improvement",
            "improved_percentage",
            "average_drug_likeness",
            "intdiv_avg",
            "morgan_similarity_avg",
            "num_simulations",
            "exploration_weight",
            "max_branching",
            "max_depth",
            "pruning_patience",
        ],
        [
            {
                "table_id": row["table_id"],
                "markdown_file": row["markdown_file"],
                "section": row["section"],
                "row_order": row["row_order"],
                "batch_dir": row["batch_dir"],
                "target_prop": row["target_prop"],
                "direction": row["direction"],
                "source": row["source"],
                "average_improvement": row["average_improvement"],
                "improved_percentage": row["improved_percentage"],
                "average_drug_likeness": row["average_drug_likeness"],
                "intdiv_avg": row["intdiv_avg"],
                "morgan_similarity_avg": row["morgan_similarity_avg"],
                "num_simulations": row.get("num_simulations", ""),
                "exploration_weight": row.get("exploration_weight", ""),
                "max_branching": row.get("max_branching", ""),
                "max_depth": row.get("max_depth", ""),
                "pruning_patience": row.get("pruning_patience", ""),
            }
            for row in long_rows
        ],
    )

    total_csv_files = len(TABLE_SPECS) + 1
    print(f"Source: {SOURCE_CSV}")
    print(f"Generated {total_csv_files} table CSV files in {THIS_DIR}")
    print(f"Manifest: {THIS_DIR / 'table_manifest.csv'}")
    print(f"Long-form data: {THIS_DIR / 'table_data_long.csv'}")


if __name__ == "__main__":
    main()
