from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path("/home/ubuntu/mol_opt/mol-ofo/mol_evo")
PILOT_TSV = ROOT / "output/paper/ablations/20260413_1536_pilot_mcts/true_eval_summary_aggregated.tsv"
OFFICIAL_TSV = ROOT / "output/paper/ablations/20260414_0115_official_mcts/run_summary.tsv"
OUTPUT_DIR = ROOT / "docs/paper/ablations/images"
OUTPUT_PNG = OUTPUT_DIR / "mcts_ablation_summary.png"
OUTPUT_PDF = OUTPUT_DIR / "mcts_ablation_summary.pdf"
OUTPUT_TSV = OUTPUT_DIR / "mcts_ablation_plot_data.tsv"

TASK_ORDER = ["lumo_up", "homo_down"]
TASK_LABELS = {
    "lumo_up": "LUMO(U)",
    "homo_down": "HOMO(D)",
}
VARIANT_ORDER = ["full", "wo_prior", "wo_leaf_value", "random_topb"]
VARIANT_LABELS = {
    "full": "full",
    "wo_prior": "w/o prior",
    "wo_leaf_value": "w/o leaf value",
    "random_topb": "random top-b",
}

# ── Colors: each variant gets its own color ──
VARIANT_STYLE = {
    "full":           {"fill": "#C6DDF0", "edge": "#4E75A8", "text": "#355E96"},
    "wo_prior":       {"fill": "#E9D8C5", "edge": "#C6692C", "text": "#B95D1F"},
    "wo_leaf_value":  {"fill": "#D4E8D0", "edge": "#4E8A47", "text": "#3C7535"},
    "random_topb":    {"fill": "#E3D1E8", "edge": "#8B5AA0", "text": "#7A4990"},
}

FIG_BG = "#FFFFFF"
TEXT_PRIMARY = "#222630"
TEXT_MUTED = "#5E6470"
GRID_COLOR = "#D0D0D0"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_pilot_metrics() -> dict[tuple[str, str], dict[str, float]]:
    rows = read_tsv(PILOT_TSV)
    metrics: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        task = row["task"]
        variant = row["variant"]
        metrics[(task, variant)] = {
            "start_count": float(row["start_count_mean"]),
            "improvement": float(row["paper_avg_improvement_mean"]),
            "success_rate": float(row["paper_success_rate_mean"]),
            "runtime": float(row["avg_runtime_mean"]),
            "expanded_nodes": float(row["avg_expanded_nodes_mean"]),
        }

    for task in TASK_ORDER:
        full_runtime = metrics[(task, "full")]["runtime"]
        full_nodes = metrics[(task, "full")]["expanded_nodes"]
        for variant in VARIANT_ORDER:
            metrics[(task, variant)]["runtime_fold"] = (
                metrics[(task, variant)]["runtime"] / full_runtime
            )
            metrics[(task, variant)]["expanded_nodes_fold"] = (
                metrics[(task, variant)]["expanded_nodes"] / full_nodes
            )
    return metrics


def load_official_success() -> dict[tuple[str, str], dict[str, float]]:
    rows = read_tsv(OFFICIAL_TSV)
    summary: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        key = (row["task"], row["variant"])
        slot = summary.setdefault(key, {"success": 0.0, "total": 0.0})
        slot["total"] += 1
        if row["status"] == "success":
            slot["success"] += 1

    for key, slot in summary.items():
        total = slot["total"]
        slot["success_rate"] = slot["success"] / total if total else 0.0
    return summary


def write_plot_data_tsv(
    pilot: dict[tuple[str, str], dict[str, float]],
    official: dict[tuple[str, str], dict[str, float]],
) -> None:
    fieldnames = [
        "task", "task_label", "variant", "variant_label",
        "pilot_start_count", "pilot_improvement", "pilot_success_rate",
        "pilot_runtime", "pilot_runtime_fold_vs_full",
        "pilot_expanded_nodes", "pilot_expanded_nodes_fold_vs_full",
        "official_successes", "official_total", "official_success_rate",
    ]
    with OUTPUT_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for task in TASK_ORDER:
            for variant in VARIANT_ORDER:
                pilot_row = pilot[(task, variant)]
                official_row = official.get((task, variant), {})
                writer.writerow({
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "pilot_start_count": f"{pilot_row['start_count']:.0f}",
                    "pilot_improvement": f"{pilot_row['improvement']:.6f}",
                    "pilot_success_rate": f"{pilot_row['success_rate']:.6f}",
                    "pilot_runtime": f"{pilot_row['runtime']:.6f}",
                    "pilot_runtime_fold_vs_full": f"{pilot_row['runtime_fold']:.6f}",
                    "pilot_expanded_nodes": f"{pilot_row['expanded_nodes']:.6f}",
                    "pilot_expanded_nodes_fold_vs_full": f"{pilot_row['expanded_nodes_fold']:.6f}",
                    "official_successes": int(official_row.get("success", 0)),
                    "official_total": int(official_row.get("total", 0)),
                    "official_success_rate": (
                        f"{official_row['success_rate']:.6f}"
                        if "success_rate" in official_row else "NA"
                    ),
                })


# ── Plotting helpers ──

def _draw_panel(
    ax,
    pilot: dict[tuple[str, str], dict[str, float]],
    metric_key: str,
    *,
    ylabel: str,
    title: str,
    log_scale: bool = False,
    fmt: str = "{:.2f}",
    ylim: tuple[float, float] | None = None,
) -> None:
    """Draw a panel with X = tasks (LUMO/HOMO), grouped bars = variants (colored)."""
    n_tasks = len(TASK_ORDER)
    n_variants = len(VARIANT_ORDER)
    width = 0.18
    # Center the group of bars around each task position
    total_group_width = n_variants * width + (n_variants - 1) * 0.03
    offsets = [
        -total_group_width / 2 + i * (width + 0.03) + width / 2
        for i in range(n_variants)
    ]

    ax.set_facecolor(FIG_BG)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("#444444")
    ax.tick_params(colors=TEXT_PRIMARY, direction="out")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, color=GRID_COLOR, alpha=0.7)

    group_centers = list(range(n_tasks))

    for v_idx, variant in enumerate(VARIANT_ORDER):
        style = VARIANT_STYLE[variant]
        xs = [c + offsets[v_idx] for c in group_centers]
        values = [pilot[(task, variant)][metric_key] for task in TASK_ORDER]

        bars = ax.bar(
            xs,
            values,
            width=width,
            color=style["fill"],
            edgecolor=style["edge"],
            linewidth=1.4,
            zorder=3,
        )

        # Bar labels
        for bar, value in zip(bars, values):
            x = bar.get_x() + bar.get_width() / 2
            label_text = fmt.format(value)
            if log_scale:
                y = value * 1.15
            else:
                y = value + (ylim[1] - ylim[0]) * 0.02 if ylim else value * 1.02
            ax.text(
                x, y, label_text,
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold",
                color=style["text"],
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [TASK_LABELS[t] for t in TASK_ORDER],
        fontsize=12, fontweight="bold", color=TEXT_PRIMARY,
    )
    ax.set_ylabel(ylabel, fontsize=11.5, color=TEXT_PRIMARY)
    ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT_PRIMARY, pad=10)

    if log_scale:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pilot = load_pilot_metrics()
    official = load_official_success()
    write_plot_data_tsv(pilot, official)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "figure.facecolor": FIG_BG,
        "savefig.facecolor": FIG_BG,
        "savefig.edgecolor": FIG_BG,
    })

    # 1×3 横排: (a) Avg improvement | (b) Runtime fold | (c) Expanded nodes fold
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14, 5.5), facecolor=FIG_BG)

    _draw_panel(
        ax_a, pilot, "improvement",
        ylabel="Average improvement",
        title="(a) Avg. improvement",
        ylim=(0, 2.5),
        fmt="{:.2f}",
    )

    _draw_panel(
        ax_b, pilot, "runtime_fold",
        ylabel="Relative runtime (×full, log)",
        title="(b) Runtime (fold vs full)",
        log_scale=True,
        ylim=(0.5, 300),
        fmt="{:.1f}",
    )

    _draw_panel(
        ax_c, pilot, "expanded_nodes_fold",
        ylabel="Relative expanded nodes (×full, log)",
        title="(c) Expanded nodes (fold vs full)",
        log_scale=True,
        ylim=(0.5, 40),
        fmt="{:.1f}",
    )

    # ── Bottom legend: one entry per variant ──
    legend_handles = [
        Patch(
            facecolor=VARIANT_STYLE[v]["fill"],
            edgecolor=VARIANT_STYLE[v]["edge"],
            linewidth=1.3,
            label=VARIANT_LABELS[v],
        )
        for v in VARIANT_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(VARIANT_ORDER),
        frameon=True,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=11,
        fancybox=True,
        borderpad=0.6,
        columnspacing=1.8,
        handlelength=1.6,
    )

    plt.tight_layout(rect=[0.01, 0.08, 0.99, 0.98], w_pad=2.5)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(OUTPUT_PNG)
    print(OUTPUT_PDF)
    print(OUTPUT_TSV)


if __name__ == "__main__":
    main()
