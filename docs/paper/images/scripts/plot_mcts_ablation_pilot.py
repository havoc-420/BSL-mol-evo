#!/usr/bin/env python3
"""Plot pilot-stage MCTS ablation figures for the paper.

This script reads the manually curated pilot ablation CSVs and exports two figures:
1. A raw overview figure with four metrics.
2. A relative-to-full figure highlighting effect and cost differences.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "other.md" / "ablations" / "csv"
RAW_CSV = DATA_DIR / "mcts_ablation_pilot_metrics.csv"
RELATIVE_CSV = DATA_DIR / "mcts_ablation_pilot_relative.csv"
OUT_RAW_PDF = HERE.parent / "mcts_ablation_pilot_overview.pdf"
OUT_RAW_PNG = HERE.parent / "mcts_ablation_pilot_overview.png"
OUT_REL_PDF = HERE.parent / "mcts_ablation_pilot_relative.pdf"
OUT_REL_PNG = HERE.parent / "mcts_ablation_pilot_relative.png"

TASK_ORDER = ["LUMO(U)", "HOMO(D)"]
VARIANT_ORDER = ["full", "w/o prior", "w/o leaf value", "random top-b"]
FOCUS_VARIANT_ORDER = ["w/o prior", "w/o leaf value", "random top-b"]
TASK_STYLE = {
    "LUMO(U)": {"color": "#1f77b4", "hatch": ""},
    "HOMO(D)": {"color": "#ff7f0e", "hatch": "//"},
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.dpi": 150,
    }
)


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["task_label"] = pd.Categorical(df["task_label"], categories=TASK_ORDER, ordered=True)
    if "variant_label" in df.columns:
        df["variant_label"] = pd.Categorical(df["variant_label"], categories=VARIANT_ORDER, ordered=True)
    return df.sort_values(["variant_label", "task_label"]).reset_index(drop=True)


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _pivot_metric(df: pd.DataFrame, value_col: str, variant_order: list[str]) -> pd.DataFrame:
    pivot = df.pivot(index="variant_label", columns="task_label", values=value_col)
    pivot = pivot.reindex(index=variant_order, columns=TASK_ORDER)
    return pivot


def _format_value(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    if abs_value >= 1:
        return f"{value:.2f}"
    return f"{value:.2f}"


def _format_ratio(value: float) -> str:
    if value >= 10:
        return f"{value:.1f}×"
    return f"{value:.2f}×"


def _annotate_bars(ax, bars, formatter, log_scale: bool = False) -> None:
    ymin, ymax = ax.get_ylim()
    y_span = ymax - ymin if ymax > ymin else 1.0
    for bar in bars:
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        if log_scale:
            y = height * 1.08
            va = "bottom"
        else:
            offset = y_span * 0.025
            if height >= 0:
                y = height + offset
                va = "bottom"
            else:
                y = height - offset
                va = "top"
        ax.text(x, y, formatter(height), ha="center", va=va, fontsize=8)


def _plot_grouped_bars(
    ax,
    pivot: pd.DataFrame,
    title: str,
    ylabel: str,
    formatter,
    *,
    log_scale: bool = False,
    baseline: float | None = None,
):
    x = list(range(len(pivot.index)))
    width = 0.34
    legend_handles = []

    for idx, task in enumerate(TASK_ORDER):
        offset = (idx - 0.5) * width
        values = pivot[task].tolist()
        bars = ax.bar(
            [item + offset for item in x],
            values,
            width=width,
            color=TASK_STYLE[task]["color"],
            hatch=TASK_STYLE[task]["hatch"],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.92,
            label=task,
            zorder=3,
        )
        _annotate_bars(ax, bars, formatter, log_scale=log_scale)
        legend_handles.append(bars)

    if baseline is not None:
        ax.axhline(baseline, color="black", lw=1.0, ls=":", alpha=0.7, zorder=1)

    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.grid(axis="y", ls=":", alpha=0.35, zorder=0)
    if log_scale:
        ax.set_yscale("log")
    _despine(ax)
    return legend_handles


def _add_shared_legend(fig, handles) -> None:
    fig.legend(
        [handle[0] for handle in handles],
        TASK_ORDER,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        frameon=False,
    )


def _plot_overview(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))
    axes = axes.ravel()

    handles = _plot_grouped_bars(
        axes[0],
        _pivot_metric(df, "avg_improvement", VARIANT_ORDER),
        title="(a) Avg. improvement",
        ylabel="Average improvement",
        formatter=_format_value,
    )
    _plot_grouped_bars(
        axes[1],
        _pivot_metric(df, "success_rate", VARIANT_ORDER),
        title="(b) Success rate",
        ylabel="Success rate",
        formatter=lambda value: f"{value * 100:.0f}%",
        baseline=1.0,
    )
    _plot_grouped_bars(
        axes[2],
        _pivot_metric(df, "avg_runtime", VARIANT_ORDER),
        title="(c) Runtime",
        ylabel="Average runtime (log scale)",
        formatter=_format_value,
        log_scale=True,
    )
    _plot_grouped_bars(
        axes[3],
        _pivot_metric(df, "avg_expanded_nodes", VARIANT_ORDER),
        title="(d) Expanded nodes",
        ylabel="Average expanded nodes (log scale)",
        formatter=_format_value,
        log_scale=True,
    )

    _add_shared_legend(fig, handles)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT_RAW_PDF, bbox_inches="tight")
    fig.savefig(OUT_RAW_PNG, bbox_inches="tight")
    plt.close(fig)


def _plot_relative(df: pd.DataFrame) -> None:
    focus_df = df[df["variant_label"].isin(FOCUS_VARIANT_ORDER)].copy()
    focus_df["variant_label"] = pd.Categorical(
        focus_df["variant_label"], categories=FOCUS_VARIANT_ORDER, ordered=True
    )
    focus_df = focus_df.sort_values(["variant_label", "task_label"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))

    handles = _plot_grouped_bars(
        axes[0],
        _pivot_metric(focus_df, "delta_improvement_vs_full", FOCUS_VARIANT_ORDER),
        title="(a) Δ improvement vs. full",
        ylabel="Δ average improvement",
        formatter=_format_value,
        baseline=0.0,
    )
    _plot_grouped_bars(
        axes[1],
        _pivot_metric(focus_df, "runtime_ratio_vs_full", FOCUS_VARIANT_ORDER),
        title="(b) Runtime ratio vs. full",
        ylabel="Runtime ratio (log scale)",
        formatter=_format_ratio,
        log_scale=True,
        baseline=1.0,
    )
    _plot_grouped_bars(
        axes[2],
        _pivot_metric(focus_df, "expanded_nodes_ratio_vs_full", FOCUS_VARIANT_ORDER),
        title="(c) Node ratio vs. full",
        ylabel="Expanded-node ratio (log scale)",
        formatter=_format_ratio,
        log_scale=True,
        baseline=1.0,
    )

    _add_shared_legend(fig, handles)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT_REL_PDF, bbox_inches="tight")
    fig.savefig(OUT_REL_PNG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    raw_df = _load_csv(RAW_CSV)
    relative_df = _load_csv(RELATIVE_CSV)

    _plot_overview(raw_df)
    _plot_relative(relative_df)

    print(f"Saved: {OUT_RAW_PDF}")
    print(f"Saved: {OUT_RAW_PNG}")
    print(f"Saved: {OUT_REL_PDF}")
    print(f"Saved: {OUT_REL_PNG}")


if __name__ == "__main__":
    main()
