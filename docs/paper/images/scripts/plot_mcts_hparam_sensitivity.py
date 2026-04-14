#!/usr/bin/env python3
"""Plot the canonical v1 MCTS hyperparameter sensitivity figure for the paper.

Current paper default:
    - use the v1 `LUMO(D)` sweep as the main figure source
    - output to `mcts_hparam_sensitivity-v1.pdf/png`

Design note:
    - keep the three subplots visually consistent
    - each subplot shows the same metric set: optimization gain, success rate, IntDiv

The newer v2 cross-task figures are kept in `images/` as archived alternatives,
but are not the current paper default.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
V1_CSV = HERE.parent.parent / "other.md" / "sweep" / "v1" / "mcts_hparam_sensitivity_lumo_down.csv"
OUT_PDF = HERE.parent / "mcts_hparam_sensitivity-v1.pdf"
OUT_PNG = HERE.parent / "mcts_hparam_sensitivity-v1.png"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.dpi": 150,
    }
)


C_GAIN = "#1f77b4"
C_SUCCESS = "#7f7f7f"
C_INTDIV = "#2ca02c"


def _load() -> pd.DataFrame:
    return pd.read_csv(V1_CSV)


def _subset(df: pd.DataFrame, param: str) -> pd.DataFrame:
    part = df[df["sweep_param"] == param].copy()
    part["optimization_gain"] = -part["average_improvement"]
    return part.sort_values("param_value")


def _despine(ax, right: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(right)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)


def _style_secondary_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="y", colors="black", labelcolor="black")
    ax.set_ylabel("Success rate (%)", color="black")


def _plot_consistent_panel(ax, data: pd.DataFrame, x_col: str, xlabel: str, title: str) -> None:
    x = data[x_col].to_numpy()
    line_gain, = ax.plot(
        x,
        data["optimization_gain"],
        "o-",
        color=C_GAIN,
        lw=2.0,
        ms=5,
        label="Optimization gain",
        zorder=3,
    )
    line_intdiv, = ax.plot(
        x,
        data["IntDiv"],
        "s--",
        color=C_INTDIV,
        lw=1.5,
        ms=5,
        label="IntDiv",
        zorder=2,
    )
    ax2 = ax.twinx()
    line_success, = ax2.plot(
        x,
        data["success_rate"],
        "^:",
        color=C_SUCCESS,
        lw=1.5,
        ms=5,
        label="Success rate (%)",
        zorder=2,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Optimization gain / IntDiv")
    ax.set_xticks(x)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", ls=":", alpha=0.35)
    _despine(ax, right=True)
    _style_secondary_axis(ax2)
    ax.legend(handles=[line_gain, line_intdiv, line_success], loc="best", framealpha=0.92)


def main() -> None:
    df = _load()
    budget = _subset(df, "num_simulations")
    explore = _subset(df, "exploration_weight")
    depth = _subset(df, "max_depth")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)

    _plot_consistent_panel(
        axes[0],
        budget,
        x_col="param_value",
        xlabel="num_simulations",
        title="(a) num_simulations",
    )
    _plot_consistent_panel(
        axes[1],
        explore,
        x_col="param_value",
        xlabel="exploration_weight",
        title="(b) exploration_weight",
    )
    _plot_consistent_panel(
        axes[2],
        depth,
        x_col="param_value",
        xlabel="max_depth",
        title="(c) max_depth",
    )

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_PDF}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
