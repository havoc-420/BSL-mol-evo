#!/usr/bin/env python3
"""Plot the canonical v1 MCTS hyperparameter sensitivity figure for the paper.

Current paper default:
    - use the v1 `LUMO(D)` sweep as the main figure source
    - save `mcts_hparam_sensitivity-v1.pdf/png` to the local paper image dir
      and, when available, sync the same files to `whu-thesis/pages/images`

Design note:
    - keep the three subplots visually consistent
    - each subplot shows the same metric set: optimization gain and success rate

The newer v2 cross-task figures are kept in `images/` as archived alternatives,
but are not the current paper default.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


HERE = Path(__file__).resolve().parent
V1_CSV = HERE.parent.parent / "other.md" / "sweep" / "v1" / "mcts_hparam_sensitivity_lumo_down.csv"
LOCAL_IMAGE_DIR = HERE.parent
THESIS_IMAGE_DIR = HERE.parents[5] / "whu-thesis" / "pages" / "images"
OUTPUT_DIRS = [LOCAL_IMAGE_DIR]
if THESIS_IMAGE_DIR.exists():
    OUTPUT_DIRS.append(THESIS_IMAGE_DIR)


def _pick_cjk_font() -> str:
    candidates = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Songti SC",
        "Heiti SC",
        "STHeiti",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"


CJK_FONT = _pick_cjk_font()
X_LABEL_FONT = font_manager.FontProperties(family=CJK_FONT, weight="bold", size=12)


plt.rcParams.update(
    {
        "font.family": [CJK_FONT, "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.dpi": 150,
        "axes.unicode_minus": False,
    }
)


C_GAIN = "#1f77b4"
C_SUCCESS = "#7f7f7f"
PARAM_LABELS = {
    "num_simulations": "搜索预算",
    "max_depth": "最大深度",
    "pruning_patience": "剪枝耐心值",
    "exploration_weight": "探索系数",
    "exploration_constant": "探索系数",
}


def _load() -> pd.DataFrame:
    return pd.read_csv(V1_CSV)


def _subset(df: pd.DataFrame, param: str) -> pd.DataFrame:
    part = df[df["sweep_param"] == param].copy()
    part["optimization_gain"] = -part["average_improvement"]
    sorted_index = sorted(part.index, key=lambda idx: float(part.at[idx, "param_value"]))
    return part.loc[sorted_index]


def _format_ticklabels(values) -> list[str]:
    labels = []
    for value in values:
        if float(value).is_integer():
            labels.append(str(int(value)))
        else:
            labels.append(f"{value:.1f}")
    return labels


def _despine(ax, right: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(right)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)


def _style_secondary_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="y", colors="black", labelcolor="black")
    ax.set_ylabel("Success rate (%)", color="black")


def _plot_consistent_panel(ax, data: pd.DataFrame, x_col: str, xlabel: str) -> None:
    x = data[x_col].to_numpy()
    positions = range(len(x))

    line_gain, = ax.plot(
        positions,
        data["optimization_gain"],
        "o-",
        color=C_GAIN,
        lw=2.0,
        ms=5,
        label="Optimization gain",
        zorder=3,
    )
    ax2 = ax.twinx()
    line_success, = ax2.plot(
        positions,
        data["success_rate"],
        "^:",
        color=C_SUCCESS,
        lw=1.5,
        ms=5,
        label="Success rate (%)",
        zorder=2,
    )

    ax.set_xlabel(xlabel, fontproperties=X_LABEL_FONT)
    ax.set_ylabel("Optimization gain")
    ax.set_xticks(list(positions), _format_ticklabels(x))
    ax.set_xlim(-0.25, len(x) - 0.75)
    ax.grid(axis="y", ls=":", alpha=0.35)
    _despine(ax, right=True)
    _style_secondary_axis(ax2)
    ax.legend(handles=[line_gain, line_success], loc="best", framealpha=0.92)


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
        xlabel=f"(a) {PARAM_LABELS['num_simulations']}",
    )
    _plot_consistent_panel(
        axes[1],
        explore,
        x_col="param_value",
        xlabel=f"(b) {PARAM_LABELS['exploration_weight']}",
    )
    _plot_consistent_panel(
        axes[2],
        depth,
        x_col="param_value",
        xlabel=f"(c) {PARAM_LABELS['max_depth']}",
    )

    saved_paths = []
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = output_dir / "mcts_hparam_sensitivity-v1.pdf"
        out_png = output_dir / "mcts_hparam_sensitivity-v1.png"
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, bbox_inches="tight")
        saved_paths.extend([out_pdf, out_png])

    plt.close(fig)

    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
