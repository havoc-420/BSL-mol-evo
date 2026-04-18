#!/usr/bin/env python3
"""Part 1 — BFS vs MCTS-full method-level comparison (main experiment data).
Dual-Y: left=Avg. Improvement, right=Success Rate, both as bars.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

base = Path(__file__).parent

# ── Style constants (matching mcts_ablation_summary) ──────────────────
FIG_BG = "#FFFFFF"
TEXT_PRIMARY = "#222630"
TEXT_MUTED = "#5E6470"
GRID_COLOR = "#D0D0D0"

METHOD_ORDER = ["OFO-BFS", "OFO-MCTS"]
METHOD_LABELS = {
    "OFO-BFS": "BFS",
    "OFO-MCTS": "MCTS",
}
METHOD_STYLE = {
    "OFO-BFS":     {"fill": "#C6DDF0", "edge": "#4E75A8", "text": "#355E96"},
    "OFO-MCTS":    {"fill": "#D4E8D0", "edge": "#4E8A47", "text": "#3C7535"},
}

# ── Data ──────────────────────────────────────────────────────────────
csv_path = base / "part1_data.csv"
with open(csv_path) as f:
    rows = list(csv.DictReader(f))

tasks = ["HOMO(D)", "LUMO(U)"]

# ── Plot ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.facecolor": FIG_BG,
    "savefig.facecolor": FIG_BG,
    "savefig.edgecolor": FIG_BG,
})

fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor=FIG_BG)

for ax, task in zip(axes, tasks):
    sub = [r for r in rows if r["task"] == task]
    methods = [r["method"] for r in sub]
    improvements = [float(r["avg_improvement"]) for r in sub]
    successes = [float(r["success_rate"]) for r in sub]
    folds = [float(r["improvement_fold_vs_bfs"]) for r in sub]

    x = np.arange(len(sub))
    bar_w = 0.35          # each bar width
    gap = 0.08            # gap between imp and sr bars within a method
    group_w = bar_w * 2 + gap  # total width per method

    ax.set_facecolor(FIG_BG)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("#444444")
    ax.tick_params(colors=TEXT_PRIMARY, direction="out")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, color=GRID_COLOR, alpha=0.7)

    # ── Left Y axis: Average Improvement bars ────────────────────────
    for i, (m, v) in enumerate(zip(methods, improvements)):
        style = METHOD_STYLE[m]
        left_x = x[i] - bar_w / 2 - gap / 2
        ax.bar(left_x, v, bar_w,
               color=style["fill"], edgecolor=style["edge"],
               linewidth=1.4, zorder=3)
        ax.text(left_x, v + 0.04, f"{v:.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=style["text"])

    # ── Right Y axis: Success Rate bars ──────────────────────────────
    ax_r = ax.twinx()
    ax_r.set_facecolor("none")
    for i, (m, s, fold) in enumerate(zip(methods, successes, folds)):
        style = METHOD_STYLE[m]   # same color family
        right_x = x[i] + bar_w / 2 + gap / 2
        ax_r.bar(right_x, s, bar_w,
                 color=style["fill"], edgecolor=style["edge"],
                 linewidth=1.4, alpha=0.45, zorder=3, hatch="//")
        label = f"{s:.1f}%"
        if fold > 1.0:
            label += f"\n({fold:.2f}×)"
        ax_r.text(right_x, s + 3.5, label, ha="center", va="bottom",
                  fontsize=8, fontweight="bold", color=style["text"],
                  linespacing=1.05)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                       fontsize=11, fontweight="bold", color=TEXT_PRIMARY)
    ax.set_ylabel("Average Improvement", fontsize=11.5, color=TEXT_PRIMARY)
    ax_r.set_ylabel("Success Rate (%)", fontsize=11.5, color=TEXT_MUTED)
    ax_r.tick_params(axis="y", colors=TEXT_MUTED)
    ax_r.spines["right"].set_visible(True)
    ax_r.spines["right"].set_linewidth(0.8)
    ax_r.spines["right"].set_color("#444444")
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["left"].set_visible(False)
    ax_r.spines["bottom"].set_visible(False)
    ax.set_ylim(0, max(improvements) * 1.45)
    ax_r.set_ylim(0, 110)

    ax.set_title(task, fontsize=14, fontweight="bold", color=TEXT_PRIMARY, pad=10)

plt.tight_layout(rect=[0.01, 0.07, 0.99, 0.97], w_pad=2.5)

# ── Legend ─────────────────────────────────────────────────────────────
legend_handles = [
    Patch(facecolor=METHOD_STYLE[m]["fill"],
          edgecolor=METHOD_STYLE[m]["edge"],
          linewidth=1.3, label=f"{METHOD_LABELS[m]} (Imp)")
    for m in METHOD_ORDER if m in {r["method"] for r in rows}
]
legend_handles.append(
    Patch(facecolor="#DDDDDD", edgecolor="#888888", hatch="//",
          linewidth=1.3, label="SR (%)")
)
fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
           frameon=True, bbox_to_anchor=(0.5, 0.005), fontsize=10.5,
           fancybox=True, borderpad=0.6, columnspacing=1.8, handlelength=1.6)

# ── Save ──────────────────────────────────────────────────────────────
for ext in ["png", "pdf"]:
    fig.savefig(base / f"part1_bfs_vs_mcts.{ext}",
                dpi=300, bbox_inches="tight")
print(f"Saved {base}/part1_bfs_vs_mcts.png and .pdf")
