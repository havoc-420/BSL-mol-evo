#!/usr/bin/env python3
"""Part 2 — MCTS internal ablation: OFO component contributions."""

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

VARIANT_ORDER = ["full", "wo_prior", "wo_leaf_value", "random_topb"]
VARIANT_LABELS = {
    "full": "full",
    "wo_prior": "w/o prior",
    "wo_leaf_value": "w/o leaf value",
    "random_topb": "random top-k",
}
VARIANT_STYLE = {
    "full":           {"fill": "#C6DDF0", "edge": "#4E75A8", "text": "#355E96"},
    "wo_prior":       {"fill": "#E9D8C5", "edge": "#C6692C", "text": "#B95D1F"},
    "wo_leaf_value":  {"fill": "#D4E8D0", "edge": "#4E8A47", "text": "#3C7535"},
    "random_topb":    {"fill": "#E3D1E8", "edge": "#8B5AA0", "text": "#7A4990"},
}

TASK_ORDER = ["homo_down", "lumo_up"]
TASK_LABELS = {"homo_down": "HOMO(D)", "lumo_up": "LUMO(U)"}
bfs_ref = {"homo_down": 0.9125, "lumo_up": 0.4726}

# ── Helpers ───────────────────────────────────────────────────────────
def read_csv(name):
    with open(base / name) as f:
        return list(csv.DictReader(f))

def lookup(rows, task, variant, key):
    for r in rows:
        if r["task"] == task and r["variant"] == variant:
            v = r[key]
            return None if v == "NA" else float(v)
    return None

# ── Data ──────────────────────────────────────────────────────────────
imp_data = read_csv("part2_data_improvement.csv")
eff_data = read_csv("part2_data_efficiency.csv")

# ── Plot setup ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.facecolor": FIG_BG,
    "savefig.facecolor": FIG_BG,
    "savefig.edgecolor": FIG_BG,
})

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=FIG_BG)

# ── Panel A: Pilot Truth Improvement ──────────────────────────────────
n_tasks = len(TASK_ORDER)
n_variants = len(VARIANT_ORDER)
width = 0.18
total_group_width = n_variants * width + (n_variants - 1) * 0.03
offsets = [-total_group_width / 2 + i * (width + 0.03) + width / 2
           for i in range(n_variants)]

ax_a.set_facecolor(FIG_BG)
for spine in ("top", "right"):
    ax_a.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax_a.spines[spine].set_linewidth(0.8)
    ax_a.spines[spine].set_color("#444444")
ax_a.tick_params(colors=TEXT_PRIMARY, direction="out")
ax_a.set_axisbelow(True)
ax_a.yaxis.grid(True, linestyle="-", linewidth=0.6, color=GRID_COLOR, alpha=0.7)

group_centers = list(range(n_tasks))

for v_idx, variant in enumerate(VARIANT_ORDER):
    style = VARIANT_STYLE[variant]
    xs = [c + offsets[v_idx] for c in group_centers]
    vals = [lookup(imp_data, t, variant, "avg_improvement") for t in TASK_ORDER]

    bars = ax_a.bar(xs, vals, width=width, color=style["fill"],
                    edgecolor=style["edge"], linewidth=1.4, zorder=3,
                    label=VARIANT_LABELS[variant])
    for bar, v in zip(bars, vals):
        ax_a.text(bar.get_x() + bar.get_width() / 2, v + 0.04, f"{v:.2f}",
                  ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                  color=style["text"])

# BFS reference lines
for i, t in enumerate(TASK_ORDER):
    ref = bfs_ref[t]
    ax_a.plot([group_centers[i] - 0.4, group_centers[i] + 0.4], [ref, ref],
              color="#888888", linestyle="--", linewidth=1.2, zorder=2)
    ax_a.text(group_centers[i], ref - 0.1, f"BFS ref: {ref:.2f}",
              ha="center", va="top", fontsize=7, color="#888888", style="italic")

ax_a.set_xticks(group_centers)
ax_a.set_xticklabels([TASK_LABELS[t] for t in TASK_ORDER],
                      fontsize=12, fontweight="bold", color=TEXT_PRIMARY)
ax_a.set_ylabel("Average Improvement", fontsize=11.5, color=TEXT_PRIMARY)
ax_a.set_title("(a) Avg. improvement", fontsize=14, fontweight="bold",
               color=TEXT_PRIMARY, pad=10)
ax_a.set_ylim(0, 2.5)

# ── Panel B: Efficiency (log scale) ──────────────────────────────────
ax_b.set_facecolor(FIG_BG)
for spine in ("top", "right"):
    ax_b.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax_b.spines[spine].set_linewidth(0.8)
    ax_b.spines[spine].set_color("#444444")
ax_b.tick_params(colors=TEXT_PRIMARY, direction="out")
ax_b.set_axisbelow(True)
ax_b.yaxis.grid(True, linestyle="-", linewidth=0.6, color=GRID_COLOR, alpha=0.7)

eff_variants = ["wo_prior", "wo_leaf_value", "random_topb"]
metrics = [("runtime_fold_vs_full", "Runtime"), ("expanded_nodes_fold_vs_full", "Nodes")]

# Use the same grouped-bar layout as Panel A:
# x groups = [HOMO(D)-RT, HOMO(D)-Nodes, LUMO(U)-RT, LUMO(U)-Nodes]
n_eff_groups = len(TASK_ORDER) * len(metrics)  # 4 groups
n_eff_vars = len(eff_variants)  # 3 bars per group
eff_width = 0.18
eff_total = n_eff_vars * eff_width + (n_eff_vars - 1) * 0.03
eff_offsets = [-eff_total / 2 + i * (eff_width + 0.03) + eff_width / 2
               for i in range(n_eff_vars)]

# Group positions: 0, 1 for HOMO(D), then 2.2, 3.2 for LUMO(U) (gap between tasks)
group_positions = []
x_cursor = 0
for t_idx in range(len(TASK_ORDER)):
    for m_idx in range(len(metrics)):
        group_positions.append(x_cursor)
        x_cursor += 1.0
    x_cursor += 0.4  # extra gap between tasks

group_labels = []
for t in TASK_ORDER:
    for _, mname in metrics:
        group_labels.append(f"{TASK_LABELS[t]}\n{mname}")

for g_idx, (task, (metric_key, _)) in enumerate(
        [(t, m) for t in TASK_ORDER for m in metrics]):
    cx = group_positions[g_idx]
    # Determine alpha: runtime=0.9, nodes=0.55
    alpha = 0.9 if "runtime" in metric_key else 0.55
    for v_idx, variant in enumerate(eff_variants):
        val = lookup(eff_data, task, variant, metric_key)
        if val is None:
            continue
        style = VARIANT_STYLE[variant]
        x_pos = cx + eff_offsets[v_idx]
        ax_b.bar(x_pos, val, eff_width, color=style["fill"],
                 edgecolor=style["edge"], linewidth=1.2, alpha=alpha, zorder=3)
        y_pos = val * 1.15
        ax_b.text(x_pos, y_pos, f"{val:.1f}",
                  ha="center", va="bottom", fontsize=7, fontweight="bold",
                  color=style["text"])

ax_b.set_xticks(group_positions)
ax_b.set_xticklabels(group_labels, fontsize=9, fontweight="bold", color=TEXT_PRIMARY)
ax_b.set_yscale("log")
ax_b.set_ylabel("Relative cost (×full, log)", fontsize=11.5, color=TEXT_PRIMARY)
ax_b.set_title("(b) Efficiency (fold vs full)", fontsize=14, fontweight="bold",
               color=TEXT_PRIMARY, pad=10)
ax_b.axhline(1.0, color="#888888", linewidth=0.8, linestyle="-", zorder=1)
ax_b.set_ylim(0.5, 300)

# ── Bottom legend: one entry per variant ──────────────────────────────
legend_handles = [
    Patch(facecolor=VARIANT_STYLE[v]["fill"],
          edgecolor=VARIANT_STYLE[v]["edge"],
          linewidth=1.3, label=VARIANT_LABELS[v])
    for v in VARIANT_ORDER
]
fig.legend(handles=legend_handles, loc="lower center", ncol=len(VARIANT_ORDER),
           frameon=True, bbox_to_anchor=(0.5, 0.005), fontsize=11,
           fancybox=True, borderpad=0.6, columnspacing=1.8, handlelength=1.6)

plt.tight_layout(rect=[0.01, 0.08, 0.99, 0.98], w_pad=2.5)

# ── Save ──────────────────────────────────────────────────────────────
for ext in ["png", "pdf"]:
    fig.savefig(base / f"part2_mcts_ablation.{ext}",
                dpi=300, bbox_inches="tight")
print(f"Saved {base}/part2_mcts_ablation.png and .pdf")
