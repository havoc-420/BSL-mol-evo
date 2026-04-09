#!/usr/bin/env bash
set -euo pipefail

OUT_BASE="/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl"
RUN_ID="lumo_eval_grid50_parallel_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$OUT_BASE/$RUN_ID"
LATEST_PTR="$OUT_BASE/latest_lumo_eval_grid50_run.txt"
SUMMARY_TSV="$RUN_ROOT/summary.tsv"
RUNNER="$OUT_BASE/run_lumo_eval_grid50_method.sh"

mkdir -p "$RUN_ROOT"
echo "$RUN_ROOT" > "$LATEST_PTR"
echo -e "method\tbudget\tprefilter\tmols\tnonempty_topk\ttop1_mean\ttop1_median\tactual_expansions_mean\tofo_calls_mean\toutput_json" > "$SUMMARY_TSV"

if tmux has-session -t lumo-grid50 2>/dev/null; then
  tmux kill-session -t lumo-grid50
fi
if tmux has-session -t lumo-grid50-bc 2>/dev/null; then
  tmux kill-session -t lumo-grid50-bc
fi
if tmux has-session -t lumo-grid50-rl 2>/dev/null; then
  tmux kill-session -t lumo-grid50-rl
fi

pkill -f "python -m mol_evo.scripts.batch_optimizer --input-csv /home/ubuntu/mol_opt/mol-ofo/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv" 2>/dev/null || true
sleep 2

chmod +x "$RUNNER"
tmux new-session -d -s lumo-grid50-bc "$RUNNER bc $RUN_ROOT"
tmux new-session -d -s lumo-grid50-rl "$RUNNER rl $RUN_ROOT"

echo "RUN_ROOT=$RUN_ROOT"
echo "START_TIME=$(date '+%F %T')"
echo "SESSIONS=lumo-grid50-bc,lumo-grid50-rl"
