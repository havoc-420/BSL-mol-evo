#!/usr/bin/env bash
set -euo pipefail

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "conda not found"
  exit 127
fi

conda activate mol-ofo
cd /home/ubuntu/mol_opt/mol-ofo

MODEL_DIR="/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200"
MODEL_PATH="$MODEL_DIR/last.pth"
CONFIG_PATH="/home/ubuntu/mol_opt/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml"
HOLDOUT_CSV="/home/ubuntu/mol_opt/mol-ofo/mol_evo/dataset/eval-data/plan_a_a0/lumo_plan_a_holdout50_start0.csv"
TREE_DIR="/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/evo-mo/batch_optimization_20260410_010348"
PIPE_ROOT="/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/plan_a_a1_small_fixed"
RL_DATA_JSON="/home/ubuntu/mol_opt/mol-ofo/mol_evo/dataset/rl_demo/lumo_plan_a_a1_small_bc_transitions_fixed_20260411.json"
BC_OUT_ROOT="/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/lumo_plan_a_a1_small_bc_fixed"
EVAL_OUT_DIR="/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/plan_a_a1_small_fixed_eval"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PIPE_LOG="$PIPE_ROOT/pipeline_${TIMESTAMP}.log"
LATEST_BC_PTR="$PIPE_ROOT/latest_bc_dir.txt"
LATEST_EVAL_PTR="$PIPE_ROOT/latest_eval_dir.txt"

mkdir -p "$PIPE_ROOT" "$EVAL_OUT_DIR"
exec > >(tee -a "$PIPE_LOG") 2>&1

printf 'START_TIME=%s\n' "$(date '+%F %T')"
printf 'PIPE_LOG=%s\n' "$PIPE_LOG"
printf 'TREE_DIR=%s\n' "$TREE_DIR"
printf 'HOLDOUT_CSV=%s\n' "$HOLDOUT_CSV"

python -m mol_evo.dataset.export_rl_demo_transitions \
  --input-dir "$TREE_DIR" \
  --output-json "$RL_DATA_JSON" \
  --target-property lumo \
  --direction decrease \
  --max-depth 3 \
  --max-trees 50

python mol_evo/train_bc_pretrain.py \
  --data-json "$RL_DATA_JSON" \
  --output-dir "$BC_OUT_ROOT" \
  --direction decrease \
  --epochs 100 \
  --batch-size 64 \
  --max-depth 3 \
  --device auto

LATEST_BC_DIR=$(find "$BC_OUT_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'bc_*' | sort | tail -n 1)
printf '%s\n' "$LATEST_BC_DIR" > "$LATEST_BC_PTR"
printf 'LATEST_BC_DIR=%s\n' "$LATEST_BC_DIR"

python mol_evo/scripts/eval_astar_rl_holdout.py \
  --input-csv "$HOLDOUT_CSV" \
  --output-dir "$EVAL_OUT_DIR" \
  --label a1_small_fixed_bc_holdout50_budget200_pref50 \
  --model-dir "$MODEL_DIR" \
  --model-path "$MODEL_PATH" \
  --config-file "$CONFIG_PATH" \
  --policy-path "$LATEST_BC_DIR/policy_best.pth" \
  --value-path "$LATEST_BC_DIR/value_best.pth" \
  --target-property lumo \
  --optimization-mode sub \
  --direction decrease \
  --max-depth 3 \
  --max-branching 8 \
  --open-set-budget 200 \
  --top-n-prefilter 50

printf '%s\n' "$EVAL_OUT_DIR" > "$LATEST_EVAL_PTR"
printf 'LATEST_EVAL_DIR=%s\n' "$EVAL_OUT_DIR"
printf 'END_TIME=%s\n' "$(date '+%F %T')"
