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
TRAIN_POOL_CSV="/home/ubuntu/mol_opt/mol-ofo/mol_evo/dataset/eval-data/plan_a_a0/lumo_plan_a_train_pool_excluding_holdout.csv"
PIPE_ROOT="/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/plan_a_a1_small"
RL_DATA_JSON="/home/ubuntu/mol_opt/mol-ofo/mol_evo/dataset/rl_demo/lumo_plan_a_a1_small_bc_transitions.json"
BC_OUT_ROOT="/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/lumo_plan_a_a1_small_bc"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PIPE_LOG="$PIPE_ROOT/pipeline_${TIMESTAMP}.log"
LATEST_BFS_PTR="$PIPE_ROOT/latest_bfs_dir.txt"
LATEST_BC_PTR="$PIPE_ROOT/latest_bc_dir.txt"

mkdir -p "$PIPE_ROOT"
exec > >(tee -a "$PIPE_LOG") 2>&1

printf 'START_TIME=%s\n' "$(date '+%F %T')"
printf 'PIPE_LOG=%s\n' "$PIPE_LOG"
printf 'TRAIN_POOL_CSV=%s\n' "$TRAIN_POOL_CSV"

python -m mol_evo.scripts.batch_optimizer \
  --input-csv "$TRAIN_POOL_CSV" \
  --model-path "$MODEL_PATH" \
  --model-dir "$MODEL_DIR" \
  --config-file "$CONFIG_PATH" \
  --target-property lumo \
  --optimization-mode sub \
  --search-mode bfs \
  --direction decrease \
  --max-depth 3 \
  --max-branching 8 \
  --start-index 0 \
  --end-index 50

LATEST_BFS_DIR=$(find /home/ubuntu/mol_opt/mol-ofo/mol_evo/output/evo-mo -maxdepth 1 -mindepth 1 -type d -name 'batch_optimization_*' | sort | tail -n 1)
printf '%s\n' "$LATEST_BFS_DIR" > "$LATEST_BFS_PTR"
printf 'LATEST_BFS_DIR=%s\n' "$LATEST_BFS_DIR"

python -m mol_evo.dataset.export_rl_demo_transitions \
  --input-dir "$LATEST_BFS_DIR" \
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
printf 'END_TIME=%s\n' "$(date '+%F %T')"
