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
HOLDOUT_CSV="/home/ubuntu/mol_opt/mol-ofo/mol_evo/dataset/eval-data/plan_a_a0/lumo_plan_a_holdout50_start0.csv"

PIPE_ROOT="/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/plan_a_mcts_expand_rl_seed11"
SEARCH_JSON="/home/ubuntu/mol_opt/mol-ofo/mol_evo/dataset/rl_demo/lumo_plan_a_mcts_expand_bc_transitions_seed11.json"
BC_OUT_ROOT="/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/lumo_plan_a_mcts_expand_bc"
BC_EVAL_OUT_DIR="/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/plan_a_mcts_expand_bc_eval"
RL_OUT_ROOT="/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/lumo_plan_a_mcts_expand_rl_seed11"
RL_EVAL_OUT_DIR="/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/plan_a_mcts_expand_rl_seed11_eval"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PIPE_LOG="$PIPE_ROOT/pipeline_${TIMESTAMP}.log"
LATEST_SEARCH_PTR="$PIPE_ROOT/latest_search_dir.txt"
LATEST_BC_PTR="$PIPE_ROOT/latest_bc_dir.txt"
LATEST_BC_EVAL_PTR="$PIPE_ROOT/latest_bc_eval_dir.txt"
LATEST_RL_PTR="$PIPE_ROOT/latest_rl_dir.txt"
LATEST_RL_EVAL_PTR="$PIPE_ROOT/latest_rl_eval_dir.txt"

NUM_SIMULATIONS="1000"
EXPLORATION_WEIGHT="2.5"
MAX_DEPTH="3"
MAX_BRANCHING="16"
PRUNING_PATIENCE="4"
LOGP_PATIENCE="5"
RL_EPISODES="300"
RL_SEED="11"
RL_PREFILTER="50"
RL_BUDGET="50"
CHECKPOINT_EVERY="25"

mkdir -p "$PIPE_ROOT" "$BC_EVAL_OUT_DIR" "$RL_EVAL_OUT_DIR"
exec > >(tee -a "$PIPE_LOG") 2>&1

printf 'START_TIME=%s\n' "$(date '+%F %T')"
printf 'PIPE_LOG=%s\n' "$PIPE_LOG"
printf 'TRAIN_POOL_CSV=%s\n' "$TRAIN_POOL_CSV"
printf 'HOLDOUT_CSV=%s\n' "$HOLDOUT_CSV"
printf 'SEARCH_MODE=mcts\n'
printf 'NUM_SIMULATIONS=%s\n' "$NUM_SIMULATIONS"
printf 'EXPLORATION_WEIGHT=%s\n' "$EXPLORATION_WEIGHT"
printf 'MAX_DEPTH=%s\n' "$MAX_DEPTH"
printf 'MAX_BRANCHING=%s\n' "$MAX_BRANCHING"
printf 'PRUNING_PATIENCE=%s\n' "$PRUNING_PATIENCE"
printf 'LOGP_PATIENCE=%s\n' "$LOGP_PATIENCE"
printf 'RL_EPISODES=%s\n' "$RL_EPISODES"
printf 'RL_SEED=%s\n' "$RL_SEED"

python -m mol_evo.scripts.batch_optimizer \
  --input-csv "$TRAIN_POOL_CSV" \
  --model-path "$MODEL_PATH" \
  --model-dir "$MODEL_DIR" \
  --config-file "$CONFIG_PATH" \
  --target-property lumo \
  --optimization-mode sub \
  --search-mode mcts \
  --num-simulations "$NUM_SIMULATIONS" \
  --exploration-weight "$EXPLORATION_WEIGHT" \
  --direction decrease \
  --max-depth "$MAX_DEPTH" \
  --max-branching "$MAX_BRANCHING" \
  --pruning-patience "$PRUNING_PATIENCE" \
  --logp-patience "$LOGP_PATIENCE" \
  --start-index 0

LATEST_SEARCH_DIR=$(find /home/ubuntu/mol_opt/mol-ofo/mol_evo/output/evo-mo -maxdepth 1 -mindepth 1 -type d -name 'batch_optimization_*' | sort | tail -n 1)
printf '%s\n' "$LATEST_SEARCH_DIR" > "$LATEST_SEARCH_PTR"
printf 'LATEST_SEARCH_DIR=%s\n' "$LATEST_SEARCH_DIR"

python -m mol_evo.dataset.export_rl_demo_transitions \
  --input-dir "$LATEST_SEARCH_DIR" \
  --output-json "$SEARCH_JSON" \
  --target-property lumo \
  --direction decrease \
  --max-depth "$MAX_DEPTH"

python mol_evo/train_bc_pretrain.py \
  --data-json "$SEARCH_JSON" \
  --output-dir "$BC_OUT_ROOT" \
  --direction decrease \
  --epochs 100 \
  --batch-size 64 \
  --max-depth "$MAX_DEPTH" \
  --device auto

LATEST_BC_DIR=$(find "$BC_OUT_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'bc_*' | sort | tail -n 1)
printf '%s\n' "$LATEST_BC_DIR" > "$LATEST_BC_PTR"
printf 'LATEST_BC_DIR=%s\n' "$LATEST_BC_DIR"

python mol_evo/scripts/eval_astar_rl_holdout.py \
  --input-csv "$HOLDOUT_CSV" \
  --output-dir "$BC_EVAL_OUT_DIR" \
  --label mcts_expand_bc_holdout50_budget50_pref50 \
  --model-dir "$MODEL_DIR" \
  --model-path "$MODEL_PATH" \
  --config-file "$CONFIG_PATH" \
  --policy-path "$LATEST_BC_DIR/policy_best.pth" \
  --value-path "$LATEST_BC_DIR/value_best.pth" \
  --target-property lumo \
  --optimization-mode sub \
  --direction decrease \
  --max-depth "$MAX_DEPTH" \
  --max-branching 8 \
  --open-set-budget "$RL_BUDGET" \
  --top-n-prefilter "$RL_PREFILTER"

printf '%s\n' "$BC_EVAL_OUT_DIR" > "$LATEST_BC_EVAL_PTR"
printf 'LATEST_BC_EVAL_DIR=%s\n' "$BC_EVAL_OUT_DIR"

python mol_evo/train_astar_rl_demo.py \
  --input-csv "$TRAIN_POOL_CSV" \
  --model-path "$MODEL_PATH" \
  --model-dir "$MODEL_DIR" \
  --config-file "$CONFIG_PATH" \
  --policy-path "$LATEST_BC_DIR/policy_best.pth" \
  --value-path "$LATEST_BC_DIR/value_best.pth" \
  --output-dir "$RL_OUT_ROOT" \
  --target-property lumo \
  --optimization-mode sub \
  --direction decrease \
  --num-episodes "$RL_EPISODES" \
  --max-depth "$MAX_DEPTH" \
  --max-branching 8 \
  --top-n-prefilter "$RL_PREFILTER" \
  --open-set-budget "$RL_BUDGET" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --seed "$RL_SEED"

LATEST_RL_DIR=$(find "$RL_OUT_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'rl_*' | sort | tail -n 1)
printf '%s\n' "$LATEST_RL_DIR" > "$LATEST_RL_PTR"
printf 'LATEST_RL_DIR=%s\n' "$LATEST_RL_DIR"

python mol_evo/scripts/eval_astar_rl_holdout.py \
  --input-csv "$HOLDOUT_CSV" \
  --output-dir "$RL_EVAL_OUT_DIR" \
  --label mcts_expand_rl_seed11_holdout50_budget50_pref50 \
  --model-dir "$MODEL_DIR" \
  --model-path "$MODEL_PATH" \
  --config-file "$CONFIG_PATH" \
  --policy-path "$LATEST_RL_DIR/policy_best.pth" \
  --value-path "$LATEST_RL_DIR/value_best.pth" \
  --target-property lumo \
  --optimization-mode sub \
  --direction decrease \
  --max-depth "$MAX_DEPTH" \
  --max-branching 8 \
  --open-set-budget "$RL_BUDGET" \
  --top-n-prefilter "$RL_PREFILTER"

printf '%s\n' "$RL_EVAL_OUT_DIR" > "$LATEST_RL_EVAL_PTR"
printf 'LATEST_RL_EVAL_DIR=%s\n' "$RL_EVAL_OUT_DIR"
printf 'END_TIME=%s\n' "$(date '+%F %T')"
