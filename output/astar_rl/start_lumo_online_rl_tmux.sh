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

RL_ROOT="/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/lumo_rl_bfs15_depth3_run100"
BC_DIR="/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/lumo_bc_bfs15_depth3/bc_20260409_145233"
MODEL_DIR="/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200"
MODEL_PATH="$MODEL_DIR/last.pth"
CONFIG_PATH="/home/ubuntu/mol_opt/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml"
INPUT_CSV="/home/ubuntu/mol_opt/mol-ofo/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv"

mkdir -p "$RL_ROOT"

echo "=== start online rl ==="
python mol_evo/train_astar_rl_demo.py \
  --input-csv "$INPUT_CSV" \
  --model-path "$MODEL_PATH" \
  --model-dir "$MODEL_DIR" \
  --config-file "$CONFIG_PATH" \
  --policy-path "$BC_DIR/policy_best.pth" \
  --value-path "$BC_DIR/value_best.pth" \
  --output-dir "$RL_ROOT" \
  --target-property lumo \
  --optimization-mode sub \
  --direction decrease \
  --num-episodes 100 \
  --max-depth 3 \
  --max-branching 8 \
  --top-n-prefilter 20 \
  --open-set-budget 50 \
  --checkpoint-every 25

LATEST_RL_DIR=$(find "$RL_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'rl_*' | sort | tail -n 1)
echo "LATEST_RL_DIR=$LATEST_RL_DIR"

if [ ! -f "$LATEST_RL_DIR/policy_best.pth" ] || [ ! -f "$LATEST_RL_DIR/value_best.pth" ]; then
  echo "missing best checkpoints in $LATEST_RL_DIR"
  exit 1
fi

echo "=== start post-train eval ==="
python -m mol_evo.scripts.batch_optimizer \
  --input-csv "$INPUT_CSV" \
  --output-json /home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/lumo_rl_eval_5_after_run100.json \
  --model-path "$MODEL_PATH" \
  --model-dir "$MODEL_DIR" \
  --config-file "$CONFIG_PATH" \
  --target-property lumo \
  --optimization-mode sub \
  --search-mode astar_demo \
  --rl-eval \
  --policy-path "$LATEST_RL_DIR/policy_best.pth" \
  --value-path "$LATEST_RL_DIR/value_best.pth" \
  --direction decrease \
  --max-depth 3 \
  --max-branching 8 \
  --open-set-budget 50 \
  --top-n-prefilter 20 \
  --start-index 0 \
  --end-index 5

echo "=== online rl pipeline finished ==="
