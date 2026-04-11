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

EVAL_OUT_DIR="/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/plan_a_a1_small_eval"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$EVAL_OUT_DIR/tmux_eval_${TIMESTAMP}.log"
mkdir -p "$EVAL_OUT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

printf 'START_TIME=%s\n' "$(date '+%F %T')"
printf 'LOG_FILE=%s\n' "$LOG_FILE"

python mol_evo/scripts/eval_astar_rl_holdout.py \
  --input-csv /home/ubuntu/mol_opt/mol-ofo/mol_evo/dataset/eval-data/plan_a_a0/lumo_plan_a_holdout50_start0.csv \
  --output-dir "$EVAL_OUT_DIR" \
  --label a1_small_bc_holdout50_budget200_pref50_vs_a0_bc200_pref50 \
  --reference-json /home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/lumo_eval_grid50_parallel_20260409_194105/bc_budget200_pref50.json \
  --model-dir /home/ubuntu/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200 \
  --model-path /home/ubuntu/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200/last.pth \
  --config-file /home/ubuntu/mol_opt/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml \
  --policy-path /home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/lumo_plan_a_a1_small_bc/bc_20260410_054557/policy_best.pth \
  --value-path /home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/lumo_plan_a_a1_small_bc/bc_20260410_054557/value_best.pth \
  --target-property lumo \
  --optimization-mode sub \
  --direction decrease \
  --max-depth 3 \
  --max-branching 8 \
  --open-set-budget 200 \
  --top-n-prefilter 50

printf 'END_TIME=%s\n' "$(date '+%F %T')"
