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

echo "waiting-for-bc-training"
while pgrep -f "python mol_evo/train_bc_pretrain.py --data-json /home/ubuntu/mol_opt/mol-ofo/mol_evo/dataset/rl_demo/lumo_bfs15_depth3_bc_transitions_20260409.json" >/dev/null; do
  sleep 30
done

BC_DIR=$(find /home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/lumo_bc_bfs15_depth3 -maxdepth 1 -mindepth 1 -type d -name 'bc_*' | sort | tail -n 1)
echo "using-bc-dir=$BC_DIR"

if [ ! -f "$BC_DIR/policy_best.pth" ] || [ ! -f "$BC_DIR/value_best.pth" ]; then
  echo "missing best checkpoints in $BC_DIR"
  exit 1
fi

python -m mol_evo.scripts.batch_optimizer \
  --input-csv /home/ubuntu/mol_opt/mol-ofo/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv \
  --output-json /home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/lumo_bc_eval_5_results.json \
  --model-path /home/ubuntu/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200/last.pth \
  --model-dir /home/ubuntu/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200 \
  --config-file /home/ubuntu/mol_opt/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml \
  --target-property lumo \
  --optimization-mode sub \
  --search-mode astar_demo \
  --rl-eval \
  --policy-path "$BC_DIR/policy_best.pth" \
  --value-path "$BC_DIR/value_best.pth" \
  --direction decrease \
  --max-depth 3 \
  --max-branching 8 \
  --open-set-budget 50 \
  --top-n-prefilter 20 \
  --start-index 0 \
  --end-index 5
