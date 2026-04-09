#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-}"
RUN_ROOT="${2:-}"

if [ -z "$METHOD" ] || [ -z "$RUN_ROOT" ]; then
  echo "usage: $0 <bc|rl> <run_root>"
  exit 2
fi

if [ "$METHOD" != "bc" ] && [ "$METHOD" != "rl" ]; then
  echo "invalid method: $METHOD"
  exit 2
fi

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
INPUT_CSV="/home/ubuntu/mol_opt/mol-ofo/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv"
BC_DIR="/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/lumo_bc_bfs15_depth3/bc_20260409_145233"
RL_DIR="/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/astar_rl/lumo_rl_bfs15_depth3_run100/rl_20260409_161937"
SUMMARY_TSV="$RUN_ROOT/summary.tsv"
LOG_FILE="$RUN_ROOT/pipeline_${METHOD}.log"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "RUN_ROOT=$RUN_ROOT"
echo "METHOD=$METHOD"
echo "START_TIME=$(date '+%F %T')"

if [ ! -f "$SUMMARY_TSV" ]; then
  echo -e "method\tbudget\tprefilter\tmols\tnonempty_topk\ttop1_mean\ttop1_median\tactual_expansions_mean\tofo_calls_mean\toutput_json" > "$SUMMARY_TSV"
fi

run_eval() {
  local budget="$1"
  local prefilter="$2"
  local policy_path
  local value_path
  local output_json="$RUN_ROOT/${METHOD}_budget${budget}_pref${prefilter}.json"

  if python - "$SUMMARY_TSV" "$METHOD" "$budget" "$prefilter" <<'PY'
import pathlib, sys
summary_tsv, method, budget, prefilter = sys.argv[1:5]
needle = f"{method}\t{budget}\t{prefilter}\t"
path = pathlib.Path(summary_tsv)
if path.exists() and needle in path.read_text(encoding='utf-8'):
    sys.exit(0)
sys.exit(1)
PY
  then
    echo "SKIP summary-exists method=$METHOD budget=$budget prefilter=$prefilter"
    return 0
  fi

  if [ "$METHOD" = "bc" ]; then
    policy_path="$BC_DIR/policy_best.pth"
    value_path="$BC_DIR/value_best.pth"
  else
    policy_path="$RL_DIR/policy_best.pth"
    value_path="$RL_DIR/value_best.pth"
  fi

  echo "=== RUN method=$METHOD budget=$budget prefilter=$prefilter ==="
  python -m mol_evo.scripts.batch_optimizer \
    --input-csv "$INPUT_CSV" \
    --output-json "$output_json" \
    --model-path "$MODEL_PATH" \
    --model-dir "$MODEL_DIR" \
    --config-file "$CONFIG_PATH" \
    --target-property lumo \
    --optimization-mode sub \
    --search-mode astar_demo \
    --rl-eval \
    --policy-path "$policy_path" \
    --value-path "$value_path" \
    --direction decrease \
    --max-depth 3 \
    --max-branching 8 \
    --open-set-budget "$budget" \
    --top-n-prefilter "$prefilter" \
    --start-index 0 \
    --end-index 50

  python - "$output_json" "$SUMMARY_TSV" "$METHOD" "$budget" "$prefilter" <<'PY'
import json
import statistics
import sys
import fcntl

json_path, summary_tsv, method, budget, prefilter = sys.argv[1:6]
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

rows = []
for _, item in data.items():
    opt = (item or {}).get('optimization_result') or {}
    tk_block = opt.get('topk_results') or {}
    topk = tk_block.get('topK_results') or []
    init = tk_block.get('initial_property_value')
    improvement = 0.0
    if topk and init is not None:
        improvement = float(init) - float(topk[0]['property_value'])
    stats = (opt.get('optimized_result') or {}).get('astar_stats') or {}
    rows.append({
        'improvement': improvement,
        'nonempty': 1 if topk else 0,
        'actual_expansions': float(stats.get('actual_expansions', 0.0)),
        'ofo_calls': float(stats.get('ofo_scored_candidates', 0.0)),
    })

mols = len(rows)
nonempty = sum(r['nonempty'] for r in rows)
improvements = [r['improvement'] for r in rows]
actual_expansions = [r['actual_expansions'] for r in rows]
ofo_calls = [r['ofo_calls'] for r in rows]
line = (
    f"{method}\t{budget}\t{prefilter}\t{mols}\t{nonempty}\t"
    f"{statistics.mean(improvements):.6f}\t{statistics.median(improvements):.6f}\t"
    f"{statistics.mean(actual_expansions):.6f}\t{statistics.mean(ofo_calls):.6f}\t{json_path}\n"
)
with open(summary_tsv, 'a+', encoding='utf-8') as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    f.seek(0)
    text = f.read()
    needle = f"{method}\t{budget}\t{prefilter}\t"
    if needle not in text:
        f.write(line)
        f.flush()
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

print(
    f"SUMMARY method={method} budget={budget} prefilter={prefilter} "
    f"mols={mols} nonempty_topk={nonempty} top1_mean={statistics.mean(improvements):.6f} "
    f"top1_median={statistics.median(improvements):.6f} actual_expansions_mean={statistics.mean(actual_expansions):.4f} "
    f"ofo_calls_mean={statistics.mean(ofo_calls):.4f}"
)
PY
}

for budget in 50 100 200; do
  for prefilter in 20 50; do
    run_eval "$budget" "$prefilter"
  done
done

echo "END_TIME=$(date '+%F %T')"
