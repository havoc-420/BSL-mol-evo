#!/usr/bin/env bash
set -euo pipefail

# 用法示例：
#   bash mol_evo/scripts/run_mcts_ablations.sh
#   PROFILE=pilot TASKS=lumo_up,homo_down VARIANTS=full,wo_prior,wo_leaf_value,random_topb bash mol_evo/scripts/run_mcts_ablations.sh
#   CONDA_ENV=mol-ofo CUDA_VISIBLE_DEVICES=0 PROFILE=official SEEDS=42,43,44 bash mol_evo/scripts/run_mcts_ablations.sh
#
# 说明：
# 1. 默认执行主文最小版本：full / w/o prior / w/o leaf value / random_topb；
# 2. 默认任务为 LUMO(U) + HOMO(D)；
# 3. profile=smoke/pilot/official 控制样本切片与搜索预算；
# 4. 每个 run unit 会写入独立目录，包含 search / evaluation / manifest / log。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO"

if [[ -n "${CONDA_ENV:-}" ]]; then
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    eval "$(conda shell.bash hook)"
  fi
  conda activate "$CONDA_ENV"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
PROFILE="${PROFILE:-smoke}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
TASKS="${TASKS:-lumo_up,homo_down}"
VARIANTS="${VARIANTS:-full,wo_prior,wo_leaf_value,random_topb}"
SEEDS="${SEEDS:-42}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/mol_evo/output/paper/ablations/$TIMESTAMP}"
mkdir -p "$OUTPUT_ROOT"

INPUT_CSV="${INPUT_CSV:-$REPO/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv}"
CONFIG_FILE="${CONFIG_FILE:-$REPO/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml}"
OPTIMIZATION_MODE="${OPTIMIZATION_MODE:-sub}"
TOPK="${TOPK:-20}"
MAX_FILES="${MAX_FILES:-}"

LUMO_MODEL_DIR="${LUMO_MODEL_DIR:-$REPO/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200}"
LUMO_MODEL_PATH="${LUMO_MODEL_PATH:-$LUMO_MODEL_DIR/last.pth}"
HOMO_MODEL_DIR="${HOMO_MODEL_DIR:-$REPO/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251127_121257-homo_change-120000-200}"
HOMO_MODEL_PATH="${HOMO_MODEL_PATH:-$HOMO_MODEL_DIR/last.pth}"

case "$PROFILE" in
  smoke)
    START_INDEX="${START_INDEX:-0}"
    END_INDEX="${END_INDEX:-5}"
    NUM_SIMULATIONS="${NUM_SIMULATIONS:-60}"
    MAX_DEPTH="${MAX_DEPTH:-6}"
    MAX_BRANCHING="${MAX_BRANCHING:-12}"
    EXPLORATION_WEIGHT="${EXPLORATION_WEIGHT:-2.0}"
    PRUNING_PATIENCE="${PRUNING_PATIENCE:-3}"
    LOGP_MIN="${LOGP_MIN:--0.5}"
    LOGP_MAX="${LOGP_MAX:-6}"
    LOGP_PATIENCE="${LOGP_PATIENCE:-5}"
    ITEM_SIZE="${ITEM_SIZE:-10}"
    ;;
  pilot)
    START_INDEX="${START_INDEX:-0}"
    END_INDEX="${END_INDEX:-20}"
    NUM_SIMULATIONS="${NUM_SIMULATIONS:-400}"
    MAX_DEPTH="${MAX_DEPTH:-10}"
    MAX_BRANCHING="${MAX_BRANCHING:-20}"
    EXPLORATION_WEIGHT="${EXPLORATION_WEIGHT:-2.0}"
    PRUNING_PATIENCE="${PRUNING_PATIENCE:-3}"
    LOGP_MIN="${LOGP_MIN:--0.5}"
    LOGP_MAX="${LOGP_MAX:-6}"
    LOGP_PATIENCE="${LOGP_PATIENCE:-5}"
    ITEM_SIZE="${ITEM_SIZE:-20}"
    ;;
  official)
    START_INDEX="${START_INDEX:-0}"
    END_INDEX="${END_INDEX:-50}"
    NUM_SIMULATIONS="${NUM_SIMULATIONS:-800}"
    MAX_DEPTH="${MAX_DEPTH:-10}"
    MAX_BRANCHING="${MAX_BRANCHING:-20}"
    EXPLORATION_WEIGHT="${EXPLORATION_WEIGHT:-2.0}"
    PRUNING_PATIENCE="${PRUNING_PATIENCE:-3}"
    LOGP_MIN="${LOGP_MIN:--0.5}"
    LOGP_MAX="${LOGP_MAX:-6}"
    LOGP_PATIENCE="${LOGP_PATIENCE:-5}"
    ITEM_SIZE="${ITEM_SIZE:-20}"
    ;;
  *)
    echo "不支持的 PROFILE: $PROFILE" >&2
    exit 1
    ;;
esac

SUMMARY_TSV="$OUTPUT_ROOT/run_summary.tsv"
COMMANDS_TSV="$OUTPUT_ROOT/run_commands.tsv"
ROOT_MANIFEST="$OUTPUT_ROOT/run_manifest.txt"
printf 'task\tvariant\tseed\tstatus\tsearch_dir\tbatch_json\tbatch_eval_csv\tcsv_eval_dir\tsearch_log\teval_log\tcsv_eval_log\n' > "$SUMMARY_TSV"
printf 'label\tcommand\n' > "$COMMANDS_TSV"

cat > "$ROOT_MANIFEST" <<EOF
REPO=$REPO
PROFILE=$PROFILE
OUTPUT_ROOT=$OUTPUT_ROOT
INPUT_CSV=$INPUT_CSV
CONFIG_FILE=$CONFIG_FILE
TASKS=$TASKS
VARIANTS=$VARIANTS
SEEDS=$SEEDS
START_INDEX=$START_INDEX
END_INDEX=$END_INDEX
NUM_SIMULATIONS=$NUM_SIMULATIONS
MAX_DEPTH=$MAX_DEPTH
MAX_BRANCHING=$MAX_BRANCHING
EXPLORATION_WEIGHT=$EXPLORATION_WEIGHT
PRUNING_PATIENCE=$PRUNING_PATIENCE
LOGP_MIN=$LOGP_MIN
LOGP_MAX=$LOGP_MAX
LOGP_PATIENCE=$LOGP_PATIENCE
TOPK=$TOPK
RUN_EVAL=$RUN_EVAL
RUN_SUMMARY=$RUN_SUMMARY
EOF

slugify() {
  local s="$1"
  s="${s// /_}"
  s="${s//,/__}"
  echo "$s"
}

resolve_task() {
  local task="$1"
  case "$task" in
    lumo_up)
      TARGET_PROP="lumo"
      DIRECTION="increase"
      MODEL_DIR="$LUMO_MODEL_DIR"
      MODEL_PATH="$LUMO_MODEL_PATH"
      ;;
    lumo_down)
      TARGET_PROP="lumo"
      DIRECTION="decrease"
      MODEL_DIR="$LUMO_MODEL_DIR"
      MODEL_PATH="$LUMO_MODEL_PATH"
      ;;
    homo_up)
      TARGET_PROP="homo"
      DIRECTION="increase"
      MODEL_DIR="$HOMO_MODEL_DIR"
      MODEL_PATH="$HOMO_MODEL_PATH"
      ;;
    homo_down)
      TARGET_PROP="homo"
      DIRECTION="decrease"
      MODEL_DIR="$HOMO_MODEL_DIR"
      MODEL_PATH="$HOMO_MODEL_PATH"
      ;;
    *)
      echo "不支持的 task: $task" >&2
      exit 1
      ;;
  esac
}

run_and_log() {
  local label="$1"
  local log_file="$2"
  shift 2
  printf '%s\t' "$label" >> "$COMMANDS_TSV"
  printf '%q ' "$@" >> "$COMMANDS_TSV"
  printf '\n' >> "$COMMANDS_TSV"

  set +e
  "$@" 2>&1 | tee "$log_file"
  local exit_code=${PIPESTATUS[0]}
  set -e
  return $exit_code
}

run_unit() {
  local task="$1"
  local variant="$2"
  local seed="$3"

  resolve_task "$task"

  local prior_mode="softmax"
  local value_mode="accumulated"
  local expansion_mode="topk"
  local variant_pruning_patience="$PRUNING_PATIENCE"
  local variant_logp_min="$LOGP_MIN"
  local variant_logp_max="$LOGP_MAX"
  local variant_logp_patience="$LOGP_PATIENCE"

  case "$variant" in
    full)
      ;;
    wo_prior)
      prior_mode="uniform"
      ;;
    wo_leaf_value)
      value_mode="zero"
      ;;
    random_topb)
      expansion_mode="random_topk"
      ;;
    full_expand)
      expansion_mode="full"
      ;;
    wo_pruning)
      variant_pruning_patience="0"
      ;;
    wo_logp)
      variant_logp_patience="0"
      variant_logp_min="-999"
      variant_logp_max="999"
      ;;
    *)
      echo "不支持的 variant: $variant" >&2
      exit 1
      ;;
  esac

  local run_root="$OUTPUT_ROOT/$task/$variant/seed${seed}"
  local search_dir="$run_root/search"
  local search_log="$run_root/search.log"
  local eval_log="$run_root/evaluate_batch.log"
  local csv_eval_log="$run_root/evaluate_csv.log"
  local manifest_json="$run_root/manifest.json"
  mkdir -p "$run_root" "$search_dir"

  local batch_json="$search_dir/batch_results.json"
  local batch_eval_csv=""
  local csv_eval_dir=""
  local status="success"

  local -a search_cmd=(
    "$PYTHON_BIN" -m mol_evo.scripts.optimization.batch_optimizer
    --input-csv "$INPUT_CSV"
    --output-dir "$search_dir"
    --output-json "$batch_json"
    --model-path "$MODEL_PATH"
    --model-dir "$MODEL_DIR"
    --config-file "$CONFIG_FILE"
    --target-property "$TARGET_PROP"
    --optimization-mode "$OPTIMIZATION_MODE"
    --search-mode mcts
    --num-simulations "$NUM_SIMULATIONS"
    --exploration-weight "$EXPLORATION_WEIGHT"
    --mcts-prior-mode "$prior_mode"
    --mcts-value-mode "$value_mode"
    --mcts-expansion-mode "$expansion_mode"
    --mcts-random-seed "$seed"
    --direction "$DIRECTION"
    --max-depth "$MAX_DEPTH"
    --max-branching "$MAX_BRANCHING"
    --pruning-patience "$variant_pruning_patience"
    --logp-min "$variant_logp_min"
    --logp-max "$variant_logp_max"
    --logp-patience "$variant_logp_patience"
    --topK "$TOPK"
    --start-index "$START_INDEX"
    --end-index "$END_INDEX"
  )

  echo
  echo "============================================================"
  echo "task=$task variant=$variant seed=$seed"
  echo "search_dir=$search_dir"
  echo "============================================================"

  if ! run_and_log "${task}_${variant}_seed${seed}_search" "$search_log" "${search_cmd[@]}"; then
    status="search_failed"
  fi

  if [[ "$status" == "success" && "$RUN_EVAL" == "1" ]]; then
    local -a eval_cmd=(
      "$PYTHON_BIN" /home/ubuntu/mol_opt/utils/evaluate_batch_mo.py
      --result-dir "$search_dir"
      --target-prop "$TARGET_PROP"
      --direction "$DIRECTION"
      --item-size "$ITEM_SIZE"
      --tag ofo
    )
    if [[ -n "$MAX_FILES" ]]; then
      eval_cmd+=(--max-files "$MAX_FILES")
    fi

    if ! run_and_log "${task}_${variant}_seed${seed}_eval_batch" "$eval_log" "${eval_cmd[@]}"; then
      status="eval_batch_failed"
    else
      batch_eval_csv="$(find "$search_dir/.evaluation_results_${TARGET_PROP}_${DIRECTION}" -type f -name 'batch_evaluation_results.csv' | sort | tail -n 1 || true)"
      if [[ -n "$batch_eval_csv" ]]; then
        local -a csv_eval_cmd=(
          "$PYTHON_BIN" /home/ubuntu/mol_opt/utils/evaluate_csv_results.py
          --csv-file "$batch_eval_csv"
          --target-prop "$TARGET_PROP"
          --direction "$DIRECTION"
          --max-opt-molecules "$ITEM_SIZE"
        )
        if ! run_and_log "${task}_${variant}_seed${seed}_eval_csv" "$csv_eval_log" "${csv_eval_cmd[@]}"; then
          status="eval_csv_failed"
        else
          csv_eval_dir="$(find "$(dirname "$batch_eval_csv")" -mindepth 1 -maxdepth 1 -type d -name 'eval_*' | sort | tail -n 1 || true)"
        fi
      else
        status="eval_csv_missing_input"
      fi
    fi
  fi

  cat > "$manifest_json" <<EOF
{
  "task": "$task",
  "variant": "$variant",
  "seed": $seed,
  "profile": "$PROFILE",
  "target_property": "$TARGET_PROP",
  "direction": "$DIRECTION",
  "model_dir": "$MODEL_DIR",
  "model_path": "$MODEL_PATH",
  "input_csv": "$INPUT_CSV",
  "config_file": "$CONFIG_FILE",
  "search_dir": "$search_dir",
  "batch_json": "$batch_json",
  "batch_eval_csv": "$batch_eval_csv",
  "csv_eval_dir": "$csv_eval_dir",
  "num_simulations": $NUM_SIMULATIONS,
  "exploration_weight": $EXPLORATION_WEIGHT,
  "max_depth": $MAX_DEPTH,
  "max_branching": $MAX_BRANCHING,
  "pruning_patience": $variant_pruning_patience,
  "logp_min": $variant_logp_min,
  "logp_max": $variant_logp_max,
  "logp_patience": $variant_logp_patience,
  "topK": $TOPK,
  "mcts_prior_mode": "$prior_mode",
  "mcts_value_mode": "$value_mode",
  "mcts_expansion_mode": "$expansion_mode",
  "run_eval": "$RUN_EVAL",
  "status": "$status"
}
EOF

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$task" "$variant" "$seed" "$status" "$search_dir" "$batch_json" "$batch_eval_csv" "$csv_eval_dir" "$search_log" "$eval_log" "$csv_eval_log" \
    >> "$SUMMARY_TSV"
}

echo "输出目录: $OUTPUT_ROOT"
echo "summary: $SUMMARY_TSV"
echo "commands: $COMMANDS_TSV"

aTasks="${TASKS//,/ }"
aVariants="${VARIANTS//,/ }"
aSeeds="${SEEDS//,/ }"

for task in $aTasks; do
  for variant in $aVariants; do
    for seed in $aSeeds; do
      run_unit "$task" "$variant" "$seed"
    done
  done
done

if [[ "$RUN_SUMMARY" == "1" ]]; then
  "$PYTHON_BIN" "$REPO/mol_evo/scripts/summarize_mcts_ablation_runs.py" --run-root "$OUTPUT_ROOT"
fi

echo
echo "全部 run unit 执行完成。"
echo "结果汇总: $SUMMARY_TSV"
