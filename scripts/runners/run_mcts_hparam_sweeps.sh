#!/usr/bin/env bash
set -euo pipefail

# 用法示例：
#   bash mol_evo/scripts/run_mcts_hparam_sweeps.sh
#   CONDA_ENV=mol-ofo CUDA_VISIBLE_DEVICES=0 bash mol_evo/scripts/run_mcts_hparam_sweeps.sh
#   TASKS=lumo_up,homo_down SWEEP_PRESET=paper_minimal bash mol_evo/scripts/run_mcts_hparam_sweeps.sh
#   TASKS=lumo_down SWEEP_PRESET=full bash mol_evo/scripts/run_mcts_hparam_sweeps.sh
#
# 说明：
# 1. 默认执行论文“最小可行集”：`LUMO(U)` + `HOMO(D)`，只跑主文三组 sweep；
# 2. `SWEEP_PRESET=full` 可切回更完整的主文/附录 sweep；
# 3. `logP` 相关约束在当前论文主线中仅作为固定背景，默认不展开成 sweep；
# 4. 如需自定义取值，可通过 `*_VALUES_CSV` 环境变量覆盖默认数组。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
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
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/mol_evo/output/evo-mo/hparam_sweeps_$TIMESTAMP}"
mkdir -p "$OUTPUT_ROOT"

# ===== 基础数据与模型配置 =====
INPUT_CSV="${INPUT_CSV:-$REPO/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv}"
CONFIG_FILE="${CONFIG_FILE:-$REPO/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml}"
OPTIMIZATION_MODE="${OPTIMIZATION_MODE:-sub}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-50}"
SEARCH_MODE="mcts"
TASKS="${TASKS:-lumo_up,homo_down}"
SWEEP_PRESET="${SWEEP_PRESET:-paper_minimal}"

LUMO_MODEL_DIR="${LUMO_MODEL_DIR:-$REPO/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200}"
LUMO_MODEL_PATH="${LUMO_MODEL_PATH:-$LUMO_MODEL_DIR/last.pth}"
HOMO_MODEL_DIR="${HOMO_MODEL_DIR:-$REPO/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251127_121257-homo_change-120000-200}"
HOMO_MODEL_PATH="${HOMO_MODEL_PATH:-$HOMO_MODEL_DIR/last.pth}"

# ===== 论文推荐基线参数 =====
BASE_MAX_DEPTH="${BASE_MAX_DEPTH:-10}"
BASE_MAX_BRANCHING="${BASE_MAX_BRANCHING:-20}"
BASE_NUM_SIMULATIONS="${BASE_NUM_SIMULATIONS:-800}"
BASE_EXPLORATION_WEIGHT="${BASE_EXPLORATION_WEIGHT:-2.0}"
BASE_PRUNING_PATIENCE="${BASE_PRUNING_PATIENCE:-3}"
BASE_LOGP_MIN="${BASE_LOGP_MIN:--0.5}"
BASE_LOGP_MAX="${BASE_LOGP_MAX:-6}"
BASE_LOGP_PATIENCE="${BASE_LOGP_PATIENCE:-5}"
BASE_TOPK="${BASE_TOPK:-20}"

join_by() {
  local sep="$1"
  shift || true
  local first="${1:-}"
  shift || true
  printf '%s' "$first"
  for arg in "$@"; do
    printf '%s%s' "$sep" "$arg"
  done
}

csv_to_array() {
  local raw="$1"
  local -n ref="$2"
  if [[ -n "$raw" ]]; then
    IFS=',' read -r -a ref <<< "$raw"
  fi
}

case "$SWEEP_PRESET" in
  paper_minimal)
    : "${RUN_NUM_SIMULATIONS:=1}"
    : "${RUN_EXPLORATION_WEIGHT:=1}"
    : "${RUN_MAX_BRANCHING:=1}"
    : "${RUN_MAX_DEPTH:=0}"
    : "${RUN_PRUNING_PATIENCE:=0}"
    : "${RUN_LOGP_PATIENCE:=0}"
    : "${RUN_LOGP_RANGE:=0}"

    NUM_SIMULATION_VALUES=(200 400 800)
    EXPLORATION_WEIGHT_VALUES=(1.0 1.4 2.0)
    MAX_BRANCHING_VALUES=(8 20)
    MAX_DEPTH_VALUES=(2 4 6 8 10 12)
    PRUNING_PATIENCE_VALUES=(1 2 3 4)
    LOGP_PATIENCE_VALUES=(1 3 5)
    LOGP_RANGE_VALUES=("0.0:5.0" "-0.5:6.0" "-1.0:6.5")
    ;;
  full)
    : "${RUN_NUM_SIMULATIONS:=1}"
    : "${RUN_EXPLORATION_WEIGHT:=1}"
    : "${RUN_MAX_BRANCHING:=1}"
    : "${RUN_MAX_DEPTH:=1}"
    : "${RUN_PRUNING_PATIENCE:=1}"
    : "${RUN_LOGP_PATIENCE:=0}"
    : "${RUN_LOGP_RANGE:=0}"

    NUM_SIMULATION_VALUES=(100 200 400 800 1600)
    EXPLORATION_WEIGHT_VALUES=(0.5 1.0 1.4 2.0 3.0)
    MAX_BRANCHING_VALUES=(5 10 20 30)
    MAX_DEPTH_VALUES=(2 4 6 8 10 12)
    PRUNING_PATIENCE_VALUES=(1 2 3 4)
    LOGP_PATIENCE_VALUES=(1 3 5)
    LOGP_RANGE_VALUES=("0.0:5.0" "-0.5:6.0" "-1.0:6.5")
    ;;
  *)
    echo "不支持的 SWEEP_PRESET: $SWEEP_PRESET" >&2
    echo "可选值: paper_minimal / full" >&2
    exit 1
    ;;
esac

csv_to_array "${NUM_SIMULATION_VALUES_CSV:-}" NUM_SIMULATION_VALUES
csv_to_array "${EXPLORATION_WEIGHT_VALUES_CSV:-}" EXPLORATION_WEIGHT_VALUES
csv_to_array "${MAX_BRANCHING_VALUES_CSV:-}" MAX_BRANCHING_VALUES
csv_to_array "${MAX_DEPTH_VALUES_CSV:-}" MAX_DEPTH_VALUES
csv_to_array "${PRUNING_PATIENCE_VALUES_CSV:-}" PRUNING_PATIENCE_VALUES
csv_to_array "${LOGP_PATIENCE_VALUES_CSV:-}" LOGP_PATIENCE_VALUES
csv_to_array "${LOGP_RANGE_VALUES_CSV:-}" LOGP_RANGE_VALUES

SUMMARY_TSV="$OUTPUT_ROOT/run_summary.tsv"
COMMANDS_TSV="$OUTPUT_ROOT/run_commands.tsv"
MANIFEST_TXT="$OUTPUT_ROOT/run_manifest.txt"

printf 'task\ttarget_property\tdirection\tgroup\tparameter\tvalue\toutput_json\tlog_file\tstatus\tstarted_at\tfinished_at\n' > "$SUMMARY_TSV"
printf 'label\tcommand\n' > "$COMMANDS_TSV"

cat > "$MANIFEST_TXT" <<EOF
REPO=$REPO
OUTPUT_ROOT=$OUTPUT_ROOT
INPUT_CSV=$INPUT_CSV
CONFIG_FILE=$CONFIG_FILE
OPTIMIZATION_MODE=$OPTIMIZATION_MODE
START_INDEX=$START_INDEX
END_INDEX=$END_INDEX
SEARCH_MODE=$SEARCH_MODE
TASKS=$TASKS
SWEEP_PRESET=$SWEEP_PRESET
LUMO_MODEL_DIR=$LUMO_MODEL_DIR
LUMO_MODEL_PATH=$LUMO_MODEL_PATH
HOMO_MODEL_DIR=$HOMO_MODEL_DIR
HOMO_MODEL_PATH=$HOMO_MODEL_PATH
BASE_MAX_DEPTH=$BASE_MAX_DEPTH
BASE_MAX_BRANCHING=$BASE_MAX_BRANCHING
BASE_NUM_SIMULATIONS=$BASE_NUM_SIMULATIONS
BASE_EXPLORATION_WEIGHT=$BASE_EXPLORATION_WEIGHT
BASE_PRUNING_PATIENCE=$BASE_PRUNING_PATIENCE
BASE_LOGP_MIN=$BASE_LOGP_MIN
BASE_LOGP_MAX=$BASE_LOGP_MAX
BASE_LOGP_PATIENCE=$BASE_LOGP_PATIENCE
BASE_TOPK=$BASE_TOPK
RUN_NUM_SIMULATIONS=$RUN_NUM_SIMULATIONS
RUN_EXPLORATION_WEIGHT=$RUN_EXPLORATION_WEIGHT
RUN_MAX_BRANCHING=$RUN_MAX_BRANCHING
RUN_MAX_DEPTH=$RUN_MAX_DEPTH
RUN_PRUNING_PATIENCE=$RUN_PRUNING_PATIENCE
RUN_LOGP_PATIENCE=$RUN_LOGP_PATIENCE
RUN_LOGP_RANGE=$RUN_LOGP_RANGE
NUM_SIMULATION_VALUES=$(join_by ',' "${NUM_SIMULATION_VALUES[@]}")
EXPLORATION_WEIGHT_VALUES=$(join_by ',' "${EXPLORATION_WEIGHT_VALUES[@]}")
MAX_BRANCHING_VALUES=$(join_by ',' "${MAX_BRANCHING_VALUES[@]}")
MAX_DEPTH_VALUES=$(join_by ',' "${MAX_DEPTH_VALUES[@]}")
PRUNING_PATIENCE_VALUES=$(join_by ',' "${PRUNING_PATIENCE_VALUES[@]}")
LOGP_PATIENCE_VALUES=$(join_by ',' "${LOGP_PATIENCE_VALUES[@]}")
LOGP_RANGE_VALUES=$(join_by ',' "${LOGP_RANGE_VALUES[@]}")
EOF

TOTAL_RUNS=0
FAILED_RUNS=0
CURRENT_TASK=""
CURRENT_TARGET_PROPERTY=""
CURRENT_DIRECTION=""
CURRENT_MODEL_PATH=""
CURRENT_MODEL_DIR=""
CURRENT_TASK_ROOT=""

slugify() {
  local s="$1"
  s="${s//-/m}"
  s="${s//./p}"
  s="${s//:/_to_}"
  s="${s//,/_}"
  s="${s//[/}"
  s="${s//]/}"
  s="${s// /_}"
  echo "$s"
}

resolve_task() {
  local task="$1"
  case "$task" in
    lumo_up)
      CURRENT_TASK="$task"
      CURRENT_TARGET_PROPERTY="lumo"
      CURRENT_DIRECTION="increase"
      CURRENT_MODEL_DIR="$LUMO_MODEL_DIR"
      CURRENT_MODEL_PATH="$LUMO_MODEL_PATH"
      ;;
    lumo_down)
      CURRENT_TASK="$task"
      CURRENT_TARGET_PROPERTY="lumo"
      CURRENT_DIRECTION="decrease"
      CURRENT_MODEL_DIR="$LUMO_MODEL_DIR"
      CURRENT_MODEL_PATH="$LUMO_MODEL_PATH"
      ;;
    homo_up)
      CURRENT_TASK="$task"
      CURRENT_TARGET_PROPERTY="homo"
      CURRENT_DIRECTION="increase"
      CURRENT_MODEL_DIR="$HOMO_MODEL_DIR"
      CURRENT_MODEL_PATH="$HOMO_MODEL_PATH"
      ;;
    homo_down)
      CURRENT_TASK="$task"
      CURRENT_TARGET_PROPERTY="homo"
      CURRENT_DIRECTION="decrease"
      CURRENT_MODEL_DIR="$HOMO_MODEL_DIR"
      CURRENT_MODEL_PATH="$HOMO_MODEL_PATH"
      ;;
    *)
      echo "不支持的 task: $task" >&2
      echo "可选值: lumo_up / lumo_down / homo_up / homo_down" >&2
      exit 1
      ;;
  esac
  CURRENT_TASK_ROOT="$OUTPUT_ROOT/$CURRENT_TASK"
  mkdir -p "$CURRENT_TASK_ROOT"
}

run_case() {
  local group="$1"
  local parameter="$2"
  local value_label="$3"
  local run_label="$4"
  shift 4

  local output_json="$CURRENT_TASK_ROOT/${run_label}.json"
  local log_file="$CURRENT_TASK_ROOT/${run_label}.log"
  local started_at finished_at status exit_code

  local -a cmd=(
    "$PYTHON_BIN" -m mol_evo.scripts.batch_optimizer
    --input-csv "$INPUT_CSV"
    --output-json "$output_json"
    --model-path "$CURRENT_MODEL_PATH"
    --model-dir "$CURRENT_MODEL_DIR"
    --config-file "$CONFIG_FILE"
    --target-property "$CURRENT_TARGET_PROPERTY"
    --optimization-mode "$OPTIMIZATION_MODE"
    --search-mode "$SEARCH_MODE"
    --direction "$CURRENT_DIRECTION"
    --start-index "$START_INDEX"
    --end-index "$END_INDEX"
    --topK "$BASE_TOPK"
    "$@"
  )

  TOTAL_RUNS=$((TOTAL_RUNS + 1))
  started_at="$(date '+%F %T')"

  echo
  echo "============================================================"
  echo "[$TOTAL_RUNS] task=$CURRENT_TASK group=$group parameter=$parameter value=$value_label"
  echo "target_property=$CURRENT_TARGET_PROPERTY direction=$CURRENT_DIRECTION"
  echo "output_json=$output_json"
  echo "log_file=$log_file"
  echo "============================================================"

  {
    printf '%s\t' "$run_label"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } >> "$COMMANDS_TSV"

  set +e
  "${cmd[@]}" 2>&1 | tee "$log_file"
  exit_code=${PIPESTATUS[0]}
  set -e

  finished_at="$(date '+%F %T')"
  if [[ $exit_code -eq 0 ]]; then
    status="success"
  else
    status="failed($exit_code)"
    FAILED_RUNS=$((FAILED_RUNS + 1))
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$CURRENT_TASK" "$CURRENT_TARGET_PROPERTY" "$CURRENT_DIRECTION" "$group" "$parameter" "$value_label" "$output_json" "$log_file" "$status" "$started_at" "$finished_at" \
    >> "$SUMMARY_TSV"
}

echo "输出目录: $OUTPUT_ROOT"
echo "summary: $SUMMARY_TSV"
echo "commands: $COMMANDS_TSV"
echo "tasks: $TASKS"
echo "sweep preset: $SWEEP_PRESET"

aTasks="${TASKS//,/ }"
for task in $aTasks; do
  resolve_task "$task"

  echo
  echo "################ 当前任务: $CURRENT_TASK (${CURRENT_TARGET_PROPERTY}, ${CURRENT_DIRECTION}) ################"

  if [[ "$RUN_NUM_SIMULATIONS" == "1" ]]; then
    for value in "${NUM_SIMULATION_VALUES[@]}"; do
      run_case \
        "main" "num_simulations" "$value" "${CURRENT_TASK}_sweep_num_simulations_$(slugify "$value")" \
        --max-depth "$BASE_MAX_DEPTH" \
        --max-branching "$BASE_MAX_BRANCHING" \
        --num-simulations "$value" \
        --exploration-weight "$BASE_EXPLORATION_WEIGHT" \
        --pruning-patience "$BASE_PRUNING_PATIENCE" \
        --logp-min "$BASE_LOGP_MIN" \
        --logp-max "$BASE_LOGP_MAX" \
        --logp-patience "$BASE_LOGP_PATIENCE"
    done
  fi

  if [[ "$RUN_EXPLORATION_WEIGHT" == "1" ]]; then
    for value in "${EXPLORATION_WEIGHT_VALUES[@]}"; do
      run_case \
        "main" "exploration_weight" "$value" "${CURRENT_TASK}_sweep_exploration_weight_$(slugify "$value")" \
        --max-depth "$BASE_MAX_DEPTH" \
        --max-branching "$BASE_MAX_BRANCHING" \
        --num-simulations "$BASE_NUM_SIMULATIONS" \
        --exploration-weight "$value" \
        --pruning-patience "$BASE_PRUNING_PATIENCE" \
        --logp-min "$BASE_LOGP_MIN" \
        --logp-max "$BASE_LOGP_MAX" \
        --logp-patience "$BASE_LOGP_PATIENCE"
    done
  fi

  if [[ "$RUN_MAX_BRANCHING" == "1" ]]; then
    for value in "${MAX_BRANCHING_VALUES[@]}"; do
      run_case \
        "main" "max_branching" "$value" "${CURRENT_TASK}_sweep_max_branching_$(slugify "$value")" \
        --max-depth "$BASE_MAX_DEPTH" \
        --max-branching "$value" \
        --num-simulations "$BASE_NUM_SIMULATIONS" \
        --exploration-weight "$BASE_EXPLORATION_WEIGHT" \
        --pruning-patience "$BASE_PRUNING_PATIENCE" \
        --logp-min "$BASE_LOGP_MIN" \
        --logp-max "$BASE_LOGP_MAX" \
        --logp-patience "$BASE_LOGP_PATIENCE"
    done
  fi

  if [[ "$RUN_MAX_DEPTH" == "1" ]]; then
    for value in "${MAX_DEPTH_VALUES[@]}"; do
      run_case \
        "appendix" "max_depth" "$value" "${CURRENT_TASK}_sweep_max_depth_$(slugify "$value")" \
        --max-depth "$value" \
        --max-branching "$BASE_MAX_BRANCHING" \
        --num-simulations "$BASE_NUM_SIMULATIONS" \
        --exploration-weight "$BASE_EXPLORATION_WEIGHT" \
        --pruning-patience "$BASE_PRUNING_PATIENCE" \
        --logp-min "$BASE_LOGP_MIN" \
        --logp-max "$BASE_LOGP_MAX" \
        --logp-patience "$BASE_LOGP_PATIENCE"
    done
  fi

  if [[ "$RUN_PRUNING_PATIENCE" == "1" ]]; then
    for value in "${PRUNING_PATIENCE_VALUES[@]}"; do
      run_case \
        "appendix" "pruning_patience" "$value" "${CURRENT_TASK}_sweep_pruning_patience_$(slugify "$value")" \
        --max-depth "$BASE_MAX_DEPTH" \
        --max-branching "$BASE_MAX_BRANCHING" \
        --num-simulations "$BASE_NUM_SIMULATIONS" \
        --exploration-weight "$BASE_EXPLORATION_WEIGHT" \
        --pruning-patience "$value" \
        --logp-min "$BASE_LOGP_MIN" \
        --logp-max "$BASE_LOGP_MAX" \
        --logp-patience "$BASE_LOGP_PATIENCE"
    done
  fi

  if [[ "$RUN_LOGP_PATIENCE" == "1" ]]; then
    for value in "${LOGP_PATIENCE_VALUES[@]}"; do
      run_case \
        "constraint" "logp_patience" "$value" "${CURRENT_TASK}_sweep_logp_patience_$(slugify "$value")" \
        --max-depth "$BASE_MAX_DEPTH" \
        --max-branching "$BASE_MAX_BRANCHING" \
        --num-simulations "$BASE_NUM_SIMULATIONS" \
        --exploration-weight "$BASE_EXPLORATION_WEIGHT" \
        --pruning-patience "$BASE_PRUNING_PATIENCE" \
        --logp-min "$BASE_LOGP_MIN" \
        --logp-max "$BASE_LOGP_MAX" \
        --logp-patience "$value"
    done
  fi

  if [[ "$RUN_LOGP_RANGE" == "1" ]]; then
    for value in "${LOGP_RANGE_VALUES[@]}"; do
      IFS=':' read -r logp_min logp_max <<< "$value"
      run_case \
        "constraint" "logp_range" "[$logp_min,$logp_max]" "${CURRENT_TASK}_sweep_logp_range_$(slugify "$value")" \
        --max-depth "$BASE_MAX_DEPTH" \
        --max-branching "$BASE_MAX_BRANCHING" \
        --num-simulations "$BASE_NUM_SIMULATIONS" \
        --exploration-weight "$BASE_EXPLORATION_WEIGHT" \
        --pruning-patience "$BASE_PRUNING_PATIENCE" \
        --logp-min "$logp_min" \
        --logp-max "$logp_max" \
        --logp-patience "$BASE_LOGP_PATIENCE"
    done
  fi
done

echo
echo "全部 sweep 结束。"
echo "总运行数: $TOTAL_RUNS"
echo "失败数: $FAILED_RUNS"
echo "汇总文件: $SUMMARY_TSV"
echo "命令记录: $COMMANDS_TSV"

if [[ $FAILED_RUNS -gt 0 ]]; then
  exit 1
fi
