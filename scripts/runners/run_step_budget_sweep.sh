#!/usr/bin/env bash
set -euo pipefail

# step_budget 超参数扫描脚本
#
# 用法示例：
#   bash mol_evo/scripts/runners/run_step_budget_sweep.sh
#   CONDA_ENV=mol-ofo CUDA_VISIBLE_DEVICES=0 bash mol_evo/scripts/runners/run_step_budget_sweep.sh
#   TASKS=lumo_up,homo_down SWEEP_PRESET=pilot bash mol_evo/scripts/runners/run_step_budget_sweep.sh
#   STEP_BUDGET_VALUES_CSV=50,100,200,500 bash mol_evo/scripts/runners/run_step_budget_sweep.sh
#
# 说明：
# 1. 扫描多个 step_budget 取值，同时保留一组 baseline（不设 step_budget，仅用 num_simulations）；
# 2. 所有 step_budget 运行均使用 --num-simulations 9999，让 step_budget 成为真正的约束；
# 3. 三档预设：smoke（快速冒烟）/ pilot（中等规模）/ official（论文正式规模）；
# 4. 通过 STEP_BUDGET_VALUES_CSV 环境变量可覆盖默认扫描取值，如：
#      STEP_BUDGET_VALUES_CSV=50,100,300 bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO"

# ===== conda 激活（可选）=====
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
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/mol_evo/output/evo-mo/step_budget_sweep_$TIMESTAMP}"
mkdir -p "$OUTPUT_ROOT"

# ===== 基础数据与模型配置 =====
INPUT_CSV="${INPUT_CSV:-$REPO/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv}"
CONFIG_FILE="${CONFIG_FILE:-$REPO/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml}"
OPTIMIZATION_MODE="${OPTIMIZATION_MODE:-sub}"
SEARCH_MODE="mcts"
TASKS="${TASKS:-lumo_up,homo_down}"
SWEEP_PRESET="${SWEEP_PRESET:-smoke}"

LUMO_MODEL_DIR="${LUMO_MODEL_DIR:-$REPO/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200}"
LUMO_MODEL_PATH="${LUMO_MODEL_PATH:-$LUMO_MODEL_DIR/last.pth}"
HOMO_MODEL_DIR="${HOMO_MODEL_DIR:-$REPO/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251127_121257-homo_change-120000-200}"
HOMO_MODEL_PATH="${HOMO_MODEL_PATH:-$HOMO_MODEL_DIR/last.pth}"

# ===== 三档预设参数 =====
# smoke  : 极少分子 + 小搜索空间，用于快速验证逻辑是否正确（约 2–5 min）
# pilot  : 中等规模，用于判断趋势（约 30–60 min）
# official: 论文正式规模（约数小时）
case "${SWEEP_PRESET}" in
  smoke)
    START_INDEX="${START_INDEX:-0}"
    END_INDEX="${END_INDEX:-5}"
    BASE_MAX_DEPTH="${BASE_MAX_DEPTH:-4}"
    BASE_MAX_BRANCHING="${BASE_MAX_BRANCHING:-8}"
    BASE_EXPLORATION_WEIGHT="${BASE_EXPLORATION_WEIGHT:-2.0}"
    BASE_PRUNING_PATIENCE="${BASE_PRUNING_PATIENCE:-3}"
    BASE_LOGP_MIN="${BASE_LOGP_MIN:--0.5}"
    BASE_LOGP_MAX="${BASE_LOGP_MAX:-6}"
    BASE_LOGP_PATIENCE="${BASE_LOGP_PATIENCE:-5}"
    BASE_TOPK="${BASE_TOPK:-10}"
    # smoke 下用小预算让实验秒级完成
    STEP_BUDGET_VALUES=(5 10 20 50)
    # baseline：不设 step_budget，用少量 simulations
    BASELINE_NUM_SIMULATIONS="${BASELINE_NUM_SIMULATIONS:-50}"
    # step_budget 组的 num_simulations 上限（设大，让 step_budget 生效）
    BUDGET_NUM_SIMULATIONS="${BUDGET_NUM_SIMULATIONS:-9999}"
    ;;
  pilot)
    START_INDEX="${START_INDEX:-0}"
    END_INDEX="${END_INDEX:-20}"
    BASE_MAX_DEPTH="${BASE_MAX_DEPTH:-8}"
    BASE_MAX_BRANCHING="${BASE_MAX_BRANCHING:-15}"
    BASE_EXPLORATION_WEIGHT="${BASE_EXPLORATION_WEIGHT:-2.0}"
    BASE_PRUNING_PATIENCE="${BASE_PRUNING_PATIENCE:-3}"
    BASE_LOGP_MIN="${BASE_LOGP_MIN:--0.5}"
    BASE_LOGP_MAX="${BASE_LOGP_MAX:-6}"
    BASE_LOGP_PATIENCE="${BASE_LOGP_PATIENCE:-5}"
    BASE_TOPK="${BASE_TOPK:-20}"
    STEP_BUDGET_VALUES=(50 100 200 300 500)
    BASELINE_NUM_SIMULATIONS="${BASELINE_NUM_SIMULATIONS:-800}"
    BUDGET_NUM_SIMULATIONS="${BUDGET_NUM_SIMULATIONS:-9999}"
    ;;
  official)
    START_INDEX="${START_INDEX:-0}"
    END_INDEX="${END_INDEX:-50}"
    BASE_MAX_DEPTH="${BASE_MAX_DEPTH:-10}"
    BASE_MAX_BRANCHING="${BASE_MAX_BRANCHING:-20}"
    BASE_EXPLORATION_WEIGHT="${BASE_EXPLORATION_WEIGHT:-2.0}"
    BASE_PRUNING_PATIENCE="${BASE_PRUNING_PATIENCE:-3}"
    BASE_LOGP_MIN="${BASE_LOGP_MIN:--0.5}"
    BASE_LOGP_MAX="${BASE_LOGP_MAX:-6}"
    BASE_LOGP_PATIENCE="${BASE_LOGP_PATIENCE:-5}"
    BASE_TOPK="${BASE_TOPK:-20}"
    # 上限设到 10000：
    #   - 若 step/sim 比 ≈ 0.1–0.3（UCT 集中走少数路径），1600 sims ≈ 160–480 steps；2000 已够
    #   - 若 step/sim 比 ≈ 1.0（几乎每轮展开新节点），1600 sims ≈ 1600 steps；需 >2000
    #   - 实际比值未校准，保守上限 10000 可覆盖所有情况
    #   建议先跑 calibration（--run-calibration-only 或查看 baseline 日志的 actual_steps）再收窄范围
    STEP_BUDGET_VALUES=(50 100 200 500 1000 2000 5000 10000)
    BASELINE_NUM_SIMULATIONS="${BASELINE_NUM_SIMULATIONS:-1600}"
    BUDGET_NUM_SIMULATIONS="${BUDGET_NUM_SIMULATIONS:-99999}"
    ;;
  *)
    echo "不支持的 SWEEP_PRESET: $SWEEP_PRESET" >&2
    echo "可选值: smoke / pilot / official" >&2
    exit 1
    ;;
esac

# 允许通过环境变量覆盖扫描取值（逗号分隔）
if [[ -n "${STEP_BUDGET_VALUES_CSV:-}" ]]; then
  IFS=',' read -r -a STEP_BUDGET_VALUES <<< "$STEP_BUDGET_VALUES_CSV"
fi

# ===== 辅助函数 =====
join_by() {
  local sep="$1"; shift || true
  local first="${1:-}"; shift || true
  printf '%s' "$first"
  for arg in "$@"; do printf '%s%s' "$sep" "$arg"; done
}

slugify() {
  local s="$1"
  s="${s//-/m}"
  s="${s//./p}"
  s="${s//:/_to_}"
  s="${s//,/_}"
  s="${s// /_}"
  echo "$s"
}

# ===== 摘要文件初始化 =====
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
SEARCH_MODE=$SEARCH_MODE
TASKS=$TASKS
SWEEP_PRESET=$SWEEP_PRESET
START_INDEX=$START_INDEX
END_INDEX=$END_INDEX
LUMO_MODEL_PATH=$LUMO_MODEL_PATH
HOMO_MODEL_PATH=$HOMO_MODEL_PATH
BASE_MAX_DEPTH=$BASE_MAX_DEPTH
BASE_MAX_BRANCHING=$BASE_MAX_BRANCHING
BASE_EXPLORATION_WEIGHT=$BASE_EXPLORATION_WEIGHT
BASE_PRUNING_PATIENCE=$BASE_PRUNING_PATIENCE
BASE_LOGP_MIN=$BASE_LOGP_MIN
BASE_LOGP_MAX=$BASE_LOGP_MAX
BASE_LOGP_PATIENCE=$BASE_LOGP_PATIENCE
BASE_TOPK=$BASE_TOPK
BASELINE_NUM_SIMULATIONS=$BASELINE_NUM_SIMULATIONS
BUDGET_NUM_SIMULATIONS=$BUDGET_NUM_SIMULATIONS
STEP_BUDGET_VALUES=$(join_by ',' "${STEP_BUDGET_VALUES[@]}")
EOF

# ===== 任务解析 =====
CURRENT_TASK=""
CURRENT_TARGET_PROPERTY=""
CURRENT_DIRECTION=""
CURRENT_MODEL_PATH=""
CURRENT_MODEL_DIR=""
CURRENT_TASK_ROOT=""

resolve_task() {
  local task="$1"
  case "$task" in
    lumo_up)
      CURRENT_TARGET_PROPERTY="lumo"; CURRENT_DIRECTION="increase"
      CURRENT_MODEL_DIR="$LUMO_MODEL_DIR"; CURRENT_MODEL_PATH="$LUMO_MODEL_PATH" ;;
    lumo_down)
      CURRENT_TARGET_PROPERTY="lumo"; CURRENT_DIRECTION="decrease"
      CURRENT_MODEL_DIR="$LUMO_MODEL_DIR"; CURRENT_MODEL_PATH="$LUMO_MODEL_PATH" ;;
    homo_up)
      CURRENT_TARGET_PROPERTY="homo"; CURRENT_DIRECTION="increase"
      CURRENT_MODEL_DIR="$HOMO_MODEL_DIR"; CURRENT_MODEL_PATH="$HOMO_MODEL_PATH" ;;
    homo_down)
      CURRENT_TARGET_PROPERTY="homo"; CURRENT_DIRECTION="decrease"
      CURRENT_MODEL_DIR="$HOMO_MODEL_DIR"; CURRENT_MODEL_PATH="$HOMO_MODEL_PATH" ;;
    *)
      echo "不支持的 task: $task (可选: lumo_up/lumo_down/homo_up/homo_down)" >&2
      exit 1 ;;
  esac
  CURRENT_TASK="$task"
  CURRENT_TASK_ROOT="$OUTPUT_ROOT/$CURRENT_TASK"
  mkdir -p "$CURRENT_TASK_ROOT"
}

# ===== 单次运行函数 =====
TOTAL_RUNS=0
FAILED_RUNS=0

run_case() {
  local group="$1"      # baseline / step_budget
  local parameter="$2"  # num_simulations / step_budget
  local value_label="$3"
  local run_label="$4"
  shift 4               # 剩余参数直接传给 batch_optimizer

  local output_json="$CURRENT_TASK_ROOT/${run_label}.json"
  local log_file="$CURRENT_TASK_ROOT/${run_label}.log"
  local started_at finished_at status exit_code

  local -a cmd=(
    "$PYTHON_BIN" -m mol_evo.scripts.batch_optimizer
    --input-csv       "$INPUT_CSV"
    --output-json     "$output_json"
    --model-path      "$CURRENT_MODEL_PATH"
    --model-dir       "$CURRENT_MODEL_DIR"
    --config-file     "$CONFIG_FILE"
    --target-property "$CURRENT_TARGET_PROPERTY"
    --optimization-mode "$OPTIMIZATION_MODE"
    --search-mode     "$SEARCH_MODE"
    --direction       "$CURRENT_DIRECTION"
    --start-index     "$START_INDEX"
    --end-index       "$END_INDEX"
    --topK            "$BASE_TOPK"
    --max-depth       "$BASE_MAX_DEPTH"
    --max-branching   "$BASE_MAX_BRANCHING"
    --exploration-weight "$BASE_EXPLORATION_WEIGHT"
    --pruning-patience   "$BASE_PRUNING_PATIENCE"
    --logp-min        "$BASE_LOGP_MIN"
    --logp-max        "$BASE_LOGP_MAX"
    --logp-patience   "$BASE_LOGP_PATIENCE"
    "$@"
  )

  TOTAL_RUNS=$((TOTAL_RUNS + 1))
  started_at="$(date '+%F %T')"

  echo
  echo "============================================================"
  echo "[$TOTAL_RUNS] task=$CURRENT_TASK  group=$group  $parameter=$value_label"
  echo "target_property=$CURRENT_TARGET_PROPERTY  direction=$CURRENT_DIRECTION"
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
    "$CURRENT_TASK" "$CURRENT_TARGET_PROPERTY" "$CURRENT_DIRECTION" \
    "$group" "$parameter" "$value_label" \
    "$output_json" "$log_file" "$status" "$started_at" "$finished_at" \
    >> "$SUMMARY_TSV"
}

# ===== 启动提示 =====
echo "======================================================"
echo "step_budget 超参数扫描"
echo "======================================================"
echo "输出目录   : $OUTPUT_ROOT"
echo "summary    : $SUMMARY_TSV"
echo "commands   : $COMMANDS_TSV"
echo "任务       : $TASKS"
echo "预设       : $SWEEP_PRESET  (分子 ${START_INDEX}–${END_INDEX})"
echo "基线 sims  : $BASELINE_NUM_SIMULATIONS  (无 step_budget)"
echo "预算取值   : $(join_by ', ' "${STEP_BUDGET_VALUES[@]}")"
echo "  num_simulations 上限: $BUDGET_NUM_SIMULATIONS（让 step_budget 生效）"
echo "======================================================"

# ===== 主循环 =====
aTasks="${TASKS//,/ }"
for task in $aTasks; do
  resolve_task "$task"

  echo
  echo "########## 任务: $CURRENT_TASK (${CURRENT_TARGET_PROPERTY}, ${CURRENT_DIRECTION}) ##########"

  # ------------------------------------------------------------------
  # 第 1 步：baseline 组（使用 num_simulations，不设 step_budget）
  # ------------------------------------------------------------------
  echo
  echo "--- [baseline] num_simulations=${BASELINE_NUM_SIMULATIONS} ---"
  run_case \
    "baseline" "num_simulations" "$BASELINE_NUM_SIMULATIONS" \
    "${CURRENT_TASK}_baseline_sims${BASELINE_NUM_SIMULATIONS}" \
    --num-simulations "$BASELINE_NUM_SIMULATIONS"

  # ------------------------------------------------------------------
  # 第 1.5 步：从 baseline 日志读取 actual_steps，提示合理的 step_budget 范围
  # ------------------------------------------------------------------
  baseline_log="$CURRENT_TASK_ROOT/${CURRENT_TASK}_baseline_sims${BASELINE_NUM_SIMULATIONS}.log"
  if [[ -f "$baseline_log" ]]; then
    echo
    echo "--- [calibration] 从 baseline 日志提取 actual_steps ---"
    # 提取最后一次出现的 actual_steps 字段（来自 mcts_stats 的 JSON 输出）
    actual_steps_sample=$(grep -oP '"actual_steps"\s*:\s*\K[0-9]+' "$baseline_log" | tail -5 | sort -n)
    if [[ -n "$actual_steps_sample" ]]; then
      echo "  各分子 actual_steps（最后 5 个）: $(echo "$actual_steps_sample" | tr '\n' ' ')"
      max_steps=$(echo "$actual_steps_sample" | tail -1)
      echo "  最大 actual_steps = $max_steps"
      echo "  → 建议 step_budget 上限至少设为 ${max_steps}（等价于 baseline）"
      echo "  → sweep 区间建议覆盖 [max_steps×0.1, max_steps×1.5]"
    else
      echo "  （未在日志中找到 actual_steps 字段，请手动检查 $baseline_log）"
    fi
    echo
  fi

  # ------------------------------------------------------------------
  # 第 2 步：step_budget 扫描组
  # ------------------------------------------------------------------
  for budget in "${STEP_BUDGET_VALUES[@]}"; do
    echo
    echo "--- [step_budget] step_budget=${budget} ---"
    run_case \
      "step_budget" "step_budget" "$budget" \
      "${CURRENT_TASK}_step_budget_$(slugify "$budget")" \
      --num-simulations "$BUDGET_NUM_SIMULATIONS" \
      --step-budget     "$budget"
  done
done

# ===== 结束摘要 =====
echo
echo "======================================================"
echo "全部 sweep 结束"
echo "总运行数 : $TOTAL_RUNS"
echo "失败数   : $FAILED_RUNS"
echo "汇总文件 : $SUMMARY_TSV"
echo "命令记录 : $COMMANDS_TSV"
echo "======================================================"

# ===== 简易对比输出（仅有 python/jq 时生效）=====
if command -v python &>/dev/null; then
  echo
  echo "======================================================"
  echo "结果对比（各组 batch_results.json → 汇总）"
  echo "======================================================"
  "$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PYEOF'
import sys, os, json, glob

output_root = sys.argv[1]

# 收集每个 run 的关键指标
rows = []
for json_path in sorted(glob.glob(os.path.join(output_root, "**", "*.json"), recursive=True)):
    # 跳过非 batch_results 文件
    fname = os.path.basename(json_path)
    if fname.startswith("batch_results") or fname.endswith("_topK.csv"):
        continue
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        continue

    # 尝试从 mcts_stats 拿 actual_steps
    # batch_optimizer 输出的 JSON 结构：顶层是分子列表
    if not isinstance(data, (list, dict)):
        continue

    molecules = data if isinstance(data, list) else data.get("results", [])
    if not molecules:
        continue

    actual_steps_list = []
    top1_scores = []
    for mol in molecules:
        if not isinstance(mol, dict):
            continue
        tree = mol.get("evolution_tree") or mol.get("result") or {}
        stats = tree.get("mcts_stats", {})
        if stats.get("actual_steps") is not None:
            actual_steps_list.append(stats["actual_steps"])
        # top1 score
        topk = mol.get("topK_results") or mol.get("top_k_results") or []
        if topk:
            scores = [r.get("score", r.get("property_change", None)) for r in topk if isinstance(r, dict)]
            scores = [s for s in scores if s is not None]
            if scores:
                top1_scores.append(max(scores))

    run_label = os.path.splitext(fname)[0]
    avg_steps = sum(actual_steps_list) / len(actual_steps_list) if actual_steps_list else float("nan")
    avg_top1  = sum(top1_scores) / len(top1_scores) if top1_scores else float("nan")
    rows.append((run_label, avg_steps, avg_top1, len(molecules)))

if not rows:
    print("（未找到可解析的结果 JSON，请手动检查输出目录）")
else:
    print(f"{'run_label':<55}  {'avg_steps':>10}  {'avg_top1':>10}  {'n_mols':>6}")
    print("-" * 90)
    for label, steps, top1, n in rows:
        print(f"{label:<55}  {steps:>10.1f}  {top1:>10.4f}  {n:>6}")
PYEOF
fi

if [[ $FAILED_RUNS -gt 0 ]]; then
  exit 1
fi
