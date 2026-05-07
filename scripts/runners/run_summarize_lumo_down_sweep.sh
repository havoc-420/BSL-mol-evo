#!/usr/bin/env bash
set -euo pipefail

# ===== 批量总结 lumo_down_sweep 评估结果 =====
# 对每个评估结果的 batch_evaluation_results.csv 调用 evaluate_csv_results.py 生成统计报告

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ===== 配置 =====
SWEEP_DIR="${SWEEP_DIR:-$REPO/mol_evo/output/evo-mo/lumo_down_sweep_20260502_131242}"
EVAL_SCRIPT="$REPO/../utils/evaluate_csv_results.py"
CONDA_ENV="${CONDA_ENV:-mol-opt-tdc}"
TARGET_PROP="lumo"
DIRECTION="decrease"
MAX_OPT_MOLECULES="${MAX_OPT_MOLECULES:-}"  # 可选：每个分子的优化后分子数量上限

# ===== 日志 =====
log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

# ===== 构建 conda 激活命令 =====
build_conda_init() {
  local init_cmd=""
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    init_cmd="source $HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    init_cmd="source $HOME/anaconda3/etc/profile.d/conda.sh"
  elif [[ -n "${CONDA_EXE:-}" ]]; then
    local conda_base="$(dirname "$(dirname "$CONDA_EXE")")"
    init_cmd="source $conda_base/etc/profile.d/conda.sh"
  else
    init_cmd='eval "$(conda shell.bash hook)"'
  fi
  echo "$init_cmd && conda activate $CONDA_ENV"
}

CONDA_INIT="$(build_conda_init)"

# ===== 收集所有评估结果 CSV =====
declare -A CSV_MAP=()

# 定义所有评估任务标签及其对应的 CSV 文件路径
LABELS=(
  "lumo_down_baseline_sims1600"
  "lumo_down_step_budget_100"
  "lumo_down_step_budget_10"
  "lumo_down_step_budget_200"
  "lumo_down_step_budget_25"
  "lumo_down_step_budget_400"
  "lumo_down_step_budget_50"
)

log "======================================================"
log "lumo_down_sweep 批量统计总结"
log "SWEEP_DIR: $SWEEP_DIR"
log "======================================================"

# 自动发现每个标签的 batch_evaluation_results.csv
FOUND=0
NOT_FOUND=0
for label in "${LABELS[@]}"; do
  eval_base="$SWEEP_DIR/$label/.evaluation_results_${TARGET_PROP}_${DIRECTION}"
  batch_csv=""
  if [[ -d "$eval_base" ]]; then
    batch_csv="$(find "$eval_base" -type f -name 'batch_evaluation_results.csv' 2>/dev/null | sort | tail -n 1 || true)"
  fi

  if [[ -n "$batch_csv" ]]; then
    CSV_MAP["$label"]="$batch_csv"
    log "  ✅ [$label] → $batch_csv"
    FOUND=$((FOUND + 1))
  else
    log "  ❌ [$label] → 未找到评估结果"
    NOT_FOUND=$((NOT_FOUND + 1))
  fi
done

log "发现 ${FOUND} 个评估结果, ${NOT_FOUND} 个缺失"

if [[ $FOUND -eq 0 ]]; then
  log "❌ 未找到任何评估结果，退出"
  exit 1
fi

# ===== 逐个调用 evaluate_csv_results.py =====
log ""
log "======================================================"
log "开始逐个生成统计报告..."
log "======================================================"

SUCCEEDED=0
FAILED=0

for label in "${LABELS[@]}"; do
  csv_file="${CSV_MAP[$label]:-}"
  if [[ -z "$csv_file" ]]; then
    continue
  fi

  log ""
  log "▶ 正在处理 [$label]..."
  log "  CSV: $csv_file"

  # 构建命令
  CMD="$CONDA_INIT && cd $REPO && python $EVAL_SCRIPT --csv-file '$csv_file' --target-prop $TARGET_PROP --direction $DIRECTION"

  # 添加可选参数
  if [[ -n "$MAX_OPT_MOLECULES" ]]; then
    CMD+=" --max-opt-molecules $MAX_OPT_MOLECULES"
  fi

  # 执行
  if eval "$CMD"; then
    log "  ✅ [$label] 统计报告生成成功"
    SUCCEEDED=$((SUCCEEDED + 1))
  else
    log "  ❌ [$label] 统计报告生成失败"
    FAILED=$((FAILED + 1))
  fi
done

# ===== 汇总 =====
log ""
log "======================================================"
log "批量统计总结完成"
log "成功: $SUCCEEDED"
log "失败: $FAILED"
log "======================================================"

if [[ $FAILED -gt 0 ]]; then
  exit 1
fi
