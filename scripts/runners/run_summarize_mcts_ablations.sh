#!/usr/bin/env bash
set -euo pipefail

# ===== 批量总结 MCTS 消融实验评估结果 =====
# 对 ablations 目录下所有 batch_evaluation_results.csv 调用 evaluate_csv_results.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ===== 配置 =====
ABL_ROOT="${ABL_ROOT:-$REPO/mol_evo/output/paper/ablations}"
RUN_ROOTS="${RUN_ROOTS:-}"          # 逗号分隔，为空则扫描 ABL_ROOT 下所有
TASKS="${TASKS:-lumo_up,homo_down}"
VARIANTS="${VARIANTS:-full,wo_prior,wo_leaf_value,random_topb}"
SEEDS="${SEEDS:-42}"

EVAL_SCRIPT="${EVAL_SCRIPT:-/home/rhj/projects/mol_opt/utils/evaluate_csv_results.py}"
CONDA_ENV="${CONDA_ENV:-mol-opt-tdc}"
MAX_OPT_MOLECULES="${MAX_OPT_MOLECULES:-10}"
MIN_LINES="${MIN_LINES:-500}"       # CSV 行数 ≥ 此值才进行统计（否则认为 eval 未完成）

# ===== 日志 =====
log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

# ===== task → target_prop/direction =====
resolve_task() {
  local task="$1"
  case "$task" in
    lumo_up)    echo "lumo|increase" ;;
    lumo_down)  echo "lumo|decrease" ;;
    homo_up)    echo "homo|increase" ;;
    homo_down)  echo "homo|decrease" ;;
    *) echo "unknown|unknown" ;;
  esac
}

# ===== 发现 run_roots =====
discover_run_roots() {
  if [[ -n "$RUN_ROOTS" ]]; then
    local normalized="${RUN_ROOTS//,/ }"
    for root in $normalized; do
      [[ -n "$root" ]] && echo "$root"
    done
  else
    find "$ABL_ROOT" -mindepth 1 -maxdepth 1 -type d | sort
  fi
}

# ===== conda 初始化 =====
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

# ===== 主逻辑 =====
main() {
  mapfile -t run_roots < <(discover_run_roots)
  if [[ ${#run_roots[@]} -eq 0 ]]; then
    log "未找到 run root。ABL_ROOT=$ABL_ROOT RUN_ROOTS=$RUN_ROOTS"
    exit 1
  fi

  local aTasks="${TASKS//,/ }"
  local aVariants="${VARIANTS//,/ }"
  local aSeeds="${SEEDS//,/ }"

  TOTAL=0
  SUCCEEDED=0
  FAILED=0
  SKIPPED_INCOMPLETE=0
  SKIPPED_NO_CSV=0

  log "======================================================"
  log "MCTS 消融实验 — 批量统计总结"
  log "EVAL_SCRIPT: $EVAL_SCRIPT"
  log "MAX_OPT_MOLECULES: $MAX_OPT_MOLECULES"
  log "MIN_LINES: $MIN_LINES (CSV 行数低于此值跳过)"
  log "======================================================"

  for run_root in "${run_roots[@]}"; do
    [[ ! -d "$run_root" ]] && continue
    log ""
    log "📁 RUN_ROOT: $run_root"

    for task in $aTasks; do
      local task_info
      task_info=$(resolve_task "$task")
      IFS='|' read -r target_prop direction <<< "$task_info"

      for variant in $aVariants; do
        for seed in $aSeeds; do
          local seed_dir="$run_root/$task/$variant/seed${seed}"
          local search_dir="$seed_dir/search"
          local eval_base="$search_dir/.evaluation_results_${target_prop}_${direction}"

          # 找最新的 CSV
          local csv_file=""
          if [[ -d "$eval_base" ]]; then
            csv_file="$(find "$eval_base" -type f -name 'batch_evaluation_results.csv' 2>/dev/null | sort | tail -n 1 || true)"
          fi

          if [[ -z "$csv_file" ]]; then
            log "  ⏭️  [${task}/${variant}/s${seed}] — 无 CSV"
            SKIPPED_NO_CSV=$((SKIPPED_NO_CSV + 1))
            continue
          fi

          # 检查行数
          local csv_lines
          csv_lines="$(wc -l < "$csv_file")"
          if [[ $csv_lines -lt $MIN_LINES ]]; then
            log "  ⏭️  [${task}/${variant}/s${seed}] — CSV 行数不足 (${csv_lines} < ${MIN_LINES})"
            SKIPPED_INCOMPLETE=$((SKIPPED_INCOMPLETE + 1))
            continue
          fi

          TOTAL=$((TOTAL + 1))
          log ""
          log "  ▶ [${task}/${variant}/s${seed}] (${csv_lines} lines)"
          log "    CSV: $csv_file"

          # 构建并执行命令
          local CMD="$CONDA_INIT && cd $REPO && python $EVAL_SCRIPT"
          CMD+=" --csv-file '$csv_file'"
          CMD+=" --target-prop $target_prop"
          CMD+=" --direction $direction"
          if [[ -n "$MAX_OPT_MOLECULES" ]]; then
            CMD+=" --max-opt-molecules $MAX_OPT_MOLECULES"
          fi

          if eval "$CMD"; then
            log "    ✅ 统计完成"
            SUCCEEDED=$((SUCCEEDED + 1))
          else
            log "    ❌ 统计失败"
            FAILED=$((FAILED + 1))
          fi
        done
      done
    done
  done

  # ===== 汇总 =====
  log ""
  log "======================================================"
  log "批量统计总结完成"
  log "  成功: $SUCCEEDED"
  log "  失败: $FAILED"
  log "  跳过(未完成): $SKIPPED_INCOMPLETE"
  log "  跳过(无CSV): $SKIPPED_NO_CSV"
  log "======================================================"

  if [[ $FAILED -gt 0 ]]; then
    exit 1
  fi
}

main "$@"
