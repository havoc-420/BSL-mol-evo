#!/usr/bin/env bash
set -euo pipefail

# ===== 路径和环境 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO"

CONDA_ENV="${CONDA_ENV:-mol-edit}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/home/xxx/projects/mol_opt/utils/evaluate_batch_mo.py}"
CSV_EVAL_SCRIPT="${CSV_EVAL_SCRIPT:-/home/xxx/projects/mol_opt/utils/evaluate_csv_results.py}"

# ===== 配置参数 =====
# 输入目录（即搜索阶段的输出 root）
ABL_ROOT="${ABL_ROOT:-$REPO/mol_evo/output/paper/ablations}"
RUN_ROOTS="${RUN_ROOTS:-}"          # 逗号分隔，为空则扫描 ABL_ROOT 下所有
TASKS="${TASKS:-lumo_up,homo_down}" # 逗号分隔
VARIANTS="${VARIANTS:-full,wo_prior,wo_leaf_value,random_topb}"
SEEDS="${SEEDS:-42}"
ITEM_SIZE="${ITEM_SIZE:-10}"
TAG="${TAG:-ofo}"
MAX_FILES="${MAX_FILES:-}"
RUN_CSV_EVAL="${RUN_CSV_EVAL:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
FORCE="${FORCE:-0}"

# 并发控制
MAX_PARALLEL="${MAX_PARALLEL:-2}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"  # 秒

# ===== 日志函数 =====
log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

# ===== 解析 task → target_prop/direction =====
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

# ===== 获取 profile =====
read_profile() {
  local run_root="$1"
  local run_manifest="$run_root/run_manifest.txt"
  if [[ -f "$run_manifest" ]]; then
    local profile
    profile="$(grep '^PROFILE=' "$run_manifest" | tail -n 1 | cut -d'=' -f2- || true)"
    if [[ -n "$profile" ]]; then
      echo "$profile"
      return 0
    fi
  fi
  echo "official"
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

# ===== 构建 eval 任务队列 =====
EVAL_QUEUE=()     # 格式: "run_root|task|variant|seed"
SKIPPED=0

build_eval_queue() {
  mapfile -t run_roots < <(discover_run_roots)
  if [[ ${#run_roots[@]} -eq 0 ]]; then
    log "未找到可评估的 run root。ABL_ROOT=$ABL_ROOT RUN_ROOTS=$RUN_ROOTS"
    exit 1
  fi

  local aTasks="${TASKS//,/ }"
  local aVariants="${VARIANTS//,/ }"
  local aSeeds="${SEEDS//,/ }"

  for run_root in "${run_roots[@]}"; do
    [[ ! -d "$run_root" ]] && continue
    for task in $aTasks; do
      for variant in $aVariants; do
        for seed in $aSeeds; do
          local seed_dir="$run_root/$task/$variant/seed${seed}"
          local search_dir="$seed_dir/search"

          # 检查 search 目录是否存在
          if [[ ! -d "$search_dir" ]]; then
            log "⏭️  跳过 [${task}/${variant}/seed${seed}] — 无 search 目录"
            SKIPPED=$((SKIPPED + 1))
            continue
          fi

          # 解析 task
          local task_info
          task_info=$(resolve_task "$task")
          IFS='|' read -r target_prop direction <<< "$task_info"
          local eval_base="$search_dir/.evaluation_results_${target_prop}_${direction}"

          # 检查是否已完成（csv 行数 > 500 视为完成，否则 resume）
          MIN_LINES="${MIN_LINES:-500}"
          if [[ "$FORCE" != "1" && -d "$eval_base" ]]; then
            local existing_csv
            existing_csv="$(find "$eval_base" -type f -name 'batch_evaluation_results.csv' 2>/dev/null | sort | tail -n 1 || true)"
            if [[ -n "$existing_csv" ]]; then
              local csv_lines
              csv_lines="$(wc -l < "$existing_csv")"
              if [[ $csv_lines -gt $MIN_LINES ]]; then
                log "⏭️  跳过已完成 [${task}/${variant}/seed${seed}] (${csv_lines} lines > ${MIN_LINES})"
                SKIPPED=$((SKIPPED + 1))
                continue
              else
                log "🔄 需要 resume [${task}/${variant}/seed${seed}] (${csv_lines} lines ≤ ${MIN_LINES})"
              fi
            fi
          fi

          EVAL_QUEUE+=("${run_root}|${task}|${variant}|${seed}")
        done
      done
    done
  done
}

# ===== 启动单个 eval 任务 =====
# 运行中的任务跟踪: SESSION_NAME -> task_entry
declare -A RUNNING_SESSIONS  # session_name -> "run_root|task|variant|seed"
RUNNING_COUNT=0
COMPLETED=0
FAILED=0
TASK_INDEX=0

launch_eval_task() {
  local task_entry="$1"
  IFS='|' read -r run_root task variant seed <<< "$task_entry"

  local seed_dir="$run_root/$task/$variant/seed${seed}"
  local search_dir="$seed_dir/search"
  local eval_log="$seed_dir/evaluate_batch.log"
  local csv_eval_log="$seed_dir/evaluate_csv.log"

  # 解析 task
  local task_info
  task_info=$(resolve_task "$task")
  IFS='|' read -r target_prop direction <<< "$task_info"

  local eval_base="$search_dir/.evaluation_results_${target_prop}_${direction}"

  # 确定 resume_dir
  local resume_dir=""
  if [[ -d "$eval_base" && "$FORCE" != "1" ]]; then
    resume_dir="$(find "$eval_base" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
  fi

  local session_name="eval_${task}_${variant}_s${seed}"

  # 构建 eval 命令
  local cmd="cd $REPO && source ~/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && "
  cmd+="$PYTHON_BIN $EVAL_SCRIPT "
  cmd+="--result-dir $search_dir "
  cmd+="--target-prop $target_prop "
  cmd+="--direction $direction "
  cmd+="--item-size $ITEM_SIZE "
  cmd+="--tag $TAG "
  if [[ -n "$MAX_FILES" ]]; then
    cmd+="--max-files $MAX_FILES "
  fi
  if [[ -n "$resume_dir" ]]; then
    cmd+="--resume-dir $resume_dir "
  fi
  cmd+="2>&1 | tee $eval_log; "

  # CSV eval（如果开启）
  if [[ "$RUN_CSV_EVAL" == "1" ]]; then
    cmd+="BATCH_CSV=\$(find '$eval_base' -type f -name 'batch_evaluation_results.csv' 2>/dev/null | sort | tail -n 1); "
    cmd+="if [[ -n \"\$BATCH_CSV\" ]]; then "
    cmd+="$PYTHON_BIN $CSV_EVAL_SCRIPT "
    cmd+="--csv-file \"\$BATCH_CSV\" "
    cmd+="--target-prop $target_prop "
    cmd+="--direction $direction "
    cmd+="--max-opt-molecules $ITEM_SIZE "
    cmd+="2>&1 | tee $csv_eval_log; "
    cmd+="fi; "
  fi

  cmd+="echo '=== EVAL_DONE ===' >> $eval_log"

  # 在 tmux 中启动
  TMUX= tmux new-session -d -s "$session_name" "$cmd"
  RUNNING_SESSIONS["$session_name"]="$task_entry"
  RUNNING_COUNT=$((RUNNING_COUNT + 1))

  log "启动 eval [${task}/${variant}/seed${seed}] → session=$session_name"
}

# ===== 检查任务是否完成 =====
check_running_tasks() {
  local session_name
  for session_name in "${!RUNNING_SESSIONS[@]}"; do
    # 检查 tmux session 是否还存在
    if ! TMUX= tmux has-session -t "$session_name" 2>/dev/null; then
      # session 已退出 — 检查结果
      local task_entry="${RUNNING_SESSIONS[$session_name]}"
      IFS='|' read -r run_root task variant seed <<< "$task_entry"
      local seed_dir="$run_root/$task/$variant/seed${seed}"
      local eval_log="$seed_dir/evaluate_batch.log"

      if [[ -f "$eval_log" ]] && grep -q "=== EVAL_DONE ===" "$eval_log" 2>/dev/null; then
        log "✅ eval 完成 [${task}/${variant}/seed${seed}]"
        COMPLETED=$((COMPLETED + 1))
      else
        log "❌ eval 失败 [${task}/${variant}/seed${seed}]"
        FAILED=$((FAILED + 1))
      fi

      unset RUNNING_SESSIONS["$session_name"]
      RUNNING_COUNT=$((RUNNING_COUNT - 1))
    fi
  done
}

# ===== 主逻辑 =====
main() {
  build_eval_queue

  local total=${#EVAL_QUEUE[@]}
  log "======================================================"
  log "MCTS 消融实验 — 批量 Eval (并发=$MAX_PARALLEL)"
  log "任务队列: $total 个待执行 (跳过=$SKIPPED)"
  log "EVAL_SCRIPT=$EVAL_SCRIPT"
  log "CSV_EVAL_SCRIPT=$CSV_EVAL_SCRIPT"
  log "RUN_CSV_EVAL=$RUN_CSV_EVAL  ITEM_SIZE=$ITEM_SIZE  TAG=$TAG"
  log "======================================================"

  if [[ $total -eq 0 ]]; then
    log "所有 eval 已完成，无需执行"
    exit 0
  fi

  # 初始分发
  while [[ $TASK_INDEX -lt $total && $RUNNING_COUNT -lt $MAX_PARALLEL ]]; do
    launch_eval_task "${EVAL_QUEUE[$TASK_INDEX]}"
    TASK_INDEX=$((TASK_INDEX + 1))
  done

  # 主循环：轮询等待 + 调度
  while [[ $RUNNING_COUNT -gt 0 || $TASK_INDEX -lt $total ]]; do
    sleep "$POLL_INTERVAL"
    check_running_tasks

    # 补充新任务
    while [[ $TASK_INDEX -lt $total && $RUNNING_COUNT -lt $MAX_PARALLEL ]]; do
      launch_eval_task "${EVAL_QUEUE[$TASK_INDEX]}"
      TASK_INDEX=$((TASK_INDEX + 1))
    done

    log "进度: 完成=$COMPLETED 失败=$FAILED 运行中=$RUNNING_COUNT 待分配=$((total - TASK_INDEX))"
  done

  # 汇总
  log "======================================================"
  log "全部完成！成功=$COMPLETED 失败=$FAILED 跳过=$SKIPPED"
  log "======================================================"

  # 运行 summary（如果开启）
  if [[ "$RUN_SUMMARY" == "1" ]]; then
    mapfile -t run_roots < <(discover_run_roots)
    for run_root in "${run_roots[@]}"; do
      if [[ -f "$REPO/mol_evo/scripts/summarize_mcts_true_eval_runs.py" ]]; then
        log "生成汇总: $run_root"
        $PYTHON_BIN "$REPO/mol_evo/scripts/summarize_mcts_true_eval_runs.py" --run-root "$run_root" || true
      fi
    done
  fi
}

main "$@"
