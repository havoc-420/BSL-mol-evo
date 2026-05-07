#!/usr/bin/env bash

# INFO cli
# RESUME_DIR=/home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/evo-mo/lumo_down_sweep_20260502_131242 bash mol_evo/scripts/runners/run_lumo_down_sweep.sh

set -euo pipefail

# lumo_down step_budget sweep — 双 GPU 逐任务调度
# 每个 GPU 同时只跑一个任务，空闲时分配下一个
# 监控 GPU 显存，超过 3500 MiB 报警并 kill

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO"

# ===== 配置 =====
CONDA_ENV="mol-edit"
GPU_LIST=(0 1)
GPU_MEM_LIMIT=3500  # MiB，超过此值 kill 并报警
POLL_INTERVAL=60    # 秒

# 任务队列：(label, 额外参数)
# baseline 用 --num-simulations 1600，无 --step-budget
# budget 点用 --num-simulations 99999 --step-budget X
TASKS_QUEUE=(
  "baseline_sims1600|--num-simulations 1600"
  "step_budget_10|--num-simulations 99999 --step-budget 10"
  "step_budget_25|--num-simulations 99999 --step-budget 25"
  "step_budget_50|--num-simulations 99999 --step-budget 50"
  "step_budget_100|--num-simulations 99999 --step-budget 100"
  "step_budget_200|--num-simulations 99999 --step-budget 200"
  "step_budget_400|--num-simulations 99999 --step-budget 400"
)

# 公共参数
# 支持断点续传：若指定 RESUME_DIR 环境变量则复用已有目录，否则新建
if [[ -n "${RESUME_DIR:-}" ]] && [[ -d "$RESUME_DIR" ]]; then
  OUTPUT_ROOT="$RESUME_DIR"
  echo "[断点续传] 复用已有输出目录: $OUTPUT_ROOT"
else
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_ROOT="$REPO/mol_evo/output/evo-mo/lumo_down_sweep_$TIMESTAMP"
fi
mkdir -p "$OUTPUT_ROOT"

INPUT_CSV="$REPO/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv"
CONFIG_FILE="$REPO/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml"
MODEL_DIR="$REPO/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200"
MODEL_PATH="$MODEL_DIR/last.pth"

COMMON_ARGS=(
  --input-csv "$INPUT_CSV"
  --model-path "$MODEL_PATH"
  --model-dir "$MODEL_DIR"
  --config-file "$CONFIG_FILE"
  --target-property lumo
  --optimization-mode sub
  --search-mode mcts
  --direction decrease
  --start-index 0 --end-index 50
  --topK 20
  --max-depth 10 --max-branching 20
  --exploration-weight 2.0
  --pruning-patience 3
  --logp-min -0.5 --logp-max 6 --logp-patience 5
)

# ===== conda 激活 =====
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)"
fi

# ===== 状态跟踪 =====
declare -A GPU_TMUX       # GPU_TMUX[gpu_id] = tmux_session_name
declare -A GPU_TASK       # GPU_TASK[gpu_id] = task_label
declare -A GPU_PID        # GPU_PID[gpu_id] = python PID (for GPU monitoring)
TASK_INDEX=0
COMPLETED=0
FAILED=0
ALARM_TRIGGERED=0

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

# ===== 启动一个任务到指定 GPU =====
launch_task() {
  local gpu_id="$1"
  local task_entry="$2"
  local label="${task_entry%%|*}"
  local extra_args="${task_entry#*|}"
  local session_name="sweep_gpu${gpu_id}_${label}"
  local output_json="$OUTPUT_ROOT/lumo_down_${label}.json"
  local log_file="$OUTPUT_ROOT/lumo_down_${label}.log"

  # 构建完整命令
  local cmd="cd $REPO && source ~/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && "
  cmd+="CUDA_VISIBLE_DEVICES=$gpu_id python -m mol_evo.scripts.optimization.batch_optimizer "
  cmd+="--output-json $output_json "
  for arg in "${COMMON_ARGS[@]}"; do
    cmd+="$arg "
  done
  cmd+="$extra_args "
  cmd+="2>&1 | tee $log_file; "
  cmd+="echo '=== TASK_DONE ===' >> $log_file"

  # 创建 tmux session
  tmux kill-session -t "$session_name" 2>/dev/null || true
  tmux new-session -d -s "$session_name" "$cmd"

  GPU_TMUX[$gpu_id]="$session_name"
  GPU_TASK[$gpu_id]="$label"

  log "GPU $gpu_id: 启动任务 [$label] → session=$session_name"
  log "  output: $output_json"
}

# ===== 检查 GPU 上任务是否完成 =====
is_task_done() {
  local gpu_id="$1"
  local session_name="${GPU_TMUX[$gpu_id]:-}"
  [[ -z "$session_name" ]] && return 0

  # 检查 tmux session 是否还存在
  if ! tmux has-session -t "$session_name" 2>/dev/null; then
    return 0  # session 已退出 = 任务完成
  fi

  # 检查 log 文件中是否有 TASK_DONE 标记
  local label="${GPU_TASK[$gpu_id]}"
  local log_file="$OUTPUT_ROOT/lumo_down_${label}.log"
  if [[ -f "$log_file" ]] && grep -q "=== TASK_DONE ===" "$log_file" 2>/dev/null; then
    return 0
  fi

  return 1
}

# ===== 检查任务是否成功 =====
check_task_result() {
  local gpu_id="$1"
  local label="${GPU_TASK[$gpu_id]}"
  local output_json="$OUTPUT_ROOT/lumo_down_${label}.json"

  if [[ -f "$output_json" ]] && [[ -s "$output_json" ]]; then
    log "GPU $gpu_id: ✅ 任务 [$label] 成功完成"
    COMPLETED=$((COMPLETED + 1))
  else
    log "GPU $gpu_id: ❌ 任务 [$label] 失败（输出文件不存在或为空）"
    FAILED=$((FAILED + 1))
  fi

  # 清理 tmux
  local session_name="${GPU_TMUX[$gpu_id]:-}"
  tmux kill-session -t "$session_name" 2>/dev/null || true
  GPU_TMUX[$gpu_id]=""
  GPU_TASK[$gpu_id]=""
}

# ===== GPU 显存监控 =====
check_gpu_memory() {
  for gpu_id in "${GPU_LIST[@]}"; do
    local session_name="${GPU_TMUX[$gpu_id]:-}"
    [[ -z "$session_name" ]] && continue

    # 通过 tmux session 获取我们启动的 Python 进程 PID
    local tmux_pid
    tmux_pid=$(tmux list-panes -t "$session_name" -F '#{pane_pid}' 2>/dev/null | head -1 || true)
    [[ -z "$tmux_pid" ]] && continue

    # 获取该 tmux session 下所有子进程 PID（包括 conda/bash → python）
    local our_pids
    our_pids=$(pstree -p "$tmux_pid" 2>/dev/null | grep -oP '\(\K[0-9]+(?=\))' | tr '\n' '|' | sed 's/|$//' || true)
    [[ -z "$our_pids" ]] && continue

    # 查询 GPU 上属于我们进程的显存占用
    local our_mem
    our_mem=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$gpu_id" 2>/dev/null | \
      awk -F',' -v pids="$our_pids" 'BEGIN{split(pids,a,"|")} {gsub(/[^0-9]/,"",$1); gsub(/[^0-9]/,"",$2); for(i in a) if($1==a[i]) sum+=$2} END {print sum+0}' || true)

    if [[ "$our_mem" -gt "$GPU_MEM_LIMIT" ]]; then
      log "🚨 GPU $gpu_id: 我们的进程占用 ${our_mem} MiB > ${GPU_MEM_LIMIT} MiB 限制！"
      log "🚨 正在 kill 任务 [${GPU_TASK[$gpu_id]}]..."
      tmux kill-session -t "$session_name" 2>/dev/null || true
      GPU_TMUX[$gpu_id]=""
      GPU_TASK[$gpu_id]=""
      FAILED=$((FAILED + 1))
      ALARM_TRIGGERED=$((ALARM_TRIGGERED + 1))
    elif [[ "$our_mem" -gt 0 ]]; then
      log "  GPU $gpu_id: 任务 [${GPU_TASK[$gpu_id]}] 占用 ${our_mem} MiB"
    fi
  done
}

# ===== 断点续传：跳过已完成的任务 =====
SKIPPED=0
PENDING_TASKS=()
for task_entry in "${TASKS_QUEUE[@]}"; do
  label="${task_entry%%|*}"
  output_json="$OUTPUT_ROOT/lumo_down_${label}.json"
  if [[ -f "$output_json" ]] && [[ -s "$output_json" ]]; then
    log "⏭️  跳过已完成任务 [$label] (输出文件已存在)"
    SKIPPED=$((SKIPPED + 1))
    COMPLETED=$((COMPLETED + 1))
  else
    PENDING_TASKS+=("$task_entry")
  fi
done
log "断点续传检查: 已完成=${SKIPPED}, 待执行=${#PENDING_TASKS[@]}"

# ===== 主循环 =====
log "======================================================"
log "lumo_down step_budget sweep — 双 GPU 调度"
log "输出目录: $OUTPUT_ROOT"
log "任务队列: ${#PENDING_TASKS[@]} 个待执行任务 (共 ${#TASKS_QUEUE[@]} 个)"
log "GPU 显存限制: ${GPU_MEM_LIMIT} MiB"
log "======================================================"

# 若没有待执行任务，直接退出
if [[ ${#PENDING_TASKS[@]} -eq 0 ]]; then
  log "所有任务已完成，无需执行"
  exit 0
fi

# 初始分发：给每个 GPU 分配一个任务
for gpu_id in "${GPU_LIST[@]}"; do
  if [[ $TASK_INDEX -lt ${#PENDING_TASKS[@]} ]]; then
    launch_task "$gpu_id" "${PENDING_TASKS[$TASK_INDEX]}"
    TASK_INDEX=$((TASK_INDEX + 1))
  fi
done

# 轮询监控
while true; do
  sleep "$POLL_INTERVAL"

  # 检查 GPU 显存
  check_gpu_memory

  # 检查各 GPU 任务完成状态
  all_idle=true
  for gpu_id in "${GPU_LIST[@]}"; do
    if [[ -n "${GPU_TMUX[$gpu_id]:-}" ]]; then
      if is_task_done "$gpu_id"; then
        check_task_result "$gpu_id"

        # 分配下一个任务
        if [[ $TASK_INDEX -lt ${#PENDING_TASKS[@]} ]]; then
          launch_task "$gpu_id" "${PENDING_TASKS[$TASK_INDEX]}"
          TASK_INDEX=$((TASK_INDEX + 1))
        fi
      else
        all_idle=false
      fi
    fi
  done

  # 所有 GPU 空闲 + 任务队列耗尽 → 结束
  if $all_idle && [[ $TASK_INDEX -ge ${#PENDING_TASKS[@]} ]]; then
    # 再检查一遍是否还有任务在跑
    still_running=false
    for gpu_id in "${GPU_LIST[@]}"; do
      [[ -n "${GPU_TMUX[$gpu_id]:-}" ]] && still_running=true
    done
    if ! $still_running; then
      break
    fi
  fi

  # 进度报告
  running_count=0
  for gpu_id in "${GPU_LIST[@]}"; do
    [[ -n "${GPU_TMUX[$gpu_id]:-}" ]] && running_count=$((running_count + 1))
  done
  log "进度: 完成=$COMPLETED 失败=$FAILED 运行中=$running_count 待分配=$((${#PENDING_TASKS[@]} - TASK_INDEX)) 报警=$ALARM_TRIGGERED"
done

# ===== 结束汇总 =====
log "======================================================"
log "全部任务结束"
log "总任务数 : ${#TASKS_QUEUE[@]}"
log "跳过(续传): $SKIPPED"
log "本次成功 : $((COMPLETED - SKIPPED))"
log "本次失败 : $FAILED"
log "显存报警 : $ALARM_TRIGGERED"
log "输出目录 : $OUTPUT_ROOT"
log "======================================================"

if [[ $FAILED -gt 0 || $ALARM_TRIGGERED -gt 0 ]]; then
  exit 1
fi

if [[ $FAILED -gt 0 || $ALARM_TRIGGERED -gt 0 ]]; then
  exit 1
fi
