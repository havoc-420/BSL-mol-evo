#!/usr/bin/env bash
set -euo pipefail

# ===== 批量评估 lumo_down_sweep 结果 =====
# 通过 tmux 并发注册外部任务，一次最多并发 2 个评估任务

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO"

# ===== 配置 =====
SWEEP_DIR="${SWEEP_DIR:-$REPO/mol_evo/output/evo-mo/lumo_down_sweep_20260502_131242}"
EVAL_SCRIPT="/home/xxx/projects/mol_opt/utils/evaluate_batch_mo.py"
CONDA_ENV="${CONDA_ENV:-mol-opt-tdc}"
TARGET_PROP="lumo"
DIRECTION="decrease"
ITEM_SIZE="${ITEM_SIZE:-10}"
TAG="smer"
MAX_CONCURRENT=2
POLL_INTERVAL=60  # 每 60 秒检查一次任务状态
FORCE="${FORCE:-0}"  # 是否强制重跑已有结果的任务

# ===== 日志 =====
log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

# ===== 构建 conda 激活命令 =====
# 使用 conda activate 方式激活环境
build_conda_init() {
  local init_cmd=""
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    init_cmd="source $HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    init_cmd="source $HOME/anaconda3/etc/profile.d/conda.sh"
  elif [[ -n "${CONDA_EXE:-}" ]]; then
    # 从 conda 可执行文件路径推导 conda.sh 位置
    local conda_base="$(dirname "$(dirname "$CONDA_EXE")")"
    init_cmd="source $conda_base/etc/profile.d/conda.sh"
  else
    init_cmd='eval "$(conda shell.bash hook)"'
  fi
  echo "$init_cmd && conda activate $CONDA_ENV"
}

CONDA_INIT="$(build_conda_init)"

# ===== 任务列表 =====
# 每个 JSON 文件对应一个评估任务
declare -a TASKS=()
for json_file in "$SWEEP_DIR"/lumo_down_*.json; do
  [[ -f "$json_file" ]] || continue
  label="$(basename "$json_file" .json)"
  TASKS+=("$label")
done

if [[ ${#TASKS[@]} -eq 0 ]]; then
  log "❌ 在 $SWEEP_DIR 下未找到任何 lumo_down_*.json 文件"
  exit 1
fi

log "发现 ${#TASKS[@]} 个评估任务: ${TASKS[*]}"

# ===== 断点续传：跳过已完成的任务 =====
declare -a PENDING_TASKS=()
SKIPPED=0
for label in "${TASKS[@]}"; do
  # 检查该 JSON 文件对应的评估结果目录是否已存在
  json_path="$SWEEP_DIR/$label.json"
  eval_base="$SWEEP_DIR/$label/.evaluation_results_${TARGET_PROP}_${DIRECTION}"
  batch_csv=""
  if [[ -d "$eval_base" && "$FORCE" != "1" ]]; then
    batch_csv="$(find "$eval_base" -type f -name 'batch_evaluation_results.csv' 2>/dev/null | sort | tail -n 1 || true)"
  fi

  if [[ -n "$batch_csv" ]]; then
    log "⏭️  跳过已完成任务 [$label] (评估结果已存在: $batch_csv)"
    SKIPPED=$((SKIPPED + 1))
  else
    PENDING_TASKS+=("$label")
  fi
done

log "断点续传检查: 已完成=${SKIPPED}, 待执行=${#PENDING_TASKS[@]}"

if [[ ${#PENDING_TASKS[@]} -eq 0 ]]; then
  log "所有任务已完成，无需执行"
  exit 0
fi

# ===== tmux 并发调度 =====
declare -A ACTIVE_SESSIONS=()  # session_name → label
TASK_INDEX=0
COMPLETED=0
FAILED=0

# 为每个任务构建 eval 命令
build_eval_cmd() {
  local label="$1"
  local json_path="$SWEEP_DIR/$label.json"
  
  # 每个 JSON 文件单独建一个目录作为 result-dir（evaluate_batch_mo 要求目录）
  local task_dir="$SWEEP_DIR/$label"
  
  # 构建命令
  cat <<EOF
$CONDA_INIT && cd $REPO && \\
mkdir -p "$task_dir" && \\
ln -sf "$json_path" "$task_dir/" 2>/dev/null || true && \\
python $EVAL_SCRIPT \\
  --result-dir "$task_dir" \\
  --target-prop $TARGET_PROP \\
  --direction $DIRECTION \\
  --item-size $ITEM_SIZE \\
  --tag $TAG \\
  --max-files 1
EOF
}

launch_task() {
  local label="$1"
  local session_name="eval_${label}"
  
  # 如果 session 已存在，先清除
  tmux kill-session -t "$session_name" 2>/dev/null || true
  
  local cmd
  cmd="$(build_eval_cmd "$label")"
  
  tmux new-session -d -s "$session_name" bash -c "$cmd; echo; echo '=== 评估完成 (exit code: '\$?') ==='; sleep 5"
  
  ACTIVE_SESSIONS["$session_name"]="$label"
  log "🚀 启动评估任务 [$label] → session=$session_name"
  log "   result-dir: $SWEEP_DIR/$label"
}

is_session_alive() {
  local session_name="$1"
  tmux has-session -t "$session_name" 2>/dev/null
}

# ===== 主循环 =====
log "======================================================"
log "lumo_down_sweep 批量评估 — tmux 并发调度"
log "SWEEP_DIR: $SWEEP_DIR"
log "并发数: $MAX_CONCURRENT"
log "待执行: ${#PENDING_TASKS[@]} 个任务"
log "参数: --target-prop $TARGET_PROP --direction $DIRECTION --item-size $ITEM_SIZE --tag $TAG"
log "======================================================"

# 初始分发
while [[ $TASK_INDEX -lt ${#PENDING_TASKS[@]} && ${#ACTIVE_SESSIONS[@]} -lt $MAX_CONCURRENT ]]; do
  launch_task "${PENDING_TASKS[$TASK_INDEX]}"
  TASK_INDEX=$((TASK_INDEX + 1))
  sleep 2  # 错开启动避免资源争抢
done

# 轮询等待
while [[ ${#ACTIVE_SESSIONS[@]} -gt 0 || $TASK_INDEX -lt ${#PENDING_TASKS[@]} ]]; do
  sleep "$POLL_INTERVAL"
  
  # 检查已完成的 session
  for session_name in "${!ACTIVE_SESSIONS[@]}"; do
    if ! is_session_alive "$session_name"; then
      label="${ACTIVE_SESSIONS[$session_name]}"
      
      # 检查评估结果是否生成
      eval_base="$SWEEP_DIR/$label/.evaluation_results_${TARGET_PROP}_${DIRECTION}"
      batch_csv=""
      if [[ -d "$eval_base" ]]; then
        batch_csv="$(find "$eval_base" -type f -name 'batch_evaluation_results.csv' 2>/dev/null | sort | tail -n 1 || true)"
      fi
      
      if [[ -n "$batch_csv" ]]; then
        log "✅ 任务 [$label] 成功完成 → $batch_csv"
        COMPLETED=$((COMPLETED + 1))
      else
        log "❌ 任务 [$label] 失败（未生成评估结果）"
        FAILED=$((FAILED + 1))
      fi
      
      unset ACTIVE_SESSIONS["$session_name"]
      
      # 分配下一个任务
      if [[ $TASK_INDEX -lt ${#PENDING_TASKS[@]} ]]; then
        launch_task "${PENDING_TASKS[$TASK_INDEX]}"
        TASK_INDEX=$((TASK_INDEX + 1))
        sleep 2
      fi
    fi
  done
  
  # 进度报告
  log "进度: 完成=$COMPLETED 失败=$FAILED 运行中=${#ACTIVE_SESSIONS[@]} 待分配=$((${#PENDING_TASKS[@]} - TASK_INDEX))"
done

# ===== 结束汇总 =====
log "======================================================"
log "全部评估任务结束"
log "总任务数   : ${#TASKS[@]}"
log "跳过(续传) : $SKIPPED"
log "本次成功   : $COMPLETED"
log "本次失败   : $FAILED"
log "输出目录   : $SWEEP_DIR"
log "======================================================"

# 列出所有评估结果
log "评估结果:"
for label in "${TASKS[@]}"; do
  eval_base="$SWEEP_DIR/$label/.evaluation_results_${TARGET_PROP}_${DIRECTION}"
  batch_csv=""
  if [[ -d "$eval_base" ]]; then
    batch_csv="$(find "$eval_base" -type f -name 'batch_evaluation_results.csv' 2>/dev/null | sort | tail -n 1 || true)"
  fi
  if [[ -n "$batch_csv" ]]; then
    log "  ✅ [$label] → $batch_csv"
  else
    log "  ❌ [$label] → 无结果"
  fi
done

if [[ $FAILED -gt 0 ]]; then
  exit 1
fi
