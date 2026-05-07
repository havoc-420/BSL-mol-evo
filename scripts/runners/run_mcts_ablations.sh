#!/usr/bin/env bash
set -euo pipefail

# 确保能从 tmux 内部启动子 session
unset TMUX

# MCTS 消融实验 — 双 GPU 逐任务调度 + 显存监控
#
# 用法示例：
#   bash mol_evo/scripts/runners/run_mcts_ablations.sh
#   PROFILE=official TASKS=lumo_up,homo_down bash mol_evo/scripts/runners/run_mcts_ablations.sh
#   GPU_LIST="0,1" GPU_MEM_LIMIT=3500 PROFILE=official bash mol_evo/scripts/runners/run_mcts_ablations.sh
#   RESUME_DIR=/path/to/existing/output bash mol_evo/scripts/runners/run_mcts_ablations.sh
#
# 说明：
# 1. 默认执行主文最小版本：full / w/o prior / w/o leaf value / random_topb；
# 2. 默认任务为 LUMO(U) + HOMO(D)；
# 3. profile=smoke/pilot/official 控制样本切片与搜索预算；
# 4. 双 GPU 调度：每个 GPU 同时只跑一个任务，空闲时分配下一个；
# 5. 监控 GPU 显存，超过 GPU_MEM_LIMIT（默认 3500 MiB）报警并 kill；
# 6. 支持断点续传（RESUME_DIR 环境变量）。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO"

# ===== conda 激活 =====
CONDA_ENV="${CONDA_ENV:-mol-edit}"
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)"
fi

# ===== GPU 调度配置 =====
IFS=',' read -r -a GPU_LIST <<< "${GPU_LIST:-0,1}"
GPU_MEM_LIMIT="${GPU_MEM_LIMIT:-5120}"  # MiB，超过此值 kill 并报警
POLL_INTERVAL="${POLL_INTERVAL:-60}"    # 秒

# ===== 实验参数 =====
PYTHON_BIN="${PYTHON_BIN:-python}"
PROFILE="${PROFILE:-official}"
RUN_EVAL="${RUN_EVAL:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-0}"
TASKS="${TASKS:-lumo_up,homo_down}"
VARIANTS="${VARIANTS:-full,wo_prior,wo_leaf_value,random_topb}"
SEEDS="${SEEDS:-42}"

# 输出目录：支持断点续传
if [[ -n "${RESUME_DIR:-}" ]] && [[ -d "$RESUME_DIR" ]]; then
  OUTPUT_ROOT="$RESUME_DIR"
  echo "[断点续传] 复用已有输出目录: $OUTPUT_ROOT"
else
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/mol_evo/output/paper/ablations/$TIMESTAMP}"
fi
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

# ===== profile 参数 =====
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

# ===== 辅助函数 =====
log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

# ===== 构建任务队列 =====
# 每个条目格式: "task|variant|seed"
TASKS_QUEUE=()
aTasks="${TASKS//,/ }"
aVariants="${VARIANTS//,/ }"
aSeeds="${SEEDS//,/ }"

for task in $aTasks; do
  for variant in $aVariants; do
    for seed in $aSeeds; do
      TASKS_QUEUE+=("${task}|${variant}|${seed}")
    done
  done
done

log "构建任务队列: ${#TASKS_QUEUE[@]} 个任务"
log "  tasks=$TASKS  variants=$VARIANTS  seeds=$SEEDS"

# ===== 解析 task → model/direction =====
get_task_params() {
  local task="$1"
  case "$task" in
    lumo_up)    echo "lumo|increase|$LUMO_MODEL_DIR|$LUMO_MODEL_PATH" ;;
    lumo_down)  echo "lumo|decrease|$LUMO_MODEL_DIR|$LUMO_MODEL_PATH" ;;
    homo_up)    echo "homo|increase|$HOMO_MODEL_DIR|$HOMO_MODEL_PATH" ;;
    homo_down)  echo "homo|decrease|$HOMO_MODEL_DIR|$HOMO_MODEL_PATH" ;;
    *) echo "unknown" ;;
  esac
}

# ===== 解析 variant → mcts 参数 =====
get_variant_args() {
  local variant="$1"
  local prior_mode="softmax"
  local value_mode="accumulated"
  local expansion_mode="topk"
  local v_pruning_patience="$PRUNING_PATIENCE"
  local v_logp_min="$LOGP_MIN"
  local v_logp_max="$LOGP_MAX"
  local v_logp_patience="$LOGP_PATIENCE"

  case "$variant" in
    full)           ;;
    wo_prior)       prior_mode="uniform" ;;
    wo_leaf_value)  value_mode="zero" ;;
    random_topb)    expansion_mode="random_topk" ;;
    full_expand)    expansion_mode="full" ;;
    wo_pruning)     v_pruning_patience="0" ;;
    wo_logp)        v_logp_patience="0"; v_logp_min="-999"; v_logp_max="999" ;;
    *)
      echo "不支持的 variant: $variant" >&2
      return 1
      ;;
  esac

  echo "${prior_mode}|${value_mode}|${expansion_mode}|${v_pruning_patience}|${v_logp_min}|${v_logp_max}|${v_logp_patience}"
}

# ===== 获取 run 标签和输出路径 =====
get_run_paths() {
  local task="$1" variant="$2" seed="$3"
  local run_root="$OUTPUT_ROOT/$task/$variant/seed${seed}"
  local search_dir="$run_root/search"
  local batch_json="$search_dir/batch_results.json"
  echo "${run_root}|${search_dir}|${batch_json}"
}

# ===== 状态跟踪 =====
declare -A GPU_TMUX       # GPU_TMUX[gpu_id] = tmux_session_name
declare -A GPU_TASK       # GPU_TASK[gpu_id] = task_entry
TASK_INDEX=0
COMPLETED=0
FAILED=0
SKIPPED=0
ALARM_TRIGGERED=0

# ===== 启动一个任务到指定 GPU =====
launch_task() {
  local gpu_id="$1"
  local task_entry="$2"

  IFS='|' read -r task variant seed <<< "$task_entry"

  # 解析 task 参数
  local task_params
  task_params=$(get_task_params "$task")
  IFS='|' read -r target_prop direction model_dir model_path <<< "$task_params"

  # 解析 variant 参数
  local variant_args
  variant_args=$(get_variant_args "$variant")
  IFS='|' read -r prior_mode value_mode expansion_mode v_pruning v_logp_min v_logp_max v_logp_patience <<< "$variant_args"

  # 输出路径
  local paths
  paths=$(get_run_paths "$task" "$variant" "$seed")
  IFS='|' read -r run_root search_dir batch_json <<< "$paths"

  local search_log="$run_root/search.log"
  local eval_log="$run_root/evaluate_batch.log"
  local csv_eval_log="$run_root/evaluate_csv.log"
  local session_name="ablation_gpu${gpu_id}_${task}_${variant}_s${seed}"

  mkdir -p "$run_root" "$search_dir"

  # 构建搜索命令
  local cmd="cd $REPO && source ~/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && "
  cmd+="CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON_BIN -m mol_evo.scripts.optimization.batch_optimizer "
  cmd+="--input-csv $INPUT_CSV "
  cmd+="--output-dir $search_dir "
  cmd+="--output-json $batch_json "
  cmd+="--model-path $model_path "
  cmd+="--model-dir $model_dir "
  cmd+="--config-file $CONFIG_FILE "
  cmd+="--target-property $target_prop "
  cmd+="--optimization-mode $OPTIMIZATION_MODE "
  cmd+="--search-mode mcts "
  cmd+="--num-simulations $NUM_SIMULATIONS "
  cmd+="--exploration-weight $EXPLORATION_WEIGHT "
  cmd+="--mcts-prior-mode $prior_mode "
  cmd+="--mcts-value-mode $value_mode "
  cmd+="--mcts-expansion-mode $expansion_mode "
  cmd+="--mcts-random-seed $seed "
  cmd+="--direction $direction "
  cmd+="--max-depth $MAX_DEPTH "
  cmd+="--max-branching $MAX_BRANCHING "
  cmd+="--pruning-patience $v_pruning "
  cmd+="--logp-min $v_logp_min "
  cmd+="--logp-max $v_logp_max "
  cmd+="--logp-patience $v_logp_patience "
  cmd+="--topK $TOPK "
  cmd+="--start-index $START_INDEX "
  cmd+="--end-index $END_INDEX "
  cmd+="2>&1 | tee $search_log; "

  # 搜索完成后跑评估（如果开启）
  if [[ "$RUN_EVAL" == "1" ]]; then
    cmd+="if [ \$? -eq 0 ] || true; then "
    cmd+="$PYTHON_BIN $REPO/../utils/evaluate_batch_mo.py "
    cmd+="--result-dir $search_dir "
    cmd+="--target-prop $target_prop "
    cmd+="--direction $direction "
    cmd+="--item-size $ITEM_SIZE "
    cmd+="--tag ofo "
    if [[ -n "$MAX_FILES" ]]; then
      cmd+="--max-files $MAX_FILES "
    fi
    cmd+="2>&1 | tee $eval_log; "
    cmd+="fi; "
  fi

  cmd+="echo '=== TASK_DONE ===' >> $search_log"

  # 创建 tmux session（unset TMUX 以支持从 tmux 内部启动子 session）
  TMUX= tmux kill-session -t "$session_name" 2>/dev/null || true
  TMUX= tmux new-session -d -s "$session_name" "$cmd"

  GPU_TMUX[$gpu_id]="$session_name"
  GPU_TASK[$gpu_id]="$task_entry"

  log "GPU $gpu_id: 启动任务 [${task}/${variant}/seed${seed}] → session=$session_name"
  log "  search_dir: $search_dir"
}

# ===== 检查 GPU 上任务是否完成 =====
is_task_done() {
  local gpu_id="$1"
  local session_name="${GPU_TMUX[$gpu_id]:-}"
  [[ -z "$session_name" ]] && return 0

  # 检查 tmux session 是否还存在
  if ! TMUX= tmux has-session -t "$session_name" 2>/dev/null; then
    return 0  # session 已退出 = 任务完成
  fi

  # 检查 log 文件中是否有 TASK_DONE 标记
  local task_entry="${GPU_TASK[$gpu_id]}"
  IFS='|' read -r task variant seed <<< "$task_entry"
  local paths
  paths=$(get_run_paths "$task" "$variant" "$seed")
  IFS='|' read -r run_root search_dir batch_json <<< "$paths"
  local search_log="$run_root/search.log"

  if [[ -f "$search_log" ]] && grep -q "=== TASK_DONE ===" "$search_log" 2>/dev/null; then
    return 0
  fi

  return 1
}

# ===== 检查任务是否成功 =====
check_task_result() {
  local gpu_id="$1"
  local task_entry="${GPU_TASK[$gpu_id]}"
  IFS='|' read -r task variant seed <<< "$task_entry"
  local paths
  paths=$(get_run_paths "$task" "$variant" "$seed")
  IFS='|' read -r run_root search_dir batch_json <<< "$paths"

  if [[ -f "$batch_json" ]] && [[ -s "$batch_json" ]]; then
    log "GPU $gpu_id: ✅ 任务 [${task}/${variant}/seed${seed}] 成功完成"
    COMPLETED=$((COMPLETED + 1))
  else
    log "GPU $gpu_id: ❌ 任务 [${task}/${variant}/seed${seed}] 失败（输出文件不存在或为空）"
    FAILED=$((FAILED + 1))
  fi

  # 清理 tmux
  local session_name="${GPU_TMUX[$gpu_id]:-}"
  TMUX= tmux kill-session -t "$session_name" 2>/dev/null || true
  GPU_TMUX[$gpu_id]=""
  GPU_TASK[$gpu_id]=""
}

# ===== GPU 显存监控 =====
check_gpu_memory() {
  for gpu_id in "${GPU_LIST[@]}"; do
    local session_name="${GPU_TMUX[$gpu_id]:-}"
    [[ -z "$session_name" ]] && continue

    # 通过 tmux session 获取我们启动的进程 PID
    local tmux_pid
    tmux_pid=$(TMUX= tmux list-panes -t "$session_name" -F '#{pane_pid}' 2>/dev/null | head -1 || true)
    [[ -z "$tmux_pid" ]] && continue

    # 获取该 tmux session 下所有子进程 PID
    local our_pids
    our_pids=$(pstree -p "$tmux_pid" 2>/dev/null | grep -oP '\(\K[0-9]+(?=\))' | tr '\n' '|' | sed 's/|$//' || true)
    [[ -z "$our_pids" ]] && continue

    # 查询 GPU 上属于我们进程的显存占用
    local our_mem
    our_mem=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$gpu_id" 2>/dev/null | \
      awk -F',' -v pids="$our_pids" 'BEGIN{split(pids,a,"|")} {gsub(/[^0-9]/,"",$1); gsub(/[^0-9]/,"",$2); for(i in a) if($1==a[i]) sum+=$2} END {print sum+0}' || true)

    if [[ "$our_mem" -gt "$GPU_MEM_LIMIT" ]]; then
      log "🚨 GPU $gpu_id: 进程占用 ${our_mem} MiB > ${GPU_MEM_LIMIT} MiB 限制！"
      log "🚨 正在 kill 任务 [${GPU_TASK[$gpu_id]}]..."
      TMUX= tmux kill-session -t "$session_name" 2>/dev/null || true
      GPU_TMUX[$gpu_id]=""
      GPU_TASK[$gpu_id]=""
      FAILED=$((FAILED + 1))
      ALARM_TRIGGERED=$((ALARM_TRIGGERED + 1))
    elif [[ "$our_mem" -gt 0 ]]; then
      local task_entry="${GPU_TASK[$gpu_id]}"
      IFS='|' read -r t v s <<< "$task_entry"
      log "  GPU $gpu_id: [${t}/${v}/seed${s}] 占用 ${our_mem} MiB"
    fi
  done
}

# ===== 断点续传：跳过已完成的任务 =====
# 判断标准：search 目录中 _topK.csv 文件数 >= 预期分子数 (END_INDEX - START_INDEX)
EXPECTED_MOLECULES=$((END_INDEX - START_INDEX))
PENDING_TASKS=()
for task_entry in "${TASKS_QUEUE[@]}"; do
  IFS='|' read -r task variant seed <<< "$task_entry"
  local_paths=$(get_run_paths "$task" "$variant" "$seed")
  IFS='|' read -r run_root search_dir batch_json <<< "$local_paths"

  if [[ -d "$search_dir" ]]; then
    topk_count=$(find "$search_dir" -maxdepth 1 -name "*_topK.csv" 2>/dev/null | wc -l)
    if [[ $topk_count -ge $EXPECTED_MOLECULES ]]; then
      log "⏭️  跳过已完成任务 [${task}/${variant}/seed${seed}] (${topk_count}/${EXPECTED_MOLECULES} topK files)"
      SKIPPED=$((SKIPPED + 1))
      COMPLETED=$((COMPLETED + 1))
    else
      log "🔄 需要 resume [${task}/${variant}/seed${seed}] (${topk_count}/${EXPECTED_MOLECULES} topK files)"
      PENDING_TASKS+=("$task_entry")
    fi
  else
    PENDING_TASKS+=("$task_entry")
  fi
done

# ===== 写入 manifest =====
cat > "$OUTPUT_ROOT/run_manifest.txt" <<EOF
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
GPU_LIST=${GPU_LIST[*]}
GPU_MEM_LIMIT=$GPU_MEM_LIMIT
RUN_EVAL=$RUN_EVAL
RUN_SUMMARY=$RUN_SUMMARY
EOF

# ===== 主循环启动 =====
log "======================================================"
log "MCTS 消融实验 — 双 GPU 调度"
log "输出目录: $OUTPUT_ROOT"
log "Profile: $PROFILE (分子 ${START_INDEX}–${END_INDEX}, sims=${NUM_SIMULATIONS})"
log "任务队列: ${#PENDING_TASKS[@]} 个待执行 (共 ${#TASKS_QUEUE[@]} 个, 跳过=${SKIPPED})"
log "GPU: ${GPU_LIST[*]}  显存限制: ${GPU_MEM_LIMIT} MiB"
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

# ===== 汇总脚本（可选）=====
if [[ "$RUN_SUMMARY" == "1" ]] && [[ -f "$REPO/mol_evo/scripts/summarize_mcts_ablation_runs.py" ]]; then
  log "运行汇总脚本..."
  "$PYTHON_BIN" "$REPO/mol_evo/scripts/summarize_mcts_ablation_runs.py" --run-root "$OUTPUT_ROOT" || true
fi

# ===== 结束汇总 =====
log "======================================================"
log "全部任务结束"
log "总任务数   : ${#TASKS_QUEUE[@]}"
log "跳过(续传) : $SKIPPED"
log "本次成功   : $((COMPLETED - SKIPPED))"
log "本次失败   : $FAILED"
log "显存报警   : $ALARM_TRIGGERED"
log "输出目录   : $OUTPUT_ROOT"
log "======================================================"

if [[ $FAILED -gt 0 || $ALARM_TRIGGERED -gt 0 ]]; then
  exit 1
fi
