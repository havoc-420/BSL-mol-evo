#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO"

CONDA_ENV="${CONDA_ENV:-mol-tdc}"
if [[ -n "$CONDA_ENV" ]]; then
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
ABL_ROOT="${ABL_ROOT:-$REPO/mol_evo/output/paper/ablations}"
RUN_ROOTS="${RUN_ROOTS:-}"
TASKS="${TASKS:-}"
VARIANTS="${VARIANTS:-}"
SEEDS="${SEEDS:-}"
MAX_FILES="${MAX_FILES:-}"
RUN_CSV_EVAL="${RUN_CSV_EVAL:-1}"
FORCE="${FORCE:-0}"
TAG="${TAG:-ofo}"

contains_token() {
  local raw_list="$1"
  local token="$2"
  if [[ -z "$raw_list" ]]; then
    return 0
  fi
  local normalized="${raw_list//,/ }"
  for item in $normalized; do
    if [[ "$item" == "$token" ]]; then
      return 0
    fi
  done
  return 1
}

resolve_task() {
  local task="$1"
  case "$task" in
    lumo_up)
      TARGET_PROP="lumo"
      DIRECTION="increase"
      ;;
    lumo_down)
      TARGET_PROP="lumo"
      DIRECTION="decrease"
      ;;
    homo_up)
      TARGET_PROP="homo"
      DIRECTION="increase"
      ;;
    homo_down)
      TARGET_PROP="homo"
      DIRECTION="decrease"
      ;;
    *)
      echo "不支持的 task: $task" >&2
      return 1
      ;;
  esac
}

profile_to_item_size() {
  local profile="$1"
  case "$profile" in
    smoke)
      echo 10
      ;;
    pilot|official)
      echo 20
      ;;
    *)
      echo 20
      ;;
  esac
}

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
  echo "pilot"
}

run_and_log() {
  local label="$1"
  local log_file="$2"
  shift 2

  echo
  echo "[$(date '+%F %T')] $label"
  printf 'CMD: '
  printf '%q ' "$@"
  printf '\n'

  set +e
  "$@" 2>&1 | tee "$log_file"
  local exit_code=${PIPESTATUS[0]}
  set -e
  return "$exit_code"
}

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

main() {
  mapfile -t run_roots < <(discover_run_roots)
  if [[ ${#run_roots[@]} -eq 0 ]]; then
    echo "未找到可评估的 run root。ABL_ROOT=$ABL_ROOT RUN_ROOTS=$RUN_ROOTS" >&2
    exit 1
  fi

  for run_root in "${run_roots[@]}"; do
    if [[ ! -d "$run_root" ]]; then
      echo "跳过不存在的 run root: $run_root" >&2
      continue
    fi

    local profile item_size eval_summary_tsv
    profile="$(read_profile "$run_root")"
    item_size="$(profile_to_item_size "$profile")"
    eval_summary_tsv="$run_root/eval_summary.tsv"
    printf 'task\tvariant\tseed\tstatus\tprofile\titem_size\tsearch_dir\tbatch_eval_csv\tcsv_eval_dir\teval_log\tcsv_eval_log\n' > "$eval_summary_tsv"

    echo "============================================================"
    echo "run_root=$run_root"
    echo "profile=$profile item_size=$item_size"
    echo "eval_summary=$eval_summary_tsv"
    echo "============================================================"

    local task_dir
    while IFS= read -r -d '' task_dir; do
      local task
      task="$(basename "$task_dir")"
      if ! contains_token "$TASKS" "$task"; then
        continue
      fi
      resolve_task "$task"

      local variant_dir
      while IFS= read -r -d '' variant_dir; do
        local variant
        variant="$(basename "$variant_dir")"
        if ! contains_token "$VARIANTS" "$variant"; then
          continue
        fi

        local seed_dir
        while IFS= read -r -d '' seed_dir; do
          local seed_name seed search_dir eval_log csv_eval_log eval_base resume_dir batch_eval_csv csv_eval_dir status
          seed_name="$(basename "$seed_dir")"
          seed="${seed_name#seed}"
          if ! contains_token "$SEEDS" "$seed"; then
            continue
          fi

          search_dir="$seed_dir/search"
          if [[ ! -d "$search_dir" ]]; then
            echo "跳过缺少 search 目录的 run unit: $seed_dir" >&2
            continue
          fi

          eval_log="$seed_dir/evaluate_batch.log"
          csv_eval_log="$seed_dir/evaluate_csv.log"
          eval_base="$search_dir/.evaluation_results_${TARGET_PROP}_${DIRECTION}"
          resume_dir=""
          batch_eval_csv=""
          csv_eval_dir=""
          status="success"

          if [[ -d "$eval_base" ]]; then
            if [[ "$FORCE" != "1" ]]; then
              resume_dir="$(find "$eval_base" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1 || true)"
              batch_eval_csv="$(find "$eval_base" -type f -name 'batch_evaluation_results.csv' | sort | tail -n 1 || true)"
            fi
          fi

          if [[ -z "$batch_eval_csv" || "$FORCE" == "1" ]]; then
            local -a eval_cmd=(
              "$PYTHON_BIN" /home/ubuntu/mol_opt/utils/evaluate_batch_mo.py
              --result-dir "$search_dir"
              --target-prop "$TARGET_PROP"
              --direction "$DIRECTION"
              --item-size "$item_size"
              --tag "$TAG"
            )
            if [[ -n "$MAX_FILES" ]]; then
              eval_cmd+=(--max-files "$MAX_FILES")
            fi
            if [[ -n "$resume_dir" && "$FORCE" != "1" ]]; then
              eval_cmd+=(--resume-dir "$resume_dir")
            fi

            if ! run_and_log "${task}_${variant}_${seed_name}_eval_batch" "$eval_log" "${eval_cmd[@]}"; then
              status="eval_batch_failed"
            fi
          else
            echo "复用已有 batch 评估结果: $batch_eval_csv"
          fi

          if [[ -d "$eval_base" ]]; then
            batch_eval_csv="$(find "$eval_base" -type f -name 'batch_evaluation_results.csv' | sort | tail -n 1 || true)"
          fi

          if [[ "$status" == "success" && "$RUN_CSV_EVAL" == "1" ]]; then
            if [[ -z "$batch_eval_csv" ]]; then
              status="eval_csv_missing_input"
            else
              local -a csv_eval_cmd=(
                "$PYTHON_BIN" /home/ubuntu/mol_opt/utils/evaluate_csv_results.py
                --csv-file "$batch_eval_csv"
                --target-prop "$TARGET_PROP"
                --direction "$DIRECTION"
                --max-opt-molecules "$item_size"
              )
              if ! run_and_log "${task}_${variant}_${seed_name}_eval_csv" "$csv_eval_log" "${csv_eval_cmd[@]}"; then
                status="eval_csv_failed"
              else
                csv_eval_dir="$(find "$(dirname "$batch_eval_csv")" -mindepth 1 -maxdepth 1 -type d -name 'eval_*' | sort | tail -n 1 || true)"
              fi
            fi
          fi

          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$task" "$variant" "$seed" "$status" "$profile" "$item_size" "$search_dir" "$batch_eval_csv" "$csv_eval_dir" "$eval_log" "$csv_eval_log" \
            >> "$eval_summary_tsv"
        done < <(find "$variant_dir" -mindepth 1 -maxdepth 1 -type d -name 'seed*' -print0 | sort -z)
      done < <(find "$task_dir" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
    done < <(find "$run_root" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

    if [[ "$RUN_SUMMARY" == "1" ]]; then
      "$PYTHON_BIN" "$REPO/mol_evo/scripts/summarize_mcts_true_eval_runs.py" --run-root "$run_root"
    fi

    echo
    echo "run_root 评估完成: $run_root"
    echo "汇总文件: $eval_summary_tsv"
    if [[ "$RUN_SUMMARY" == "1" ]]; then
      echo "真值汇总: $run_root/true_eval_summary.tsv"
    fi
  done
}

main "$@"
