#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/toy/m3.yaml}"
SEEDS="${SEEDS:-0,1,2}"
TOTAL_STEPS="${TOTAL_STEPS:-30000}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SKIP_COMPLETE="${SKIP_COMPLETE:-0}"
CLEAN_RUN_DIR="${CLEAN_RUN_DIR:-0}"

IFS=',' read -r -a seed_list <<< "$SEEDS"

echo "[info] config=${CONFIG_PATH}"
echo "[info] seeds=${SEEDS}"
echo "[info] total_steps=${TOTAL_STEPS}"
echo "[info] force_rerun=${FORCE_RERUN}"
echo "[info] skip_complete=${SKIP_COMPLETE}"
echo "[info] clean_run_dir=${CLEAN_RUN_DIR}"

for raw_seed in "${seed_list[@]}"; do
  seed="${raw_seed//[[:space:]]/}"
  if [[ -z "$seed" ]]; then
    continue
  fi

  run_dir="runs/toy/M3/seed${seed}"
  metrics_path="${run_dir}/metrics.csv"
  last_step=""
  is_complete=0

  if [[ -f "$metrics_path" ]]; then
    last_step="$(tail -n 1 "$metrics_path" | cut -d, -f1 || true)"
    if [[ "$last_step" =~ ^[0-9]+$ && "$last_step" -ge "$TOTAL_STEPS" ]]; then
      is_complete=1
    fi
  fi

  if [[ "$is_complete" == "1" && "$FORCE_RERUN" != "1" && "$SKIP_COMPLETE" == "1" ]]; then
    echo "[skip] ${run_dir} already complete (last_step=${last_step})"
    continue
  fi

  if [[ -d "$run_dir" && "$CLEAN_RUN_DIR" == "1" ]]; then
    echo "[clean] removing existing run dir: ${run_dir}"
    rm -rf "$run_dir"
  fi

  if [[ -f "$metrics_path" && "$FORCE_RERUN" != "1" && "$is_complete" == "0" ]]; then
    echo "[rerun] ${run_dir} is incomplete (last_step=${last_step:-unknown}); restarting training"
  fi
  if [[ "$is_complete" == "1" && ("$FORCE_RERUN" == "1" || "$SKIP_COMPLETE" == "0") ]]; then
    echo "[rerun] ${run_dir} is complete (last_step=${last_step}); running again by policy"
  fi

  echo "[train] M3 toy seed=${seed}"
  python -m src.main_train --config "$CONFIG_PATH" --seed "$seed"
done
