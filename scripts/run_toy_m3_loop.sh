#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/toy/m3.yaml}"
SEEDS="${SEEDS:-0,1,2}"
TOTAL_STEPS="${TOTAL_STEPS:-30000}"
FORCE_RERUN="${FORCE_RERUN:-0}"

IFS=',' read -r -a seed_list <<< "$SEEDS"

echo "[info] config=${CONFIG_PATH}"
echo "[info] seeds=${SEEDS}"
echo "[info] total_steps=${TOTAL_STEPS}"
echo "[info] force_rerun=${FORCE_RERUN}"

for raw_seed in "${seed_list[@]}"; do
  seed="${raw_seed//[[:space:]]/}"
  if [[ -z "$seed" ]]; then
    continue
  fi

  run_dir="runs/toy/M3/seed${seed}"
  metrics_path="${run_dir}/metrics.csv"

  if [[ -f "$metrics_path" && "$FORCE_RERUN" != "1" ]]; then
    last_step="$(tail -n 1 "$metrics_path" | cut -d, -f1 || true)"
    if [[ "$last_step" =~ ^[0-9]+$ && "$last_step" -ge "$TOTAL_STEPS" ]]; then
      echo "[skip] ${run_dir} already complete (last_step=${last_step})"
      continue
    fi
    echo "[skip] ${run_dir} exists (last_step=${last_step:-unknown}); set FORCE_RERUN=1 to overwrite"
    continue
  fi

  echo "[train] M3 toy seed=${seed}"
  python -m src.main_train --config "$CONFIG_PATH" --seed "$seed"
done
