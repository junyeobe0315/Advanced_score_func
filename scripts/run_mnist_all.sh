#!/usr/bin/env bash
set -euo pipefail

configs=(m0 m1 m2_epoch_matched m3 m4)

for seed in 0 1 2; do
  for config_name in "${configs[@]}"; do
    python -m src.main_train --config "configs/mnist/${config_name}.yaml" --seed "$seed"
  done
done

model_ids=(M0 M1 M2 M3 M4)
for seed in 0 1 2; do
  for model_id in "${model_ids[@]}"; do
    python -m src.main_eval --run_dir "runs/mnist/${model_id}/seed${seed}" --nfe_list 8,18,32,64,128
  done
done
