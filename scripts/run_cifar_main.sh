#!/usr/bin/env bash
set -euo pipefail

models=(m0 m1 m2 m3 m4)
for seed in 0 1 2; do
  for model in "${models[@]}"; do
    python -m src.main_train --config "configs/cifar10/${model}.yaml" --seed "$seed"
  done
done

model_ids=(M0 M1 M2 M3 M4)
for seed in 0 1 2; do
  for model_id in "${model_ids[@]}"; do
    python -m src.main_eval --run_dir "runs/cifar10/${model_id}/seed${seed}" --nfe_list 10,20,50,100,200
  done
done
