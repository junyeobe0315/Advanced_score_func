#!/usr/bin/env bash
set -euo pipefail

for seed in 0 1 2; do
  python -m src.main_train --config configs/imagenet128_baseline.yaml --seed "$seed"
  python -m src.main_train --config configs/imagenet128_reg.yaml --seed "$seed"
  python -m src.main_train --config configs/imagenet128_struct.yaml --seed "$seed"
done

for seed in 0 1 2; do
  python -m src.main_eval --run_dir "runs/imagenet128/M0/seed${seed}" --nfe_list 10,20,50,100,200
  python -m src.main_eval --run_dir "runs/imagenet128/M1/seed${seed}" --nfe_list 10,20,50,100,200
  python -m src.main_eval --run_dir "runs/imagenet128/M2/seed${seed}" --nfe_list 10,20,50,100,200
done
