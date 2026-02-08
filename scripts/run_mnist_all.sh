#!/usr/bin/env bash
set -euo pipefail

for seed in 0 1 2; do
  python -m src.main_train --config configs/mnist_baseline.yaml --seed "$seed"
  python -m src.main_train --config configs/mnist_reg.yaml --seed "$seed"
  python -m src.main_train --config configs/mnist_struct.yaml --seed "$seed"

done

for seed in 0 1 2; do
  python -m src.main_eval --run_dir "runs/mnist/baseline/seed${seed}" --nfe_list 10,20,50,100,200
  python -m src.main_eval --run_dir "runs/mnist/reg/seed${seed}" --nfe_list 10,20,50,100,200
  python -m src.main_eval --run_dir "runs/mnist/struct/seed${seed}" --nfe_list 10,20,50,100,200
done
