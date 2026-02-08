#!/usr/bin/env bash
set -euo pipefail

python -m src.main_train --config configs/cifar_baseline.yaml --seed 0 --override train.total_steps=2000 train.ckpt_every=1000 compute.tier='smoke'
python -m src.main_train --config configs/cifar_reg.yaml --seed 0 --override train.total_steps=2000 train.ckpt_every=1000 compute.tier='smoke'
python -m src.main_train --config configs/cifar_struct.yaml --seed 0 --override train.total_steps=2000 train.ckpt_every=1000 compute.tier='smoke'

python -m src.main_eval --run_dir runs/cifar10/baseline/seed0 --nfe_list 10,20,50
python -m src.main_eval --run_dir runs/cifar10/reg/seed0 --nfe_list 10,20,50
python -m src.main_eval --run_dir runs/cifar10/struct/seed0 --nfe_list 10,20,50
