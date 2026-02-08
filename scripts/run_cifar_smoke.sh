#!/usr/bin/env bash
set -euo pipefail

for model in m0 m1 m2 m3 m4; do
  python -m src.main_train --config "configs/cifar10/${model}.yaml" --seed 0 --override train.total_steps=2000 train.ckpt_every=1000 compute.tier='smoke'
done

for model_id in M0 M1 M2 M3 M4; do
  python -m src.main_eval --run_dir "runs/cifar10/${model_id}/seed0" --nfe_list 10,20,50
done
