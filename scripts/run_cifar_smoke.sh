#!/usr/bin/env bash
set -euo pipefail

for config_name in m0 m1 m2_epoch_matched m3 m4; do
  python -m src.main_train --config "configs/cifar10/${config_name}.yaml" --seed 0 --override train.total_steps=2000 train.ckpt_every=1000 compute.tier='smoke'
done

for model_id in M0 M1 M2 M3 M4; do
  python -m src.main_eval --run_dir "runs/cifar10/${model_id}/seed0" --nfe_list 8,18,32
done
