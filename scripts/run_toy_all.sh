#!/usr/bin/env bash
set -euo pipefail

for model in m0 m1 m2 m3 m4; do
  python -m src.main_train --config "configs/toy/${model}.yaml" --seed 0
done

for model_id in M0 M1 M2 M3 M4; do
  python -m src.main_eval --run_dir "runs/toy/${model_id}/seed0" --nfe_list 10,20,50,100,200
done
