#!/usr/bin/env bash
set -euo pipefail

python -m src.main_train --config configs/toy_baseline.yaml --seed 0
python -m src.main_train --config configs/toy_reg.yaml --seed 0
python -m src.main_train --config configs/toy_struct.yaml --seed 0

python -m src.main_eval --run_dir runs/toy/baseline/seed0 --nfe_list 10,20,50,100,200
python -m src.main_eval --run_dir runs/toy/reg/seed0 --nfe_list 10,20,50,100,200
python -m src.main_eval --run_dir runs/toy/struct/seed0 --nfe_list 10,20,50,100,200
