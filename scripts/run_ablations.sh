#!/usr/bin/env bash
set -euo pipefail

python -m src.main_sweep --sweep configs/ablations/reg_lambda_only.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/reg_mu_only.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/reg_both.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/estimator_k1.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/estimator_k2.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/loop_delta_small.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/loop_delta_smaller.yaml --seeds 0,1,2
