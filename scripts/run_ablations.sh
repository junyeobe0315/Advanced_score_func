#!/usr/bin/env bash
set -euo pipefail

python -m src.main_sweep --sweep configs/ablations/reg_lambda_only.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/reg_mu_only.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/reg_both.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/estimator_k1.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/estimator_k2.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/loop_delta_small.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/loop_delta_smaller.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/m3_mu1_only.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/m3_mu2_only.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/m3_single_scale.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/m3_cycle_len3_only.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/m3_all_noise.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/m4_alpha0.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/m4_beta0.yaml --seeds 0,1,2
python -m src.main_sweep --sweep configs/ablations/m4_sigma_c_075.yaml --seeds 0,1,2
