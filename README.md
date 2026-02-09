# Jacobian-Free Nonlocal Integrability Benchmark (M0-M4)

PyTorch-only codebase for 5-way score-model comparison:
- `M0`: EDM-preconditioned DSM baseline score model
- `M1`: Jacobian-asymmetry regularized score model (QCSBM-style)
- `M2`: fully conservative model (`s = grad_x phi`)
- `M3`: Jacobian-free nonlocal integrability regularization (multi-scale loops + graph cycles)
- `M4`: hybrid low-noise hard-conservative model with boundary matching

GitHub repository:
- https://github.com/junyeobe0315/Advanced_score_func.git

## Environment

```bash
conda env create -f environment.yml
conda activate advance_score
```

If CUDA package resolution fails in conda, install PyTorch wheels first and then install dependencies:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
pip install -r requirements.txt
```

## Training / Evaluation CLI

```bash
python -m src.main_train --config configs/cifar10/m3.yaml --seed 0
python -m src.main_eval --run_dir runs/cifar10/M3/seed0 --nfe_list 8,18,32,64,128
python -m src.main_sweep --sweep configs/toy/m1.yaml --seeds 0,1,2
```

Outputs are saved under:

```text
runs/{dataset}/{model_id}/seed{n}/
```

Artifacts per run:
- `config_resolved.yaml`
- `metrics.json`
- `metrics.csv`
- `tb/`
- `checkpoints/step_*.pt`
- `eval/fid_vs_nfe.csv`
- `eval/integrability_vs_sigma.csv`
- `eval/compute_summary.json`
- `reports/steps_to_target_fid*.csv` (via `scripts/make_report_tables.py`)
- `reports/compute_matched_fid.csv` (via `scripts/make_report_tables.py`)

## Quick Run Scripts

```bash
bash scripts/run_toy_all.sh
bash scripts/run_mnist_all.sh
bash scripts/run_cifar_smoke.sh
bash scripts/run_cifar_main.sh
bash scripts/run_imagenet128_all.sh
bash scripts/run_imagenet256_all.sh
bash scripts/run_imagenet512_all.sh
bash scripts/run_lsun256_all.sh
bash scripts/run_ffhq256_all.sh
```

`run_toy_all.sh` / `run_mnist_all.sh` / `run_cifar_main.sh` default to 3 seeds (`0,1,2`).
`run_cifar_smoke.sh` runs the 5 models with short steps for pipeline validation.

## Notes (EDM + M3/M4)

- Score branches (`M0/M1/M3` and `M4` high-noise branch) use EDM-style preconditioning wrappers from config.
- Common base training objective is EDM denoiser loss:
  - `sigma ~ exp(N(P_mean, P_std))` (configurable, clamped to `[sigma_min, sigma_max]`)
  - `w(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2`
  - denoiser is recovered from score by `D(x,sigma) = x + sigma^2 s(x,sigma)`
- Samplers (`Euler`, `Heun`) follow EDM denoiser-based drift
  - `dx/dsigma = (x - D(x,sigma))/sigma`
- M1 defaults to QCSBM-style asymmetry estimator on image runs:
  - variant: `traceJJt - traceJJ`
  - probe distribution: Rademacher
  - per-sample `sigma^2` scaling before batch reduction
  - default M1 regularization uses all noise levels (`reg_low_noise_only: false`)
- M3 graph-cycle loss supports same-`t` training (`loss.cycle_same_sigma: true`).
- M3/M4 expose per-term frequencies for speed/compute tradeoff:
  - `loss.loop_freq`, `loss.cycle_freq`, `loss.match_freq`

## Notebook Tools

- `notebooks/toy_visual_dashboard.ipynb`: 기존 run artifact를 읽어 Toy 결과를 한눈에 비교.
- `notebooks/toy_tiny_training_viz.ipynb`: 아주 작은 학습 루프를 epoch마다 직접 시각화해 학습 속도/품질 비교.

## 5-Way Config Sets

- Toy: `configs/toy/m0.yaml` ... `configs/toy/m4.yaml`
- MNIST: `configs/mnist/m0.yaml` ... `configs/mnist/m4.yaml`
- CIFAR-10: `configs/cifar10/m0.yaml` ... `configs/cifar10/m4.yaml`
- ImageNet-128: `configs/imagenet128/m0.yaml` ... `configs/imagenet128/m4.yaml`
- ImageNet-256: `configs/imagenet256/m0.yaml` ... `configs/imagenet256/m4.yaml`
- ImageNet-512: `configs/imagenet512/m0.yaml` ... `configs/imagenet512/m4.yaml`
- LSUN-256: `configs/lsun256/m0.yaml` ... `configs/lsun256/m4.yaml`
- FFHQ-256: `configs/ffhq256/m0.yaml` ... `configs/ffhq256/m4.yaml`

## References (Paper + Code)

### Diffusion / Score Frameworks
- EDM (Karras et al., 2022) code: https://github.com/NVlabs/edm
- Score SDE (Song et al., 2021) code: https://github.com/yang-song/score_sde_pytorch
- Score SDE legacy repo: https://github.com/yang-song/score_sde

### Prior-inspired Baselines
- QCSBM project: https://chen-hao-chao.github.io/qcsbm/
- QCSBM code: https://github.com/chen-hao-chao/qcsbm
- Reduce-Reuse-Recycle project: https://energy-based-model.github.io/reduce-reuse-recycle/
- Reduce-Reuse-Recycle code: https://github.com/yilundu/reduce_reuse_recycle

### Related Energy/Composition
- Thornton et al. (AISTATS 2025): https://proceedings.mlr.press/v258/thornton25a.html
- arXiv: https://arxiv.org/abs/2502.12786

### Evaluation Tooling
- torch-fidelity: https://github.com/toshas/torch-fidelity
- clean-fid: https://github.com/GaParmar/clean-fid
- pytorch-fid: https://github.com/mseitzer/pytorch-fid

### Sampling Utilities
- k-diffusion: https://github.com/crowsonkb/k-diffusion

## Report Tables

```bash
python scripts/make_report_tables.py --run_root runs
```

Generated files:
- `reports/fid_summary.csv`
- `reports/integrability_summary.csv`
- `reports/steps_to_target_fid.csv`
- `reports/steps_to_target_fid_summary.csv`
- `reports/compute_budget_by_dataset.csv`
- `reports/compute_matched_fid.csv`
