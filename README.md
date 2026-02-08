# Integrability-Regularized vs Structurally Conservative Score Models

PyTorch-only implementation for comparing three score-model families:
- `baseline`: standard score network
- `reg`: score network with integrability regularizers (`R_sym`, `R_loop`)
- `struct`: potential network with score defined as gradient of scalar potential

GitHub repository:
- https://github.com/junyeobe0315/Advanced_score_func.git

## Environment

Conda (recommended):

```bash
conda env create -f environment.yml
conda activate advance_score
```

If your machine needs explicit CUDA wheels, install PyTorch separately first:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
pip install -r requirements.txt
```

## Train

```bash
python -m src.main_train --config configs/cifar_reg.yaml --seed 0
```

Outputs are written under:

```text
runs/{dataset}/{variant}/seed{n}/
```

## Eval

```bash
python -m src.main_eval --run_dir runs/cifar10/reg/seed0 --nfe_list 10,20,50,100,200
```

This writes:
- `eval/fid_vs_nfe.csv`
- `eval/integrability_vs_sigma.csv`

## Sweep

```bash
python -m src.main_sweep --sweep configs/ablations/reg_both.yaml --seeds 0,1,2
```

## Notes

- `R_sym` uses randomized probing with `JVP` + `VJP`.
- `R_loop` uses small rectangle-loop approximation.
- Integrability metrics are logged by sigma bins (log-scale).
- CIFAR runs support 2-tier execution (`smoke` and `main`) using dedicated scripts.

## Added Datasets

### ImageNet
- Supported configs:
  - `configs/imagenet128_baseline.yaml`
  - `configs/imagenet128_reg.yaml`
  - `configs/imagenet128_struct.yaml`
  - `configs/imagenet256_baseline.yaml`
  - `configs/imagenet256_reg.yaml`
  - `configs/imagenet256_struct.yaml`
  - `configs/imagenet512_baseline.yaml`
  - `configs/imagenet512_reg.yaml`
  - `configs/imagenet512_struct.yaml`
- Run directory keys:
  - ImageNet-128 -> `runs/imagenet128/...`
  - ImageNet-256 -> `runs/imagenet256/...`
  - ImageNet-512 -> `runs/imagenet512/...`
- Expected directory layout:
```text
data/
  imagenet/
    train/
      class_x/*.jpg
    val/
      class_x/*.jpg
```

### LSUN
- Supported configs:
  - `configs/lsun256_baseline.yaml`
  - `configs/lsun256_reg.yaml`
  - `configs/lsun256_struct.yaml`
- Run directory key: `runs/lsun256/...`
- Default class setup is bedroom (`bedroom_train`, `bedroom_val`), configurable under `dataset.lsun.*`.

### FFHQ
- Supported configs:
  - `configs/ffhq256_baseline.yaml`
  - `configs/ffhq256_reg.yaml`
  - `configs/ffhq256_struct.yaml`
- Run directory key: `runs/ffhq256/...`
- Expected directory layout:
```text
data/
  ffhq/
    train/
      class_or_dummy/*.png
    val/
      class_or_dummy/*.png
```

## New Run Scripts
- `scripts/run_imagenet128_all.sh`
- `scripts/run_imagenet256_all.sh`
- `scripts/run_imagenet512_all.sh`
- `scripts/run_lsun256_all.sh`
- `scripts/run_ffhq256_all.sh`
