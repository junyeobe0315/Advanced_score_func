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
