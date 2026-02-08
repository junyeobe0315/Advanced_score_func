from __future__ import annotations

import torch

from src.losses import reg_loop_estimator, reg_sym_estimator

from .common import compute_dsm_for_score


def _should_apply_regularizer(step: int, cfg: dict, sigma: torch.Tensor) -> bool:
    freq = int(cfg["loss"].get("reg_freq", 1))
    if step % max(freq, 1) != 0:
        return False

    if bool(cfg["loss"].get("reg_low_noise_only", False)):
        thr = float(cfg["loss"].get("reg_low_noise_threshold", 0.25))
        return bool((sigma <= thr).float().mean().item() > 0.5)
    return True


def train_step_reg(
    model: torch.nn.Module,
    x0: torch.Tensor,
    cfg: dict,
    step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    sigma_min = float(cfg["loss"]["sigma_min"])
    sigma_max = float(cfg["loss"]["sigma_max"])
    weight_mode = str(cfg["loss"].get("weight_mode", "sigma2"))

    lam = float(cfg["loss"].get("lambda_sym", 0.0))
    mu = float(cfg["loss"].get("mu_loop", 0.0))
    reg_k = int(cfg["loss"].get("reg_k", 1))
    delta = float(cfg["loss"].get("loop_delta", 0.01))
    sparse_ratio = float(cfg["loss"].get("loop_sparse_ratio", 1.0))

    dsm, cache = compute_dsm_for_score(
        score_fn=model,
        x0=x0,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        weight_mode=weight_mode,
    )

    x = cache["x"]
    sigma = cache["sigma"]

    sym = torch.zeros((), device=x0.device)
    loop = torch.zeros((), device=x0.device)

    if _should_apply_regularizer(step=step, cfg=cfg, sigma=sigma):
        if lam > 0.0:
            sym = reg_sym_estimator(model, x=x, sigma=sigma, K=reg_k)
        if mu > 0.0:
            loop = reg_loop_estimator(model, x=x, sigma=sigma, delta=delta, sparse_ratio=sparse_ratio)

    total = dsm + lam * sym + mu * loop
    metrics = {
        "loss_total": float(total.detach().item()),
        "loss_dsm": float(dsm.detach().item()),
        "loss_sym": float(sym.detach().item()),
        "loss_loop": float(loop.detach().item()),
    }
    return total, metrics
