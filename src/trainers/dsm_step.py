from __future__ import annotations

from typing import Callable

import torch

from .common import compute_dsm_for_score


def run_dsm_only_step(
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one DSM-only objective step and return standard metrics payload."""
    loss_cfg = cfg["loss"]
    dsm, _ = compute_dsm_for_score(
        score_fn=score_fn,
        x0=x0,
        sigma_min=float(loss_cfg["sigma_min"]),
        sigma_max=float(loss_cfg["sigma_max"]),
        weight_mode=str(loss_cfg.get("weight_mode", "sigma2")),
        objective=str(loss_cfg.get("objective", "dsm_score")),
        sigma_sampling=str(loss_cfg.get("sigma_sampling", "log_uniform")),
        edm_p_mean=float(loss_cfg.get("edm_p_mean", -1.2)),
        edm_p_std=float(loss_cfg.get("edm_p_std", 1.2)),
        sigma_sample_clamp=bool(loss_cfg.get("sigma_sample_clamp", True)),
        sigma_data=float(loss_cfg.get("sigma_data", cfg.get("model", {}).get("preconditioning", {}).get("sigma_data", 0.5))),
        return_cache=False,
    )
    metrics = {
        "loss_total": float(dsm.detach().item()),
        "loss_dsm": float(dsm.detach().item()),
        "loss_sym": 0.0,
        "loss_loop": 0.0,
        "loss_loop_multi": 0.0,
        "loss_cycle": 0.0,
        "loss_match": 0.0,
    }
    return dsm, metrics
