from __future__ import annotations

import torch

from src.models.potential_net import score_from_potential

from .common import compute_dsm_for_score


def train_step_struct(model: torch.nn.Module, x0: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one training-step objective for structural integrability model.

    Args:
        model: Potential model.
        x0: Clean training batch.
        cfg: Resolved config dictionary.

    Returns:
        Tuple ``(loss, metrics)`` where loss is DSM over ``grad_x phi``.

    How it works:
        Wraps potential model with gradient-based score function and computes
        standard DSM objective without extra soft regularizers.
    """
    loss_cfg = cfg["loss"]
    sigma_min = float(loss_cfg["sigma_min"])
    sigma_max = float(loss_cfg["sigma_max"])
    weight_mode = str(loss_cfg.get("weight_mode", "sigma2"))
    objective = str(loss_cfg.get("objective", "dsm_score"))
    sigma_sampling = str(loss_cfg.get("sigma_sampling", "log_uniform"))
    edm_p_mean = float(loss_cfg.get("edm_p_mean", -1.2))
    edm_p_std = float(loss_cfg.get("edm_p_std", 1.2))
    sigma_sample_clamp = bool(loss_cfg.get("sigma_sample_clamp", True))
    sigma_data = float(loss_cfg.get("sigma_data", cfg.get("model", {}).get("preconditioning", {}).get("sigma_data", 0.5)))

    dsm, _ = compute_dsm_for_score(
        score_fn=lambda x, s: score_from_potential(model, x, s, create_graph=True),
        x0=x0,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        weight_mode=weight_mode,
        objective=objective,
        sigma_sampling=sigma_sampling,
        edm_p_mean=edm_p_mean,
        edm_p_std=edm_p_std,
        sigma_sample_clamp=sigma_sample_clamp,
        sigma_data=sigma_data,
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
