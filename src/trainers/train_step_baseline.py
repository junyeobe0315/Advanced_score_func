from __future__ import annotations

import torch

from .common import compute_dsm_for_score


def train_step_baseline(model: torch.nn.Module, x0: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one baseline training step objective computation.

    Args:
        model: Score model that directly outputs ``s_theta(x, sigma)``.
        x0: Clean training batch.
        cfg: Resolved config dictionary.

    Returns:
        Tuple ``(loss, metrics)`` where ``loss`` is scalar tensor used for
        backprop and ``metrics`` contains detached scalar logs.

    How it works:
        Computes pure DSM loss without integrability regularizers.
    """
    sigma_min = float(cfg["loss"]["sigma_min"])
    sigma_max = float(cfg["loss"]["sigma_max"])
    weight_mode = str(cfg["loss"].get("weight_mode", "sigma2"))

    dsm, _ = compute_dsm_for_score(
        score_fn=model,
        x0=x0,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        weight_mode=weight_mode,
    )
    metrics = {
        "loss_total": float(dsm.detach().item()),
        "loss_dsm": float(dsm.detach().item()),
        "loss_sym": 0.0,
        "loss_loop": 0.0,
    }
    return dsm, metrics
