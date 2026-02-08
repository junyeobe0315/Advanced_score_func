from __future__ import annotations

import torch

from .common import compute_dsm_for_score


def _score_from_potential(model: torch.nn.Module, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Convert potential network output into score via input gradient.

    Args:
        model: Potential model returning ``phi(x, sigma)``.
        x: Noisy input tensor.
        sigma: Per-sample sigma tensor.

    Returns:
        Score tensor with same shape as ``x``.

    How it works:
        Enables input gradients, evaluates scalar potential, then computes
        ``grad_x phi`` with ``torch.autograd.grad``.
    """
    x_req = x.requires_grad_(True)
    phi = model(x_req, sigma)
    if phi.ndim == 1:
        phi = phi[:, None]
    score = torch.autograd.grad(phi.sum(), x_req, create_graph=True)[0]
    return score


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
    sigma_min = float(cfg["loss"]["sigma_min"])
    sigma_max = float(cfg["loss"]["sigma_max"])
    weight_mode = str(cfg["loss"].get("weight_mode", "sigma2"))

    dsm, _ = compute_dsm_for_score(
        score_fn=lambda x, s: _score_from_potential(model, x, s),
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
