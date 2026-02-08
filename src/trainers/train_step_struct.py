from __future__ import annotations

import torch

from .common import compute_dsm_for_score


def _score_from_potential(model: torch.nn.Module, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    x_req = x.requires_grad_(True)
    phi = model(x_req, sigma)
    if phi.ndim == 1:
        phi = phi[:, None]
    score = torch.autograd.grad(phi.sum(), x_req, create_graph=True)[0]
    return score


def train_step_struct(model: torch.nn.Module, x0: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict[str, float]]:
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
