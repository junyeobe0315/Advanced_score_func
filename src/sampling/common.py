from __future__ import annotations

import torch


def sigma_view(sigma: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast per-sample sigma tensor to match sample tensor dimensions."""
    return sigma.view(sigma.shape[0], *([1] * (x.ndim - 1)))


def edm_drift_from_score(
    x: torch.Tensor,
    score: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Compute EDM ODE drift ``(x - D(x,sigma)) / sigma`` from score."""
    return -sigma_view(sigma, x) * score
