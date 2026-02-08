from __future__ import annotations

import torch


def sigma_weight(sigma: torch.Tensor, mode: str = "sigma2") -> torch.Tensor:
    if mode == "sigma2":
        return sigma ** 2
    if mode == "none":
        return torch.ones_like(sigma)
    raise ValueError(f"unknown weight mode: {mode}")


def dsm_target(eps: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    view_shape = [sigma.shape[0]] + [1] * (eps.ndim - 1)
    sigma_view = sigma.view(*view_shape)
    return -eps / sigma_view


def dsm_loss(
    score: torch.Tensor,
    x0: torch.Tensor,
    sigma: torch.Tensor,
    eps: torch.Tensor,
    weight_mode: str,
) -> torch.Tensor:
    del x0
    target = dsm_target(eps, sigma)
    err = (score - target) ** 2
    per_sample = err.flatten(start_dim=1).mean(dim=1)
    w = sigma_weight(sigma, mode=weight_mode)
    return (w * per_sample).mean()
