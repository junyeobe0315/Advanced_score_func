from __future__ import annotations

from typing import Callable

import torch

from src.losses import dsm_loss
from src.sampling.sigma_schedule import sample_log_uniform_sigmas


def make_noisy_input(
    x0: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma = sample_log_uniform_sigmas(
        batch_size=x0.shape[0],
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=x0.device,
        dtype=x0.dtype,
    )
    eps = torch.randn_like(x0)
    sigma_view = sigma.view(x0.shape[0], *([1] * (x0.ndim - 1)))
    x = x0 + sigma_view * eps
    return x, sigma, eps


def compute_dsm_for_score(
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    weight_mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x, sigma, eps = make_noisy_input(x0, sigma_min=sigma_min, sigma_max=sigma_max)
    score = score_fn(x, sigma)
    dsm = dsm_loss(score, x0=x0, sigma=sigma, eps=eps, weight_mode=weight_mode)
    cache = {
        "x": x,
        "sigma": sigma,
        "eps": eps,
        "score": score,
    }
    return dsm, cache
