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
    """Generate noisy inputs used by DSM objective.

    Args:
        x0: Clean data batch.
        sigma_min: Minimum noise scale.
        sigma_max: Maximum noise scale.

    Returns:
        Tuple ``(x, sigma, eps)`` where ``x = x0 + sigma * eps``.

    How it works:
        Samples sigma from log-uniform distribution and Gaussian noise ``eps``,
        then constructs noisy observations by broadcasted perturbation.
    """
    sigma = sample_log_uniform_sigmas(
        batch_size=x0.shape[0],
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=x0.device,
        dtype=x0.dtype,
    )
    # Standard Gaussian perturbation used in DSM target.
    eps = torch.randn_like(x0)
    # Broadcast sigma to sample tensor shape.
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
    """Compute DSM loss and cache intermediate tensors for extra regularizers.

    Args:
        score_fn: Callable score model ``s_theta(x, sigma)``.
        x0: Clean data batch.
        sigma_min: Minimum sigma for sampling.
        sigma_max: Maximum sigma for sampling.
        weight_mode: Sigma weighting mode for DSM.

    Returns:
        Tuple ``(loss, cache)`` where cache includes noisy input, sigma, eps,
        and predicted score.

    How it works:
        Creates noisy inputs, evaluates score function once, then computes DSM
        loss and returns both loss and intermediate tensors for reuse.
    """
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
