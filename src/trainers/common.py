from __future__ import annotations

from typing import Callable

import torch

from src.losses import dsm_loss, edm_denoiser_loss_from_score
from src.sampling.sigma_schedule import sample_training_sigmas


def make_noisy_input(
    x0: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    *,
    sigma_sampling: str = "log_uniform",
    edm_p_mean: float = -1.2,
    edm_p_std: float = 1.2,
    sigma_sample_clamp: bool = True,
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
    sigma = sample_training_sigmas(
        batch_size=x0.shape[0],
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=x0.device,
        dtype=x0.dtype,
        mode=sigma_sampling,
        edm_p_mean=edm_p_mean,
        edm_p_std=edm_p_std,
        clamp=bool(sigma_sample_clamp),
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
    objective: str = "dsm_score",
    sigma_sampling: str = "log_uniform",
    edm_p_mean: float = -1.2,
    edm_p_std: float = 1.2,
    sigma_sample_clamp: bool = True,
    sigma_data: float = 0.5,
    *,
    return_cache: bool = True,
    cache_eps: bool = False,
    cache_score: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute DSM loss and cache intermediate tensors for extra regularizers.

    Args:
        score_fn: Callable score model ``s_theta(x, sigma)``.
        x0: Clean data batch.
        sigma_min: Minimum sigma for sampling.
        sigma_max: Maximum sigma for sampling.
        weight_mode: Sigma weighting mode for DSM.
        return_cache: Whether to return intermediate cache tensors.
        cache_eps: Whether to include ``eps`` in returned cache.
        cache_score: Whether to include predicted score in returned cache.

    Returns:
        Tuple ``(loss, cache)`` where cache always includes noisy input and
        sigma, plus optional tensors controlled by ``cache_eps`` and
        ``cache_score``.

    How it works:
        Creates noisy inputs, evaluates score function once, then computes DSM
        loss and returns both loss and intermediate tensors for reuse.
    """
    x, sigma, eps = make_noisy_input(
        x0,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_sampling=sigma_sampling,
        edm_p_mean=edm_p_mean,
        edm_p_std=edm_p_std,
        sigma_sample_clamp=sigma_sample_clamp,
    )
    score = score_fn(x, sigma)
    objective_token = str(objective).lower()
    if objective_token in {"dsm", "dsm_score", "score"}:
        dsm = dsm_loss(score, x0=x0, sigma=sigma, eps=eps, weight_mode=weight_mode)
    elif objective_token in {"edm", "edm_denoiser", "denoiser"}:
        dsm = edm_denoiser_loss_from_score(score=score, x=x, x0=x0, sigma=sigma, sigma_data=float(sigma_data))
    else:
        raise ValueError(f"unknown training objective: {objective}")
    if not return_cache:
        return dsm, {}
    cache = {
        "x": x,
        "sigma": sigma,
    }
    if cache_eps:
        cache["eps"] = eps
    if cache_score:
        cache["score"] = score
    return dsm, cache
