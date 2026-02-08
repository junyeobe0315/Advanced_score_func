from __future__ import annotations

import torch


def sigma_weight(sigma: torch.Tensor, mode: str = "sigma2") -> torch.Tensor:
    """Compute per-sample DSM weighting from sigma values.

    Args:
        sigma: Noise level tensor with shape ``[B]``.
        mode: Weighting policy name. ``"sigma2"`` returns ``sigma^2`` and
            ``"none"`` returns ones.

    Returns:
        A tensor of shape ``[B]`` used to weight per-sample DSM errors.

    How it works:
        The function applies a deterministic mapping from each noise scale to
        a scalar weight. This keeps all model variants on the same weighting
        rule during comparisons.
    """
    if mode == "sigma2":
        return sigma ** 2
    if mode == "none":
        return torch.ones_like(sigma)
    raise ValueError(f"unknown weight mode: {mode}")


def dsm_target(eps: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Build the denoising score matching target ``-eps / sigma``.

    Args:
        eps: Gaussian perturbation with shape matching the noisy input.
        sigma: Per-sample noise levels with shape ``[B]``.

    Returns:
        Target score tensor with the same shape as ``eps``.

    How it works:
        ``sigma`` is reshaped for broadcasting across spatial/feature
        dimensions, then the closed-form DSM target is computed.
    """
    # Broadcast sigma from [B] to [B,1,...,1] so division matches eps shape.
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
    """Compute weighted denoising score matching loss.

    Args:
        score: Model-predicted score field ``s_theta(x, sigma)``.
        x0: Clean samples. Unused in the closed-form loss but kept for API
            consistency across training code paths.
        sigma: Per-sample noise levels with shape ``[B]``.
        eps: Gaussian perturbation used to form noisy inputs.
        weight_mode: Name of sigma-based weighting policy.

    Returns:
        Scalar DSM loss tensor.

    How it works:
        1) Compute the analytic target ``-eps/sigma``.
        2) Compute per-element squared error between prediction and target.
        3) Average each sample's error over non-batch dimensions.
        4) Apply sigma-dependent weights and average over batch.
    """
    del x0
    target = dsm_target(eps, sigma)
    # Element-wise mismatch between predicted and analytic target score.
    err = (score - target) ** 2
    # Reduce feature/spatial dimensions to one error value per sample.
    per_sample = err.flatten(start_dim=1).mean(dim=1)
    # Apply the selected noise-level weighting policy.
    w = sigma_weight(sigma, mode=weight_mode)
    return (w * per_sample).mean()
