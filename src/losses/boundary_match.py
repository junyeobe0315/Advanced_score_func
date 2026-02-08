from __future__ import annotations

import torch

from src.models.hybrid_wrapper import HybridWrapper


def boundary_match_estimator(
    model: HybridWrapper,
    x: torch.Tensor,
    sigma_c: float,
    bandwidth: float = 0.05,
    create_graph: bool = True,
) -> torch.Tensor:
    """Compute M4 boundary matching loss around split threshold ``sigma_c``.

    Args:
        model: Hybrid model wrapper with low/high score branches.
        x: Input batch tensor.
        sigma_c: Split threshold for low/high noise branches.
        bandwidth: Relative perturbation width around ``sigma_c``.
        create_graph: Whether low-branch gradient path keeps graph.

    Returns:
        Scalar mean-squared mismatch between low/high branch scores.

    How it works:
        Samples boundary sigmas in ``[sigma_c*(1-band), sigma_c*(1+band)]``
        and penalizes ``||s_low - s_high||^2`` to enforce branch continuity.
    """
    if not isinstance(model, HybridWrapper):
        raise TypeError("boundary_match_estimator expects HybridWrapper model")

    bsz = x.shape[0]
    # Boundary sigma samples near split threshold.
    noise = (2.0 * torch.rand((bsz,), device=x.device, dtype=x.dtype) - 1.0) * float(bandwidth)
    sigma = torch.full((bsz,), float(sigma_c), device=x.device, dtype=x.dtype) * (1.0 + noise)

    low, high = model.boundary_score_pair(x, sigma, create_graph=create_graph)
    diff = low - high
    return (diff.flatten(start_dim=1) ** 2).mean()
