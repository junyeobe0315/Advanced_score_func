from __future__ import annotations

from typing import Callable

import torch

from .jacobian_asymmetry import jacobian_asymmetry_estimator


TensorFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def reg_sym_estimator(
    score_fn: TensorFn,
    x: torch.Tensor,
    sigma: torch.Tensor,
    K: int = 1,
    method: str = "jvp_vjp",
    eps_fd: float = 1.0e-3,
) -> torch.Tensor:
    """Backward-compatible wrapper for Jacobian asymmetry estimator.

    Args:
        score_fn: Callable returning score tensor from ``(x, sigma)``.
        x: Noisy input batch.
        sigma: Per-sample noise levels with shape ``[B]``.
        K: Number of Gaussian probe vectors.
        method: ``\"jvp_vjp\"``, ``\"finite_diff\"``, or ``\"auto\"``.
        eps_fd: Finite-difference epsilon for fallback mode.

    Returns:
        Scalar estimate of asymmetry regularizer.
    """
    return jacobian_asymmetry_estimator(
        score_fn=score_fn,
        x=x,
        sigma=sigma,
        num_probes=int(K),
        method=method,
        eps_fd=float(eps_fd),
    )
