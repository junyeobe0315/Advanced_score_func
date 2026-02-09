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
    variant: str = "skew_fro",
    probe_dist: str = "gaussian",
    eps_fd: float = 1.0e-3,
    return_per_sample: bool = False,
) -> torch.Tensor:
    """Backward-compatible wrapper for Jacobian asymmetry estimator.

    Args:
        score_fn: Callable returning score tensor from ``(x, sigma)``.
        x: Noisy input batch.
        sigma: Per-sample noise levels with shape ``[B]``.
        K: Number of Gaussian probe vectors.
        method: ``\"jvp_vjp\"``, ``\"finite_diff\"``, ``\"auto\"``, or
            ``\"auto_fast\"``.
        variant: Asymmetry estimator form (``"skew_fro"`` or
            ``"qcsbm_trace"``).
        probe_dist: Hutchinson probe distribution token.
        eps_fd: Finite-difference epsilon for fallback mode.
        return_per_sample: Return per-sample values when ``True``.

    Returns:
        Scalar estimate of asymmetry regularizer.
    """
    return jacobian_asymmetry_estimator(
        score_fn=score_fn,
        x=x,
        sigma=sigma,
        num_probes=int(K),
        method=method,
        variant=variant,
        probe_dist=probe_dist,
        eps_fd=float(eps_fd),
        return_per_sample=bool(return_per_sample),
    )
