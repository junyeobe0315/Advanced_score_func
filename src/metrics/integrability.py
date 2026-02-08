from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from src.losses import reg_loop_estimator, reg_sym_estimator


ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def sigma_bin_edges(sigma_min: float, sigma_max: float, bins: int) -> np.ndarray:
    """Create logarithmic sigma-bin edges."""
    return np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), bins + 1))


def _bucket_index(sigmas: torch.Tensor, edges: np.ndarray) -> torch.Tensor:
    """Assign each sigma value to logarithmic bin index."""
    edges_t = torch.tensor(edges, device=sigmas.device, dtype=sigmas.dtype)
    idx = torch.bucketize(sigmas, edges_t, right=True) - 1
    return idx.clamp(min=0, max=len(edges) - 2)


def integrability_by_sigma_bins(
    score_fn: ScoreFn,
    x: torch.Tensor,
    sigma: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    bins: int,
    reg_k: int,
    loop_delta: float,
    loop_sparse_ratio: float,
) -> list[dict[str, float]]:
    """Compute integrability diagnostics grouped by sigma bins.

    Args:
        score_fn: Callable score function.
        x: Noisy input batch.
        sigma: Per-sample sigma tensor.
        sigma_min: Minimum sigma for bins.
        sigma_max: Maximum sigma for bins.
        bins: Number of logarithmic bins.
        reg_k: Probe count for symmetry estimator.
        loop_delta: Loop side length for loop estimator.
        loop_sparse_ratio: Active-coordinate ratio for loop directions.

    Returns:
        List of dictionaries, one per sigma bin, containing counts and
        ``r_sym``/``r_loop`` values.

    How it works:
        Buckets samples by sigma, then runs regularizer estimators on each bin
        subset independently to expose noise-level dependence.
    """
    edges = sigma_bin_edges(sigma_min, sigma_max, bins)
    idx = _bucket_index(sigma, edges)

    out: list[dict[str, float]] = []
    for b in range(bins):
        mask = idx == b
        if int(mask.sum().item()) == 0:
            out.append(
                {
                    "bin": b,
                    "sigma_lo": float(edges[b]),
                    "sigma_hi": float(edges[b + 1]),
                    "count": 0,
                    "r_sym": float("nan"),
                    "r_loop": float("nan"),
                }
            )
            continue

        xb = x[mask]
        sb = sigma[mask]
        r_sym = reg_sym_estimator(score_fn, xb, sb, K=reg_k)
        r_loop = reg_loop_estimator(score_fn, xb, sb, delta=loop_delta, sparse_ratio=loop_sparse_ratio)

        out.append(
            {
                "bin": b,
                "sigma_lo": float(edges[b]),
                "sigma_hi": float(edges[b + 1]),
                "count": int(mask.sum().item()),
                "r_sym": float(r_sym.item()),
                "r_loop": float(r_loop.item()),
            }
        )
    return out


def exact_jacobian_asymmetry_2d(score_fn: ScoreFn, x: torch.Tensor, sigma: torch.Tensor) -> float:
    """Compute exact Jacobian asymmetry in 2D by explicit derivatives.

    Args:
        score_fn: Callable score function.
        x: 2D input batch with shape ``[B, 2]``.
        sigma: Per-sample sigma tensor.

    Returns:
        Mean squared Frobenius norm of ``J - J^T`` across batch.
    """
    if x.shape[-1] != 2:
        raise ValueError("exact_jacobian_asymmetry_2d expects shape [B,2]")

    vals: list[torch.Tensor] = []
    for i in range(x.shape[0]):
        xi = x[i : i + 1].requires_grad_(True)
        si = sigma[i : i + 1]

        y = score_fn(xi, si)
        j_rows = []
        for d in range(2):
            grad = torch.autograd.grad(y[0, d], xi, retain_graph=True, create_graph=False)[0]
            j_rows.append(grad[0])
        jac = torch.stack(j_rows, dim=0)  # [2,2]
        asym = jac - jac.t()
        vals.append((asym ** 2).sum())
    return float(torch.stack(vals).mean().item())


def path_variance(
    score_fn: ScoreFn,
    x_ref: torch.Tensor,
    x_tgt: torch.Tensor,
    sigma: torch.Tensor,
    num_paths: int,
    num_segments: int,
) -> float:
    """Estimate path-dependence variance of line integral diagnostics.

    Args:
        score_fn: Callable score function.
        x_ref: Path start points batch.
        x_tgt: Path end points batch.
        sigma: Sigma tensor aligned with sample batch.
        num_paths: Number of random piecewise-linear paths.
        num_segments: Number of segments per random path.

    Returns:
        Mean variance of line integrals across random paths.

    How it works:
        Samples random interpolation knots between ``x_ref`` and ``x_tgt``,
        approximates line integrals with midpoint rule, and reports variance.
    """
    # Piecewise-linear random paths between x_ref and x_tgt.
    integrals = []
    for _ in range(num_paths):
        t = torch.sort(torch.rand(num_segments - 1, device=x_ref.device))[0]
        t = torch.cat([torch.zeros(1, device=x_ref.device), t, torch.ones(1, device=x_ref.device)])

        points = [x_ref + tt * (x_tgt - x_ref) for tt in t]
        total = torch.zeros(x_ref.shape[0], device=x_ref.device)
        for a, b in zip(points[:-1], points[1:]):
            xm = 0.5 * (a + b)
            dx = b - a
            s = score_fn(xm, sigma)
            total = total + (s.flatten(start_dim=1) * dx.flatten(start_dim=1)).sum(dim=1)
        integrals.append(total)

    stacked = torch.stack(integrals, dim=0)
    return float(stacked.var(dim=0).mean().item())
