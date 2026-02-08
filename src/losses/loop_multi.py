from __future__ import annotations

from typing import Callable, Iterable

import torch


ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _dot_per_sample(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute per-sample inner products for batched tensors."""
    return (a.flatten(start_dim=1) * b.flatten(start_dim=1)).sum(dim=1)


def random_unit_direction(x: torch.Tensor, sparse_ratio: float) -> torch.Tensor:
    """Sample normalized random directions with optional coordinate sparsity.

    Args:
        x: Reference tensor whose shape sets output direction shape.
        sparse_ratio: Active-coordinate ratio in ``(0, 1]``.

    Returns:
        Direction tensor with same shape as ``x`` and per-sample unit norm.
    """
    direction = torch.randn_like(x)
    if sparse_ratio < 1.0:
        mask = (torch.rand_like(x) < float(sparse_ratio)).to(dtype=x.dtype)
        direction = direction * mask

    # Ensure unit norm per sample so loop magnitude is controlled by delta.
    norm = direction.flatten(start_dim=1).norm(dim=1).clamp_min(1.0e-8)
    reshape = [x.shape[0]] + [1] * (x.ndim - 1)
    return direction / norm.view(*reshape)


def rectangle_circulation(
    score_fn: ScoreFn,
    x: torch.Tensor,
    sigma: torch.Tensor,
    delta: float,
    sparse_ratio: float,
) -> torch.Tensor:
    """Approximate loop circulation on small rectangles around each sample.

    Args:
        score_fn: Score function ``s(x, sigma)``.
        x: Anchor points for loop construction.
        sigma: Per-sample sigma tensor.
        delta: Rectangle edge length.
        sparse_ratio: Active-coordinate ratio for random directions.

    Returns:
        Per-sample loop circulation tensor ``[B]``.

    How it works:
        Uses the fixed four-point scheme from the project spec:
        ``x0=x``, ``x1=x+Δu``, ``x3=x+Δv`` and
        ``circ ≈ Δ s(x0)·u + Δ s(x1)·v - Δ s(x3)·u - Δ s(x0)·v``.
    """
    u = random_unit_direction(x, sparse_ratio=sparse_ratio)
    v = random_unit_direction(x, sparse_ratio=sparse_ratio)

    x0 = x
    x1 = x + float(delta) * u
    x3 = x + float(delta) * v

    s0 = score_fn(x0, sigma)
    s1 = score_fn(x1, sigma)
    s3 = score_fn(x3, sigma)

    circ = (
        float(delta) * _dot_per_sample(s0, u)
        + float(delta) * _dot_per_sample(s1, v)
        - float(delta) * _dot_per_sample(s3, u)
        - float(delta) * _dot_per_sample(s0, v)
    )
    return circ


def loop_multi_scale_estimator(
    score_fn: ScoreFn,
    x: torch.Tensor,
    sigma: torch.Tensor,
    delta_set: Iterable[float],
    sparse_ratio: float = 1.0,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """Compute multi-scale loop energy regularizer for Jacobian-free M3.

    Args:
        score_fn: Score function ``s(x, sigma)``.
        x: Noisy input batch.
        sigma: Per-sample sigma tensor.
        delta_set: Collection of loop scales.
        sparse_ratio: Active-coordinate ratio for random directions.

    Returns:
        Tuple ``(mean_energy, per_scale)`` where ``per_scale`` maps each delta
        to a scalar loop energy tensor.

    How it works:
        For each ``delta`` computes mean ``circ(delta)^2`` and averages across
        all scales to form ``R_loop_multi``.
    """
    per_scale: dict[float, torch.Tensor] = {}
    values = []

    for delta in delta_set:
        d = float(delta)
        circ = rectangle_circulation(score_fn, x=x, sigma=sigma, delta=d, sparse_ratio=sparse_ratio)
        energy = (circ**2).mean()
        per_scale[d] = energy
        values.append(energy)

    if not values:
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        return zero, per_scale

    total = torch.stack(values).mean()
    return total, per_scale
