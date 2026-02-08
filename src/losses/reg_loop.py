from __future__ import annotations

from typing import Callable

import torch


TensorFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute per-sample inner product between two batched tensors.

    Args:
        a: Batch tensor.
        b: Batch tensor with same shape as ``a``.

    Returns:
        Tensor of shape ``[B]`` containing one dot product per sample.

    How it works:
        Flattens each sample and sums element-wise products.
    """
    return (a.flatten(start_dim=1) * b.flatten(start_dim=1)).sum(dim=1)


def _random_direction_like(x: torch.Tensor, sparse_ratio: float) -> torch.Tensor:
    """Sample normalized random direction vectors with optional sparsity.

    Args:
        x: Reference tensor whose shape defines output direction shape.
        sparse_ratio: Fraction of active coordinates in each direction.

    Returns:
        Normalized random direction tensor with same shape as ``x``.

    How it works:
        Draws Gaussian directions, applies Bernoulli sparsity mask when
        requested, then L2-normalizes each sample direction.
    """
    # Base random direction before optional sparsification.
    d = torch.randn_like(x)
    if sparse_ratio < 1.0:
        # Keep only a subset of coordinates for local loop probes.
        mask = (torch.rand_like(x) < sparse_ratio).to(x.dtype)
        d = d * mask
    # Normalize each direction so loop scale is controlled by delta only.
    norm = d.flatten(start_dim=1).norm(dim=1).clamp_min(1e-8)
    shape = [x.shape[0]] + [1] * (x.ndim - 1)
    d = d / norm.view(*shape)
    return d


def reg_loop_estimator(
    score_fn: TensorFn,
    x: torch.Tensor,
    sigma: torch.Tensor,
    delta: float,
    sparse_ratio: float = 1.0,
) -> torch.Tensor:
    """Estimate loop-integral consistency penalty with rectangle approximation.

    Args:
        score_fn: Callable returning score tensor from ``(x, sigma)``.
        x: Anchor points for loop construction.
        sigma: Per-sample noise levels.
        delta: Side length of the small rectangle loop.
        sparse_ratio: Ratio of active dimensions in random loop directions.

    Returns:
        Scalar loop regularization estimate.

    How it works:
        Builds a small rectangle around each sample using directions ``u`` and
        ``v`` and approximates ``oint s·dx`` by signed edge contributions. The
        returned value is the batch mean of squared loop integrals.
    """
    # Two random unit directions defining rectangle edges.
    u = _random_direction_like(x, sparse_ratio=sparse_ratio)
    v = _random_direction_like(x, sparse_ratio=sparse_ratio)

    # Rectangle vertices used in the edge-based line-integral approximation.
    x0 = x
    x1 = x + delta * u
    x3 = x + delta * v

    s0 = score_fn(x0, sigma)
    s1 = score_fn(x1, sigma)
    s3 = score_fn(x3, sigma)

    # Signed sum of four directed edge contributions.
    loop = (
        delta * _dot(s0, u)
        + delta * _dot(s1, v)
        - delta * _dot(s3, u)
        - delta * _dot(s0, v)
    )
    return (loop ** 2).mean()
