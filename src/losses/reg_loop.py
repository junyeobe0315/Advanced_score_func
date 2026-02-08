from __future__ import annotations

from typing import Callable

import torch


TensorFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.flatten(start_dim=1) * b.flatten(start_dim=1)).sum(dim=1)


def _random_direction_like(x: torch.Tensor, sparse_ratio: float) -> torch.Tensor:
    d = torch.randn_like(x)
    if sparse_ratio < 1.0:
        mask = (torch.rand_like(x) < sparse_ratio).to(x.dtype)
        d = d * mask
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
    u = _random_direction_like(x, sparse_ratio=sparse_ratio)
    v = _random_direction_like(x, sparse_ratio=sparse_ratio)

    x0 = x
    x1 = x + delta * u
    x3 = x + delta * v

    s0 = score_fn(x0, sigma)
    s1 = score_fn(x1, sigma)
    s3 = score_fn(x3, sigma)

    loop = (
        delta * _dot(s0, u)
        + delta * _dot(s1, v)
        - delta * _dot(s3, u)
        - delta * _dot(s0, v)
    )
    return (loop ** 2).mean()
