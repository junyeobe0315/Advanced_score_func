from __future__ import annotations

from typing import Callable

import torch

from .loop_multi import rectangle_circulation


TensorFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def reg_loop_estimator(
    score_fn: TensorFn,
    x: torch.Tensor,
    sigma: torch.Tensor,
    delta: float,
    sparse_ratio: float = 1.0,
) -> torch.Tensor:
    """Backward-compatible single-scale loop consistency estimator.

    Args:
        score_fn: Callable returning score tensor from ``(x, sigma)``.
        x: Anchor points for loop construction.
        sigma: Per-sample noise levels.
        delta: Side length of the small rectangle loop.
        sparse_ratio: Ratio of active dimensions in random loop directions.

    Returns:
        Scalar loop regularization estimate.
    """
    circ = rectangle_circulation(
        score_fn=score_fn,
        x=x,
        sigma=sigma,
        delta=float(delta),
        sparse_ratio=float(sparse_ratio),
    )
    return (circ**2).mean()
