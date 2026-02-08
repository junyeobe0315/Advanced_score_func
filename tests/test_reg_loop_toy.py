from __future__ import annotations

import torch

from src.losses.reg_loop import reg_loop_estimator


def test_loop_regularizer_small_for_conservative_field() -> None:
    def score_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        del sigma
        return x

    x = torch.randn(512, 2)
    sigma = torch.ones(512)
    val = reg_loop_estimator(score_fn, x, sigma, delta=0.01, sparse_ratio=1.0)
    assert val.item() < 1e-3


def test_loop_regularizer_higher_for_rotational_field() -> None:
    def score_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        del sigma
        return torch.stack([-x[:, 1], x[:, 0]], dim=1)

    x = torch.randn(512, 2)
    sigma = torch.ones(512)
    val = reg_loop_estimator(score_fn, x, sigma, delta=0.1, sparse_ratio=1.0)
    assert val.item() > 1e-4
