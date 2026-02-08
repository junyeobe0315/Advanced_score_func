from __future__ import annotations

import torch

from src.losses.reg_sym import reg_sym_estimator


def test_reg_sym_near_zero_for_conservative_linear_field() -> None:
    A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])

    def score_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        del sigma
        return x @ A.t()

    x = torch.randn(256, 2, requires_grad=True)
    sigma = torch.ones(256)
    val = reg_sym_estimator(score_fn, x, sigma, K=8)
    assert val.item() < 1e-4


def test_reg_sym_large_for_rotational_field() -> None:
    def score_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        del sigma
        return torch.stack([-x[:, 1], x[:, 0]], dim=1)

    x = torch.randn(512, 2, requires_grad=True)
    sigma = torch.ones(512)
    val = reg_sym_estimator(score_fn, x, sigma, K=4)
    assert val.item() > 1.0
