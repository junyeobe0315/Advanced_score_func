from __future__ import annotations

import torch

from src.losses.dsm import dsm_loss, dsm_target


def test_dsm_loss_zero_on_exact_target() -> None:
    """DSM loss should be near zero when score equals analytic target."""
    b, d = 32, 4
    x0 = torch.randn(b, d)
    eps = torch.randn(b, d)
    sigma = torch.exp(torch.empty(b).uniform_(-2.0, 1.0))

    score = dsm_target(eps, sigma)
    loss = dsm_loss(score=score, x0=x0, sigma=sigma, eps=eps, weight_mode="sigma2")
    assert loss.item() < 1e-6


def test_dsm_loss_increases_when_perturbed() -> None:
    """DSM loss should increase when prediction is perturbed away from target."""
    b, d = 32, 4
    x0 = torch.randn(b, d)
    eps = torch.randn(b, d)
    sigma = torch.exp(torch.empty(b).uniform_(-2.0, 1.0))

    score_true = dsm_target(eps, sigma)
    loss_true = dsm_loss(score=score_true, x0=x0, sigma=sigma, eps=eps, weight_mode="sigma2")

    score_bad = score_true + 0.5 * torch.randn_like(score_true)
    loss_bad = dsm_loss(score=score_bad, x0=x0, sigma=sigma, eps=eps, weight_mode="sigma2")

    assert loss_bad.item() > loss_true.item()
