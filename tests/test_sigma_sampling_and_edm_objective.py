from __future__ import annotations

import torch

from src.losses.dsm import dsm_target, edm_denoiser_loss_from_score
from src.sampling.sigma_schedule import sample_training_sigmas


def test_sample_training_sigmas_edm_lognormal_clamped() -> None:
    """EDM sigma sampler should honor configured clamp bounds."""
    sigma = sample_training_sigmas(
        batch_size=4096,
        sigma_min=0.01,
        sigma_max=2.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        mode="edm_lognormal",
        edm_p_mean=-1.2,
        edm_p_std=1.2,
        clamp=True,
    )
    assert float(sigma.min().item()) >= 0.01 - 1e-8
    assert float(sigma.max().item()) <= 2.0 + 1e-8


def test_edm_denoiser_loss_zero_for_exact_score_target() -> None:
    """EDM denoiser objective should vanish when score target is exact."""
    b, c, h, w = 32, 1, 8, 8
    x0 = torch.randn(b, c, h, w)
    eps = torch.randn_like(x0)
    sigma = torch.exp(torch.empty(b).uniform_(-2.0, 0.7))
    sigma_view = sigma.view(b, 1, 1, 1)
    x = x0 + sigma_view * eps
    score = dsm_target(eps=eps, sigma=sigma)

    loss = edm_denoiser_loss_from_score(score=score, x=x, x0=x0, sigma=sigma, sigma_data=0.5)
    assert loss.item() < 1e-6
