from __future__ import annotations

import torch

from src.sampling import sample_euler, sample_heun


def test_sampler_reproducibility_and_nfe() -> None:
    """Euler sampler should be deterministic for fixed init and report NFE."""

    def score_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Linear contraction field for deterministic integration."""
        del sigma
        return -0.1 * x

    x0 = torch.randn(16, 2)
    out1, st1 = sample_euler(
        score_fn=score_fn,
        shape=x0.shape,
        sigma_min=0.01,
        sigma_max=1.0,
        nfe=10,
        device=torch.device("cpu"),
        init_x=x0.clone(),
    )
    out2, st2 = sample_euler(
        score_fn=score_fn,
        shape=x0.shape,
        sigma_min=0.01,
        sigma_max=1.0,
        nfe=10,
        device=torch.device("cpu"),
        init_x=x0.clone(),
    )

    assert torch.allclose(out1, out2)
    assert st1["nfe"] == 10
    assert st2["nfe"] == 10


def test_heun_matches_euler_for_constant_field() -> None:
    """Heun and Euler should match on constant slope vector fields."""

    def score_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Constant vector field independent of state and sigma."""
        del x, sigma
        return torch.ones(8, 2) * 0.25

    init = torch.zeros(8, 2)
    out_e, _ = sample_euler(
        score_fn=score_fn,
        shape=init.shape,
        sigma_min=0.01,
        sigma_max=1.0,
        nfe=5,
        device=torch.device("cpu"),
        init_x=init.clone(),
    )
    out_h, _ = sample_heun(
        score_fn=score_fn,
        shape=init.shape,
        sigma_min=0.01,
        sigma_max=1.0,
        nfe=5,
        device=torch.device("cpu"),
        init_x=init.clone(),
    )

    assert torch.allclose(out_e, out_h, atol=1e-6)
