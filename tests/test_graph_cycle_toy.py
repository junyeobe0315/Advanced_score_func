from __future__ import annotations

import torch

from src.losses.graph_cycle import graph_cycle_estimator


def test_graph_cycle_energy_distinguishes_rotational_field() -> None:
    """Graph-cycle energy should be larger for rotational fields than conservative fields."""
    torch.manual_seed(0)

    x = 0.2 * torch.randn(128, 2)
    sigma = torch.ones(128)
    features = x.clone()

    def conservative_score(x_in: torch.Tensor, sigma_in: torch.Tensor) -> torch.Tensor:
        """Conservative zero field from constant potential."""
        del sigma_in
        return torch.zeros_like(x_in)

    def rotational_score(x_in: torch.Tensor, sigma_in: torch.Tensor) -> torch.Tensor:
        """Simple 2D rotational non-conservative field."""
        del sigma_in
        return torch.stack([-x_in[:, 1], x_in[:, 0]], dim=1)

    e_cons, _ = graph_cycle_estimator(
        score_fn=conservative_score,
        x=x,
        sigma=sigma,
        features=features,
        k=8,
        cycle_lengths=[3, 4],
        num_cycles=32,
    )
    e_rot, _ = graph_cycle_estimator(
        score_fn=rotational_score,
        x=x,
        sigma=sigma,
        features=features,
        k=8,
        cycle_lengths=[3, 4],
        num_cycles=32,
    )

    assert e_rot.item() > e_cons.item() + 1e-6
