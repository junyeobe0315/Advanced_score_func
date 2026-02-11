from __future__ import annotations

import torch

from src.losses.graph_cycle import _circulation_from_cycles, graph_cycle_estimator


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


def test_cycle_path_length_normalization_reduces_scale_sensitivity() -> None:
    """Normalized circulation should stay stable when geometry is uniformly scaled."""
    score = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    x = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    cycles = torch.tensor([[0, 1, 2]], dtype=torch.long)

    raw = _circulation_from_cycles(score, x, cycles, path_length_normalization=False)
    raw_scaled = _circulation_from_cycles(score, 10.0 * x, cycles, path_length_normalization=False)
    norm = _circulation_from_cycles(score, x, cycles, path_length_normalization=True)
    norm_scaled = _circulation_from_cycles(score, 10.0 * x, cycles, path_length_normalization=True)

    assert torch.allclose(raw_scaled, 10.0 * raw, atol=1e-6, rtol=1e-6)
    assert torch.allclose(norm_scaled, norm, atol=1e-6, rtol=1e-6)
