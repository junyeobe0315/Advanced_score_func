from __future__ import annotations

import torch

from src.losses.boundary_match import boundary_match_estimator
from src.models.hybrid_wrapper import HybridWrapper
from src.models.potential_mlp_toy import PotentialMLPToy
from src.models.score_mlp_toy import ScoreMLPToy


def test_hybrid_wrapper_sigma_gate_behavior() -> None:
    """Hybrid wrapper should route low/high sigma inputs to the right branch."""
    torch.manual_seed(0)

    high = ScoreMLPToy(dim=2, hidden_dim=32, depth=2, sigma_embed_dim=16)
    low = PotentialMLPToy(dim=2, hidden_dim=32, depth=2, sigma_embed_dim=16)
    hybrid = HybridWrapper(score_high=high, potential_low=low, sigma_c=0.5)

    x = torch.randn(8, 2)
    sigma_low = torch.full((8,), 0.2)
    sigma_high = torch.full((8,), 1.0)

    out_low = hybrid.score(x, sigma_low, create_graph=True)
    ref_low = hybrid.low_score(x, sigma_low, create_graph=True)
    assert torch.allclose(out_low, ref_low, atol=1e-6)

    out_high = hybrid.score(x, sigma_high, create_graph=True)
    ref_high = hybrid.high_score(x, sigma_high)
    assert torch.allclose(out_high, ref_high, atol=1e-6)


def test_boundary_match_estimator_is_finite() -> None:
    """Boundary matching loss should produce a finite non-negative scalar."""
    high = ScoreMLPToy(dim=2, hidden_dim=32, depth=2, sigma_embed_dim=16)
    low = PotentialMLPToy(dim=2, hidden_dim=32, depth=2, sigma_embed_dim=16)
    hybrid = HybridWrapper(score_high=high, potential_low=low, sigma_c=0.5)

    x = torch.randn(16, 2)
    val = boundary_match_estimator(hybrid, x=x, sigma_c=0.5, bandwidth=0.05, create_graph=True)

    assert torch.isfinite(val)
    assert val.item() >= 0.0
