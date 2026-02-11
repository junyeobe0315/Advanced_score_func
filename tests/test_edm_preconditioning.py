from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.models import build_model
from src.models.edm_precond import EDMPreconditionedScore
from src.utils.config import ensure_experiment_defaults, load_experiment_config


class ZeroBackbone(nn.Module):
    """Backbone that predicts zero residual for all inputs."""

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        del sigma
        return torch.zeros_like(x)


def test_edm_preconditioned_score_matches_closed_form_for_zero_backbone() -> None:
    """Wrapper output should match analytic expression when residual is zero."""
    sigma_data = 0.5
    model = EDMPreconditionedScore(backbone=ZeroBackbone(), sigma_data=sigma_data, sigma_eps=1.0e-8)

    x = torch.randn(32, 2)
    sigma = torch.exp(torch.empty(32).uniform_(-1.0, 0.5))

    out = model(x, sigma)

    sigma_v = sigma.view(-1, 1)
    sd2 = sigma_data * sigma_data
    c_skip = sd2 / (sigma_v * sigma_v + sd2)
    expected = ((c_skip - 1.0) / (sigma_v * sigma_v)) * x
    assert torch.allclose(out, expected, atol=1.0e-6, rtol=1.0e-5)


def test_toy_m0_builds_with_edm_preconditioning() -> None:
    """Toy M0 config should resolve to EDM-preconditioned score wrapper."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_experiment_config(str(repo_root / "configs" / "toy" / "experiment.yaml"), model="m0", ablation="none")
    cfg = ensure_experiment_defaults(cfg)
    model = build_model(cfg)
    assert isinstance(model, EDMPreconditionedScore)
