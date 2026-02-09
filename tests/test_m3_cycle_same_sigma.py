from __future__ import annotations

from pathlib import Path

import torch

import src.trainers.train_step_m3 as train_step_m3_mod
from src.models.score_mlp_toy import ScoreMLPToy
from src.utils.config import ensure_experiment_defaults, load_config


class IdentityFeatureEncoder(torch.nn.Module):
    """Simple feature encoder for toy tensors."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_m3_cycle_uses_shared_sigma_when_enabled(monkeypatch) -> None:
    """M3 cycle path should pass a single shared sigma value to cycle estimator."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(str(repo_root / "configs" / "toy" / "m3.yaml"))
    cfg = ensure_experiment_defaults(cfg)
    cfg["loss"]["mu1"] = 0.0
    cfg["loss"]["mu2"] = 0.5
    cfg["loss"]["reg_freq"] = 1
    cfg["loss"]["cycle_freq"] = 1
    cfg["loss"]["cycle_same_sigma"] = True
    cfg["loss"]["reg_low_noise_only"] = False
    cfg["loss"]["cycle_subset"] = 16
    cfg["loss"]["cycle_subset_cap"] = 16
    cfg["loss"]["cycle_samples"] = 4
    cfg["loss"]["cycle_samples_cap"] = 4

    captured: dict[str, torch.Tensor] = {}

    def fake_cycle_estimator(*, score_fn, x, sigma, features, k, cycle_lengths, num_cycles, subset_size, precomputed_score):
        del score_fn, features, k, cycle_lengths, num_cycles, subset_size, precomputed_score
        captured["sigma"] = sigma.detach().clone()
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        return zero, {}

    monkeypatch.setattr(train_step_m3_mod, "graph_cycle_estimator", fake_cycle_estimator)

    model = ScoreMLPToy(dim=2, hidden_dim=32, depth=2, sigma_embed_dim=16)
    x0 = torch.randn(32, 2)
    feature_encoder = IdentityFeatureEncoder()
    train_step_m3_mod.train_step_m3(model=model, x0=x0, cfg=cfg, step=1, feature_encoder=feature_encoder)

    assert "sigma" in captured
    assert torch.unique(captured["sigma"]).numel() == 1
