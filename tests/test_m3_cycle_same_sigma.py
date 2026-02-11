from __future__ import annotations

from pathlib import Path

import torch

import src.trainers.train_step_m3 as train_step_m3_mod
from src.models.score_mlp_toy import ScoreMLPToy
from src.utils.config import ensure_experiment_defaults, load_experiment_config


class IdentityFeatureEncoder(torch.nn.Module):
    """Simple feature encoder for toy tensors."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_m3_cycle_uses_shared_sigma_when_enabled(monkeypatch) -> None:
    """M3 cycle path should pass a single shared sigma value to cycle estimator."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_experiment_config(str(repo_root / "configs" / "toy" / "experiment.yaml"), model="m3", ablation="none")
    cfg = ensure_experiment_defaults(cfg)
    cfg["loss"]["mu1"] = 0.0
    cfg["loss"]["start_mu2"] = 5.0e-1
    cfg["loss"]["target_r"] = 0.1
    cfg["loss"]["update_step"] = 1000
    cfg["loss"]["reg_freq"] = 1
    cfg["loss"]["cycle_freq"] = 1
    cfg["loss"]["cycle_same_sigma"] = True
    cfg["loss"]["reg_low_noise_only"] = False
    cfg["loss"]["cycle_subset"] = 16
    cfg["loss"]["cycle_subset_cap"] = 16
    cfg["loss"]["cycle_samples"] = 4
    cfg["loss"]["cycle_samples_cap"] = 4

    captured: dict[str, torch.Tensor] = {}

    def fake_cycle_estimator(
        *,
        score_fn,
        x,
        sigma,
        features,
        k,
        cycle_lengths,
        num_cycles,
        subset_size,
        precomputed_score,
        path_length_normalization,
    ):
        del score_fn, features, k, cycle_lengths, num_cycles, subset_size, precomputed_score
        captured["sigma"] = sigma.detach().clone()
        captured["path_length_normalization"] = torch.tensor(bool(path_length_normalization))
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        return zero, {}

    monkeypatch.setattr(train_step_m3_mod, "graph_cycle_estimator", fake_cycle_estimator)

    model = ScoreMLPToy(dim=2, hidden_dim=32, depth=2, sigma_embed_dim=16)
    x0 = torch.randn(32, 2)
    feature_encoder = IdentityFeatureEncoder()
    train_step_m3_mod.train_step_m3(model=model, x0=x0, cfg=cfg, step=1, feature_encoder=feature_encoder)

    assert "sigma" in captured
    assert torch.unique(captured["sigma"]).numel() == 1
    assert bool(captured["path_length_normalization"].item()) is True


def test_m3_dynamic_mu2_updates_from_recent_window(monkeypatch) -> None:
    """M3 should update mu2 from moving-average DSM/cycle statistics."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_experiment_config(str(repo_root / "configs" / "toy" / "experiment.yaml"), model="m3", ablation="none")
    cfg = ensure_experiment_defaults(cfg)
    cfg["loss"]["mu1"] = 0.0
    cfg["loss"]["start_mu2"] = 1.0e-3
    cfg["loss"]["target_r"] = 0.1
    cfg["loss"]["update_step"] = 2
    cfg["loss"]["reg_freq"] = 1
    cfg["loss"]["cycle_freq"] = 1
    cfg["loss"]["cycle_same_sigma"] = True
    cfg["loss"]["reg_low_noise_only"] = False
    cfg["loss"]["cycle_subset"] = 16
    cfg["loss"]["cycle_subset_cap"] = 16
    cfg["loss"]["cycle_samples"] = 4
    cfg["loss"]["cycle_samples_cap"] = 4

    def fake_compute_dsm_for_score(*, score_fn, x0, sigma_min, sigma_max, weight_mode, objective, sigma_sampling, edm_p_mean, edm_p_std, sigma_sample_clamp, sigma_data, cache_score):
        del score_fn, sigma_min, sigma_max, weight_mode, objective, sigma_sampling, edm_p_mean, edm_p_std, sigma_sample_clamp, sigma_data, cache_score
        dsm = torch.tensor(4.0, device=x0.device, dtype=x0.dtype)
        sigma = torch.ones(x0.shape[0], device=x0.device, dtype=x0.dtype)
        return dsm, {"x": x0, "sigma": sigma, "score": None}

    def fake_cycle_estimator(
        *,
        score_fn,
        x,
        sigma,
        features,
        k,
        cycle_lengths,
        num_cycles,
        subset_size,
        precomputed_score,
        path_length_normalization,
    ):
        del score_fn, sigma, features, k, cycle_lengths, num_cycles, subset_size, precomputed_score
        assert path_length_normalization is True
        cycle = torch.tensor(2.0, device=x.device, dtype=x.dtype)
        return cycle, {}

    monkeypatch.setattr(train_step_m3_mod, "compute_dsm_for_score", fake_compute_dsm_for_score)
    monkeypatch.setattr(train_step_m3_mod, "graph_cycle_estimator", fake_cycle_estimator)

    model = ScoreMLPToy(dim=2, hidden_dim=32, depth=2, sigma_embed_dim=16)
    x0 = torch.randn(32, 2)
    feature_encoder = IdentityFeatureEncoder()

    loss_1, _ = train_step_m3_mod.train_step_m3(model=model, x0=x0, cfg=cfg, step=1, feature_encoder=feature_encoder)
    loss_2, _ = train_step_m3_mod.train_step_m3(model=model, x0=x0, cfg=cfg, step=2, feature_encoder=feature_encoder)

    assert torch.isclose(loss_1, torch.tensor(4.002, dtype=loss_1.dtype), atol=1e-4, rtol=0.0)
    assert torch.isclose(loss_2, torch.tensor(4.4, dtype=loss_2.dtype), atol=1e-4, rtol=0.0)
