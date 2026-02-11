from __future__ import annotations

import torch

from src.losses import boundary_match_estimator, graph_cycle_estimator
from src.models.hybrid_wrapper import HybridWrapper
from src.sampling.sigma_schedule import sample_training_sigmas

from .common import compute_dsm_for_score
from .step_utils import (
    cap_positive_int,
    cap_subset,
    is_due,
    make_noisy_from_clean,
    should_apply_regularizer,
    take_subset,
    take_subset_x,
)


def _sample_cycle_sigma(
    loss_cfg: dict,
    sigma_c: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample one low-noise shared sigma used by M4 cycle loss."""
    sigma_min = float(loss_cfg["sigma_min"])
    sigma_max = min(float(loss_cfg["sigma_max"]), float(sigma_c))
    if sigma_max <= sigma_min:
        return torch.full((1,), sigma_min, device=device, dtype=dtype)

    return sample_training_sigmas(
        batch_size=1,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=device,
        dtype=dtype,
        mode=str(loss_cfg.get("sigma_sampling", "log_uniform")),
        edm_p_mean=float(loss_cfg.get("edm_p_mean", -1.2)),
        edm_p_std=float(loss_cfg.get("edm_p_std", 1.2)),
        clamp=bool(loss_cfg.get("sigma_sample_clamp", True)),
    )


def train_step_m4(
    model: HybridWrapper,
    x0: torch.Tensor,
    cfg: dict,
    step: int,
    feature_encoder: torch.nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one M4 hybrid training step.

    Args:
        model: Hybrid wrapper containing high-noise score and low-noise
            potential branches.
        x0: Clean training batch.
        cfg: Resolved config dictionary.
        step: Global optimization step.
        feature_encoder: Frozen encoder for optional low-noise cycle loss.

    Returns:
        Tuple ``(loss, metrics)`` with DSM, boundary-match, and optional cycle.

    How it works:
        Uses sigma-gated hybrid score for DSM, then applies boundary matching
        near ``sigma_c`` and optional low-noise graph cycle consistency.
    """
    if not isinstance(model, HybridWrapper):
        raise TypeError("train_step_m4 expects HybridWrapper model")

    loss_cfg = cfg["loss"]
    sigma_c = float(loss_cfg.get("sigma_c", loss_cfg.get("sigma0", 0.25)))
    alpha = float(loss_cfg.get("alpha", loss_cfg.get("lambda0", loss_cfg.get("lambda_sym", 0.0))))
    beta = float(loss_cfg.get("beta", 0.0))
    objective = str(loss_cfg.get("objective", "dsm_score"))
    sigma_sampling = str(loss_cfg.get("sigma_sampling", "log_uniform"))
    edm_p_mean = float(loss_cfg.get("edm_p_mean", -1.2))
    edm_p_std = float(loss_cfg.get("edm_p_std", 1.2))
    sigma_sample_clamp = bool(loss_cfg.get("sigma_sample_clamp", True))
    sigma_data = float(loss_cfg.get("sigma_data", cfg.get("model", {}).get("preconditioning", {}).get("sigma_data", 0.5)))
    apply_reg = should_apply_regularizer(step=step, cfg=cfg)
    apply_boundary = apply_reg and is_due(step, loss_cfg.get("match_freq", loss_cfg.get("boundary_freq")))
    apply_cycle = apply_reg and is_due(step, loss_cfg.get("cycle_freq"))
    cycle_same_sigma = bool(loss_cfg.get("cycle_same_sigma", True))
    # Reuse DSM score cache for cycle path when sampling from the same batch.
    need_score_cache = apply_cycle and beta > 0.0 and (not cycle_same_sigma)

    dsm, cache = compute_dsm_for_score(
        score_fn=lambda x, s: model.score(x, s, create_graph=True),
        x0=x0,
        sigma_min=float(loss_cfg["sigma_min"]),
        sigma_max=float(loss_cfg["sigma_max"]),
        weight_mode=str(loss_cfg.get("weight_mode", "sigma2")),
        objective=objective,
        sigma_sampling=sigma_sampling,
        edm_p_mean=edm_p_mean,
        edm_p_std=edm_p_std,
        sigma_sample_clamp=sigma_sample_clamp,
        sigma_data=sigma_data,
        cache_score=need_score_cache,
    )

    x = cache["x"]
    sigma = cache["sigma"]
    score = cache.get("score")

    match = torch.zeros((), device=x.device, dtype=x.dtype)
    cycle = torch.zeros((), device=x.device, dtype=x.dtype)

    if apply_reg:
        if apply_boundary and alpha > 0.0:
            boundary_subset = loss_cfg.get("boundary_subset", loss_cfg.get("cycle_subset"))
            boundary_subset = cap_subset(boundary_subset, loss_cfg.get("boundary_subset_cap"))
            # Match branches near sigma_c on freshly noised clean points
            # to better approximate E_{x, t≈t_c}.
            x_match = take_subset_x(x0, boundary_subset)
            match = boundary_match_estimator(
                model=model,
                x=x_match,
                sigma_c=sigma_c,
                bandwidth=float(loss_cfg.get("boundary_bandwidth", 0.05)),
                create_graph=True,
                treat_input_as_clean=True,
            )

        if apply_cycle and beta > 0.0:
            cycle_subset = cap_subset(loss_cfg.get("cycle_subset"), loss_cfg.get("cycle_subset_cap"))
            score_cycle = None

            if cycle_same_sigma:
                x0_cycle = take_subset_x(x0, cycle_subset)
                if x0_cycle.shape[0] >= 3:
                    sigma_shared = _sample_cycle_sigma(loss_cfg=loss_cfg, sigma_c=sigma_c, device=x0_cycle.device, dtype=x0_cycle.dtype)
                    sigma_cycle = sigma_shared.expand(x0_cycle.shape[0])
                    x_cycle = make_noisy_from_clean(x0_cycle, sigma_cycle)
                else:
                    x_cycle = x0_cycle
                    sigma_cycle = torch.empty((0,), device=x0.device, dtype=x0.dtype)
            else:
                low_mask = sigma <= sigma_c
                x_low = x[low_mask]
                sigma_low = sigma[low_mask]
                score_low = None if score is None else score[low_mask]
                x_cycle, sigma_cycle, score_cycle = take_subset(
                    x_low,
                    sigma_low,
                    cycle_subset,
                    score=score_low,
                )

            if x_cycle.shape[0] >= 3:
                with torch.no_grad():
                    features = feature_encoder(x_cycle.detach())
                cycle_samples = cap_positive_int(
                    int(loss_cfg.get("cycle_samples", 16)),
                    loss_cfg.get("cycle_samples_cap"),
                )
                cycle, _ = graph_cycle_estimator(
                    score_fn=lambda x_in, s_in: model.low_score(x_in, s_in, create_graph=True),
                    x=x_cycle,
                    sigma=sigma_cycle,
                    features=features,
                    k=int(loss_cfg.get("cycle_knn_k", 8)),
                    cycle_lengths=loss_cfg.get("cycle_lengths", [3, 4, 5]),
                    num_cycles=cycle_samples,
                    subset_size=None,
                    precomputed_score=score_cycle,
                )

    total = dsm + alpha * match + beta * cycle

    metrics = {
        "loss_total": float(total.detach().item()),
        "loss_dsm": float(dsm.detach().item()),
        "loss_sym": 0.0,
        "loss_loop": 0.0,
        "loss_loop_multi": 0.0,
        "loss_cycle": float(cycle.detach().item()),
        "loss_match": float(match.detach().item()),
    }
    return total, metrics
