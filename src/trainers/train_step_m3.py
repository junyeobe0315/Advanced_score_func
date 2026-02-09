from __future__ import annotations

import torch

from src.losses import graph_cycle_estimator, loop_multi_scale_estimator
from src.sampling.sigma_schedule import sample_training_sigmas

from .common import compute_dsm_for_score


def _parse_float_list(value, fallback: list[float]) -> list[float]:
    """Parse list-like config value into float list."""
    if value is None:
        return fallback
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if isinstance(value, str):
        out = [float(v.strip()) for v in value.split(",") if v.strip()]
        return out if out else fallback
    return fallback


def _should_apply_regularizer(step: int, cfg: dict) -> bool:
    """Return whether optional M3 regularizers should run at this step."""
    freq = int(cfg["loss"].get("reg_freq", cfg["loss"].get("regularizer_every", 1)))
    return step % max(freq, 1) == 0


def _is_due(step: int, freq: int | None) -> bool:
    """Return whether per-term frequency gate is active at this step."""
    if freq is None:
        return True
    f = max(int(freq), 1)
    return step % f == 0


def _regularizer_batch(
    x: torch.Tensor,
    sigma: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor | None]:
    """Select batch slice used by optional M3 regularizers."""
    if not bool(cfg["loss"].get("reg_low_noise_only", True)):
        return x, sigma, 1.0, None

    sigma0 = float(cfg["loss"].get("sigma0", cfg["loss"].get("reg_low_noise_threshold", 0.25)))
    low_mask = sigma <= sigma0
    low_factor = float(low_mask.float().mean().item())
    if low_factor <= 0.0:
        return x[:0], sigma[:0], 0.0, low_mask
    return x[low_mask], sigma[low_mask], low_factor, low_mask


def _take_subset(
    x: torch.Tensor,
    sigma: torch.Tensor,
    subset_size: int | None,
    score: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Optionally sample a random batch subset for expensive regularizers."""
    if subset_size is None:
        return x, sigma, score
    n = int(subset_size)
    if n <= 0 or x.shape[0] <= n:
        return x, sigma, score
    idx = torch.randperm(x.shape[0], device=x.device)[:n]
    score_sub = None if score is None else score[idx]
    return x[idx], sigma[idx], score_sub


def _take_subset_x(x: torch.Tensor, subset_size: int | None) -> torch.Tensor:
    """Optionally sample a random subset from a batch tensor."""
    if subset_size is None:
        return x
    n = int(subset_size)
    if n <= 0 or x.shape[0] <= n:
        return x
    idx = torch.randperm(x.shape[0], device=x.device)[:n]
    return x[idx]


def _cap_subset(subset_size: int | None, cap: int | None) -> int | None:
    """Apply optional upper cap to subset size."""
    if cap is None:
        return subset_size
    cap_i = int(cap)
    if cap_i <= 0:
        return subset_size
    if subset_size is None:
        return cap_i
    n = int(subset_size)
    if n <= 0:
        return subset_size
    return min(n, cap_i)


def _cap_positive_int(value: int, cap: int | None) -> int:
    """Apply optional positive upper cap to integer value."""
    base = max(int(value), 1)
    if cap is None:
        return base
    cap_i = int(cap)
    if cap_i <= 0:
        return base
    return min(base, cap_i)


def _sample_cycle_sigma(loss_cfg: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sample one shared sigma for M3 cycle batches (same t)."""
    sigma_min = float(loss_cfg["sigma_min"])
    sigma_max = float(loss_cfg["sigma_max"])
    if bool(loss_cfg.get("reg_low_noise_only", True)):
        sigma0 = float(loss_cfg.get("sigma0", loss_cfg.get("reg_low_noise_threshold", 0.25)))
        sigma_max = min(sigma_max, sigma0)

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


def _make_noisy_from_clean(x0: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Create noisy samples from clean inputs and per-sample sigma."""
    eps = torch.randn_like(x0)
    sigma_view = sigma.view(x0.shape[0], *([1] * (x0.ndim - 1)))
    return x0 + sigma_view * eps


def train_step_m3(
    model: torch.nn.Module,
    x0: torch.Tensor,
    cfg: dict,
    step: int,
    feature_encoder: torch.nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one M3 training step with Jacobian-free nonlocal regularizers.

    Args:
        model: Unconstrained score model.
        x0: Clean training batch.
        cfg: Resolved config dictionary.
        step: Global step index.
        feature_encoder: Frozen feature extractor for kNN cycle graph.

    Returns:
        Tuple ``(loss, metrics)`` including DSM and M3 regularizer components.

    How it works:
        Computes DSM on noisy batch, then optionally adds:
        - multi-scale loop circulation energy ``R_loop_multi``
        - graph-cycle consistency energy ``R_cycle``
        with configurable low-noise gating and evaluation frequency.
    """
    loss_cfg = cfg["loss"]
    mu1 = float(loss_cfg.get("mu1", loss_cfg.get("mu_loop", 0.0)))
    mu2 = float(loss_cfg.get("mu2", 0.0))
    objective = str(loss_cfg.get("objective", "dsm_score"))
    sigma_sampling = str(loss_cfg.get("sigma_sampling", "log_uniform"))
    edm_p_mean = float(loss_cfg.get("edm_p_mean", -1.2))
    edm_p_std = float(loss_cfg.get("edm_p_std", 1.2))
    sigma_sample_clamp = bool(loss_cfg.get("sigma_sample_clamp", True))
    sigma_data = float(loss_cfg.get("sigma_data", cfg.get("model", {}).get("preconditioning", {}).get("sigma_data", 0.5)))
    apply_reg = _should_apply_regularizer(step=step, cfg=cfg)
    apply_loop = apply_reg and _is_due(step, loss_cfg.get("loop_freq"))
    apply_cycle = apply_reg and _is_due(step, loss_cfg.get("cycle_freq"))
    # Only loop term can reuse DSM score cache.
    need_score_cache = apply_loop and mu1 > 0.0

    dsm, cache = compute_dsm_for_score(
        score_fn=model,
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
    x_reg, sigma_reg, low_noise_factor, low_mask = _regularizer_batch(x, sigma, cfg)
    score_reg = None
    if score is not None:
        score_reg = score if low_mask is None else score[low_mask]

    loop_multi = torch.zeros((), device=x.device, dtype=x.dtype)
    cycle = torch.zeros((), device=x.device, dtype=x.dtype)

    if apply_reg and low_noise_factor > 0.0:
        if apply_loop and mu1 > 0.0 and x_reg.shape[0] > 0:
            loop_subset = loss_cfg.get("loop_subset", loss_cfg.get("cycle_subset"))
            loop_subset = _cap_subset(loop_subset, loss_cfg.get("loop_subset_cap"))
            x_loop, sigma_loop, score_loop = _take_subset(x_reg, sigma_reg, loop_subset, score=score_reg)
            delta_set = _parse_float_list(loss_cfg.get("delta_set"), fallback=[float(loss_cfg.get("loop_delta", 0.01))])
            loop_multi, _ = loop_multi_scale_estimator(
                score_fn=model,
                x=x_loop,
                sigma=sigma_loop,
                delta_set=delta_set,
                sparse_ratio=float(loss_cfg.get("loop_sparse_ratio", 1.0)),
                base_score=score_loop,
            )

        if apply_cycle and mu2 > 0.0:
            cycle_subset = _cap_subset(loss_cfg.get("cycle_subset"), loss_cfg.get("cycle_subset_cap"))
            cycle_same_sigma = bool(loss_cfg.get("cycle_same_sigma", True))
            score_cycle = None

            if cycle_same_sigma:
                x0_cycle = _take_subset_x(x0, cycle_subset)
                if x0_cycle.shape[0] >= 3:
                    sigma_shared = _sample_cycle_sigma(loss_cfg=loss_cfg, device=x0_cycle.device, dtype=x0_cycle.dtype)
                    sigma_cycle = sigma_shared.expand(x0_cycle.shape[0])
                    x_cycle = _make_noisy_from_clean(x0_cycle, sigma_cycle)
                else:
                    x_cycle = x0_cycle
                    sigma_cycle = torch.empty((0,), device=x0.device, dtype=x0.dtype)
            else:
                x_cycle, sigma_cycle, score_cycle = _take_subset(x_reg, sigma_reg, cycle_subset, score=score_reg)

            if x_cycle.shape[0] >= 3:
                with torch.no_grad():
                    features = feature_encoder(x_cycle.detach())
                cycle_samples = _cap_positive_int(
                    int(loss_cfg.get("cycle_samples", 16)),
                    loss_cfg.get("cycle_samples_cap"),
                )
                cycle, _ = graph_cycle_estimator(
                    score_fn=model,
                    x=x_cycle,
                    sigma=sigma_cycle,
                    features=features,
                    k=int(loss_cfg.get("cycle_knn_k", 8)),
                    cycle_lengths=loss_cfg.get("cycle_lengths", [3, 4, 5]),
                    num_cycles=cycle_samples,
                    subset_size=None,
                    precomputed_score=score_cycle,
                )

    total = dsm + (mu1 * low_noise_factor) * loop_multi + (mu2 * low_noise_factor) * cycle

    metrics = {
        "loss_total": float(total.detach().item()),
        "loss_dsm": float(dsm.detach().item()),
        "loss_sym": 0.0,
        "loss_loop": float(loop_multi.detach().item()),
        "loss_loop_multi": float(loop_multi.detach().item()),
        "loss_cycle": float(cycle.detach().item()),
        "loss_match": 0.0,
    }
    return total, metrics
