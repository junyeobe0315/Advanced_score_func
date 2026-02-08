from __future__ import annotations

import torch

from src.losses import graph_cycle_estimator, loop_multi_scale_estimator

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


def _low_noise_weight(sigma: torch.Tensor, cfg: dict) -> float:
    """Compute scalar low-noise gate weight for M3 regularizers.

    Args:
        sigma: Per-sample sigma tensor.
        cfg: Resolved config dictionary.

    Returns:
        Scalar in ``[0,1]`` representing fraction of low-noise samples.
    """
    if not bool(cfg["loss"].get("reg_low_noise_only", True)):
        return 1.0
    sigma0 = float(cfg["loss"].get("sigma0", cfg["loss"].get("reg_low_noise_threshold", 0.25)))
    return float((sigma <= sigma0).float().mean().item())


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

    dsm, cache = compute_dsm_for_score(
        score_fn=model,
        x0=x0,
        sigma_min=float(loss_cfg["sigma_min"]),
        sigma_max=float(loss_cfg["sigma_max"]),
        weight_mode=str(loss_cfg.get("weight_mode", "sigma2")),
    )

    x = cache["x"]
    sigma = cache["sigma"]

    mu1 = float(loss_cfg.get("mu1", loss_cfg.get("mu_loop", 0.0)))
    mu2 = float(loss_cfg.get("mu2", 0.0))
    low_noise_factor = _low_noise_weight(sigma, cfg)

    loop_multi = torch.zeros((), device=x.device, dtype=x.dtype)
    cycle = torch.zeros((), device=x.device, dtype=x.dtype)

    if _should_apply_regularizer(step=step, cfg=cfg):
        if mu1 > 0.0:
            delta_set = _parse_float_list(loss_cfg.get("delta_set"), fallback=[float(loss_cfg.get("loop_delta", 0.01))])
            loop_multi, _ = loop_multi_scale_estimator(
                score_fn=model,
                x=x,
                sigma=sigma,
                delta_set=delta_set,
                sparse_ratio=float(loss_cfg.get("loop_sparse_ratio", 1.0)),
            )

        if mu2 > 0.0:
            with torch.no_grad():
                features = feature_encoder(x.detach())
            cycle, _ = graph_cycle_estimator(
                score_fn=model,
                x=x,
                sigma=sigma,
                features=features,
                k=int(loss_cfg.get("cycle_knn_k", 8)),
                cycle_lengths=loss_cfg.get("cycle_lengths", [3, 4, 5]),
                num_cycles=int(loss_cfg.get("cycle_samples", 16)),
                subset_size=loss_cfg.get("cycle_subset"),
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
