from __future__ import annotations

import torch

from src.losses import boundary_match_estimator, graph_cycle_estimator
from src.models.hybrid_wrapper import HybridWrapper

from .common import compute_dsm_for_score


def _should_apply_regularizer(step: int, cfg: dict) -> bool:
    """Return whether optional M4 regularizers should run at this step."""
    freq = int(cfg["loss"].get("reg_freq", cfg["loss"].get("regularizer_every", 1)))
    return step % max(freq, 1) == 0


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

    dsm, cache = compute_dsm_for_score(
        score_fn=lambda x, s: model.score(x, s, create_graph=True),
        x0=x0,
        sigma_min=float(loss_cfg["sigma_min"]),
        sigma_max=float(loss_cfg["sigma_max"]),
        weight_mode=str(loss_cfg.get("weight_mode", "sigma2")),
    )

    x = cache["x"]
    sigma = cache["sigma"]

    alpha = float(loss_cfg.get("alpha", loss_cfg.get("lambda0", loss_cfg.get("lambda_sym", 0.0))))
    beta = float(loss_cfg.get("beta", 0.0))

    match = torch.zeros((), device=x.device, dtype=x.dtype)
    cycle = torch.zeros((), device=x.device, dtype=x.dtype)

    if _should_apply_regularizer(step=step, cfg=cfg):
        if alpha > 0.0:
            match = boundary_match_estimator(
                model=model,
                x=x,
                sigma_c=sigma_c,
                bandwidth=float(loss_cfg.get("boundary_bandwidth", 0.05)),
                create_graph=True,
            )

        if beta > 0.0:
            low_mask = sigma <= sigma_c
            if int(low_mask.sum().item()) >= 3:
                x_low = x[low_mask]
                sigma_low = sigma[low_mask]
                with torch.no_grad():
                    features = feature_encoder(x_low.detach())
                cycle, _ = graph_cycle_estimator(
                    score_fn=lambda x_in, s_in: model.low_score(x_in, s_in, create_graph=True),
                    x=x_low,
                    sigma=sigma_low,
                    features=features,
                    k=int(loss_cfg.get("cycle_knn_k", 8)),
                    cycle_lengths=loss_cfg.get("cycle_lengths", [3, 4, 5]),
                    num_cycles=int(loss_cfg.get("cycle_samples", 16)),
                    subset_size=loss_cfg.get("cycle_subset"),
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
