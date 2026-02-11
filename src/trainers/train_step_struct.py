from __future__ import annotations

import torch

from src.models.potential_net import score_from_potential

from .dsm_step import run_dsm_only_step


def train_step_struct(model: torch.nn.Module, x0: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one training-step objective for structural integrability model.

    Args:
        model: Potential model.
        x0: Clean training batch.
        cfg: Resolved config dictionary.

    Returns:
        Tuple ``(loss, metrics)`` where loss is DSM over ``grad_x phi``.

    How it works:
        Wraps potential model with gradient-based score function and computes
        standard DSM objective without extra soft regularizers.
    """
    return run_dsm_only_step(
        score_fn=lambda x, s: score_from_potential(model, x, s, create_graph=True),
        x0=x0,
        cfg=cfg,
    )
