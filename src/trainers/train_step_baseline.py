from __future__ import annotations

import torch

from .dsm_step import run_dsm_only_step


def train_step_baseline(model: torch.nn.Module, x0: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one baseline training step objective computation.

    Args:
        model: Score model that directly outputs ``s_theta(x, sigma)``.
        x0: Clean training batch.
        cfg: Resolved config dictionary.

    Returns:
        Tuple ``(loss, metrics)`` where ``loss`` is scalar tensor used for
        backprop and ``metrics`` contains detached scalar logs.

    How it works:
        Computes pure DSM loss without integrability regularizers.
    """
    return run_dsm_only_step(score_fn=model, x0=x0, cfg=cfg)
