from __future__ import annotations

import math

import torch


def grad_norm_stats(parameters) -> dict[str, float]:
    norms = []
    for p in parameters:
        if p.grad is not None:
            norms.append(p.grad.detach().norm().item())

    if not norms:
        return {"grad_norm_mean": 0.0, "grad_norm_max": 0.0}

    return {
        "grad_norm_mean": float(sum(norms) / len(norms)),
        "grad_norm_max": float(max(norms)),
    }


def has_nan_or_inf(t: torch.Tensor) -> bool:
    return not torch.isfinite(t).all().item()


def model_has_nan_or_inf(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if p is None:
            continue
        if not torch.isfinite(p).all().item():
            return True
        if p.grad is not None and not torch.isfinite(p.grad).all().item():
            return True
    return False


def curvature_proxy(trajectory: list[torch.Tensor]) -> float:
    if len(trajectory) < 3:
        return 0.0
    vals = []
    for i in range(1, len(trajectory) - 1):
        d1 = trajectory[i] - trajectory[i - 1]
        d2 = trajectory[i + 1] - trajectory[i]
        cos = torch.nn.functional.cosine_similarity(
            d1.flatten(start_dim=1),
            d2.flatten(start_dim=1),
            dim=1,
            eps=1e-8,
        )
        vals.append((1.0 - cos).mean())
    return float(torch.stack(vals).mean().item())


def safe_float(value: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    return float(value)
