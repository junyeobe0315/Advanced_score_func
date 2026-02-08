from __future__ import annotations

from typing import Callable

import torch

from .sigma_schedule import make_sigma_schedule, sigma_to_tensor


ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def sample_euler(
    score_fn: ScoreFn,
    shape: tuple[int, ...],
    sigma_min: float,
    sigma_max: float,
    nfe: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    init_x: torch.Tensor | None = None,
    score_requires_grad: bool = False,
    return_trajectory: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Sample from score field using first-order Euler integration.

    Args:
        score_fn: Callable returning score from ``(x, sigma)``.
        shape: Output sample shape including batch dimension.
        sigma_min: Minimum sigma for schedule.
        sigma_max: Maximum sigma for schedule and initial noise scale.
        nfe: Number of integration steps.
        device: Target device.
        dtype: Output dtype.
        init_x: Optional initial state. If ``None``, starts from Gaussian noise.
        score_requires_grad: Enable input-grad path for structural models.
        return_trajectory: Whether to return full trajectory snapshots.

    Returns:
        Tuple ``(samples, stats)`` where ``samples`` is final state tensor and
        ``stats`` contains trajectory length summary and optional path states.

    How it works:
        Iterates sigma schedule from high to low noise and performs explicit
        Euler updates ``x_{i+1} = x_i + (sigma_{i+1}-sigma_i) * score(x_i)``.
    """
    bsz = shape[0]
    # Start from Gaussian prior scaled by highest noise level when no init given.
    x = init_x if init_x is not None else torch.randn(shape, device=device, dtype=dtype) * float(sigma_max)
    sigmas = make_sigma_schedule(sigma_min, sigma_max, nfe, include_zero=True, device=device)

    trajectory = [x.clone()] if return_trajectory else None
    # Tracks per-sample path length for dynamics diagnostics.
    traj_len = torch.zeros(bsz, device=device, dtype=dtype)

    for i in range(len(sigmas) - 1):
        sigma_i = sigma_to_tensor(float(sigmas[i]), bsz, device, x.dtype)
        sigma_j = float(sigmas[i + 1])
        if score_requires_grad:
            with torch.enable_grad():
                x_req = x.detach().requires_grad_(True)
                score = score_fn(x_req, sigma_i).detach()
        else:
            with torch.no_grad():
                score = score_fn(x, sigma_i)
        # Euler drift update with sigma-step as integration interval.
        step = (sigma_j - float(sigmas[i])) * score
        x_next = x + step
        traj_len += (x_next - x).flatten(start_dim=1).norm(dim=1)
        x = x_next
        if trajectory is not None:
            trajectory.append(x.clone())

    stats = {
        "trajectory_length_mean": float(traj_len.mean().item()),
        "trajectory_length_std": float(traj_len.std().item()),
        "nfe": int(nfe),
    }
    if trajectory is not None:
        stats["trajectory"] = trajectory
    return x, stats
