from __future__ import annotations

from typing import Callable

import torch

from .sigma_schedule import make_sigma_schedule, sigma_to_tensor


ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _sigma_view(sigma: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast per-sample sigma tensor to match ``x`` dimensions."""
    return sigma.view(sigma.shape[0], *([1] * (x.ndim - 1)))


def _edm_drift_from_score(
    x: torch.Tensor,
    score: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Compute EDM ODE drift ``(x - D(x,sigma)) / sigma`` from score.

    Notes:
        Since ``D(x,sigma) = x + sigma^2 * s(x,sigma)``, this is equivalent to
        ``-sigma * s(x,sigma)``. We keep denoiser form explicitly for clarity
        and alignment with the EDM update equation.
    """
    # Equivalent to (x - (x + sigma^2 * score)) / sigma.
    sigma_v = _sigma_view(sigma, x)
    return -sigma_v * score


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
    sigma_vals = [float(v) for v in sigmas]
    sigma_tensors = [sigma_to_tensor(v, bsz, device, x.dtype) for v in sigma_vals[:-1]]

    trajectory = [x.clone()] if return_trajectory else None
    # Tracks per-sample path length for dynamics diagnostics.
    traj_len = torch.zeros(bsz, device=device, dtype=dtype)

    if score_requires_grad:
        for i in range(len(sigmas) - 1):
            sigma_i = sigma_tensors[i]
            sigma_j = sigma_vals[i + 1]
            dt = sigma_j - sigma_vals[i]
            with torch.enable_grad():
                score = score_fn(x.detach(), sigma_i).detach()
            drift = _edm_drift_from_score(x=x, score=score, sigma=sigma_i)
            x_next = x + dt * drift
            traj_len += (x_next - x).flatten(start_dim=1).norm(dim=1)
            x = x_next
            if trajectory is not None:
                trajectory.append(x.clone())
    else:
        with torch.inference_mode():
            for i in range(len(sigmas) - 1):
                sigma_i = sigma_tensors[i]
                sigma_j = sigma_vals[i + 1]
                dt = sigma_j - sigma_vals[i]
                score = score_fn(x, sigma_i)
                drift = _edm_drift_from_score(x=x, score=score, sigma=sigma_i)
                x_next = x + dt * drift
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
