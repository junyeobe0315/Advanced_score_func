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
    """Compute EDM ODE drift ``(x - D(x,sigma)) / sigma`` from score."""
    # Equivalent to (x - (x + sigma^2 * score)) / sigma.
    sigma_v = _sigma_view(sigma, x)
    return -sigma_v * score


def sample_heun(
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
    """Sample from score field using second-order Heun predictor-corrector.

    Args:
        score_fn: Callable returning score from ``(x, sigma)``.
        shape: Output sample shape including batch dimension.
        sigma_min: Minimum sigma for schedule.
        sigma_max: Maximum sigma for schedule and initial state scale.
        nfe: Number of function evaluations.
        device: Target device.
        dtype: Output dtype.
        init_x: Optional initial state tensor.
        score_requires_grad: Enable input-grad path for structural models.
        return_trajectory: Whether to save trajectory snapshots.

    Returns:
        Tuple ``(samples, stats)`` with final tensor and dynamics metrics.

    How it works:
        Uses Euler predictor at each step and then corrects with average of
        start/end scores, giving a second-order update in sigma domain.
    """
    bsz = shape[0]
    # Default prior initialization at maximum noise scale.
    x = init_x if init_x is not None else torch.randn(shape, device=device, dtype=dtype) * float(sigma_max)
    sigmas = make_sigma_schedule(sigma_min, sigma_max, nfe, include_zero=True, device=device)
    sigma_vals = [float(v) for v in sigmas]
    sigma_tensors = [sigma_to_tensor(v, bsz, device, x.dtype) for v in sigma_vals]

    trajectory = [x.clone()] if return_trajectory else None
    # Per-sample path-length accumulator for analysis.
    traj_len = torch.zeros(bsz, device=device, dtype=dtype)

    if score_requires_grad:
        for i in range(len(sigmas) - 1):
            sigma_i = sigma_vals[i]
            sigma_j = sigma_vals[i + 1]
            dt = sigma_j - sigma_i

            sigma_i_t = sigma_tensors[i]
            with torch.enable_grad():
                score_i = score_fn(x.detach(), sigma_i_t).detach()
            d_cur = _edm_drift_from_score(x=x, score=score_i, sigma=sigma_i_t)

            # Predictor step (Euler) in EDM ODE parameterization.
            x_euler = x + dt * d_cur
            if sigma_j > 0.0:
                sigma_j_t = sigma_tensors[i + 1]
                with torch.enable_grad():
                    score_j = score_fn(x_euler.detach(), sigma_j_t).detach()
                d_prime = _edm_drift_from_score(x=x_euler, score=score_j, sigma=sigma_j_t)
                # Corrector step: average EDM drift at start/end.
                x_next = x + dt * 0.5 * (d_cur + d_prime)
            else:
                x_next = x_euler

            traj_len += (x_next - x).flatten(start_dim=1).norm(dim=1)
            x = x_next
            if trajectory is not None:
                trajectory.append(x.clone())
    else:
        with torch.inference_mode():
            for i in range(len(sigmas) - 1):
                sigma_i = sigma_vals[i]
                sigma_j = sigma_vals[i + 1]
                dt = sigma_j - sigma_i

                sigma_i_t = sigma_tensors[i]
                score_i = score_fn(x, sigma_i_t)
                d_cur = _edm_drift_from_score(x=x, score=score_i, sigma=sigma_i_t)

                # Predictor step (Euler) in EDM ODE parameterization.
                x_euler = x + dt * d_cur
                if sigma_j > 0.0:
                    sigma_j_t = sigma_tensors[i + 1]
                    score_j = score_fn(x_euler, sigma_j_t)
                    d_prime = _edm_drift_from_score(x=x_euler, score=score_j, sigma=sigma_j_t)
                    # Corrector step: average EDM drift at start/end.
                    x_next = x + dt * 0.5 * (d_cur + d_prime)
                else:
                    x_next = x_euler

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
