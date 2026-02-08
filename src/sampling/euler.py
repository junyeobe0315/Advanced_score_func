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
    bsz = shape[0]
    x = init_x if init_x is not None else torch.randn(shape, device=device, dtype=dtype) * float(sigma_max)
    sigmas = make_sigma_schedule(sigma_min, sigma_max, nfe, include_zero=True, device=device)

    trajectory = [x.clone()] if return_trajectory else None
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
