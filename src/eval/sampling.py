from __future__ import annotations

from typing import Callable

import torch

from src.metrics import curvature_proxy
from src.sampling import sample_euler, sample_heun


ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def sampler_by_name(name: str):
    """Resolve sampler function from sampler name token."""
    token = str(name).lower()
    if token == "heun":
        return sample_heun
    if token == "euler":
        return sample_euler
    raise ValueError(f"unknown sampler: {name}")


def generate_samples_batched(
    sampler_name: str,
    score_fn: ScoreFn,
    shape_per_sample: tuple[int, ...],
    total: int,
    batch_size: int,
    sigma_min: float,
    sigma_max: float,
    nfe: int,
    device: torch.device,
    score_requires_grad: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Generate samples in chunks and aggregate trajectory diagnostics.

    Args:
        sampler_name: Name in ``{"heun", "euler"}``.
        score_fn: Callable score function.
        shape_per_sample: Sample shape excluding batch dimension.
        total: Number of samples to generate.
        batch_size: Chunk size per sampler call.
        sigma_min: Minimum sigma.
        sigma_max: Maximum sigma.
        nfe: Number of function evaluations.
        device: Sampling device.
        score_requires_grad: Enable grad path for structural/hybrid score.

    Returns:
        Tuple ``(samples_cpu, dynamics)`` where ``dynamics`` contains averaged
        trajectory length and curvature proxy.
    """
    sampler = sampler_by_name(sampler_name)

    samples_cpu = []
    traj_lengths = []
    curvatures = []

    produced = 0
    while produced < total:
        bsz = min(int(batch_size), total - produced)
        shape = (bsz, *shape_per_sample)
        fake, stats = sampler(
            score_fn=score_fn,
            shape=shape,
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            nfe=int(nfe),
            device=device,
            score_requires_grad=score_requires_grad,
            return_trajectory=(produced == 0),
        )

        samples_cpu.append(fake.detach().cpu())
        traj_lengths.append(float(stats["trajectory_length_mean"]))
        if "trajectory" in stats:
            curvatures.append(float(curvature_proxy(stats["trajectory"])))

        produced += bsz

    merged = torch.cat(samples_cpu, dim=0)
    dynamics = {
        "trajectory_length_mean": float(sum(traj_lengths) / max(1, len(traj_lengths))),
        "curvature_proxy": float(sum(curvatures) / max(1, len(curvatures))) if curvatures else 0.0,
    }
    return merged, dynamics
