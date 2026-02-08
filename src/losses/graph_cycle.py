from __future__ import annotations

from typing import Callable, Iterable

import torch

from src.utils.kNN_graph import knn_indices, sample_cycles


ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _flatten(x: torch.Tensor) -> torch.Tensor:
    """Flatten all non-batch dimensions into one feature axis."""
    return x.flatten(start_dim=1)


def _circulation_from_cycles(
    score_flat: torch.Tensor,
    x_flat: torch.Tensor,
    cycles: torch.Tensor,
) -> torch.Tensor:
    """Compute per-cycle circulation values for a fixed cycle set.

    Args:
        score_flat: Flattened score tensor ``[B, D]``.
        x_flat: Flattened state tensor ``[B, D]``.
        cycles: Cycle index tensor ``[N, L]``.

    Returns:
        Per-cycle circulation tensor ``[N]``.

    How it works:
        Uses ``Circ(C) = Σ s(x_i)·(x_{i+1}-x_i)`` with wrap-around at cycle end.
    """
    if cycles.numel() == 0:
        return torch.zeros((0,), device=score_flat.device, dtype=score_flat.dtype)

    n_cycles, length = int(cycles.shape[0]), int(cycles.shape[1])
    total = torch.zeros((n_cycles,), device=score_flat.device, dtype=score_flat.dtype)

    for i in range(length):
        idx_i = cycles[:, i]
        idx_j = cycles[:, (i + 1) % length]

        s_i = score_flat[idx_i]
        dx = x_flat[idx_j] - x_flat[idx_i]
        total = total + (s_i * dx).sum(dim=1)

    return total


def graph_cycle_estimator(
    score_fn: ScoreFn,
    x: torch.Tensor,
    sigma: torch.Tensor,
    features: torch.Tensor,
    k: int,
    cycle_lengths: Iterable[int],
    num_cycles: int,
    subset_size: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Estimate nonlocal graph-cycle consistency loss for M3/M4.

    Args:
        score_fn: Score function ``s(x, sigma)``.
        x: Input batch ``[B, ...]``.
        sigma: Per-sample sigma tensor.
        features: Feature tensor ``[B, F]`` used for kNN graph.
        k: Number of neighbors for graph construction.
        cycle_lengths: Cycle lengths to evaluate (e.g. ``[3,4,5]``).
        num_cycles: Number of sampled cycles per length.
        subset_size: Optional subset size to reduce compute.
        generator: Optional PRNG generator.

    Returns:
        Tuple ``(mean_energy, per_length)`` where ``per_length`` stores
        ``E[Circ(C)^2]`` for each cycle length.

    How it works:
        Builds a kNN graph in feature space, samples random valid cycles,
        computes circulation squared for each cycle, and averages by length.
    """
    if x.shape[0] < 3:
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        return zero, {int(l): zero for l in cycle_lengths}

    if subset_size is not None and int(subset_size) < x.shape[0]:
        idx = torch.randperm(x.shape[0], device=x.device, generator=generator)[: int(subset_size)]
        x_use = x[idx]
        sigma_use = sigma[idx]
        feat_use = features[idx]
    else:
        x_use = x
        sigma_use = sigma
        feat_use = features

    feat_2d = feat_use.detach()
    if feat_2d.ndim != 2:
        feat_2d = _flatten(feat_2d)

    knn = knn_indices(feat_2d, k=int(k), exclude_self=True)
    cycle_map = sample_cycles(knn, cycle_lengths=cycle_lengths, num_cycles=int(num_cycles), generator=generator)

    score = score_fn(x_use, sigma_use)
    score_flat = _flatten(score)
    x_flat = _flatten(x_use)

    per_length: dict[int, torch.Tensor] = {}
    values = []

    for length in cycle_lengths:
        l = int(length)
        cycles = cycle_map.get(l)
        if cycles is None or cycles.numel() == 0:
            zero = torch.zeros((), device=x.device, dtype=x.dtype)
            per_length[l] = zero
            continue

        circ = _circulation_from_cycles(score_flat, x_flat, cycles)
        energy = (circ**2).mean()
        per_length[l] = energy
        values.append(energy)

    if not values:
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        return zero, per_length

    total = torch.stack(values).mean()
    return total, per_length
