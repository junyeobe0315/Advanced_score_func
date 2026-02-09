from __future__ import annotations

import math
from typing import Iterable

import torch


def pairwise_squared_l2(features: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared L2 distance matrix.

    Args:
        features: Feature tensor of shape ``[B, F]``.

    Returns:
        Distance matrix ``[B, B]`` where entry ``(i,j)`` is
        ``||features[i] - features[j]||_2^2``.

    How it works:
        Uses the identity ``||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b`` to avoid
        explicit broadcasted subtraction tensors.
    """
    if features.ndim != 2:
        raise ValueError("pairwise_squared_l2 expects a [B, F] tensor")

    sq = (features**2).sum(dim=1, keepdim=True)
    dist = sq + sq.transpose(0, 1) - 2.0 * (features @ features.transpose(0, 1))
    return dist.clamp_min(0.0)


def knn_indices(features: torch.Tensor, k: int, exclude_self: bool = True) -> torch.Tensor:
    """Build k-nearest-neighbor index table from feature vectors.

    Args:
        features: Feature tensor ``[B, F]``.
        k: Number of neighbors to keep per node.
        exclude_self: Whether to disallow self-neighbor entries.

    Returns:
        Tensor of shape ``[B, k]`` containing integer neighbor indices.

    How it works:
        Computes all-pairs distances, masks diagonal when requested, and uses
        ``topk(largest=False)`` to select nearest neighbors.
    """
    bsz = int(features.shape[0])
    if bsz < 2:
        raise ValueError("knn_indices requires at least 2 samples")

    k = max(1, min(int(k), bsz - 1 if exclude_self else bsz))
    dist = pairwise_squared_l2(features)

    if exclude_self:
        eye = torch.eye(bsz, device=features.device, dtype=torch.bool)
        dist = dist.masked_fill(eye, float("inf"))

    return torch.topk(dist, k=k, largest=False, dim=1).indices


def sample_cycles(
    knn: torch.Tensor,
    cycle_lengths: Iterable[int],
    num_cycles: int,
    generator: torch.Generator | None = None,
) -> dict[int, torch.Tensor]:
    """Sample simple cycles over a kNN graph.

    Args:
        knn: kNN table with shape ``[B, k]``.
        cycle_lengths: Requested cycle lengths, typically ``[3,4,5]``.
        num_cycles: Target number of cycles per length.
        generator: Optional PRNG generator.

    Returns:
        Mapping from cycle length to tensor ``[N, L]`` of node indices.

    How it works:
        Performs random walks constrained by kNN edges, avoiding repeated
        vertices, and keeps only walks whose final node connects back to start
        in the kNN graph.
    """
    if knn.ndim != 2:
        raise ValueError("sample_cycles expects knn shape [B, k]")

    bsz, k = int(knn.shape[0]), int(knn.shape[1])
    out: dict[int, torch.Tensor] = {}

    if bsz < 3 or k <= 0 or int(num_cycles) <= 0:
        for raw_l in cycle_lengths:
            length = int(raw_l)
            if length < 3:
                continue
            out[length] = torch.empty((0, length), device=knn.device, dtype=torch.long)
        return out

    # Dense adjacency for O(1) closure checks: last node must connect to start.
    adj = torch.zeros((bsz, bsz), device=knn.device, dtype=torch.bool)
    row_idx = torch.arange(bsz, device=knn.device).view(bsz, 1).expand(bsz, k)
    adj[row_idx.reshape(-1), knn.reshape(-1)] = True

    for raw_l in cycle_lengths:
        length = int(raw_l)
        if length < 3:
            continue

        target = int(num_cycles)
        # Vectorized proposal sampler. Oversampling keeps acceptance high while
        # reducing Python-level loop overhead versus per-cycle rejection loops.
        proposal_batch = max(target * 8, 128)
        max_rounds = max(10, int(math.ceil((target * 20) / proposal_batch)))

        chunks: list[torch.Tensor] = []
        picked_count = 0

        for _ in range(max_rounds):
            if picked_count >= target:
                break

            starts = torch.randint(0, bsz, (proposal_batch,), device=knn.device, generator=generator)
            cycles = torch.empty((proposal_batch, length), device=knn.device, dtype=torch.long)
            cycles[:, 0] = starts
            current = starts

            # Build random walk over kNN edges.
            for pos in range(1, length):
                ridx = torch.randint(0, k, (proposal_batch,), device=knn.device, generator=generator)
                current = knn[current, ridx]
                cycles[:, pos] = current

            # Keep only simple cycles (no repeated vertices).
            sorted_vals = torch.sort(cycles, dim=1).values
            unique_mask = (sorted_vals[:, 1:] != sorted_vals[:, :-1]).all(dim=1)

            # Enforce explicit closure edge from last node back to start.
            closure_mask = adj[current, starts]
            valid = unique_mask & closure_mask
            if not bool(valid.any()):
                continue

            picked = cycles[valid]
            need = target - picked_count
            if picked.shape[0] > need:
                picked = picked[:need]

            chunks.append(picked)
            picked_count += int(picked.shape[0])

        if not chunks:
            out[length] = torch.empty((0, length), device=knn.device, dtype=torch.long)
        else:
            out[length] = torch.cat(chunks, dim=0)

    return out
