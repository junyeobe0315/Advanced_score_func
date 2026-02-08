from __future__ import annotations

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


def _edge_lookup(knn: torch.Tensor) -> list[set[int]]:
    """Create O(1) adjacency lookup sets from kNN table."""
    return [set(row.detach().cpu().tolist()) for row in knn]


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

    bsz = int(knn.shape[0])
    adj = _edge_lookup(knn)
    out: dict[int, torch.Tensor] = {}

    for raw_l in cycle_lengths:
        length = int(raw_l)
        if length < 3:
            continue

        picked: list[list[int]] = []
        max_attempts = max(10 * int(num_cycles), 100)
        attempts = 0

        while len(picked) < int(num_cycles) and attempts < max_attempts:
            attempts += 1
            start = int(torch.randint(0, bsz, (1,), generator=generator).item())
            cycle = [start]
            used = {start}

            valid = True
            for _ in range(length - 1):
                neigh = knn[cycle[-1]]
                cand = [int(n.item()) for n in neigh if int(n.item()) not in used]
                if not cand:
                    valid = False
                    break

                idx = int(torch.randint(0, len(cand), (1,), generator=generator).item())
                nxt = cand[idx]
                cycle.append(nxt)
                used.add(nxt)

            if not valid:
                continue

            if start not in adj[cycle[-1]]:
                continue

            picked.append(cycle)

        if picked:
            out[length] = torch.tensor(picked, device=knn.device, dtype=torch.long)
        else:
            out[length] = torch.empty((0, length), device=knn.device, dtype=torch.long)

    return out
