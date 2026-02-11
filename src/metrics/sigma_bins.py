from __future__ import annotations

import numpy as np
import torch


def sigma_bin_edges(sigma_min: float, sigma_max: float, bins: int) -> np.ndarray:
    """Create logarithmic sigma-bin edges."""
    return np.exp(np.linspace(np.log(float(sigma_min)), np.log(float(sigma_max)), int(bins) + 1))


def bucket_index(sigmas: torch.Tensor, edges: np.ndarray) -> torch.Tensor:
    """Assign each sigma value to a sigma-bin index."""
    edges_t = torch.tensor(edges, device=sigmas.device, dtype=sigmas.dtype)
    idx = torch.bucketize(sigmas, edges_t, right=True) - 1
    return idx.clamp(min=0, max=len(edges) - 2)
