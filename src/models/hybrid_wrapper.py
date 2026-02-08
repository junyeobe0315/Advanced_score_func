from __future__ import annotations

import torch
import torch.nn as nn

from .potential_net import score_from_potential


class HybridWrapper(nn.Module):
    """Hybrid M4 model combining high-noise score net and low-noise potential.

    Notes:
        For ``sigma <= sigma_c`` the model uses hard-conservative score from
        potential gradient. For ``sigma > sigma_c`` it uses an unconstrained
        score network.
    """

    def __init__(self, score_high: nn.Module, potential_low: nn.Module, sigma_c: float) -> None:
        """Initialize hybrid model wrapper.

        Args:
            score_high: High-noise unconstrained score network.
            potential_low: Low-noise potential network.
            sigma_c: Sigma threshold separating low/high branches.
        """
        super().__init__()
        self.score_high = score_high
        self.potential_low = potential_low
        self.sigma_c = float(sigma_c)

    def _sigma_mask(self, sigma: torch.Tensor, x_ndim: int) -> torch.Tensor:
        """Create broadcastable low-noise mask from sigma threshold."""
        mask = (sigma <= self.sigma_c).to(dtype=sigma.dtype)
        shape = [sigma.shape[0]] + [1] * (x_ndim - 1)
        return mask.view(*shape)

    def low_score(self, x: torch.Tensor, sigma: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
        """Compute low-noise conservative score ``grad phi_low``."""
        return score_from_potential(self.potential_low, x, sigma, create_graph=create_graph)

    def high_score(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Compute high-noise unconstrained score prediction."""
        return self.score_high(x, sigma)

    def boundary_score_pair(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        *,
        create_graph: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pair of (low_branch_score, high_branch_score)."""
        low = self.low_score(x, sigma, create_graph=create_graph)
        high = self.high_score(x, sigma)
        return low, high

    def score(self, x: torch.Tensor, sigma: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
        """Return sigma-gated hybrid score field.

        Args:
            x: Input tensor.
            sigma: Per-sample sigma tensor.
            create_graph: Graph retention flag for low-branch gradient.

        Returns:
            Gated score tensor with same shape as ``x``.
        """
        low = self.low_score(x, sigma, create_graph=create_graph)
        high = self.high_score(x, sigma)
        mask = self._sigma_mask(sigma, x.ndim)
        return mask * low + (1.0 - mask) * high

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Forward alias returning differentiable hybrid score field."""
        return self.score(x, sigma, create_graph=True)
