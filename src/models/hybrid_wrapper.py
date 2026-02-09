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

    def __init__(
        self,
        score_high: nn.Module,
        potential_low: nn.Module,
        sigma_c: float,
        mixed_mode: str = "split",
    ) -> None:
        """Initialize hybrid model wrapper.

        Args:
            score_high: High-noise unconstrained score network.
            potential_low: Low-noise potential network.
            sigma_c: Sigma threshold separating low/high branches.
            mixed_mode: Mixed-batch evaluation mode in
                ``{\"auto\", \"split\", \"full\"}``.
        """
        super().__init__()
        self.score_high = score_high
        self.potential_low = potential_low
        self.sigma_c = float(sigma_c)
        mode = str(mixed_mode).lower()
        if mode not in {"auto", "split", "full"}:
            raise ValueError(f"unknown mixed_mode: {mixed_mode}")
        self.mixed_mode = mode

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
        # Evaluate only the needed branch per sample to avoid redundant work.
        mask = sigma <= self.sigma_c
        if bool(mask.all()):
            return self.low_score(x, sigma, create_graph=create_graph)
        if bool((~mask).all()):
            return self.high_score(x, sigma)

        mode = self.mixed_mode
        if mode == "auto":
            # Small images usually run faster in full mode, while large images
            # are safer in split mode to control memory usage.
            elems_per_sample = int(x[0].numel())
            mode = "full" if elems_per_sample <= 3 * 32 * 32 and x.shape[0] <= 128 else "split"

        if mode == "full":
            low = self.low_score(x, sigma, create_graph=create_graph)
            high = self.high_score(x, sigma).to(dtype=low.dtype)
            mask_view = mask.to(dtype=low.dtype).view(sigma.shape[0], *([1] * (x.ndim - 1)))
            return mask_view * low + (1.0 - mask_view) * high

        # Mixed regime in split mode: evaluate only the required branch.
        low_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        high_idx = torch.nonzero(~mask, as_tuple=False).squeeze(1)

        x_low = x.index_select(0, low_idx)
        sigma_low = sigma.index_select(0, low_idx)
        low = self.low_score(x_low, sigma_low, create_graph=create_graph)

        x_high = x.index_select(0, high_idx)
        sigma_high = sigma.index_select(0, high_idx)
        high = self.high_score(x_high, sigma_high).to(dtype=low.dtype)

        out = torch.empty_like(x, dtype=low.dtype)
        out.index_copy_(0, low_idx, low)
        out.index_copy_(0, high_idx, high)
        return out

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Forward alias returning differentiable hybrid score field."""
        return self.score(x, sigma, create_graph=True)
