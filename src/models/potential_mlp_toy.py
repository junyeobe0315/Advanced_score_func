from __future__ import annotations

import torch
import torch.nn as nn

from .common import SigmaMLP


class PotentialMLPToy(nn.Module):
    """Toy-stage scalar potential network for structurally conservative model.

    Notes:
        This network outputs ``phi(x, sigma)``. The score field is obtained as
        ``grad_x phi``, enforcing integrability by construction.
    """

    def __init__(self, dim: int, hidden_dim: int, depth: int, sigma_embed_dim: int) -> None:
        """Initialize toy potential MLP.

        Args:
            dim: Input coordinate dimension.
            hidden_dim: Hidden layer width.
            depth: Number of hidden blocks.
            sigma_embed_dim: Sinusoidal sigma embedding dimension.
        """
        super().__init__()
        self.dim = dim
        self.sigma_proj = SigmaMLP(sigma_embed_dim=sigma_embed_dim, out_dim=hidden_dim)

        # Final layer outputs one scalar potential per sample.
        layers: list[nn.Module] = [nn.Linear(dim + hidden_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Predict scalar potential for toy data.

        Args:
            x: Input tensor with shape ``[B, dim]``.
            sigma: Noise levels tensor with shape ``[B]`` or ``[B,1]``.

        Returns:
            Potential tensor with shape ``[B, 1]``.

        How it works:
            Embeds sigma, concatenates with coordinates, and maps through MLP
            to produce scalar potential values.
        """
        s_emb = self.sigma_proj(sigma)
        h = torch.cat([x, s_emb], dim=1)
        return self.net(h)
