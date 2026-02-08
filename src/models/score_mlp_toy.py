from __future__ import annotations

import torch
import torch.nn as nn

from .common import SigmaMLP


class ScoreMLPToy(nn.Module):
    """Toy-stage MLP score network for low-dimensional data.

    Notes:
        This is a compact baseline architecture for 2D/low-D score modeling,
        conditioned on sigma through a learned embedding MLP.
    """

    def __init__(self, dim: int, hidden_dim: int, depth: int, sigma_embed_dim: int) -> None:
        """Initialize toy score MLP.

        Args:
            dim: Input/output data dimension.
            hidden_dim: Hidden layer width.
            depth: Number of hidden blocks.
            sigma_embed_dim: Sinusoidal sigma embedding dimension.
        """
        super().__init__()
        self.dim = dim
        self.sigma_proj = SigmaMLP(sigma_embed_dim=sigma_embed_dim, out_dim=hidden_dim)

        # Input features are concatenation of sample coordinates and sigma code.
        layers: list[nn.Module] = [nn.Linear(dim + hidden_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Predict score vector field for toy data.

        Args:
            x: Input tensor with shape ``[B, dim]``.
            sigma: Noise levels tensor with shape ``[B]`` or ``[B,1]``.

        Returns:
            Score tensor with shape ``[B, dim]``.

        How it works:
            Embeds sigma, concatenates with input coordinates, and maps through
            MLP to produce the denoising score estimate.
        """
        s_emb = self.sigma_proj(sigma)
        h = torch.cat([x, s_emb], dim=1)
        return self.net(h)
