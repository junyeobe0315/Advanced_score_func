from __future__ import annotations

import torch
import torch.nn as nn

from .common import SigmaMLP


class PotentialMLPToy(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, depth: int, sigma_embed_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.sigma_proj = SigmaMLP(sigma_embed_dim=sigma_embed_dim, out_dim=hidden_dim)

        layers: list[nn.Module] = [nn.Linear(dim + hidden_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        s_emb = self.sigma_proj(sigma)
        h = torch.cat([x, s_emb], dim=1)
        return self.net(h)
