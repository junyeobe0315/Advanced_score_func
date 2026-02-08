from __future__ import annotations

import torch
import torch.nn as nn

from .common import UNetBackbone


class PotentialUNet(nn.Module):
    def __init__(
        self,
        channels: int,
        image_size: int,
        base_channels: int,
        channel_mults: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        sigma_embed_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.unet = UNetBackbone(
            in_channels=channels,
            out_channels=channels,
            image_size=image_size,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            sigma_embed_dim=sigma_embed_dim,
            dropout=dropout,
        )
        self.readout = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        h = self.unet(x, sigma)
        return self.readout(h)
