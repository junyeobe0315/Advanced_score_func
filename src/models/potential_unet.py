from __future__ import annotations

import torch
import torch.nn as nn

from .common import UNetBackbone


class PotentialUNet(nn.Module):
    """UNet-based scalar potential model for hard integrability constraint.

    Notes:
        Backbone is diffusion-style UNet (DDPM-family). A scalar readout head
        maps final features to ``phi(x, sigma)``; score is ``grad_x phi``.
    """

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
        """Initialize potential UNet.

        Args:
            channels: Input image channels.
            image_size: Input spatial size.
            base_channels: Base width for UNet.
            channel_mults: Per-resolution channel multipliers.
            num_res_blocks: Residual blocks per stage.
            attn_resolutions: Resolutions where attention is enabled.
            sigma_embed_dim: Sigma embedding dimension.
            dropout: Dropout probability for residual blocks.
        """
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
        # Scalar head converts feature map to one potential value per sample.
        self.readout = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Predict scalar potential from noisy image input.

        Args:
            x: Noisy image tensor ``[B, C, H, W]``.
            sigma: Noise levels tensor ``[B]``.

        Returns:
            Potential tensor with shape ``[B, 1]``.

        How it works:
            Runs UNet feature extractor and applies scalar readout head.
        """
        h = self.unet(x, sigma)
        return self.readout(h)
