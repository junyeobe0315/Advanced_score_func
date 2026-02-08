from __future__ import annotations

import torch
import torch.nn as nn

from .common import UNetBackbone


class ScoreUNet(nn.Module):
    """Image score network built on diffusion-style UNet backbone.

    Notes:
        Architecture follows DDPM/score-model UNet family and predicts
        ``s_theta(x, sigma)`` directly for baseline and soft-regularized runs.
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
        """Initialize score UNet.

        Args:
            channels: Input/output image channels.
            image_size: Input spatial size.
            base_channels: Base width for UNet.
            channel_mults: Per-resolution channel multipliers.
            num_res_blocks: Residual blocks per stage.
            attn_resolutions: Resolutions where attention is enabled.
            sigma_embed_dim: Sigma embedding dimension.
            dropout: Dropout probability used in residual blocks.
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

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Return score prediction for noisy image input.

        Args:
            x: Noisy image tensor ``[B, C, H, W]``.
            sigma: Noise levels tensor ``[B]``.

        Returns:
            Score tensor with same shape as ``x``.
        """
        return self.unet(x, sigma)
