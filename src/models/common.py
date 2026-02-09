from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_group_count(channels: int, max_groups: int = 32) -> int:
    """Return a GroupNorm group count that always divides channels."""
    return max(1, math.gcd(int(channels), int(max_groups)))


def sigma_embedding(sigma: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal embedding for continuous noise level ``sigma``.

    Args:
        sigma: Noise-level tensor with shape ``[B]`` or ``[B, 1]``.
        dim: Output embedding dimension.

    Returns:
        Sinusoidal embedding tensor with shape ``[B, dim]``.

    How it works:
        Uses transformer-style sin/cos frequencies over log-spaced scales, as
        commonly used for diffusion timestep/noise conditioning.
    """
    if sigma.ndim == 2 and sigma.shape[1] == 1:
        sigma = sigma[:, 0]
    if sigma.ndim != 1:
        sigma = sigma.reshape(sigma.shape[0])

    # Half dimensions for sin and half for cos frequencies.
    half = dim // 2
    device = sigma.device
    exponent = torch.arange(half, device=device, dtype=sigma.dtype)
    exponent = -math.log(10000.0) * exponent / max(half - 1, 1)
    freqs = torch.exp(exponent)
    angles = sigma[:, None] * freqs[None, :]

    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class SigmaMLP(nn.Module):
    """Project raw sigma embedding into feature-conditioned vector.

    Notes:
        This module follows the standard diffusion conditioning pattern used in
        DDPM/EDM-style UNets: sinusoidal embedding followed by an MLP.
    """

    def __init__(self, sigma_embed_dim: int, out_dim: int) -> None:
        """Initialize sigma-conditioning MLP.

        Args:
            sigma_embed_dim: Dimension of sinusoidal sigma embedding.
            out_dim: Output feature dimension used by downstream blocks.
        """
        super().__init__()
        self.sigma_embed_dim = sigma_embed_dim
        self.net = nn.Sequential(
            nn.Linear(sigma_embed_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """Map noise level tensor to learned conditioning vector.

        Args:
            sigma: Noise-level tensor with shape ``[B]`` or ``[B, 1]``.

        Returns:
            Conditioning tensor with shape ``[B, out_dim]``.
        """
        emb = sigma_embedding(sigma, self.sigma_embed_dim)
        return self.net(emb)


class ResBlock2D(nn.Module):
    """Residual block with sigma conditioning for 2D feature maps.

    Notes:
        This block is adapted from DDPM-style ResNet blocks (Ho et al., 2020)
        with GroupNorm + SiLU + residual skip and additive time/noise embedding.
    """

    def __init__(self, in_ch: int, out_ch: int, temb_ch: int, dropout: float = 0.0) -> None:
        """Initialize residual block.

        Args:
            in_ch: Input channel count.
            out_ch: Output channel count.
            temb_ch: Sigma/time embedding channel count.
            dropout: Dropout probability in second conv path.
        """
        super().__init__()
        groups1 = _safe_group_count(in_ch)
        groups2 = _safe_group_count(out_ch)
        self.norm1 = nn.GroupNorm(groups1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.temb_proj = nn.Linear(temb_ch, out_ch)

        self.norm2 = nn.GroupNorm(groups2, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Apply residual transform conditioned by sigma embedding.

        Args:
            x: Feature map tensor ``[B, C, H, W]``.
            temb: Conditioning vector ``[B, temb_ch]``.

        Returns:
            Updated feature map tensor ``[B, out_ch, H, W]``.

        How it works:
            The block applies two normalized conv layers; sigma embedding is
            projected and added after first conv, then residual skip is added.
        """
        h = self.conv1(F.silu(self.norm1(x)))
        # Broadcast projected sigma embedding over spatial dimensions.
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention2D(nn.Module):
    """Single-head self-attention over spatial positions.

    Notes:
        This follows attention blocks used in diffusion UNets (DDPM and
        subsequent score-model literature) for global spatial interactions.
    """

    def __init__(self, channels: int) -> None:
        """Initialize 2D attention block.

        Args:
            channels: Number of feature channels.
        """
        super().__init__()
        groups = _safe_group_count(channels)
        self.norm = nn.GroupNorm(groups, channels)
        self.q = nn.Conv1d(channels, channels, kernel_size=1)
        self.k = nn.Conv1d(channels, channels, kernel_size=1)
        self.v = nn.Conv1d(channels, channels, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial self-attention with residual connection.

        Args:
            x: Feature map tensor ``[B, C, H, W]``.

        Returns:
            Attention-enhanced tensor with same shape as ``x``.

        How it works:
            Normalizes input, flattens spatial dimensions, computes Q/K/V
            attention, projects output, then adds residual skip.
        """
        b, c, h, w = x.shape
        # Flatten spatial grid to sequence length HW.
        z = self.norm(x).reshape(b, c, h * w)

        q = self.q(z).transpose(1, 2)  # [B,HW,C]
        k = self.k(z)  # [B,C,HW]
        v = self.v(z).transpose(1, 2)  # [B,HW,C]

        attn = torch.softmax((q @ k) * (c ** -0.5), dim=-1)
        out = attn @ v
        out = self.proj(out.transpose(1, 2)).reshape(b, c, h, w)
        return x + out


class Downsample2D(nn.Module):
    """Strided-convolution downsampling layer."""

    def __init__(self, ch: int) -> None:
        """Initialize downsampling conv.

        Args:
            ch: Input/output channel count.
        """
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample feature map by factor 2 using strided convolution."""
        return self.conv(x)


class Upsample2D(nn.Module):
    """Nearest-neighbor upsampling followed by 3x3 convolution."""

    def __init__(self, ch: int) -> None:
        """Initialize upsampling projection.

        Args:
            ch: Input/output channel count.
        """
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample feature map by factor 2 and refine with convolution."""
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNetBackbone(nn.Module):
    """Diffusion-style UNet backbone with sigma conditioning.

    Notes:
        Architecture is inspired by DDPM UNet (Ho et al., 2020) and modern
        score-model variants (e.g., EDM-style noise conditioning). It supports
        configurable channel multipliers, residual depth, and attention scales.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: int,
        base_channels: int,
        channel_mults: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        sigma_embed_dim: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize configurable UNet backbone.

        Args:
            in_channels: Input image channels.
            out_channels: Output channels.
            image_size: Input spatial size (assumed square).
            base_channels: Base channel width.
            channel_mults: Per-level channel multipliers.
            num_res_blocks: Residual blocks per level.
            attn_resolutions: Spatial resolutions where attention is enabled.
            sigma_embed_dim: Sinusoidal sigma embedding dimension.
            dropout: Dropout rate in residual blocks.
        """
        super().__init__()
        self.image_size = int(image_size)
        # Global sigma embedding projected for all residual blocks.
        self.sigma_mlp = SigmaMLP(sigma_embed_dim=sigma_embed_dim, out_dim=base_channels * 4)

        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        ch = base_channels
        # Tracks current spatial resolution while building hierarchy.
        cur_res = self.image_size
        # Stores skip-channel widths for mirrored up path construction.
        hs_channels: list[int] = [ch]

        self.down: nn.ModuleList = nn.ModuleList()
        for level, mult in enumerate(channel_mults):
            block = nn.ModuleDict()
            out_ch = base_channels * mult
            res_blocks = nn.ModuleList()
            attn_blocks = nn.ModuleList()

            for _ in range(num_res_blocks):
                res_blocks.append(ResBlock2D(ch, out_ch, base_channels * 4, dropout=dropout))
                ch = out_ch
                use_attn = cur_res in attn_resolutions
                attn_blocks.append(SelfAttention2D(ch) if use_attn else nn.Identity())
                hs_channels.append(ch)

            block["res"] = res_blocks
            block["attn"] = attn_blocks
            if level != len(channel_mults) - 1:
                block["downsample"] = Downsample2D(ch)
                hs_channels.append(ch)
                cur_res //= 2
            self.down.append(block)

        self.mid1 = ResBlock2D(ch, ch, base_channels * 4, dropout=dropout)
        self.mid_attn = SelfAttention2D(ch)
        self.mid2 = ResBlock2D(ch, ch, base_channels * 4, dropout=dropout)

        self.up: nn.ModuleList = nn.ModuleList()
        for rev_idx, mult in enumerate(reversed(channel_mults)):
            level = len(channel_mults) - 1 - rev_idx
            block = nn.ModuleDict()
            out_ch = base_channels * mult
            res_blocks = nn.ModuleList()
            attn_blocks = nn.ModuleList()

            for _ in range(num_res_blocks + 1):
                skip_ch = hs_channels.pop()
                res_blocks.append(ResBlock2D(ch + skip_ch, out_ch, base_channels * 4, dropout=dropout))
                ch = out_ch
                use_attn = cur_res in attn_resolutions
                attn_blocks.append(SelfAttention2D(ch) if use_attn else nn.Identity())

            block["res"] = res_blocks
            block["attn"] = attn_blocks
            if level != 0:
                block["upsample"] = Upsample2D(ch)
                cur_res *= 2
            self.up.append(block)

        self.out_norm = nn.GroupNorm(_safe_group_count(ch), ch)
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Run UNet forward pass with skip connections.

        Args:
            x: Input tensor ``[B, C, H, W]``.
            sigma: Noise levels tensor ``[B]`` or ``[B,1]``.

        Returns:
            Output tensor ``[B, out_channels, H, W]``.

        How it works:
            Encodes input through down path while caching skips, processes
            bottleneck blocks, then decodes with concatenated skip features.
        """
        # Shared conditioning vector reused in all residual blocks.
        temb = self.sigma_mlp(sigma)
        h = self.input_conv(x)
        hs: list[torch.Tensor] = [h]

        for block in self.down:
            for res, attn in zip(block["res"], block["attn"]):
                h = res(h, temb)
                h = attn(h)
                hs.append(h)
            if "downsample" in block:
                h = block["downsample"](h)
                hs.append(h)

        h = self.mid1(h, temb)
        h = self.mid_attn(h)
        h = self.mid2(h, temb)

        for block in self.up:
            for res, attn in zip(block["res"], block["attn"]):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = res(h, temb)
                h = attn(h)
            if "upsample" in block:
                h = block["upsample"](h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
