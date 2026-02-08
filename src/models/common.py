from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigma_embedding(sigma: torch.Tensor, dim: int) -> torch.Tensor:
    if sigma.ndim == 2 and sigma.shape[1] == 1:
        sigma = sigma[:, 0]
    if sigma.ndim != 1:
        sigma = sigma.reshape(sigma.shape[0])

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
    def __init__(self, sigma_embed_dim: int, out_dim: int) -> None:
        super().__init__()
        self.sigma_embed_dim = sigma_embed_dim
        self.net = nn.Sequential(
            nn.Linear(sigma_embed_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        emb = sigma_embedding(sigma, self.sigma_embed_dim)
        return self.net(emb)


class ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups1 = min(32, in_ch)
        groups2 = min(32, out_ch)
        self.norm1 = nn.GroupNorm(groups1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.temb_proj = nn.Linear(temb_ch, out_ch)

        self.norm2 = nn.GroupNorm(groups2, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(32, channels)
        self.norm = nn.GroupNorm(groups, channels)
        self.q = nn.Conv1d(channels, channels, kernel_size=1)
        self.k = nn.Conv1d(channels, channels, kernel_size=1)
        self.v = nn.Conv1d(channels, channels, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        z = self.norm(x).reshape(b, c, h * w)

        q = self.q(z).transpose(1, 2)  # [B,HW,C]
        k = self.k(z)  # [B,C,HW]
        v = self.v(z).transpose(1, 2)  # [B,HW,C]

        attn = torch.softmax((q @ k) * (c ** -0.5), dim=-1)
        out = attn @ v
        out = self.proj(out.transpose(1, 2)).reshape(b, c, h, w)
        return x + out


class Downsample2D(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNetBackbone(nn.Module):
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
        super().__init__()
        self.image_size = int(image_size)
        self.sigma_mlp = SigmaMLP(sigma_embed_dim=sigma_embed_dim, out_dim=base_channels * 4)

        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        ch = base_channels
        cur_res = self.image_size
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

        self.out_norm = nn.GroupNorm(min(32, ch), ch)
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
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
