from __future__ import annotations

import torch
import torch.nn as nn


class IdentityEncoder(nn.Module):
    """Feature encoder that returns flattened input vectors.

    Notes:
        Used for toy experiments where raw coordinates already form a meaningful
        manifold representation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return flattened features.

        Args:
            x: Input tensor ``[B, ...]``.

        Returns:
            Flattened feature tensor ``[B, D]``.
        """
        return x.flatten(start_dim=1)


class FrozenTinyEncoder(nn.Module):
    """Lightweight frozen convolutional feature extractor.

    Notes:
        This is intentionally small and deterministic. It is not trained during
        score-model optimization and only provides a stable space for kNN graph
        construction in cycle regularization.
    """

    def __init__(self, in_channels: int) -> None:
        """Initialize tiny CNN encoder.

        Args:
            in_channels: Number of channels of the input images.
        """
        super().__init__()

        # Compact conv stack keeps cycle-loss overhead modest.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
        )

        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract frozen low-dimensional features from image inputs.

        Args:
            x: Image tensor ``[B, C, H, W]``.

        Returns:
            Feature tensor ``[B, 64]``.
        """
        return self.encoder(x)


def build_feature_encoder(dataset_name: str, channels: int, device: torch.device) -> nn.Module:
    """Build dataset-aware frozen encoder used for graph-cycle losses.

    Args:
        dataset_name: Dataset identifier from config.
        channels: Input channel count.
        device: Runtime device for feature extraction.

    Returns:
        Frozen ``nn.Module`` feature encoder.

    How it works:
        Uses identity encoder for toy-style vector data and a tiny frozen CNN
        for image datasets (MNIST, CIFAR-10, and higher-resolution extensions).
    """
    name = str(dataset_name).lower()
    if name == "toy":
        model = IdentityEncoder()
    else:
        model = FrozenTinyEncoder(in_channels=int(channels))
    model.eval().to(device)
    return model
