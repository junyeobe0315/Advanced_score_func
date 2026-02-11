from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .common import collect_samples_from_loader, get_torchvision
from .loader_opts import make_loader_kwargs


def make_cifar10_loader(cfg: dict, train: bool = True) -> DataLoader:
    """Build DataLoader for CIFAR-10 with channel-wise normalization.

    Args:
        cfg: Resolved experiment config dictionary.
        train: Whether to use training split.

    Returns:
        ``DataLoader`` yielding tuples ``(image, label)``.

    How it works:
        Loads CIFAR-10 via torchvision, normalizes RGB channels to roughly
        [-1, 1], then configures DataLoader fields from config.
    """
    datasets, transforms = get_torchvision("CIFAR-10")

    # Symmetric normalization around 0 for score model training.
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    ds = datasets.CIFAR10(
        root=cfg["dataset"].get("data_root", "data"),
        train=train,
        transform=tfm,
        download=True,
    )

    loader = DataLoader(ds, **make_loader_kwargs(cfg, train=train, shuffle=train, default_num_workers=4))
    return loader


def sample_cifar10_data(cfg: dict, num_samples: int) -> torch.Tensor:
    """Collect a fixed number of CIFAR-10 images for evaluation metrics.

    Args:
        cfg: Resolved experiment config dictionary.
        num_samples: Number of images to collect.

    Returns:
        Tensor with shape ``[num_samples, 3, 32, 32]``.

    How it works:
        Iterates over test loader and concatenates batches until target count.
    """
    loader = make_cifar10_loader(cfg, train=False)
    return collect_samples_from_loader(loader, num_samples=num_samples)
