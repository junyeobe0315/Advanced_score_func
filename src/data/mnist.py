from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .loader_opts import make_loader_kwargs


def _get_torchvision():
    """Import torchvision dataset and transform modules lazily.

    Returns:
        Tuple ``(datasets, transforms)`` from torchvision.

    How it works:
        Delays import until runtime so environments without torchvision fail
        with a clear error only when MNIST is actually requested.
    """
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torchvision is required for MNIST dataset") from exc
    return datasets, transforms


def make_mnist_loader(cfg: dict, train: bool = True) -> DataLoader:
    """Build DataLoader for MNIST with [-1, 1] normalization.

    Args:
        cfg: Resolved experiment config dictionary.
        train: Whether to load train split or test split.

    Returns:
        ``DataLoader`` yielding tuples ``(image, label)``.

    How it works:
        Uses torchvision MNIST dataset, converts images to tensors, normalizes
        with mean/std 0.5, and applies batch/worker settings from config.
    """
    datasets, transforms = _get_torchvision()

    # Keep image range centered near zero for diffusion-style training.
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    ds = datasets.MNIST(
        root=cfg["dataset"].get("data_root", "data"),
        train=train,
        transform=tfm,
        download=True,
    )

    loader = DataLoader(ds, **make_loader_kwargs(cfg, train=train, shuffle=train, default_num_workers=4))
    return loader


def sample_mnist_data(cfg: dict, num_samples: int) -> torch.Tensor:
    """Collect a fixed-size MNIST tensor batch for metric computation.

    Args:
        cfg: Resolved experiment config dictionary.
        num_samples: Number of images to collect.

    Returns:
        Tensor with shape ``[num_samples, 1, 28, 28]``.

    How it works:
        Iterates over evaluation loader until enough images are accumulated.
    """
    loader = make_mnist_loader(cfg, train=False)
    out = []
    for images, _ in loader:
        out.append(images)
        if sum(x.shape[0] for x in out) >= num_samples:
            break
    cat = torch.cat(out, dim=0)
    return cat[:num_samples]
