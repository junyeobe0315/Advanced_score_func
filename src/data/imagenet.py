from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .common import build_image_transform, collect_samples_from_loader, get_torchvision, resolve_data_path
from .loader_opts import make_loader_kwargs


def make_imagenet_loader(cfg: dict, train: bool = True) -> DataLoader:
    """Build DataLoader for ImageNet image-folder dataset.

    Args:
        cfg: Resolved experiment config dictionary.
        train: Whether to use train split.

    Returns:
        ``DataLoader`` yielding tuples ``(image, class_index)``.

    How it works:
        Uses ``torchvision.datasets.ImageFolder`` with expected directory layout
        under configured train/val paths.
    """
    datasets, transforms = get_torchvision("ImageNet")

    ds_cfg = cfg["dataset"]
    imagenet_cfg = ds_cfg.get("imagenet", {})

    data_root = ds_cfg.get("data_root", "data")
    train_dir = imagenet_cfg.get("train_dir", "imagenet/train")
    val_dir = imagenet_cfg.get("val_dir", "imagenet/val")
    split_dir = train_dir if train else val_dir

    image_size = int(ds_cfg["image_size"])
    random_flip = bool(imagenet_cfg.get("random_flip", True))

    dataset_root = resolve_data_path(data_root, split_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"ImageNet directory not found: {dataset_root}. "
            "Expected class-subfolder imagefolder format."
        )

    transform = build_image_transform(transforms, image_size=image_size, train=train, random_flip=random_flip)
    dataset = datasets.ImageFolder(root=str(dataset_root), transform=transform)

    loader = DataLoader(dataset, **make_loader_kwargs(cfg, train=train, shuffle=train, default_num_workers=4))
    return loader


def sample_imagenet_data(cfg: dict, num_samples: int) -> torch.Tensor:
    """Collect a fixed number of ImageNet images for evaluation metrics.

    Args:
        cfg: Resolved experiment config dictionary.
        num_samples: Number of samples to collect.

    Returns:
        Tensor with shape ``[num_samples, 3, H, W]``.

    How it works:
        Iterates eval loader and concatenates image batches until enough
        examples are collected.
    """
    loader = make_imagenet_loader(cfg, train=False)
    return collect_samples_from_loader(loader, num_samples=num_samples)
