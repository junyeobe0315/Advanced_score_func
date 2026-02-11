from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .common import build_image_transform, collect_samples_from_loader, get_torchvision, resolve_data_path
from .loader_opts import make_loader_kwargs


def make_ffhq_loader(cfg: dict, train: bool = True) -> DataLoader:
    """Build DataLoader for FFHQ image-folder dataset.

    Args:
        cfg: Resolved experiment config dictionary.
        train: Whether to use train split.

    Returns:
        ``DataLoader`` yielding tuples ``(image, class_index)``.

    How it works:
        Uses ``torchvision.datasets.ImageFolder``. For eval, if configured val
        directory is missing and fallback is enabled, train directory is reused.
    """
    datasets, transforms = get_torchvision("FFHQ")

    ds_cfg = cfg["dataset"]
    ffhq_cfg = ds_cfg.get("ffhq", {})

    data_root = ds_cfg.get("data_root", "data")
    train_dir = resolve_data_path(data_root, ffhq_cfg.get("train_dir", "ffhq/train"))
    val_dir = resolve_data_path(data_root, ffhq_cfg.get("val_dir", "ffhq/val"))

    if train:
        split_dir = train_dir
    else:
        use_val = val_dir.exists()
        allow_fallback = bool(ffhq_cfg.get("allow_eval_train_fallback", True))
        if use_val:
            split_dir = val_dir
        elif allow_fallback:
            split_dir = train_dir
        else:
            raise FileNotFoundError(f"FFHQ eval directory not found: {val_dir}")

    if not split_dir.exists():
        raise FileNotFoundError(f"FFHQ directory not found: {split_dir}")

    image_size = int(ds_cfg["image_size"])
    random_flip = bool(ffhq_cfg.get("random_flip", True))
    transform = build_image_transform(transforms, image_size=image_size, train=train, random_flip=random_flip)
    dataset = datasets.ImageFolder(root=str(split_dir), transform=transform)

    loader = DataLoader(dataset, **make_loader_kwargs(cfg, train=train, shuffle=train, default_num_workers=4))
    return loader


def sample_ffhq_data(cfg: dict, num_samples: int) -> torch.Tensor:
    """Collect a fixed number of FFHQ images for evaluation metrics.

    Args:
        cfg: Resolved experiment config dictionary.
        num_samples: Number of samples to collect.

    Returns:
        Tensor with shape ``[num_samples, 3, H, W]``.
    """
    loader = make_ffhq_loader(cfg, train=False)
    return collect_samples_from_loader(loader, num_samples=num_samples)
