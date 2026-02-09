from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .loader_opts import make_loader_kwargs


def _get_torchvision():
    """Import torchvision modules lazily for optional dependency safety.

    Returns:
        Tuple of ``(datasets, transforms)`` modules.
    """
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torchvision is required for FFHQ dataset") from exc
    return datasets, transforms


def _resolve_data_path(data_root: str, maybe_relative: str) -> Path:
    """Resolve FFHQ split directory path from config values."""
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (Path(data_root) / p).resolve()


def _build_transform(image_size: int, train: bool, random_flip: bool):
    """Build FFHQ preprocessing transform for train/eval.

    Args:
        image_size: Target square image resolution.
        train: Whether to build train augmentation pipeline.
        random_flip: Whether to apply random horizontal flip.

    Returns:
        torchvision transform pipeline.
    """
    _, transforms = _get_torchvision()

    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if train:
        steps = [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        resize_size = int(round(image_size * 1.1))
        steps = [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    return transforms.Compose(steps)


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
    datasets, _ = _get_torchvision()

    ds_cfg = cfg["dataset"]
    ffhq_cfg = ds_cfg.get("ffhq", {})

    data_root = ds_cfg.get("data_root", "data")
    train_dir = _resolve_data_path(data_root, ffhq_cfg.get("train_dir", "ffhq/train"))
    val_dir = _resolve_data_path(data_root, ffhq_cfg.get("val_dir", "ffhq/val"))

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
    transform = _build_transform(image_size=image_size, train=train, random_flip=random_flip)
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
    out = []
    collected = 0
    for images, _ in loader:
        out.append(images)
        collected += images.shape[0]
        if collected >= num_samples:
            break
    cat = torch.cat(out, dim=0)
    return cat[:num_samples]
