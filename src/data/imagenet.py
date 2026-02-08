from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader


def _get_torchvision():
    """Import torchvision modules lazily for optional dependency safety.

    Returns:
        Tuple of ``(datasets, transforms)`` modules.

    How it works:
        Defers torchvision import until dataset is actually requested so
        environments without torchvision fail with a clear runtime error.
    """
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torchvision is required for ImageNet dataset") from exc
    return datasets, transforms


def _resolve_data_path(data_root: str, maybe_relative: str) -> Path:
    """Resolve dataset path from root + configured subpath.

    Args:
        data_root: Base data directory from config.
        maybe_relative: Absolute path or data-root-relative path.

    Returns:
        Resolved path object.
    """
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (Path(data_root) / p).resolve()


def _build_transform(image_size: int, train: bool, random_flip: bool):
    """Build ImageNet preprocessing transform for train/eval splits.

    Args:
        image_size: Target square image resolution.
        train: Whether to create training augmentation pipeline.
        random_flip: Whether to enable horizontal flip augmentation.

    Returns:
        torchvision transform pipeline.

    How it works:
        Training uses random resized crop; eval uses resize + center crop. Both
        normalize image range to roughly [-1, 1].
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
    datasets, _ = _get_torchvision()

    ds_cfg = cfg["dataset"]
    imagenet_cfg = ds_cfg.get("imagenet", {})

    data_root = ds_cfg.get("data_root", "data")
    train_dir = imagenet_cfg.get("train_dir", "imagenet/train")
    val_dir = imagenet_cfg.get("val_dir", "imagenet/val")
    split_dir = train_dir if train else val_dir

    image_size = int(ds_cfg["image_size"])
    random_flip = bool(imagenet_cfg.get("random_flip", True))

    dataset_root = _resolve_data_path(data_root, split_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"ImageNet directory not found: {dataset_root}. "
            "Expected class-subfolder imagefolder format."
        )

    transform = _build_transform(image_size=image_size, train=train, random_flip=random_flip)
    dataset = datasets.ImageFolder(root=str(dataset_root), transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=int(ds_cfg["batch_size"]),
        shuffle=train,
        num_workers=int(ds_cfg.get("num_workers", 4)),
        pin_memory=bool(ds_cfg.get("pin_memory", True)),
        drop_last=train,
    )
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
    out = []
    collected = 0
    for images, _ in loader:
        out.append(images)
        collected += images.shape[0]
        if collected >= num_samples:
            break
    cat = torch.cat(out, dim=0)
    return cat[:num_samples]
