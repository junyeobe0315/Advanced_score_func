from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader


def _get_torchvision():
    """Import torchvision modules lazily for optional dependency safety.

    Returns:
        Tuple of ``(datasets, transforms)`` modules.
    """
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torchvision is required for LSUN dataset") from exc
    return datasets, transforms


def _resolve_data_path(data_root: str, maybe_relative: str) -> Path:
    """Resolve LSUN root path from absolute or relative config value."""
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (Path(data_root) / p).resolve()


def _normalize_classes(raw_classes, default: list[str]) -> list[str]:
    """Normalize class specification into list of LSUN class strings.

    Args:
        raw_classes: Config value that may be str or list.
        default: Fallback class list.

    Returns:
        List of class names accepted by torchvision LSUN dataset.
    """
    if raw_classes is None:
        return list(default)
    if isinstance(raw_classes, str):
        return [raw_classes]
    return [str(v) for v in raw_classes]


def _build_transform(image_size: int, train: bool, random_flip: bool):
    """Build LSUN preprocessing transform for train/eval.

    Args:
        image_size: Target square image resolution.
        train: Whether to create train augmentation pipeline.
        random_flip: Whether to enable horizontal flip augmentation.

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


def make_lsun_loader(cfg: dict, train: bool = True) -> DataLoader:
    """Build DataLoader for LSUN dataset.

    Args:
        cfg: Resolved experiment config dictionary.
        train: Whether to use train classes.

    Returns:
        ``DataLoader`` yielding tuples ``(image, class_index)``.

    How it works:
        Uses ``torchvision.datasets.LSUN`` with class names from config. If
        eval classes are unavailable and fallback is enabled, it reuses train
        classes for evaluation.
    """
    datasets, _ = _get_torchvision()

    ds_cfg = cfg["dataset"]
    lsun_cfg = ds_cfg.get("lsun", {})

    data_root = ds_cfg.get("data_root", "data")
    lsun_root = _resolve_data_path(data_root, lsun_cfg.get("root", "lsun"))

    train_classes = _normalize_classes(lsun_cfg.get("train_classes"), ["bedroom_train"])
    val_classes = _normalize_classes(lsun_cfg.get("val_classes"), ["bedroom_val"])
    classes = train_classes if train else val_classes

    if not lsun_root.exists():
        raise FileNotFoundError(f"LSUN root directory not found: {lsun_root}")

    image_size = int(ds_cfg["image_size"])
    random_flip = bool(lsun_cfg.get("random_flip", True))
    transform = _build_transform(image_size=image_size, train=train, random_flip=random_flip)

    try:
        dataset = datasets.LSUN(root=str(lsun_root), classes=classes, transform=transform)
    except Exception:
        use_fallback = (not train) and bool(lsun_cfg.get("allow_eval_train_fallback", True))
        if not use_fallback:
            raise
        dataset = datasets.LSUN(root=str(lsun_root), classes=train_classes, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=int(ds_cfg["batch_size"]),
        shuffle=train,
        num_workers=int(ds_cfg.get("num_workers", 4)),
        pin_memory=bool(ds_cfg.get("pin_memory", True)),
        drop_last=train,
    )
    return loader


def sample_lsun_data(cfg: dict, num_samples: int) -> torch.Tensor:
    """Collect a fixed number of LSUN images for evaluation metrics.

    Args:
        cfg: Resolved experiment config dictionary.
        num_samples: Number of samples to collect.

    Returns:
        Tensor with shape ``[num_samples, 3, H, W]``.
    """
    loader = make_lsun_loader(cfg, train=False)
    out = []
    collected = 0
    for images, _ in loader:
        out.append(images)
        collected += images.shape[0]
        if collected >= num_samples:
            break
    cat = torch.cat(out, dim=0)
    return cat[:num_samples]
