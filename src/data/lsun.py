from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .common import build_image_transform, collect_samples_from_loader, get_torchvision, resolve_data_path
from .loader_opts import make_loader_kwargs


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
    datasets, transforms = get_torchvision("LSUN")

    ds_cfg = cfg["dataset"]
    lsun_cfg = ds_cfg.get("lsun", {})

    data_root = ds_cfg.get("data_root", "data")
    lsun_root = resolve_data_path(data_root, lsun_cfg.get("root", "lsun"))

    train_classes = _normalize_classes(lsun_cfg.get("train_classes"), ["bedroom_train"])
    val_classes = _normalize_classes(lsun_cfg.get("val_classes"), ["bedroom_val"])
    classes = train_classes if train else val_classes

    if not lsun_root.exists():
        raise FileNotFoundError(f"LSUN root directory not found: {lsun_root}")

    image_size = int(ds_cfg["image_size"])
    random_flip = bool(lsun_cfg.get("random_flip", True))
    transform = build_image_transform(transforms, image_size=image_size, train=train, random_flip=random_flip)

    try:
        dataset = datasets.LSUN(root=str(lsun_root), classes=classes, transform=transform)
    except Exception:
        use_fallback = (not train) and bool(lsun_cfg.get("allow_eval_train_fallback", True))
        if not use_fallback:
            raise
        dataset = datasets.LSUN(root=str(lsun_root), classes=train_classes, transform=transform)

    loader = DataLoader(dataset, **make_loader_kwargs(cfg, train=train, shuffle=train, default_num_workers=4))
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
    return collect_samples_from_loader(loader, num_samples=num_samples)
