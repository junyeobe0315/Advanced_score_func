from __future__ import annotations

from pathlib import Path

import torch


def get_torchvision(dataset_label: str):
    """Import torchvision datasets/transforms lazily with dataset-scoped errors."""
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"torchvision is required for {dataset_label} dataset") from exc
    return datasets, transforms


def resolve_data_path(data_root: str, maybe_relative: str) -> Path:
    """Resolve dataset path from absolute or data-root-relative config value."""
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return (Path(data_root) / path).resolve()


def build_image_transform(
    transforms,
    image_size: int,
    train: bool,
    random_flip: bool,
):
    """Build common train/eval image-folder transform used by image datasets."""
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


def collect_samples_from_loader(loader, num_samples: int) -> torch.Tensor:
    """Collect a fixed-size tensor from a loader yielding (images, labels)."""
    out = []
    collected = 0
    for images, _ in loader:
        out.append(images)
        collected += int(images.shape[0])
        if collected >= int(num_samples):
            break
    cat = torch.cat(out, dim=0)
    return cat[:num_samples]
