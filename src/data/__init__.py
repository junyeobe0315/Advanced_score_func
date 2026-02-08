from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .cifar10 import make_cifar10_loader, sample_cifar10_data
from .ffhq import make_ffhq_loader, sample_ffhq_data
from .imagenet import make_imagenet_loader, sample_imagenet_data
from .lsun import make_lsun_loader, sample_lsun_data
from .mnist import make_mnist_loader, sample_mnist_data
from .toy import make_toy_loader, sample_toy_data


def _canonical_dataset_name(name: str) -> str:
    """Map dataset aliases (e.g. resolution-tagged names) to canonical id.

    Args:
        name: Dataset name from config.

    Returns:
        Canonical dataset identifier used by loader dispatch.

    How it works:
        Accepts names such as ``imagenet128`` or ``ffhq256`` and maps them to
        base identifiers ``imagenet`` and ``ffhq``.
    """
    lower = name.lower()
    if lower.startswith("imagenet"):
        return "imagenet"
    if lower.startswith("lsun"):
        return "lsun"
    if lower.startswith("ffhq"):
        return "ffhq"
    return lower


def make_loader(cfg: dict, train: bool = True) -> DataLoader:
    """Dispatch dataset-specific DataLoader construction.

    Args:
        cfg: Resolved experiment config dictionary.
        train: Whether to build a training loader.

    Returns:
        Dataset-specific ``DataLoader``.

    How it works:
        Reads ``cfg['dataset']['name']`` and routes to the matching loader
        factory for toy, MNIST, CIFAR-10, ImageNet, LSUN, or FFHQ.
    """
    name = _canonical_dataset_name(cfg["dataset"]["name"])
    if name == "toy":
        return make_toy_loader(cfg, train=train)
    if name == "mnist":
        return make_mnist_loader(cfg, train=train)
    if name == "cifar10":
        return make_cifar10_loader(cfg, train=train)
    if name == "imagenet":
        return make_imagenet_loader(cfg, train=train)
    if name == "lsun":
        return make_lsun_loader(cfg, train=train)
    if name == "ffhq":
        return make_ffhq_loader(cfg, train=train)
    raise ValueError(f"unknown dataset: {name}")


def sample_real_data(cfg: dict, num_samples: int) -> torch.Tensor:
    """Collect reference real data tensor for metric comparisons.

    Args:
        cfg: Resolved experiment config dictionary.
        num_samples: Number of real examples to gather.

    Returns:
        Tensor of real data samples.

    How it works:
        Dispatches to dataset-specific sampling helper using dataset name.
    """
    name = _canonical_dataset_name(cfg["dataset"]["name"])
    if name == "toy":
        return sample_toy_data(cfg, num_samples)
    if name == "mnist":
        return sample_mnist_data(cfg, num_samples)
    if name == "cifar10":
        return sample_cifar10_data(cfg, num_samples)
    if name == "imagenet":
        return sample_imagenet_data(cfg, num_samples)
    if name == "lsun":
        return sample_lsun_data(cfg, num_samples)
    if name == "ffhq":
        return sample_ffhq_data(cfg, num_samples)
    raise ValueError(f"unknown dataset: {name}")


def unpack_batch(batch) -> torch.Tensor:
    """Extract input tensor from heterogeneous DataLoader batch formats.

    Args:
        batch: Loader output, either tensor or tuple/list ``(x, y, ...)``.

    Returns:
        Input tensor ``x``.

    How it works:
        For supervised datasets it returns the first tuple element; for
        synthetic datasets it returns the batch tensor directly.
    """
    if isinstance(batch, (list, tuple)):
        return batch[0]
    if torch.is_tensor(batch):
        return batch
    raise TypeError(f"unsupported batch type: {type(batch)}")
