from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from .loader_opts import make_loader_kwargs


@dataclass
class ToyParams:
    """Parameter bundle for synthetic 2D/low-D Gaussian mixture data.

    Notes:
        This dataclass stores dataset geometry used in the toy-stage
        experiments (number of modes, radius, and per-mode variance).
    """

    # Feature dimension of each sample point.
    dim: int = 2
    # Number of Gaussian components arranged on a ring.
    n_modes: int = 8
    # Radius of mode centers from origin.
    radius: float = 4.0
    # Standard deviation for each Gaussian mode.
    std: float = 0.35


class ToyGMMDataset(Dataset[torch.Tensor]):
    """Infinite-style synthetic Gaussian mixture dataset.

    Args:
        params: Toy distribution parameters.
        length: Virtual dataset length used by DataLoader.

    Returns:
        ``__getitem__`` returns one sample tensor with shape ``[dim]``.

    How it works:
        Mode centers are precomputed on a circle. Each sample picks a random
        mode index and draws Gaussian noise around that center.
    """

    def __init__(self, params: ToyParams, length: int = 100_000) -> None:
        """Initialize toy Gaussian mixture dataset state.

        Args:
            params: Toy distribution parameter bundle.
            length: Virtual dataset length reported to DataLoader.
        """
        self.params = params
        self.length = length

        # Uniformly spaced mode centers on a 2D ring.
        angles = torch.linspace(0.0, 2.0 * torch.pi, params.n_modes + 1)[:-1]
        centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * params.radius
        # Pad extra dimensions with zeros for dim > 2 experiments.
        if params.dim > 2:
            pad = torch.zeros(params.n_modes, params.dim - 2)
            centers = torch.cat([centers, pad], dim=1)
        self.centers = centers

    def __len__(self) -> int:
        """Return virtual dataset length for DataLoader iteration."""
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        """Sample one point from the configured Gaussian mixture.

        Args:
            index: Ignored index argument required by ``Dataset`` API.

        Returns:
            Single random sample tensor with shape ``[dim]``.

        How it works:
            Chooses a random mode center and adds isotropic Gaussian noise.
        """
        del index
        mode = torch.randint(0, self.params.n_modes, size=(1,)).item()
        center = self.centers[mode]
        sample = center + self.params.std * torch.randn(self.params.dim)
        return sample


def _build_params(cfg: dict) -> ToyParams:
    """Create ``ToyParams`` from experiment config mapping.

    Args:
        cfg: Resolved experiment config dictionary.

    Returns:
        Parsed ``ToyParams`` instance.

    How it works:
        Reads ``cfg['dataset']['toy']`` with safe defaults for missing keys.
    """
    toy_cfg = cfg["dataset"].get("toy", {})
    return ToyParams(
        dim=int(toy_cfg.get("dim", 2)),
        n_modes=int(toy_cfg.get("n_modes", 8)),
        radius=float(toy_cfg.get("radius", 4.0)),
        std=float(toy_cfg.get("std", 0.35)),
    )


def make_toy_loader(cfg: dict, train: bool = True) -> DataLoader:
    """Build DataLoader for toy Gaussian-mixture dataset.

    Args:
        cfg: Resolved experiment config dictionary.
        train: Whether to build a training loader.

    Returns:
        ``torch.utils.data.DataLoader`` yielding toy samples.

    How it works:
        Instantiates ``ToyGMMDataset`` with longer virtual length for training
        and configures batch/worker behavior from ``cfg``.
    """
    params = _build_params(cfg)
    # Longer virtual epoch for train mode to reduce iterator restarts.
    ds = ToyGMMDataset(params=params, length=200_000 if train else 20_000)
    return DataLoader(ds, **make_loader_kwargs(cfg, train=train, shuffle=True, default_num_workers=0))


def sample_toy_data(cfg: dict, num_samples: int) -> torch.Tensor:
    """Draw a fixed number of toy samples for evaluation metrics.

    Args:
        cfg: Resolved experiment config dictionary.
        num_samples: Number of points to generate.

    Returns:
        Tensor with shape ``[num_samples, dim]``.

    How it works:
        Creates a temporary toy dataset and repeatedly calls ``__getitem__``.
    """
    params = _build_params(cfg)
    ds = ToyGMMDataset(params=params, length=num_samples)
    out = [ds[i] for i in range(num_samples)]
    return torch.stack(out, dim=0)
