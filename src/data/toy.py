from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .loader_opts import make_loader_kwargs


@dataclass
class ToyParams:
    """Parameter bundle for synthetic 2D/low-D Gaussian mixture data.

    Notes:
        This dataclass stores dataset geometry used in toy-stage experiments.
        It supports the classic ring GMM as well as configurable per-mode
        center/std variation.
    """

    # Feature dimension of each sample point.
    dim: int = 2
    # Number of Gaussian components arranged on a ring.
    n_modes: int = 8
    # Radius of mode centers from origin.
    radius: float = 4.0
    # Standard deviation for each Gaussian mode.
    std: float = 0.35
    # Deterministic seed used when sampling center/std variation.
    layout_seed: int = 0
    # Global angular offset (radians) applied to ring centers.
    angle_offset: float = 0.0
    # Per-mode angle jitter std (radians).
    angle_jitter: float = 0.0
    # Optional per-mode radius range when not using explicit centers.
    radius_min: float | None = None
    radius_max: float | None = None
    # Optional Gaussian jitter added to centers in first two dims.
    center_jitter_std: float = 0.0
    # Optional global translation applied to all centers.
    center_offset: tuple[float, ...] | None = None
    # Optional per-mode std range.
    std_min: float | None = None
    std_max: float | None = None
    # Optional explicit mode centers with shape [n_modes, <=dim].
    centers: torch.Tensor | None = None
    # Optional explicit per-mode std with shape [n_modes].
    stds: torch.Tensor | None = None


def _as_float_tuple(value: Any, name: str) -> tuple[float, ...] | None:
    """Parse optional numeric list/tuple into float tuple."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"dataset.toy.{name} must be a list/tuple or null")
    return tuple(float(v) for v in value)


def _coerce_centers(raw_centers: Any, n_modes: int, dim: int) -> torch.Tensor | None:
    """Parse explicit mode centers tensor from config."""
    if raw_centers is None:
        return None
    centers = torch.tensor(raw_centers, dtype=torch.float32)
    if centers.ndim != 2:
        raise ValueError("dataset.toy.centers must have shape [n_modes, dim_or_less]")
    if int(centers.shape[0]) != int(n_modes):
        raise ValueError(
            f"dataset.toy.centers first dim ({int(centers.shape[0])}) must match n_modes ({int(n_modes)})"
        )
    if int(centers.shape[1]) > int(dim):
        raise ValueError(
            f"dataset.toy.centers second dim ({int(centers.shape[1])}) cannot exceed toy dim ({int(dim)})"
        )
    if int(centers.shape[1]) < int(dim):
        pad = torch.zeros(int(n_modes), int(dim) - int(centers.shape[1]), dtype=torch.float32)
        centers = torch.cat([centers, pad], dim=1)
    return centers


def _coerce_stds(raw_stds: Any, n_modes: int) -> torch.Tensor | None:
    """Parse explicit per-mode std tensor from config."""
    if raw_stds is None:
        return None
    stds = torch.tensor(raw_stds, dtype=torch.float32).reshape(-1)
    if int(stds.shape[0]) != int(n_modes):
        raise ValueError(
            f"dataset.toy.stds length ({int(stds.shape[0])}) must match n_modes ({int(n_modes)})"
        )
    if bool(torch.any(stds <= 0.0)):
        raise ValueError("dataset.toy.stds must be strictly positive")
    return stds


def _build_toy_mixture(params: ToyParams) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic mode centers/stds from ToyParams."""
    generator = torch.Generator().manual_seed(int(params.layout_seed))

    if params.centers is not None:
        centers = params.centers.clone().to(dtype=torch.float32)
    else:
        angles = torch.linspace(0.0, 2.0 * torch.pi, int(params.n_modes) + 1, dtype=torch.float32)[:-1]
        angles = angles + float(params.angle_offset)
        if float(params.angle_jitter) > 0.0:
            angles = angles + float(params.angle_jitter) * torch.randn(int(params.n_modes), generator=generator)

        if params.radius_min is None and params.radius_max is None:
            radii = torch.full((int(params.n_modes),), float(params.radius), dtype=torch.float32)
        else:
            r_min = float(params.radius if params.radius_min is None else params.radius_min)
            r_max = float(params.radius if params.radius_max is None else params.radius_max)
            if r_min <= 0.0 or r_max <= 0.0 or r_max < r_min:
                raise ValueError("dataset.toy.radius_min/max must satisfy 0 < min <= max")
            radii = torch.empty(int(params.n_modes), dtype=torch.float32).uniform_(r_min, r_max, generator=generator)

        centers_2d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * radii[:, None]
        if float(params.center_jitter_std) > 0.0:
            centers_2d = centers_2d + float(params.center_jitter_std) * torch.randn(
                int(params.n_modes), 2, generator=generator
            )

        centers = torch.zeros(int(params.n_modes), int(params.dim), dtype=torch.float32)
        centers[:, :2] = centers_2d
        if params.center_offset is not None:
            if len(params.center_offset) > int(params.dim):
                raise ValueError(
                    f"dataset.toy.center_offset length ({len(params.center_offset)}) cannot exceed dim ({int(params.dim)})"
                )
            offset = torch.zeros(int(params.dim), dtype=torch.float32)
            offset[: len(params.center_offset)] = torch.tensor(params.center_offset, dtype=torch.float32)
            centers = centers + offset[None, :]

    if params.stds is not None:
        stds = params.stds.clone().to(dtype=torch.float32)
    elif params.std_min is not None or params.std_max is not None:
        s_min = float(params.std if params.std_min is None else params.std_min)
        s_max = float(params.std if params.std_max is None else params.std_max)
        if s_min <= 0.0 or s_max <= 0.0 or s_max < s_min:
            raise ValueError("dataset.toy.std_min/max must satisfy 0 < min <= max")
        stds = torch.empty(int(params.n_modes), dtype=torch.float32).uniform_(s_min, s_max, generator=generator)
    else:
        stds = torch.full((int(params.n_modes),), float(params.std), dtype=torch.float32)

    if bool(torch.any(stds <= 0.0)):
        raise ValueError("toy std must be strictly positive")
    return centers, stds


def toy_mixture_parameters(
    cfg: dict,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic toy GMM centers and per-mode stds from config."""
    params = _build_params(cfg)
    centers, stds = _build_toy_mixture(params)
    centers = centers.to(dtype=dtype, device=device)
    stds = stds.to(dtype=dtype, device=device)
    return centers, stds


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
        self.centers, self.stds = _build_toy_mixture(params)

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
        mode = torch.randint(0, int(self.params.n_modes), size=(1,)).item()
        center = self.centers[mode]
        std = self.stds[mode]
        sample = center + std * torch.randn(int(self.params.dim), dtype=center.dtype)
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
    dim = int(toy_cfg.get("dim", 2))
    n_modes = int(toy_cfg.get("n_modes", 8))
    return ToyParams(
        dim=dim,
        n_modes=n_modes,
        radius=float(toy_cfg.get("radius", 4.0)),
        std=float(toy_cfg.get("std", 0.35)),
        layout_seed=int(toy_cfg.get("layout_seed", 0)),
        angle_offset=float(toy_cfg.get("angle_offset", 0.0)),
        angle_jitter=float(toy_cfg.get("angle_jitter", 0.0)),
        radius_min=(None if toy_cfg.get("radius_min") is None else float(toy_cfg.get("radius_min"))),
        radius_max=(None if toy_cfg.get("radius_max") is None else float(toy_cfg.get("radius_max"))),
        center_jitter_std=float(toy_cfg.get("center_jitter_std", 0.0)),
        center_offset=_as_float_tuple(toy_cfg.get("center_offset"), "center_offset"),
        std_min=(None if toy_cfg.get("std_min") is None else float(toy_cfg.get("std_min"))),
        std_max=(None if toy_cfg.get("std_max") is None else float(toy_cfg.get("std_max"))),
        centers=_coerce_centers(toy_cfg.get("centers"), n_modes=n_modes, dim=dim),
        stds=_coerce_stds(toy_cfg.get("stds"), n_modes=n_modes),
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
