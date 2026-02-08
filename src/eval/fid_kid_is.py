from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from src.metrics import compute_fid_kid, compute_inception_score


@dataclass
class QualityMetrics:
    """Container for quality metrics returned by evaluation wrappers."""

    fid: float
    kid: float
    inception_score_mean: float
    inception_score_std: float
    backend: str


class _TensorImageDataset(Dataset):
    """Dataset wrapper exposing image tensors for torch-fidelity."""

    def __init__(self, images: torch.Tensor) -> None:
        """Store image tensor in `[0, 1]` format for metric extraction."""
        self.images = images

    def __len__(self) -> int:
        """Return number of images in wrapped tensor."""
        return int(self.images.shape[0])

    def __getitem__(self, idx: int):
        """Return one image and dummy class label expected by some APIs."""
        return self.images[idx], 0


def _to_unit_interval(images: torch.Tensor) -> torch.Tensor:
    """Map tensor values from `[-1, 1]` into `[0, 1]` range."""
    return ((images + 1.0) / 2.0).clamp(0.0, 1.0)


def _try_torch_fidelity(
    fake: torch.Tensor,
    real: torch.Tensor,
    want_is: bool,
) -> QualityMetrics | None:
    """Attempt to compute FID/KID/IS using torch-fidelity backend.

    Args:
        fake: Generated image tensor `[-1,1]`.
        real: Real image tensor `[-1,1]`.
        want_is: Whether to compute inception score.

    Returns:
        `QualityMetrics` when torch-fidelity succeeds, otherwise `None`.
    """
    try:
        from torch_fidelity import calculate_metrics
    except Exception:
        return None

    if fake.ndim != 4 or real.ndim != 4:
        return None

    fake_01 = _to_unit_interval(fake.detach().cpu())
    real_01 = _to_unit_interval(real.detach().cpu())

    fake_ds = _TensorImageDataset(fake_01)
    real_ds = _TensorImageDataset(real_01)

    metrics = calculate_metrics(
        input1=fake_ds,
        input2=real_ds,
        cuda=torch.cuda.is_available(),
        isc=bool(want_is),
        fid=True,
        kid=True,
        verbose=False,
    )

    return QualityMetrics(
        fid=float(metrics.get("frechet_inception_distance", float("nan"))),
        kid=float(metrics.get("kernel_inception_distance_mean", float("nan"))),
        inception_score_mean=float(metrics.get("inception_score_mean", float("nan"))),
        inception_score_std=float(metrics.get("inception_score_std", float("nan"))),
        backend="torch_fidelity",
    )


def compute_quality_metrics(
    fake: torch.Tensor,
    real: torch.Tensor,
    device: torch.device,
    want_is: bool,
    prefer_torch_fidelity: bool = True,
) -> QualityMetrics:
    """Compute FID/KID/IS using preferred backend with robust fallback.

    Args:
        fake: Generated samples tensor.
        real: Real reference tensor.
        device: Runtime device for fallback metric extraction.
        want_is: Whether to compute inception score.
        prefer_torch_fidelity: Whether to attempt torch-fidelity first.

    Returns:
        `QualityMetrics` dataclass.

    How it works:
        Tries torch-fidelity first when requested, and falls back to the local
        metric implementation (`src.metrics.*`) if unavailable.
    """
    if prefer_torch_fidelity:
        result = _try_torch_fidelity(fake, real, want_is=want_is)
        if result is not None:
            return result

    if fake.ndim != 4 or real.ndim != 4:
        return QualityMetrics(
            fid=float("nan"),
            kid=float("nan"),
            inception_score_mean=float("nan"),
            inception_score_std=float("nan"),
            backend="none_non_image",
        )

    fid_kid = compute_fid_kid(fake=fake.to(device), real=real.to(device), device=device)
    is_mean, is_std = (float("nan"), float("nan"))
    if want_is:
        is_mean, is_std = compute_inception_score(fake, device=device)

    return QualityMetrics(
        fid=float(fid_kid.fid),
        kid=float(fid_kid.kid),
        inception_score_mean=float(is_mean),
        inception_score_std=float(is_std),
        backend="fallback_local",
    )
