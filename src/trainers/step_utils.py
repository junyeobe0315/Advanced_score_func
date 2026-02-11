from __future__ import annotations

import torch


def should_apply_regularizer(step: int, cfg: dict) -> bool:
    """Return whether regularizers should run at this global step."""
    freq = int(cfg["loss"].get("reg_freq", cfg["loss"].get("regularizer_every", 1)))
    return step % max(freq, 1) == 0


def should_apply_regularizer_with_sigma(step: int, cfg: dict, sigma: torch.Tensor) -> bool:
    """Return step/frequency gate with optional low-noise presence check."""
    if not should_apply_regularizer(step=step, cfg=cfg):
        return False
    if not bool(cfg["loss"].get("reg_low_noise_only", True)):
        return True
    thr = float(cfg["loss"].get("sigma0", cfg["loss"].get("reg_low_noise_threshold", 0.25)))
    return bool((sigma <= thr).any().item())


def is_due(step: int, freq: int | None) -> bool:
    """Return whether a per-term frequency gate is active at this step."""
    if freq is None:
        return True
    return step % max(int(freq), 1) == 0


def regularizer_batch(
    x: torch.Tensor,
    sigma: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor | None]:
    """Select the low-noise batch slice used by optional regularizers."""
    if not bool(cfg["loss"].get("reg_low_noise_only", True)):
        return x, sigma, 1.0, None

    sigma0 = float(cfg["loss"].get("sigma0", cfg["loss"].get("reg_low_noise_threshold", 0.25)))
    low_mask = sigma <= sigma0
    low_factor = float(low_mask.float().mean().item())
    if low_factor <= 0.0:
        return x[:0], sigma[:0], 0.0, low_mask
    return x[low_mask], sigma[low_mask], low_factor, low_mask


def take_subset(
    x: torch.Tensor,
    sigma: torch.Tensor,
    subset_size: int | None,
    score: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Optionally sample a random aligned subset from (x, sigma, score)."""
    if subset_size is None:
        return x, sigma, score
    n = int(subset_size)
    if n <= 0 or x.shape[0] <= n:
        return x, sigma, score
    idx = torch.randperm(x.shape[0], device=x.device)[:n]
    score_sub = None if score is None else score[idx]
    return x[idx], sigma[idx], score_sub


def take_subset_x(x: torch.Tensor, subset_size: int | None) -> torch.Tensor:
    """Optionally sample a random subset from one batch tensor."""
    if subset_size is None:
        return x
    n = int(subset_size)
    if n <= 0 or x.shape[0] <= n:
        return x
    idx = torch.randperm(x.shape[0], device=x.device)[:n]
    return x[idx]


def cap_subset(subset_size: int | None, cap: int | None) -> int | None:
    """Apply optional upper cap to subset size."""
    if cap is None:
        return subset_size
    cap_i = int(cap)
    if cap_i <= 0:
        return subset_size
    if subset_size is None:
        return cap_i
    n = int(subset_size)
    if n <= 0:
        return subset_size
    return min(n, cap_i)


def cap_positive_int(value: int, cap: int | None) -> int:
    """Apply optional positive upper cap to integer value."""
    base = max(int(value), 1)
    if cap is None:
        return base
    cap_i = int(cap)
    if cap_i <= 0:
        return base
    return min(base, cap_i)


def make_noisy_from_clean(x0: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Create noisy samples from clean inputs and per-sample sigma."""
    eps = torch.randn_like(x0)
    sigma_view = sigma.view(x0.shape[0], *([1] * (x0.ndim - 1)))
    return x0 + sigma_view * eps
