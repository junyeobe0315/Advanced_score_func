from __future__ import annotations

import math

import torch


def sample_log_uniform_sigmas(
    batch_size: int,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample sigma values from a log-uniform distribution.

    Args:
        batch_size: Number of sigma samples.
        sigma_min: Minimum sigma.
        sigma_max: Maximum sigma.
        device: Target device for output tensor.
        dtype: Target dtype for output tensor.

    Returns:
        Tensor of shape ``[batch_size]`` with sampled sigmas.

    How it works:
        Uniformly samples in log-space and exponentiates back to sigma-space.
    """
    u = torch.rand(batch_size, device=device, dtype=dtype)
    log_min = math.log(float(sigma_min))
    log_max = math.log(float(sigma_max))
    return torch.exp(log_min + u * (log_max - log_min))


def sample_training_sigmas(
    batch_size: int,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
    dtype: torch.dtype,
    mode: str = "log_uniform",
    edm_p_mean: float = -1.2,
    edm_p_std: float = 1.2,
    clamp: bool = True,
) -> torch.Tensor:
    """Sample training sigmas using configurable policy.

    Args:
        batch_size: Number of sigma samples.
        sigma_min: Minimum sigma used by schedule/config.
        sigma_max: Maximum sigma used by schedule/config.
        device: Target device.
        dtype: Target dtype.
        mode: Sampling policy token. Supported values:
            ``"log_uniform"`` and ``"edm_lognormal"`` (aliases: ``"edm"``,
            ``"lognormal"``).
        edm_p_mean: EDM log-normal mean parameter.
        edm_p_std: EDM log-normal std parameter.
        clamp: Whether to clip sampled sigmas to ``[sigma_min, sigma_max]``.

    Returns:
        Tensor of shape ``[batch_size]`` with sampled sigmas.
    """
    token = str(mode).lower()
    if token in {"log_uniform", "loguniform"}:
        return sample_log_uniform_sigmas(
            batch_size=batch_size,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            device=device,
            dtype=dtype,
        )

    if token in {"edm", "edm_lognormal", "lognormal"}:
        sigma = torch.exp(
            torch.randn(batch_size, device=device, dtype=dtype) * float(edm_p_std) + float(edm_p_mean)
        )
        if clamp:
            sigma = sigma.clamp(min=float(sigma_min), max=float(sigma_max))
        return sigma

    raise ValueError(f"unknown sigma sampling mode: {mode}")


def make_sigma_schedule(
    sigma_min: float,
    sigma_max: float,
    nfe: int,
    rho: float = 7.0,
    include_zero: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create EDM-style sigma schedule for ODE/SDE sampling.

    Args:
        sigma_min: Minimum non-zero sigma.
        sigma_max: Maximum sigma.
        nfe: Number of function evaluations (steps).
        rho: Curvature parameter controlling schedule density.
        include_zero: Whether to append terminal sigma=0.
        device: Device for output tensor.

    Returns:
        Sigma schedule tensor of length ``nfe`` (or ``nfe + 1`` with zero).

    How it works:
        Interpolates in ``sigma^(1/rho)`` space and maps back with power ``rho``
        to allocate more steps in low-noise regime.
    """
    # Normalized step index in [0,1].
    i = torch.linspace(0, 1, nfe, device=device)
    inv_rho_min = sigma_min ** (1 / rho)
    inv_rho_max = sigma_max ** (1 / rho)
    sigmas = (inv_rho_max + i * (inv_rho_min - inv_rho_max)) ** rho
    if include_zero:
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device)], dim=0)
    return sigmas


def sigma_to_tensor(value: float, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Broadcast scalar sigma value to per-sample tensor.

    Args:
        value: Scalar sigma value.
        batch: Batch size.
        device: Target device.
        dtype: Output dtype.

    Returns:
        Tensor with shape ``[batch]`` filled with ``value``.
    """
    return torch.full((batch,), float(value), device=device, dtype=dtype)
