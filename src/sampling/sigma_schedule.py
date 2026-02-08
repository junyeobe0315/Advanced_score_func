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
