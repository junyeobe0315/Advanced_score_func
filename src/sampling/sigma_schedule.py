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
    i = torch.linspace(0, 1, nfe, device=device)
    inv_rho_min = sigma_min ** (1 / rho)
    inv_rho_max = sigma_max ** (1 / rho)
    sigmas = (inv_rho_max + i * (inv_rho_min - inv_rho_max)) ** rho
    if include_zero:
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device)], dim=0)
    return sigmas


def sigma_to_tensor(value: float, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.full((batch,), float(value), device=device, dtype=dtype)
