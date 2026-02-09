from __future__ import annotations

import torch
import torch.nn as nn


def _sigma_view(sigma: torch.Tensor, x_ndim: int) -> torch.Tensor:
    """Reshape per-sample sigma tensor to broadcast over sample dimensions."""
    if sigma.ndim != 1:
        sigma = sigma.reshape(sigma.shape[0])
    return sigma.view(sigma.shape[0], *([1] * (x_ndim - 1)))


class EDMPreconditionedScore(nn.Module):
    """Wrap a score backbone with EDM-style denoiser preconditioning.

    Notes:
        The wrapped backbone predicts a denoiser residual term. This wrapper
        applies EDM coefficients and converts denoised output into score:
        ``score = (x_denoised - x) / sigma^2``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        sigma_data: float = 0.5,
        sigma_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.sigma_data = float(sigma_data)
        self.sigma_eps = float(sigma_eps)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Return score field using EDM preconditioning coefficients."""
        sigma_v = _sigma_view(sigma.to(dtype=x.dtype), x.ndim)
        sd = torch.as_tensor(self.sigma_data, device=x.device, dtype=x.dtype)
        sd2 = sd * sd
        sigma2 = sigma_v * sigma_v
        denom = (sigma2 + sd2).sqrt()

        c_in = 1.0 / denom
        c_skip = sd2 / (sigma2 + sd2)
        c_out = sigma_v * sd / denom

        # EDM uses c_noise = log(sigma)/4 as network conditioning input.
        c_noise = sigma.clamp_min(self.sigma_eps).log() / 4.0
        residual = self.backbone(c_in * x, c_noise)
        denoised = c_skip * x + c_out * residual
        return (denoised - x) / sigma2.clamp_min(self.sigma_eps * self.sigma_eps)
