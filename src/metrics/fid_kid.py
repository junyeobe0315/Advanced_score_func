from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.linalg import sqrtm
except Exception:  # pragma: no cover
    sqrtm = None


@dataclass
class FIDKIDResult:
    fid: float
    kid: float


class FeatureExtractor:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._mode = "fallback"
        self.model = None
        self.pool = None
        self._init_model()

    def _init_model(self) -> None:
        try:
            from torchvision.models import inception_v3
            from torchvision.models import Inception_V3_Weights

            weights = Inception_V3_Weights.DEFAULT
            model = inception_v3(weights=weights, aux_logits=False)
            model.fc = torch.nn.Identity()
            model.eval().to(self.device)

            self.model = model
            self._mode = "inception"
        except Exception:
            self._mode = "fallback"
            self.pool = torch.nn.AdaptiveAvgPool2d((8, 8)).to(self.device)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        if self._mode == "inception" and self.model is not None:
            x = (x + 1.0) / 2.0
            x = x.clamp(0.0, 1.0)
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            feat = self.model(x)
            return feat

        # Fallback features: pooled normalized pixel embeddings.
        assert self.pool is not None
        x = self.pool(x)
        feat = x.flatten(start_dim=1)
        feat = F.normalize(feat, dim=1)
        return feat


def _to_numpy_cov(feats: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    arr = feats.detach().cpu().numpy()
    mu = np.mean(arr, axis=0)
    cov = np.cov(arr, rowvar=False)
    return mu, cov


def _frechet_distance(mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray) -> float:
    if sqrtm is None:
        # diagonal fallback
        diag_term = np.sum(np.sqrt(np.clip(np.diag(cov1), 0, None) * np.clip(np.diag(cov2), 0, None)))
        diff = mu1 - mu2
        return float(diff @ diff + np.trace(cov1) + np.trace(cov2) - 2.0 * diag_term)

    covmean = sqrtm(cov1 @ cov2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    fid = diff @ diff + np.trace(cov1) + np.trace(cov2) - 2.0 * np.trace(covmean)
    return float(np.real(fid))


def _polynomial_mmd(x: torch.Tensor, y: torch.Tensor) -> float:
    # KID with polynomial kernel (degree=3)
    dim = x.shape[1]
    c = 1.0

    k_xx = ((x @ x.t()) / dim + c) ** 3
    k_yy = ((y @ y.t()) / dim + c) ** 3
    k_xy = ((x @ y.t()) / dim + c) ** 3

    m = x.shape[0]
    n = y.shape[0]

    sum_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (m * (m - 1) + 1e-8)
    sum_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (n * (n - 1) + 1e-8)
    sum_xy = k_xy.mean()
    return float((sum_xx + sum_yy - 2.0 * sum_xy).item())


def compute_fid_kid(fake: torch.Tensor, real: torch.Tensor, device: torch.device) -> FIDKIDResult:
    extractor = FeatureExtractor(device=device)

    with torch.no_grad():
        f_fake = extractor(fake)
        f_real = extractor(real)

    mu_f, cov_f = _to_numpy_cov(f_fake)
    mu_r, cov_r = _to_numpy_cov(f_real)
    fid = _frechet_distance(mu_f, cov_f, mu_r, cov_r)

    # subsample for KID stability
    n = min(f_fake.shape[0], f_real.shape[0], 1000)
    idx_f = torch.randperm(f_fake.shape[0], device=f_fake.device)[:n]
    idx_r = torch.randperm(f_real.shape[0], device=f_real.device)[:n]
    kid = _polynomial_mmd(f_fake[idx_f], f_real[idx_r])

    return FIDKIDResult(fid=fid, kid=kid)
