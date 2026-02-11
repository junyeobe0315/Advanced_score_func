from __future__ import annotations

import math
from typing import Any

import numpy as np


def safe_float(value: Any) -> float:
    """Convert value to finite float, otherwise return NaN."""
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def toy_gmm_logpdf_mean(samples: np.ndarray, centers: np.ndarray, stds: np.ndarray) -> float:
    """Compute average log-density of samples under an isotropic toy GMM."""
    diff = samples[:, None, :] - centers[None, :, :]
    sq = np.sum(diff * diff, axis=2)
    dim = int(samples.shape[1])
    var = np.maximum(stds[None, :] ** 2, 1.0e-12)
    coef = np.power(2.0 * np.pi * var, -0.5 * float(dim))
    pdf = np.mean(coef * np.exp(-0.5 * sq / var), axis=1)
    return float(np.mean(np.log(pdf + 1.0e-12)))


def mmd_rbf(x: np.ndarray, y: np.ndarray, max_n: int = 1200) -> float:
    """Compute median-heuristic RBF-MMD between two sample sets."""
    rng = np.random.default_rng(123)
    if x.shape[0] > max_n:
        x = x[rng.choice(x.shape[0], size=max_n, replace=False)]
    if y.shape[0] > max_n:
        y = y[rng.choice(y.shape[0], size=max_n, replace=False)]

    z = np.concatenate([x, y], axis=0)
    d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    gamma = 1.0 / max(float(med), 1.0e-6)

    xx = np.exp(-gamma * np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2))
    yy = np.exp(-gamma * np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=2))
    xy = np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2))
    m, n = x.shape[0], y.shape[0]
    mmd2 = (xx.sum() - np.trace(xx)) / (m * (m - 1) + 1.0e-12)
    mmd2 += (yy.sum() - np.trace(yy)) / (n * (n - 1) + 1.0e-12)
    mmd2 -= 2.0 * xy.mean()
    return float(max(mmd2, 0.0))


def toy_mode_stats(samples: np.ndarray, centers: np.ndarray) -> tuple[int, float]:
    """Compute covered mode count and normalized mode-assignment entropy."""
    d2 = np.sum((samples[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    idx = np.argmin(d2, axis=1)
    counts = np.bincount(idx, minlength=centers.shape[0]).astype(np.float64)
    covered = int(np.sum(counts > 0))
    probs = counts / max(np.sum(counts), 1.0)
    entropy = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
    entropy_norm = float(entropy / max(np.log(float(centers.shape[0])), 1.0e-12))
    return covered, entropy_norm
