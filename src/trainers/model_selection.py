from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data import sample_real_data
from src.data.toy import toy_mixture_parameters
from src.eval import compute_quality_metrics, generate_samples_batched
from src.models import score_fn_from_model


@dataclass
class SelectionScore:
    """Single model-selection score computed at one training step."""

    metric_name: str
    metric_value: float
    primary_score: float
    secondary_score: float
    details: dict[str, Any]


def _canonical_dataset_name(name: str) -> str:
    """Map dataset aliases to canonical identifiers."""
    token = str(name).lower()
    if token.startswith("imagenet"):
        return "imagenet"
    if token.startswith("lsun"):
        return "lsun"
    if token.startswith("ffhq"):
        return "ffhq"
    return token


def _safe_float(value: Any) -> float:
    """Convert a value to finite float, otherwise NaN."""
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _toy_gmm_logpdf_mean(samples: np.ndarray, centers: np.ndarray, stds: np.ndarray) -> float:
    """Compute mean log-density under toy target GMM."""
    diff = samples[:, None, :] - centers[None, :, :]
    sq = np.sum(diff * diff, axis=2)
    dim = int(samples.shape[1])
    var = np.maximum(stds[None, :] ** 2, 1.0e-12)
    coef = np.power(2.0 * np.pi * var, -0.5 * float(dim))
    pdf = np.mean(coef * np.exp(-0.5 * sq / var), axis=1)
    return float(np.mean(np.log(pdf + 1.0e-12)))


def _mmd_rbf(x: np.ndarray, y: np.ndarray, max_n: int = 1200) -> float:
    """Compute RBF-MMD with median-heuristic bandwidth."""
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


def evaluate_selection_score(
    cfg: dict,
    model: torch.nn.Module,
    model_id: str,
    device: torch.device,
    num_samples: int,
    nfe: int,
    batch_size: int,
    sampler_name: str = "heun",
) -> SelectionScore:
    """Compute dataset-aware selection score for checkpoint ranking.

    Ranking policy:
    - ``toy``: maximize true-GMM average log-density, tie-break with lower MMD.
    - ``mnist`` / ``cifar10``: minimize FID (encoded as ``primary=-fid``).
    """
    dataset_name = _canonical_dataset_name(str(cfg.get("dataset", {}).get("name", "")))

    real = sample_real_data(cfg, num_samples=int(num_samples))
    shape_per_sample = tuple(real.shape[1:])

    score_requires_grad = str(model_id).upper() in {"M2", "M4"}
    score_fn = score_fn_from_model(model, str(model_id).upper(), create_graph=False)

    fake, dynamics = generate_samples_batched(
        sampler_name=str(sampler_name),
        score_fn=score_fn,
        shape_per_sample=shape_per_sample,
        total=int(num_samples),
        batch_size=int(batch_size),
        sigma_min=float(cfg["loss"]["sigma_min"]),
        sigma_max=float(cfg["loss"]["sigma_max"]),
        nfe=int(nfe),
        device=device,
        score_requires_grad=score_requires_grad,
    )

    if dataset_name == "toy" and fake.ndim == 2 and int(fake.shape[1]) == 2:
        fake_np = fake.detach().cpu().numpy().astype(np.float64)
        real_np = real.detach().cpu().numpy().astype(np.float64)
        centers_t, stds_t = toy_mixture_parameters(cfg, dtype=torch.float64, device=torch.device("cpu"))
        logpdf = _toy_gmm_logpdf_mean(
            fake_np,
            centers=centers_t.numpy(),
            stds=stds_t.numpy(),
        )
        mmd = _mmd_rbf(fake_np, real_np)
        return SelectionScore(
            metric_name="toy_logpdf",
            metric_value=float(logpdf),
            primary_score=float(logpdf),
            secondary_score=float(-mmd),
            details={
                "toy_mmd": float(mmd),
                "sampler": str(sampler_name),
                "nfe": int(nfe),
                "trajectory_length_mean": _safe_float(dynamics.get("trajectory_length_mean")),
                "curvature_proxy": _safe_float(dynamics.get("curvature_proxy")),
            },
        )

    if dataset_name in {"mnist", "cifar10"}:
        quality = compute_quality_metrics(
            fake=fake,
            real=real,
            device=device,
            want_is=(dataset_name == "cifar10"),
            prefer_torch_fidelity=bool(cfg.get("eval", {}).get("use_torch_fidelity", True)),
        )
        fid = _safe_float(quality.fid)
        primary = -fid if math.isfinite(fid) else float("-inf")
        return SelectionScore(
            metric_name="fid",
            metric_value=fid,
            primary_score=float(primary),
            secondary_score=0.0,
            details={
                "kid": _safe_float(quality.kid),
                "is_mean": _safe_float(quality.inception_score_mean),
                "is_std": _safe_float(quality.inception_score_std),
                "metric_backend": str(quality.backend),
                "sampler": str(sampler_name),
                "nfe": int(nfe),
                "trajectory_length_mean": _safe_float(dynamics.get("trajectory_length_mean")),
                "curvature_proxy": _safe_float(dynamics.get("curvature_proxy")),
            },
        )

    # Unsupported datasets are never promoted to top-k candidates.
    return SelectionScore(
        metric_name="unsupported",
        metric_value=float("nan"),
        primary_score=float("-inf"),
        secondary_score=float("-inf"),
        details={"dataset": dataset_name},
    )


def _rank_key(record: dict[str, Any]) -> tuple[float, float, int]:
    """Sort key where larger is better."""
    return (
        float(record.get("primary_score", float("-inf"))),
        float(record.get("secondary_score", float("-inf"))),
        int(record.get("step", -1)),
    )


def update_topk_records(
    previous: list[dict[str, Any]],
    current: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    """Insert one record and keep only top-k ranked entries."""
    by_step: dict[int, dict[str, Any]] = {}
    for item in [*previous, current]:
        step = int(item.get("step", -1))
        old = by_step.get(step)
        if old is None or _rank_key(item) > _rank_key(old):
            by_step[step] = item
    ranked = sorted(by_step.values(), key=_rank_key, reverse=True)
    return ranked[: max(int(top_k), 1)]


def save_selection_state(path: Path, payload: dict[str, Any]) -> None:
    """Write model-selection state JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_selection_state(path: Path) -> dict[str, Any]:
    """Load model-selection state JSON file if present."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def append_selection_history_csv(path: Path, row: dict[str, Any]) -> None:
    """Append one model-selection row into CSV history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    base_cols = [
        "step",
        "checkpoint_name",
        "metric_name",
        "metric_value",
        "primary_score",
        "secondary_score",
    ]
    detail_cols = [
        "toy_mmd",
        "kid",
        "is_mean",
        "is_std",
        "metric_backend",
        "sampler",
        "nfe",
        "trajectory_length_mean",
        "curvature_proxy",
        "dataset",
    ]
    extra_detail_cols = sorted(set(row.get("details", {}).keys()) - set(detail_cols))
    detail_cols.extend(extra_detail_cols)
    fieldnames = [*base_cols, *detail_cols]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        flat = {
            "step": int(row.get("step", -1)),
            "checkpoint_name": str(row.get("checkpoint_name", "")),
            "metric_name": str(row.get("metric_name", "")),
            "metric_value": _safe_float(row.get("metric_value")),
            "primary_score": _safe_float(row.get("primary_score")),
            "secondary_score": _safe_float(row.get("secondary_score")),
        }
        for key in detail_cols:
            value = row.get("details", {}).get(key)
            if isinstance(value, (int, float)):
                flat[key] = _safe_float(value)
            else:
                flat[key] = value
        writer.writerow(flat)


def prune_eval_candidate_checkpoints(run_dir: Path, keep_steps: set[int]) -> None:
    """Delete ``eval_step_*.pt`` checkpoints except ones in ``keep_steps``."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return
    for path in ckpt_dir.glob("eval_step_*.pt"):
        stem = path.stem  # eval_step_00000100
        token = stem.rsplit("_", 1)[-1]
        try:
            step = int(token)
        except ValueError:
            step = -1
        if step not in keep_steps:
            path.unlink(missing_ok=True)


def score_to_record(step: int, checkpoint_name: str, score: SelectionScore) -> dict[str, Any]:
    """Convert score object to serializable ranking record."""
    out = {
        "step": int(step),
        "checkpoint_name": str(checkpoint_name),
        "metric_name": str(score.metric_name),
        "metric_value": float(score.metric_value),
        "primary_score": float(score.primary_score),
        "secondary_score": float(score.secondary_score),
        "details": dict(score.details),
    }
    # Ensure details values are JSON serializable scalars/strings.
    safe_details = {}
    for k, v in out["details"].items():
        if isinstance(v, (bool, str, int, float)) or v is None:
            safe_details[k] = v
        else:
            safe_details[k] = str(v)
    out["details"] = safe_details
    return out
