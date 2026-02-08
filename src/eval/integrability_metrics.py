from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import torch

from src.losses import graph_cycle_estimator, loop_multi_scale_estimator, reg_sym_estimator


ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def sigma_bin_edges(sigma_min: float, sigma_max: float, bins: int) -> np.ndarray:
    """Create logarithmic sigma-bin edges used for sigma-resolved metrics."""
    return np.exp(np.linspace(np.log(float(sigma_min)), np.log(float(sigma_max)), int(bins) + 1))


def _bucket_index(sigmas: torch.Tensor, edges: np.ndarray) -> torch.Tensor:
    """Assign each sigma value to a logarithmic bin index."""
    edges_t = torch.tensor(edges, device=sigmas.device, dtype=sigmas.dtype)
    idx = torch.bucketize(sigmas, edges_t, right=True) - 1
    return idx.clamp(min=0, max=len(edges) - 2)


def _parse_float_list(value, default: list[float]) -> list[float]:
    """Parse config value into a float list."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if isinstance(value, str):
        parsed = [float(v.strip()) for v in value.split(",") if v.strip()]
        return parsed if parsed else default
    return default


def _parse_int_list(value, default: list[int]) -> list[int]:
    """Parse config value into an integer list."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, str):
        parsed = [int(v.strip()) for v in value.split(",") if v.strip()]
        return parsed if parsed else default
    return default


def _empty_row(bin_id: int, sigma_lo: float, sigma_hi: float) -> dict[str, float | int | str]:
    """Build NaN row skeleton for empty sigma bins."""
    return {
        "bin": int(bin_id),
        "sigma_lo": float(sigma_lo),
        "sigma_hi": float(sigma_hi),
        "count": 0,
        "metric_name": "",
        "scale_delta": "",
        "cycle_len": "",
        "value": float("nan"),
    }


def integrability_records_by_sigma(
    score_fn: ScoreFn,
    x: torch.Tensor,
    sigma: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    bins: int,
    reg_k: int,
    reg_sym_method: str,
    reg_sym_eps_fd: float,
    loop_delta_set: Iterable[float],
    loop_sparse_ratio: float,
    cycle_lengths: Iterable[int],
    cycle_knn_k: int,
    cycle_samples: int,
    feature_encoder: torch.nn.Module | None,
) -> list[dict[str, float | int | str]]:
    """Compute sigma-bucketed integrability metrics in long-table format.

    Args:
        score_fn: Callable ``s(x, sigma)``.
        x: Input batch.
        sigma: Per-sample sigma tensor.
        sigma_min: Minimum sigma for logarithmic bins.
        sigma_max: Maximum sigma for logarithmic bins.
        bins: Number of sigma bins.
        reg_k: Probe count for Jacobian asymmetry estimator.
        reg_sym_method: Jacobian estimator method token.
        reg_sym_eps_fd: Finite-difference epsilon fallback.
        loop_delta_set: Multi-scale loop deltas.
        loop_sparse_ratio: Sparse direction ratio.
        cycle_lengths: Graph cycle lengths.
        cycle_knn_k: kNN graph connectivity.
        cycle_samples: Number of sampled cycles per length.
        feature_encoder: Frozen feature encoder for graph cycles.

    Returns:
        Long-format row list with keys:
        ``bin,sigma_lo,sigma_hi,count,metric_name,scale_delta,cycle_len,value``.
    """
    edges = sigma_bin_edges(sigma_min=sigma_min, sigma_max=sigma_max, bins=bins)
    bucket = _bucket_index(sigma, edges)

    deltas = _parse_float_list(loop_delta_set, default=[0.01])
    lengths = _parse_int_list(cycle_lengths, default=[3, 4, 5])

    rows: list[dict[str, float | int | str]] = []

    for b in range(int(bins)):
        mask = bucket == b
        lo = float(edges[b])
        hi = float(edges[b + 1])
        count = int(mask.sum().item())

        if count == 0:
            base = _empty_row(bin_id=b, sigma_lo=lo, sigma_hi=hi)
            for metric_name in ["r_sym", "r_loop_multi", "r_cycle"]:
                row = dict(base)
                row["metric_name"] = metric_name
                rows.append(row)
            continue

        xb = x[mask]
        sb = sigma[mask]

        r_sym = reg_sym_estimator(
            score_fn=score_fn,
            x=xb,
            sigma=sb,
            K=int(reg_k),
            method=str(reg_sym_method),
            eps_fd=float(reg_sym_eps_fd),
        )
        rows.append(
            {
                "bin": b,
                "sigma_lo": lo,
                "sigma_hi": hi,
                "count": count,
                "metric_name": "r_sym",
                "scale_delta": "",
                "cycle_len": "",
                "value": float(r_sym.item()),
            }
        )

        loop_total, per_scale = loop_multi_scale_estimator(
            score_fn=score_fn,
            x=xb,
            sigma=sb,
            delta_set=deltas,
            sparse_ratio=float(loop_sparse_ratio),
        )
        rows.append(
            {
                "bin": b,
                "sigma_lo": lo,
                "sigma_hi": hi,
                "count": count,
                "metric_name": "r_loop_multi_total",
                "scale_delta": "",
                "cycle_len": "",
                "value": float(loop_total.item()),
            }
        )

        for delta in deltas:
            value = per_scale.get(float(delta))
            rows.append(
                {
                    "bin": b,
                    "sigma_lo": lo,
                    "sigma_hi": hi,
                    "count": count,
                    "metric_name": "r_loop_multi",
                    "scale_delta": float(delta),
                    "cycle_len": "",
                    "value": float(value.item()) if value is not None else float("nan"),
                }
            )

        if feature_encoder is not None and count >= 3:
            with torch.no_grad():
                feat = feature_encoder(xb.detach())
            cycle_total, per_len = graph_cycle_estimator(
                score_fn=score_fn,
                x=xb,
                sigma=sb,
                features=feat,
                k=int(cycle_knn_k),
                cycle_lengths=lengths,
                num_cycles=int(cycle_samples),
            )
            rows.append(
                {
                    "bin": b,
                    "sigma_lo": lo,
                    "sigma_hi": hi,
                    "count": count,
                    "metric_name": "r_cycle_total",
                    "scale_delta": "",
                    "cycle_len": "",
                    "value": float(cycle_total.item()),
                }
            )
            for length in lengths:
                val = per_len.get(int(length))
                rows.append(
                    {
                        "bin": b,
                        "sigma_lo": lo,
                        "sigma_hi": hi,
                        "count": count,
                        "metric_name": "r_cycle",
                        "scale_delta": "",
                        "cycle_len": int(length),
                        "value": float(val.item()) if val is not None else float("nan"),
                    }
                )
        else:
            rows.append(
                {
                    "bin": b,
                    "sigma_lo": lo,
                    "sigma_hi": hi,
                    "count": count,
                    "metric_name": "r_cycle_total",
                    "scale_delta": "",
                    "cycle_len": "",
                    "value": float("nan"),
                }
            )
            for length in lengths:
                rows.append(
                    {
                        "bin": b,
                        "sigma_lo": lo,
                        "sigma_hi": hi,
                        "count": count,
                        "metric_name": "r_cycle",
                        "scale_delta": "",
                        "cycle_len": int(length),
                        "value": float("nan"),
                    }
                )

    return rows
