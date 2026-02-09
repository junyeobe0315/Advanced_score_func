from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import yaml

from src.data import make_loader, sample_real_data, unpack_batch
from src.eval import compute_quality_metrics, generate_samples_batched, integrability_records_by_sigma
from src.metrics import exact_jacobian_asymmetry_2d, path_variance
from src.models import build_model, score_fn_from_model
from src.sampling.sigma_schedule import sample_log_uniform_sigmas
from src.trainers.engine import resolve_device
from src.utils.checkpoint import latest_checkpoint, load_checkpoint
from src.utils.config import ensure_experiment_defaults, resolve_model_id
from src.utils.feature_encoder import build_feature_encoder

DEFAULT_NFE_GRID = [8, 18, 32, 64, 128]
DEFAULT_MAIN_SAMPLER = "heun"
DEFAULT_COMPARE_SAMPLERS = ["euler"]


def _read_cfg(run_dir: Path) -> dict:
    """Load resolved run config from run directory.

    Args:
        run_dir: Run directory path.

    Returns:
        Parsed config dictionary from ``config_resolved.yaml``.
    """
    path = run_dir / "config_resolved.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return ensure_experiment_defaults(cfg)


def _parse_nfe_list(raw: str | None, fallback: list[int]) -> list[int]:
    """Parse comma-separated NFE values from CLI string.

    Args:
        raw: Raw comma-separated string. ``None`` means use fallback.
        fallback: Default list when ``raw`` is empty.

    Returns:
        Parsed integer NFE list.
    """
    if raw is None or str(raw).strip() == "":
        return fallback
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write list-of-dicts rows to CSV file.

    Args:
        path: Destination CSV file path.
        rows: Row dictionaries with identical keys.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_metrics_csv(path: Path) -> list[dict[str, str]]:
    """Read training metrics CSV rows if file exists."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _compute_summary_from_train_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    """Aggregate compute-related statistics from training metrics rows.

    Args:
        rows: Parsed rows from ``metrics.csv``.

    Returns:
        Summary dictionary with train throughput, step time, VRAM, and
        approximate GPU-hours.
    """
    if not rows:
        return {
            "train_step_time_ms_mean": float("nan"),
            "train_imgs_per_sec_mean": float("nan"),
            "peak_vram_mb": float("nan"),
            "approx_gpu_hours": float("nan"),
        }

    step_time = [float(r.get("step_time_ms", "nan")) for r in rows]
    ips = [float(r.get("imgs_per_sec", "nan")) for r in rows]
    vram = [float(r.get("vram_peak_mb", "nan")) for r in rows]

    finite_step_time = [v for v in step_time if torch.isfinite(torch.tensor(v)).item()]
    finite_ips = [v for v in ips if torch.isfinite(torch.tensor(v)).item()]
    finite_vram = [v for v in vram if torch.isfinite(torch.tensor(v)).item()]

    step_mean = float(sum(finite_step_time) / max(1, len(finite_step_time)))
    ips_mean = float(sum(finite_ips) / max(1, len(finite_ips)))
    vram_peak = float(max(finite_vram)) if finite_vram else float("nan")
    gpu_hours = float((step_mean * len(rows)) / 1000.0 / 3600.0)

    return {
        "train_step_time_ms_mean": step_mean,
        "train_imgs_per_sec_mean": ips_mean,
        "peak_vram_mb": vram_peak,
        "approx_gpu_hours": gpu_hours,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation entrypoint."""
    p = argparse.ArgumentParser(description="Evaluate trained runs")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--nfe_list", type=str, default=None)
    return p.parse_args()


def main() -> None:
    """Run full evaluation pipeline for one training run.

    Returns:
        None. Writes CSV/JSON files under ``run_dir/eval`` and prints that path.

    How it works:
        Loads checkpoint/model, generates samples for each sampler/NFE pair,
        computes quality metrics, then computes sigma-binned integrability
        diagnostics with long-format schema.
    """
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    cfg = _read_cfg(run_dir)

    device = resolve_device(cfg)
    model = build_model(cfg).to(device)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"no checkpoint found under {run_dir}")

    ckpt = load_checkpoint(ckpt_path, map_location=str(device))
    model.load_state_dict(ckpt["model"], strict=True)
    if ckpt.get("ema") and isinstance(ckpt["ema"], dict) and "shadow" in ckpt["ema"]:
        model.load_state_dict(ckpt["ema"]["shadow"], strict=False)

    model.eval()

    model_id = resolve_model_id(cfg)
    score_fn_sample = score_fn_from_model(model, model_id, create_graph=False)
    score_fn_metric = score_fn_from_model(model, model_id, create_graph=True)
    score_requires_grad = model_id in {"M2", "M4"}

    eval_cfg = cfg["eval"]
    loss_cfg = cfg["loss"]
    sampler_cfg = cfg.get("sampler", {})

    default_nfe = [int(v) for v in sampler_cfg.get("nfe_list", DEFAULT_NFE_GRID)]
    nfe_list = _parse_nfe_list(args.nfe_list, fallback=default_nfe)

    num_samples = int(eval_cfg.get("num_fid_samples", 10000))
    eval_batch = int(eval_cfg.get("batch_size", 64))
    sigma_min = float(loss_cfg["sigma_min"])
    sigma_max = float(loss_cfg["sigma_max"])

    # Reference real samples for FID/KID and output-shape definition.
    real = sample_real_data(cfg, num_samples=num_samples)
    shape_per_sample = tuple(real.shape[1:])

    main_sampler = str(sampler_cfg.get("main", DEFAULT_MAIN_SAMPLER))
    compare_samplers = [str(s) for s in sampler_cfg.get("compare", DEFAULT_COMPARE_SAMPLERS)]
    sampler_list = []
    for name in [main_sampler, *compare_samplers]:
        if name not in sampler_list:
            sampler_list.append(name)

    fid_rows = []
    want_is = str(cfg["dataset"]["name"]).lower().startswith("cifar")
    for sampler_name in sampler_list:
        for nfe in nfe_list:
            t0 = time.perf_counter()
            fake, dynamics = generate_samples_batched(
                sampler_name=sampler_name,
                score_fn=score_fn_sample,
                shape_per_sample=shape_per_sample,
                total=num_samples,
                batch_size=eval_batch,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                nfe=nfe,
                device=device,
                score_requires_grad=score_requires_grad,
            )
            elapsed = time.perf_counter() - t0

            quality = compute_quality_metrics(
                fake=fake,
                real=real,
                device=device,
                want_is=want_is,
                prefer_torch_fidelity=bool(eval_cfg.get("use_torch_fidelity", True)),
            )

            fid_rows.append(
                {
                    "sampler": sampler_name,
                    "nfe": int(nfe),
                    "fid": quality.fid,
                    "kid": quality.kid,
                    "is_mean": quality.inception_score_mean,
                    "is_std": quality.inception_score_std,
                    "metric_backend": quality.backend,
                    "latency_sec": float(elapsed),
                    "latency_per_step_ms": float(elapsed * 1000.0 / max(1, int(nfe))),
                    "trajectory_length_mean": dynamics["trajectory_length_mean"],
                    "curvature_proxy": dynamics["curvature_proxy"],
                }
            )

    _write_csv(run_dir / "eval" / "fid_vs_nfe.csv", fid_rows)

    # Integrability metrics by sigma bins.
    loader = make_loader(cfg, train=False)
    batch = next(iter(loader))
    x = unpack_batch(batch).to(device)
    max_b = int(eval_cfg.get("integrability_batch", x.shape[0]))
    x = x[:max_b]

    sigma = sample_log_uniform_sigmas(
        batch_size=x.shape[0],
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=device,
        dtype=x.dtype,
    )

    feature_encoder = None
    if bool(eval_cfg.get("enable_cycle_metrics", True)):
        feature_encoder = build_feature_encoder(
            dataset_name=str(cfg["dataset"]["name"]),
            channels=int(cfg["dataset"].get("channels", 1)),
            device=device,
        )

    integrability_rows = integrability_records_by_sigma(
        score_fn=score_fn_metric,
        x=x,
        sigma=sigma,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        bins=int(eval_cfg.get("sigma_bins", 8)),
        reg_k=int(loss_cfg.get("reg_k", 1)),
        reg_sym_method=str(loss_cfg.get("reg_sym_method", "jvp_vjp")),
        reg_sym_variant=str(loss_cfg.get("reg_sym_variant", "skew_fro")),
        reg_sym_probe_dist=str(loss_cfg.get("reg_sym_probe_dist", "gaussian")),
        reg_sym_eps_fd=float(loss_cfg.get("reg_sym_eps_fd", 1.0e-3)),
        loop_delta_set=loss_cfg.get("delta_set", [float(loss_cfg.get("loop_delta", 0.01))]),
        loop_sparse_ratio=float(loss_cfg.get("loop_sparse_ratio", 1.0)),
        cycle_lengths=loss_cfg.get("cycle_lengths", [3, 4, 5]),
        cycle_knn_k=int(loss_cfg.get("cycle_knn_k", 8)),
        cycle_samples=int(loss_cfg.get("cycle_samples", 16)),
        feature_encoder=feature_encoder,
    )

    if str(cfg["dataset"]["name"]).lower() == "toy" and x.ndim == 2 and x.shape[-1] == 2:
        exact = exact_jacobian_asymmetry_2d(score_fn_metric, x[: min(256, x.shape[0])], sigma[: min(256, x.shape[0])])
        integrability_rows.append(
            {
                "bin": -1,
                "sigma_lo": float("nan"),
                "sigma_hi": float("nan"),
                "count": int(min(256, x.shape[0])),
                "metric_name": "exact_jacobian_asymmetry_2d",
                "scale_delta": "",
                "cycle_len": "",
                "value": float(exact),
            }
        )

        path_cfg = eval_cfg.get("pathvar", {})
        if bool(path_cfg.get("enabled", False)) and x.shape[0] >= 2:
            n = min(64, x.shape[0] - 1)
            pv = path_variance(
                score_fn=score_fn_metric,
                x_ref=x[:1].expand(n, -1),
                x_tgt=x[1 : 1 + n],
                sigma=sigma[1 : 1 + n],
                num_paths=int(path_cfg.get("num_paths", 8)),
                num_segments=int(path_cfg.get("num_segments", 12)),
            )
            integrability_rows.append(
                {
                    "bin": -1,
                    "sigma_lo": float("nan"),
                    "sigma_hi": float("nan"),
                    "count": int(n),
                    "metric_name": "pathvar",
                    "scale_delta": "",
                    "cycle_len": "",
                    "value": float(pv),
                }
            )

    _write_csv(run_dir / "eval" / "integrability_vs_sigma.csv", integrability_rows)

    train_metrics_rows = _read_metrics_csv(run_dir / cfg["logging"].get("csv_filename", "metrics.csv"))
    compute_summary = _compute_summary_from_train_metrics(train_metrics_rows)
    compute_summary.update(
        {
            "dataset": str(cfg["dataset"]["name"]),
            "model_id": model_id,
            "compute_tier": str(cfg.get("compute", {}).get("tier", "main")),
            "compute_limited": bool(cfg.get("compute", {}).get("compute_limited", False)),
            "eval_num_samples": int(num_samples),
            "nfe_grid": nfe_list,
            "sampler_grid": sampler_list,
        }
    )

    out_json = run_dir / "eval" / "compute_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(compute_summary, f, indent=2)

    print(str(run_dir / "eval"))


if __name__ == "__main__":
    main()
