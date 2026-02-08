from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import yaml

from src.data import make_loader, sample_real_data, unpack_batch
from src.metrics import (
    compute_fid_kid,
    compute_inception_score,
    curvature_proxy,
    exact_jacobian_asymmetry_2d,
    integrability_by_sigma_bins,
    path_variance,
)
from src.models import build_model, score_fn_from_model
from src.sampling import sample_euler, sample_heun
from src.sampling.sigma_schedule import sample_log_uniform_sigmas
from src.trainers.engine import resolve_device
from src.utils.checkpoint import latest_checkpoint, load_checkpoint


def _read_cfg(run_dir: Path) -> dict:
    path = run_dir / "config_resolved.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_nfe_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _sampler_fn(name: str):
    if name == "heun":
        return sample_heun
    if name == "euler":
        return sample_euler
    raise ValueError(f"unknown sampler: {name}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _generate_samples(
    sampler_name: str,
    score_fn,
    shape_per_sample: tuple[int, ...],
    total: int,
    batch_size: int,
    sigma_min: float,
    sigma_max: float,
    nfe: int,
    device: torch.device,
    score_requires_grad: bool,
) -> tuple[torch.Tensor, dict]:
    sampler = _sampler_fn(sampler_name)
    out = []
    traj_lens = []
    curvatures = []

    produced = 0
    while produced < total:
        b = min(batch_size, total - produced)
        shape = (b, *shape_per_sample)
        samples, stats = sampler(
            score_fn=score_fn,
            shape=shape,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            nfe=nfe,
            device=device,
            score_requires_grad=score_requires_grad,
            return_trajectory=(produced == 0),
        )
        out.append(samples.detach().cpu())
        traj_lens.append(stats["trajectory_length_mean"])
        if "trajectory" in stats:
            curvatures.append(curvature_proxy(stats["trajectory"]))
        produced += b

    merged = torch.cat(out, dim=0)
    agg = {
        "trajectory_length_mean": float(sum(traj_lens) / max(len(traj_lens), 1)),
        "curvature_proxy": float(sum(curvatures) / max(len(curvatures), 1)) if curvatures else 0.0,
    }
    return merged, agg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained runs")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--nfe_list", type=str, default="10,20,50,100,200")
    return p.parse_args()


def main() -> None:
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

    variant = cfg["model"]["variant"]
    score_fn_sample = score_fn_from_model(model, variant, create_graph=False)
    score_fn_metric = score_fn_from_model(model, variant, create_graph=True)
    score_requires_grad = variant == "struct"

    nfe_list = _parse_nfe_list(args.nfe_list)
    eval_cfg = cfg["eval"]
    loss_cfg = cfg["loss"]

    num_samples = int(eval_cfg.get("num_fid_samples", 10000))
    eval_batch = int(eval_cfg.get("batch_size", 64))
    sigma_min = float(loss_cfg["sigma_min"])
    sigma_max = float(loss_cfg["sigma_max"])

    real = sample_real_data(cfg, num_samples=num_samples)
    shape_per_sample = tuple(real.shape[1:])

    fid_rows = []
    main_sampler = cfg["sampler"].get("main", "heun")
    for nfe in nfe_list:
        fake, dynamics = _generate_samples(
            sampler_name=main_sampler,
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

        fid_kid = compute_fid_kid(fake=fake.to(device), real=real.to(device), device=device)
        row = {
            "sampler": main_sampler,
            "nfe": int(nfe),
            "fid": fid_kid.fid,
            "kid": fid_kid.kid,
            "trajectory_length_mean": dynamics["trajectory_length_mean"],
            "curvature_proxy": dynamics["curvature_proxy"],
        }

        if cfg["dataset"]["name"] == "cifar10":
            is_mean, is_std = compute_inception_score(fake, device=device)
            row["inception_score_mean"] = is_mean
            row["inception_score_std"] = is_std

        fid_rows.append(row)

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

    integ_rows = integrability_by_sigma_bins(
        score_fn=score_fn_metric,
        x=x,
        sigma=sigma,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        bins=int(eval_cfg.get("sigma_bins", 8)),
        reg_k=int(loss_cfg.get("reg_k", 1)),
        loop_delta=float(loss_cfg.get("loop_delta", 0.01)),
        loop_sparse_ratio=float(loss_cfg.get("loop_sparse_ratio", 1.0)),
    )

    if cfg["dataset"]["name"] == "toy" and x.shape[-1] == 2:
        exact = exact_jacobian_asymmetry_2d(score_fn_metric, x[: min(256, x.shape[0])], sigma[: min(256, x.shape[0])])
        for row in integ_rows:
            row["exact_jacobian_asymmetry_2d"] = exact

        path_cfg = eval_cfg.get("pathvar", {})
        if bool(path_cfg.get("enabled", False)) and x.shape[0] >= 2:
            pv = path_variance(
                score_fn=score_fn_metric,
                x_ref=x[:1].expand(min(64, x.shape[0] - 1), -1),
                x_tgt=x[1 : 1 + min(64, x.shape[0] - 1)],
                sigma=sigma[1 : 1 + min(64, x.shape[0] - 1)],
                num_paths=int(path_cfg.get("num_paths", 8)),
                num_segments=int(path_cfg.get("num_segments", 12)),
            )
            for row in integ_rows:
                row["pathvar"] = pv

    _write_csv(run_dir / "eval" / "integrability_vs_sigma.csv", integ_rows)
    print(str(run_dir / "eval"))


if __name__ == "__main__":
    main()
