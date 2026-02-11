from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import sample_real_data
from src.data.toy import toy_mixture_parameters
from src.eval import generate_samples_batched
from src.metrics.toy_stats import mmd_rbf as _mmd_rbf
from src.metrics.toy_stats import safe_float as _safe_float
from src.metrics.toy_stats import toy_gmm_logpdf_mean as _toy_logpdf_mean
from src.metrics.toy_stats import toy_mode_stats as _mode_stats
from src.models import build_model, score_fn_from_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import ensure_experiment_defaults, resolve_model_id


@dataclass
class EvalRow:
    model: str
    seed: str
    checkpoint_step: int
    final_step: int
    avg_logpdf_true_gmm: float
    mmd_to_observed: float
    mode_covered: int
    mode_entropy_norm: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build toy best-step CSV and plots.")
    p.add_argument("--run_root", type=str, default="runs/toy")
    p.add_argument("--out_dir", type=str, default="reports/figures")
    p.add_argument("--sampler", type=str, default="heun")
    p.add_argument("--nfe", type=int, default=100)
    p.add_argument("--num_samples", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def _read_final_step(metrics_path: Path) -> int:
    if not metrics_path.exists():
        return -1
    with metrics_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return -1
    return int(rows[-1].get("step", -1))


def _extract_step_from_name(path: Path) -> int:
    # step_00026000.pt -> 26000
    stem = path.stem
    token = stem.rsplit("_", 1)[-1]
    return int(token)


def _eval_checkpoint(
    cfg: dict,
    ckpt_path: Path,
    device: torch.device,
    num_samples: int,
    batch_size: int,
    sampler_name: str,
    nfe: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int, float]:
    model_id = resolve_model_id(cfg)
    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(ckpt_path, map_location=str(device))
    model.load_state_dict(ckpt["model"], strict=True)
    if ckpt.get("ema") and isinstance(ckpt["ema"], dict) and "shadow" in ckpt["ema"]:
        model.load_state_dict(ckpt["ema"]["shadow"], strict=False)
    model.eval()

    real = sample_real_data(cfg, num_samples=int(num_samples))
    shape_per_sample = tuple(real.shape[1:])
    score_fn = score_fn_from_model(model, model_id, create_graph=False)
    fake, _ = generate_samples_batched(
        sampler_name=str(sampler_name),
        score_fn=score_fn,
        shape_per_sample=shape_per_sample,
        total=int(num_samples),
        batch_size=int(batch_size),
        sigma_min=float(cfg["loss"]["sigma_min"]),
        sigma_max=float(cfg["loss"]["sigma_max"]),
        nfe=int(nfe),
        device=device,
        score_requires_grad=(model_id in {"M2", "M4"}),
    )

    fake_np = fake.detach().cpu().numpy().astype(np.float64)
    real_np = real.detach().cpu().numpy().astype(np.float64)
    centers_t, stds_t = toy_mixture_parameters(cfg, dtype=torch.float64, device=torch.device("cpu"))
    centers = centers_t.numpy()
    stds = stds_t.numpy()

    logpdf = _toy_logpdf_mean(fake_np, centers=centers, stds=stds)
    mmd = _mmd_rbf(fake_np, real_np)
    covered, entropy_norm = _mode_stats(fake_np, centers)
    return real_np, fake_np, centers, logpdf, mmd, covered, entropy_norm


def _best_key(row: EvalRow) -> tuple[float, float, int]:
    # Higher logpdf better; tie-break lower MMD; tie-break later step.
    return (float(row.avg_logpdf_true_gmm), float(-row.mmd_to_observed), int(row.checkpoint_step))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_true_density(ax, centers: np.ndarray, stds: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    gx = np.linspace(xlim[0], xlim[1], 220)
    gy = np.linspace(ylim[0], ylim[1], 220)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    logpdf = _toy_logpdf_mean(grid, centers=centers, stds=stds)

    # Recover full PDF for contour map.
    diff = grid[:, None, :] - centers[None, :, :]
    sq = np.sum(diff * diff, axis=2)
    var = np.maximum(stds[None, :] ** 2, 1.0e-12)
    coef = np.power(2.0 * np.pi * var, -1.0)
    pdf = np.mean(coef * np.exp(-0.5 * sq / var), axis=1).reshape(xx.shape)

    ax.contourf(xx, yy, pdf, levels=24, cmap="viridis")
    ax.scatter(centers[:, 0], centers[:, 1], c="white", s=12, edgecolors="black", linewidths=0.4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(f"True PDF (mean logp={logpdf:.3f})", fontsize=9)


def _plot_seed_grid(
    out_path: Path,
    seed: str,
    rows_by_model: dict[str, dict[str, object]],
) -> None:
    models = ["M0", "M1", "M2", "M3", "M4"]
    n_rows = len(models)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11.0, 2.25 * n_rows), dpi=150)
    fig.suptitle(f"Toy Best-Step Comparison ({seed})", fontsize=12)

    for ridx, model in enumerate(models):
        ax_true, ax_real, ax_fake = axes[ridx]
        payload = rows_by_model.get(model)
        if payload is None:
            for ax in (ax_true, ax_real, ax_fake):
                ax.axis("off")
            continue

        centers = payload["centers"]
        stds = payload["stds"]
        real_np = payload["real"]
        fake_np = payload["fake"]
        best_step = int(payload["best_step"])
        logpdf = float(payload["logpdf"])
        mmd = float(payload["mmd"])

        span = float(np.max(np.linalg.norm(centers[:, :2], axis=1)) + 4.0 * np.max(stds) + 0.6)
        xlim = (-span, span)
        ylim = (-span, span)

        _plot_true_density(ax_true, centers=centers, stds=stds, xlim=xlim, ylim=ylim)
        ax_true.set_ylabel(model, fontsize=10)

        ax_real.scatter(real_np[:, 0], real_np[:, 1], s=3, alpha=0.35, color="#2f4b7c")
        ax_real.set_xlim(*xlim)
        ax_real.set_ylim(*ylim)
        ax_real.set_title("Observed Data", fontsize=9)

        ax_fake.scatter(fake_np[:, 0], fake_np[:, 1], s=3, alpha=0.35, color="#d45087")
        ax_fake.set_xlim(*xlim)
        ax_fake.set_ylim(*ylim)
        ax_fake.set_title(f"Generated (best={best_step})", fontsize=9)
        ax_fake.text(
            0.02,
            0.02,
            f"logp={logpdf:.3f}\nmmd={mmd:.4f}",
            transform=ax_fake.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

        for ax in (ax_true, ax_real, ax_fake):
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)

    checkpoint_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    plot_payload_by_seed: dict[str, dict[str, dict[str, object]]] = {}

    run_dirs = sorted(run_root.glob("M*/seed*"))
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config_resolved.yaml"
        metrics_path = run_dir / "metrics.csv"
        ckpt_paths = sorted((run_dir / "checkpoints").glob("step_*.pt"), key=_extract_step_from_name)
        if not cfg_path.exists() or not ckpt_paths:
            continue

        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = ensure_experiment_defaults(yaml.safe_load(f))
        if str(cfg.get("dataset", {}).get("name", "")).lower() != "toy":
            continue

        model = run_dir.parent.name
        seed = run_dir.name
        final_step = _read_final_step(metrics_path)
        run_eval_rows: list[EvalRow] = []
        best_payload: dict[str, object] | None = None

        for ckpt in ckpt_paths:
            step = _extract_step_from_name(ckpt)
            try:
                real_np, fake_np, centers, logpdf, mmd, covered, entropy_norm = _eval_checkpoint(
                    cfg=cfg,
                    ckpt_path=ckpt,
                    device=device,
                    num_samples=int(args.num_samples),
                    batch_size=int(args.batch_size),
                    sampler_name=str(args.sampler),
                    nfe=int(args.nfe),
                )
            except Exception:
                checkpoint_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "status": "eval_error",
                        "checkpoint_step": step,
                        "final_step": final_step,
                        "avg_logpdf_true_gmm": float("nan"),
                        "mmd_to_observed": float("nan"),
                        "mode_covered": float("nan"),
                        "mode_entropy_norm": float("nan"),
                    }
                )
                continue

            stds = toy_mixture_parameters(cfg, dtype=torch.float64, device=torch.device("cpu"))[1].numpy()
            row = EvalRow(
                model=model,
                seed=seed,
                checkpoint_step=step,
                final_step=final_step,
                avg_logpdf_true_gmm=logpdf,
                mmd_to_observed=mmd,
                mode_covered=covered,
                mode_entropy_norm=entropy_norm,
            )
            run_eval_rows.append(row)
            checkpoint_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "status": "ok",
                    "checkpoint_step": step,
                    "final_step": final_step,
                    "avg_logpdf_true_gmm": logpdf,
                    "mmd_to_observed": mmd,
                    "mode_covered": covered,
                    "mode_entropy_norm": entropy_norm,
                }
            )

            payload = {
                "real": real_np,
                "fake": fake_np,
                "centers": centers,
                "stds": stds,
                "best_step": step,
                "logpdf": logpdf,
                "mmd": mmd,
            }
            if best_payload is None:
                best_payload = payload
            else:
                old_row = EvalRow(
                    model=model,
                    seed=seed,
                    checkpoint_step=int(best_payload["best_step"]),
                    final_step=final_step,
                    avg_logpdf_true_gmm=float(best_payload["logpdf"]),
                    mmd_to_observed=float(best_payload["mmd"]),
                    mode_covered=0,
                    mode_entropy_norm=0.0,
                )
                if _best_key(row) > _best_key(old_row):
                    best_payload = payload

        if not run_eval_rows or best_payload is None:
            best_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "status": "incomplete_or_failed",
                    "best_checkpoint_step": "",
                    "final_step": final_step,
                    "avg_logpdf_true_gmm": float("nan"),
                    "mmd_to_observed": float("nan"),
                    "mode_covered": float("nan"),
                    "mode_entropy_norm": float("nan"),
                }
            )
            continue

        best_row = max(run_eval_rows, key=_best_key)
        best_rows.append(
            {
                "model": model,
                "seed": seed,
                "status": "ok",
                "best_checkpoint_step": int(best_row.checkpoint_step),
                "final_step": int(best_row.final_step),
                "avg_logpdf_true_gmm": float(best_row.avg_logpdf_true_gmm),
                "mmd_to_observed": float(best_row.mmd_to_observed),
                "mode_covered": int(best_row.mode_covered),
                "mode_entropy_norm": float(best_row.mode_entropy_norm),
            }
        )

        plot_payload_by_seed.setdefault(seed, {})[model] = best_payload

    checkpoint_rows = sorted(
        checkpoint_rows,
        key=lambda r: (str(r["model"]), str(r["seed"]), _safe_float(r["checkpoint_step"])),
    )
    best_rows = sorted(best_rows, key=lambda r: (str(r["model"]), str(r["seed"])))
    _write_csv(out_dir / "toy_checkpoint_density_scores.csv", checkpoint_rows)
    _write_csv(out_dir / "toy_beststep_density_scores.csv", best_rows)

    for seed, rows_by_model in sorted(plot_payload_by_seed.items()):
        seed_id = "".join(ch for ch in seed if ch.isdigit()) or seed
        _plot_seed_grid(
            out_path=out_dir / f"toy_seed_grid_beststep_seed{seed_id}.png",
            seed=seed,
            rows_by_model=rows_by_model,
        )

    print(str(out_dir))


if __name__ == "__main__":
    main()
