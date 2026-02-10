from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from src.models import build_model, score_fn_from_model
from src.utils.checkpoint import latest_checkpoint, load_checkpoint
from src.utils.config import ensure_experiment_defaults, resolve_model_id


@dataclass
class SelectionBest:
    step: int
    checkpoint_name: str
    selection_logpdf: float
    selection_mmd: float
    sampler: str
    nfe: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create plots and comparison CSVs for modified toy runs.")
    p.add_argument("--run_root", type=str, default="runs/toy")
    p.add_argument("--out_dir", type=str, default="reports/figures")
    p.add_argument(
        "--seed",
        type=str,
        default="all",
        help="Seed selector: all, 0, seed0, or comma list like 0,1,2",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num_samples", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--nfe", type=int, default=100)
    p.add_argument("--sampler", type=str, default="heun")
    p.add_argument("--smooth_alpha", type=float, default=0.08)
    return p.parse_args()


def _safe_float(x: Any) -> float:
    try:
        out = float(x)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _extract_step(path: Path) -> int:
    token = path.stem.rsplit("_", 1)[-1]
    return int(token)


def _toy_logpdf_mean(samples: np.ndarray, centers: np.ndarray, stds: np.ndarray) -> float:
    diff = samples[:, None, :] - centers[None, :, :]
    sq = np.sum(diff * diff, axis=2)
    dim = int(samples.shape[1])
    var = np.maximum(stds[None, :] ** 2, 1.0e-12)
    coef = np.power(2.0 * np.pi * var, -0.5 * float(dim))
    pdf = np.mean(coef * np.exp(-0.5 * sq / var), axis=1)
    return float(np.mean(np.log(pdf + 1.0e-12)))


def _toy_pdf_grid(centers: np.ndarray, stds: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float]):
    gx = np.linspace(xlim[0], xlim[1], 220)
    gy = np.linspace(ylim[0], ylim[1], 220)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    diff = grid[:, None, :] - centers[None, :, :]
    sq = np.sum(diff * diff, axis=2)
    var = np.maximum(stds[None, :] ** 2, 1.0e-12)
    coef = np.power(2.0 * np.pi * var, -1.0)
    pdf = np.mean(coef * np.exp(-0.5 * sq / var), axis=1).reshape(xx.shape)
    return xx, yy, pdf


def _mmd_rbf(x: np.ndarray, y: np.ndarray, max_n: int = 1200) -> float:
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


def _mode_stats(samples: np.ndarray, centers: np.ndarray) -> tuple[int, float]:
    d2 = np.sum((samples[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    idx = np.argmin(d2, axis=1)
    counts = np.bincount(idx, minlength=centers.shape[0]).astype(np.float64)
    covered = int(np.sum(counts > 0))
    probs = counts / max(np.sum(counts), 1.0)
    entropy = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
    entropy_norm = float(entropy / max(np.log(float(centers.shape[0])), 1.0e-12))
    return covered, entropy_norm


def _ema(y: np.ndarray, alpha: float) -> np.ndarray:
    if y.size == 0:
        return y
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, y.shape[0]):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
    return out


def _read_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {
            "final_step": -1,
            "final_loss": float("nan"),
            "best_loss_step": -1,
            "best_loss": float("nan"),
            "step": np.array([], dtype=np.int64),
            "loss_total": np.array([], dtype=np.float64),
        }
    with metrics_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {
            "final_step": -1,
            "final_loss": float("nan"),
            "best_loss_step": -1,
            "best_loss": float("nan"),
            "step": np.array([], dtype=np.int64),
            "loss_total": np.array([], dtype=np.float64),
        }

    step = np.array([int(r["step"]) for r in rows], dtype=np.int64)
    loss_total = np.array([_safe_float(r.get("loss_total")) for r in rows], dtype=np.float64)
    best_idx = int(np.nanargmin(loss_total))
    return {
        "final_step": int(step[-1]),
        "final_loss": float(loss_total[-1]),
        "best_loss_step": int(step[best_idx]),
        "best_loss": float(loss_total[best_idx]),
        "step": step,
        "loss_total": loss_total,
    }


def _read_selection_best(run_dir: Path, default_sampler: str, default_nfe: int) -> SelectionBest | None:
    topk_path = run_dir / "eval" / "selection_topk.json"
    if topk_path.exists():
        with topk_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        topk = payload.get("topk", []) if isinstance(payload, dict) else []
        if isinstance(topk, list) and topk:
            best = topk[0]
            details = best.get("details", {}) if isinstance(best, dict) else {}
            return SelectionBest(
                step=int(best.get("step", -1)),
                checkpoint_name=str(best.get("checkpoint_name", f"eval_step_{int(best.get('step', -1)):08d}.pt")),
                selection_logpdf=_safe_float(best.get("metric_value")),
                selection_mmd=_safe_float(details.get("toy_mmd")),
                sampler=str(details.get("sampler", default_sampler)),
                nfe=int(details.get("nfe", default_nfe)),
            )

    ckpt = latest_checkpoint(run_dir)
    if ckpt is None:
        return None
    return SelectionBest(
        step=int(_extract_step(ckpt)),
        checkpoint_name=str(ckpt.name),
        selection_logpdf=float("nan"),
        selection_mmd=float("nan"),
        sampler=str(default_sampler),
        nfe=int(default_nfe),
    )


def _find_checkpoint(run_dir: Path, best: SelectionBest) -> Path | None:
    ckpt_path = run_dir / "checkpoints" / best.checkpoint_name
    if ckpt_path.exists():
        return ckpt_path
    fallback = run_dir / "checkpoints" / f"eval_step_{int(best.step):08d}.pt"
    if fallback.exists():
        return fallback
    last = latest_checkpoint(run_dir)
    return last


def _plot_seed_grid(
    out_path: Path,
    seed: str,
    models: list[str],
    payload_by_model: dict[str, dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(len(models), 3, figsize=(11.0, 2.25 * len(models)), dpi=150)
    fig.suptitle(f"Toy Modified Run Comparison ({seed})", fontsize=12)

    for ridx, model in enumerate(models):
        ax_true, ax_real, ax_fake = axes[ridx]
        payload = payload_by_model.get(model)
        if payload is None:
            for ax in (ax_true, ax_real, ax_fake):
                ax.axis("off")
            continue

        centers = payload["centers"]
        stds = payload["stds"]
        real_np = payload["real"]
        fake_np = payload["fake"]
        span = float(np.max(np.linalg.norm(centers[:, :2], axis=1)) + 4.0 * np.max(stds) + 0.6)
        xlim = (-span, span)
        ylim = (-span, span)

        xx, yy, pdf = _toy_pdf_grid(centers=centers, stds=stds, xlim=xlim, ylim=ylim)
        ax_true.contourf(xx, yy, pdf, levels=24, cmap="viridis")
        ax_true.scatter(centers[:, 0], centers[:, 1], c="white", s=12, edgecolors="black", linewidths=0.4)
        ax_true.set_xlim(*xlim)
        ax_true.set_ylim(*ylim)
        ax_true.set_title("True PDF", fontsize=9)
        ax_true.set_ylabel(model, fontsize=10)

        ax_real.scatter(real_np[:, 0], real_np[:, 1], s=3, alpha=0.35, color="#2f4b7c")
        ax_real.set_xlim(*xlim)
        ax_real.set_ylim(*ylim)
        ax_real.set_title("Observed Data", fontsize=9)

        ax_fake.scatter(fake_np[:, 0], fake_np[:, 1], s=3, alpha=0.35, color="#d45087")
        ax_fake.set_xlim(*xlim)
        ax_fake.set_ylim(*ylim)
        ax_fake.set_title(f"Generated (best={payload['best_step']})", fontsize=9)
        ax_fake.text(
            0.02,
            0.02,
            f"sel_logp={payload['selection_logpdf']:.3f}\nres_logp={payload['resampled_logpdf']:.3f}\nres_mmd={payload['resampled_mmd']:.4f}",
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


def _plot_loss_by_seed(
    out_path: Path,
    seed: str,
    models: list[str],
    curve_by_model: dict[str, dict[str, np.ndarray]],
    smooth_alpha: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.2), dpi=150)
    for model in models:
        payload = curve_by_model.get(model)
        if payload is None:
            continue
        step = payload["step"]
        loss = payload["loss_total"]
        if step.size == 0:
            continue
        sm = _ema(loss.astype(np.float64), alpha=float(smooth_alpha))
        ax.plot(step, sm, linewidth=1.4, label=model)
    ax.set_title(f"Loss Curves ({seed})", fontsize=11)
    ax.set_xlabel("step")
    ax.set_ylabel("EMA(loss_total)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _seed_num(seed_name: str) -> int:
    token = "".join(ch for ch in seed_name if ch.isdigit())
    return int(token) if token else 0


def _normalize_seed_token(token: str) -> str:
    text = str(token).strip().lower()
    if text.startswith("seed"):
        text = text[4:]
    if not text or not text.isdigit():
        raise ValueError(f"invalid seed token: {token}")
    return f"seed{int(text)}"


def _select_seeds(all_seeds: list[str], selector: str) -> tuple[list[str], list[str]]:
    text = str(selector).strip()
    if not text or text.lower() == "all":
        return list(all_seeds), []

    requested: list[str] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        requested.append(_normalize_seed_token(part))
    requested = sorted(set(requested), key=_seed_num)

    selected = [seed for seed in all_seeds if seed in set(requested)]
    missing = [seed for seed in requested if seed not in set(all_seeds)]
    return selected, missing


def main() -> None:
    args = _parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)

    models = sorted({p.parent.name for p in run_root.glob("M*/seed*") if p.is_dir()})
    all_seeds = sorted({p.name for p in run_root.glob("M*/seed*") if p.is_dir()}, key=_seed_num)
    seeds, missing_seeds = _select_seeds(all_seeds=all_seeds, selector=str(args.seed))
    if missing_seeds:
        print(f"[warn] requested seeds not found: {','.join(missing_seeds)}")
    if not seeds:
        print(f"[warn] no seeds selected under {run_root}")
        return

    rows: list[dict[str, Any]] = []
    plot_payload_by_seed: dict[str, dict[str, dict[str, Any]]] = {seed: {} for seed in seeds}
    loss_payload_by_seed: dict[str, dict[str, dict[str, np.ndarray]]] = {seed: {} for seed in seeds}

    for model in models:
        for seed in seeds:
            run_dir = run_root / model / seed
            if not run_dir.exists():
                continue

            cfg_path = run_dir / "config_resolved.yaml"
            if not cfg_path.exists():
                continue
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = ensure_experiment_defaults(yaml.safe_load(f))
            if str(cfg.get("dataset", {}).get("name", "")).lower() != "toy":
                continue

            metrics = _read_metrics(run_dir / "metrics.csv")
            best_sel = _read_selection_best(run_dir, default_sampler=str(args.sampler), default_nfe=int(args.nfe))
            if best_sel is None:
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "status": "missing_checkpoint",
                        "best_checkpoint_step": "",
                        "checkpoint_name": "",
                        "final_step": metrics["final_step"],
                        "best_loss_step": metrics["best_loss_step"],
                        "best_loss": metrics["best_loss"],
                        "final_loss": metrics["final_loss"],
                        "selection_logpdf": float("nan"),
                        "selection_mmd": float("nan"),
                        "resampled_logpdf": float("nan"),
                        "resampled_mmd": float("nan"),
                        "mode_covered": float("nan"),
                        "mode_entropy_norm": float("nan"),
                    }
                )
                continue

            ckpt_path = _find_checkpoint(run_dir=run_dir, best=best_sel)
            if ckpt_path is None:
                continue

            model_num = int("".join(ch for ch in model if ch.isdigit()) or 0)
            torch_seed = 2026 + 31 * model_num + _seed_num(seed)
            np.random.seed(torch_seed)
            torch.manual_seed(torch_seed)

            model_id = resolve_model_id(cfg)
            model_net = build_model(cfg).to(device)
            ckpt = load_checkpoint(ckpt_path, map_location=str(device))
            model_net.load_state_dict(ckpt["model"], strict=True)
            if ckpt.get("ema") and isinstance(ckpt["ema"], dict) and "shadow" in ckpt["ema"]:
                model_net.load_state_dict(ckpt["ema"]["shadow"], strict=False)
            model_net.eval()

            real = sample_real_data(cfg, num_samples=int(args.num_samples))
            score_fn = score_fn_from_model(model_net, model_id, create_graph=False)
            fake, _ = generate_samples_batched(
                sampler_name=str(best_sel.sampler),
                score_fn=score_fn,
                shape_per_sample=tuple(real.shape[1:]),
                total=int(args.num_samples),
                batch_size=int(args.batch_size),
                sigma_min=float(cfg["loss"]["sigma_min"]),
                sigma_max=float(cfg["loss"]["sigma_max"]),
                nfe=int(best_sel.nfe),
                device=device,
                score_requires_grad=(model_id in {"M2", "M4"}),
            )

            real_np = real.detach().cpu().numpy().astype(np.float64)
            fake_np = fake.detach().cpu().numpy().astype(np.float64)
            centers_t, stds_t = toy_mixture_parameters(cfg, dtype=torch.float64, device=torch.device("cpu"))
            centers = centers_t.numpy()
            stds = stds_t.numpy()

            resampled_logpdf = _toy_logpdf_mean(fake_np, centers=centers, stds=stds)
            resampled_mmd = _mmd_rbf(fake_np, real_np)
            mode_covered, mode_entropy_norm = _mode_stats(fake_np, centers)

            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "status": "ok",
                    "best_checkpoint_step": int(best_sel.step),
                    "checkpoint_name": str(ckpt_path.name),
                    "final_step": int(metrics["final_step"]),
                    "best_loss_step": int(metrics["best_loss_step"]),
                    "best_loss": float(metrics["best_loss"]),
                    "final_loss": float(metrics["final_loss"]),
                    "selection_logpdf": float(best_sel.selection_logpdf),
                    "selection_mmd": float(best_sel.selection_mmd),
                    "resampled_logpdf": float(resampled_logpdf),
                    "resampled_mmd": float(resampled_mmd),
                    "mode_covered": int(mode_covered),
                    "mode_entropy_norm": float(mode_entropy_norm),
                }
            )

            plot_payload_by_seed.setdefault(seed, {})[model] = {
                "real": real_np,
                "fake": fake_np,
                "centers": centers,
                "stds": stds,
                "best_step": int(best_sel.step),
                "selection_logpdf": float(best_sel.selection_logpdf),
                "resampled_logpdf": float(resampled_logpdf),
                "resampled_mmd": float(resampled_mmd),
            }
            loss_payload_by_seed.setdefault(seed, {})[model] = {
                "step": metrics["step"],
                "loss_total": metrics["loss_total"],
            }

            print(f"[done] {model}/{seed} best_step={best_sel.step}")

    rows = sorted(rows, key=lambda r: (str(r["seed"]), str(r["model"])))

    suffix = ""
    seed_selector = str(args.seed).strip().lower()
    if seed_selector and seed_selector != "all":
        suffix = f"_{seeds[0]}" if len(seeds) == 1 else "_selected"

    run_csv = out_dir / f"toy_modified_run_metrics{suffix}.csv"
    _write_csv(run_csv, rows)

    valid_rows = [r for r in rows if r.get("status") == "ok"]
    summary_rows: list[dict[str, Any]] = []
    for model in models:
        grp = [r for r in valid_rows if r["model"] == model]
        if not grp:
            continue
        summary_rows.append(
            {
                "model": model,
                "num_runs": len(grp),
                "selection_logpdf_mean": float(np.mean([_safe_float(r["selection_logpdf"]) for r in grp])),
                "selection_logpdf_std": float(np.std([_safe_float(r["selection_logpdf"]) for r in grp], ddof=0)),
                "resampled_logpdf_mean": float(np.mean([_safe_float(r["resampled_logpdf"]) for r in grp])),
                "resampled_logpdf_std": float(np.std([_safe_float(r["resampled_logpdf"]) for r in grp], ddof=0)),
                "selection_mmd_mean": float(np.mean([_safe_float(r["selection_mmd"]) for r in grp])),
                "resampled_mmd_mean": float(np.mean([_safe_float(r["resampled_mmd"]) for r in grp])),
                "best_loss_mean": float(np.mean([_safe_float(r["best_loss"]) for r in grp])),
                "final_loss_mean": float(np.mean([_safe_float(r["final_loss"]) for r in grp])),
            }
        )
    summary_rows = sorted(summary_rows, key=lambda r: _safe_float(r["selection_logpdf_mean"]), reverse=True)
    summary_csv = out_dir / f"toy_modified_model_summary{suffix}.csv"
    _write_csv(summary_csv, summary_rows)

    seed_ranking_rows: list[dict[str, Any]] = []
    for seed in seeds:
        grp = [r for r in valid_rows if r["seed"] == seed]
        if not grp:
            continue
        by_selection = sorted(grp, key=lambda r: _safe_float(r["selection_logpdf"]), reverse=True)
        by_resampled = sorted(grp, key=lambda r: _safe_float(r["resampled_logpdf"]), reverse=True)
        sel_rank = {r["model"]: i + 1 for i, r in enumerate(by_selection)}
        res_rank = {r["model"]: i + 1 for i, r in enumerate(by_resampled)}
        for r in grp:
            seed_ranking_rows.append(
                {
                    "seed": seed,
                    "model": r["model"],
                    "rank_selection_logpdf": sel_rank[r["model"]],
                    "rank_resampled_logpdf": res_rank[r["model"]],
                    "selection_logpdf": r["selection_logpdf"],
                    "resampled_logpdf": r["resampled_logpdf"],
                    "selection_mmd": r["selection_mmd"],
                    "resampled_mmd": r["resampled_mmd"],
                }
            )
    seed_ranking_rows = sorted(seed_ranking_rows, key=lambda r: (r["seed"], r["rank_selection_logpdf"]))
    seed_rank_csv = out_dir / f"toy_modified_seed_rankings{suffix}.csv"
    _write_csv(seed_rank_csv, seed_ranking_rows)

    for seed in seeds:
        _plot_seed_grid(
            out_path=out_dir / f"toy_modified_seed_grid_beststep_{seed}.png",
            seed=seed,
            models=models,
            payload_by_model=plot_payload_by_seed.get(seed, {}),
        )
        _plot_loss_by_seed(
            out_path=out_dir / f"toy_modified_loss_by_seed_{seed}.png",
            seed=seed,
            models=models,
            curve_by_model=loss_payload_by_seed.get(seed, {}),
            smooth_alpha=float(args.smooth_alpha),
        )

    print(run_csv)
    print(summary_csv)
    print(seed_rank_csv)
    print(out_dir)


if __name__ == "__main__":
    main()
