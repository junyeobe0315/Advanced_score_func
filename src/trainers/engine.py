from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import amp
from tqdm import tqdm

from src.data import make_loader, unpack_batch
from src.metrics import grad_norm_stats, model_has_nan_or_inf
from src.models import build_model
from src.utils.checkpoint import keep_last_checkpoints, save_checkpoint
from src.utils.config import config_hash, ensure_experiment_defaults, resolve_model_id, save_config
from src.utils.csv_logger import CSVLogger
from src.utils.ema import EMA
from src.utils.feature_encoder import build_feature_encoder
from src.utils.tb_logger import TBLogger

from .train_step_baseline import train_step_baseline
from .train_step_m3 import train_step_m3
from .train_step_m4 import train_step_m4
from .train_step_reg import train_step_reg
from .train_step_struct import train_step_struct


def resolve_device(cfg: dict) -> torch.device:
    """Resolve runtime device from config and hardware availability.

    Args:
        cfg: Resolved config dictionary.

    Returns:
        ``torch.device`` selected from ``cpu``, ``cuda``, or auto detection.
    """
    mode = str(cfg["compute"].get("device", "auto"))
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_run_dir(cfg: dict, seed: int) -> Path:
    """Construct run directory path for dataset/model-id/seed tuple.

    Args:
        cfg: Resolved config dictionary.
        seed: Integer random seed.

    Returns:
        Run directory path.
    """
    dataset = str(cfg["dataset"]["name"])
    model_id = resolve_model_id(cfg)
    root = Path(cfg.get("project", {}).get("run_root", "runs"))
    return root / dataset / model_id / f"seed{seed}"


def _step_dispatch(model_id: str):
    """Return model-id specific train-step function."""
    if model_id == "M0":
        return train_step_baseline
    if model_id == "M1":
        return train_step_reg
    if model_id == "M2":
        return train_step_struct
    if model_id == "M3":
        return train_step_m3
    if model_id == "M4":
        return train_step_m4
    raise ValueError(f"unknown model_id: {model_id}")


def _run_step(
    step_fn,
    model_id: str,
    model: torch.nn.Module,
    x0: torch.Tensor,
    cfg: dict,
    step: int,
    feature_encoder: torch.nn.Module | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Execute one optimizer-step objective call for the active model id.

    Args:
        step_fn: Dispatched step function.
        model_id: Canonical model id in ``M0..M4``.
        model: Model instance.
        x0: Clean batch tensor.
        cfg: Resolved config dictionary.
        step: Global training step.
        feature_encoder: Optional frozen encoder for cycle losses.

    Returns:
        Tuple ``(loss, metrics)`` from model-specific train-step code.
    """
    if model_id in {"M0", "M2"}:
        return step_fn(model, x0, cfg)
    if model_id == "M1":
        return step_fn(model, x0, cfg, step)
    if model_id in {"M3", "M4"}:
        if feature_encoder is None:
            raise RuntimeError(f"feature encoder is required for {model_id}")
        return step_fn(model, x0, cfg, step, feature_encoder)
    raise ValueError(f"unsupported model_id: {model_id}")


def train(cfg: dict, seed: int | None = None) -> Path:
    """Execute full training loop and checkpoint logging.

    Args:
        cfg: Resolved training config.
        seed: Optional explicit seed overriding config field.

    Returns:
        Path to completed run directory.

    How it works:
        Builds model/optimizer/loggers, iterates training steps, applies mixed
        precision and grad accumulation, logs metrics, and saves checkpoints.
    """
    cfg = ensure_experiment_defaults(cfg)

    if seed is not None:
        cfg["train"]["seed"] = int(seed)
    seed = int(cfg["train"]["seed"])

    model_id = resolve_model_id(cfg)
    cfg["experiment"]["config_hash"] = config_hash(cfg)

    device = resolve_device(cfg)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cfg["compute"].get("cudnn_benchmark", True))

    run_dir = make_run_dir(cfg, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config_resolved.yaml")

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        betas=tuple(cfg["train"]["betas"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.01)),
    )

    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, decay=float(cfg["train"].get("ema_decay", 0.999)))
    ema.to(device)

    csv_logger = CSVLogger(run_dir / cfg["logging"].get("csv_filename", "metrics.csv"))
    tb_logger = TBLogger(run_dir / "tb", enabled=bool(cfg["logging"].get("use_tensorboard", True)))

    loader = make_loader(cfg, train=True)
    data_iter = iter(loader)

    feature_encoder: torch.nn.Module | None = None
    if model_id in {"M3", "M4"}:
        feature_encoder = build_feature_encoder(
            dataset_name=str(cfg["dataset"]["name"]),
            channels=int(cfg["dataset"].get("channels", 1)),
            device=device,
        )

    total_steps = int(cfg["train"]["total_steps"])
    accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    log_every = int(cfg["train"].get("log_every", 50))
    ckpt_every = int(cfg["train"].get("ckpt_every", 1000))
    keep_last_k = int(cfg["train"].get("keep_last_k", 3))
    clip_grad = float(cfg["train"].get("clip_grad_norm", 1.0))

    step_fn = _step_dispatch(model_id)
    progress = tqdm(range(1, total_steps + 1), desc=f"train/{cfg['dataset']['name']}/{model_id}")

    optimizer.zero_grad(set_to_none=True)
    # Counts steps where parameter/gradient NaN or Inf appears.
    nan_count = 0

    for step in progress:
        # Wall-clock step timer for throughput metrics.
        t0 = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x0 = unpack_batch(batch).to(device)

        with amp.autocast("cuda", enabled=use_amp):
            loss, loss_metrics = _run_step(
                step_fn=step_fn,
                model_id=model_id,
                model=model,
                x0=x0,
                cfg=cfg,
                step=step,
                feature_encoder=feature_encoder,
            )
            loss = loss / accum_steps

        # Standard AMP-safe backward path.
        scaler.scale(loss).backward()

        if step % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)

        if model_has_nan_or_inf(model):
            nan_count += 1

        gstats = grad_norm_stats(model.parameters())
        step_time_ms = (time.perf_counter() - t0) * 1000.0
        imgs_per_sec = float(x0.shape[0] / max(step_time_ms, 1e-6) * 1000.0)

        # Unified scalar row consumed by CSV and TensorBoard loggers.
        row = {
            "step": step,
            "loss_total": float(loss_metrics.get("loss_total", 0.0)),
            "loss_dsm": float(loss_metrics.get("loss_dsm", 0.0)),
            "loss_sym": float(loss_metrics.get("loss_sym", 0.0)),
            "loss_loop": float(loss_metrics.get("loss_loop", 0.0)),
            "loss_loop_multi": float(loss_metrics.get("loss_loop_multi", 0.0)),
            "loss_cycle": float(loss_metrics.get("loss_cycle", 0.0)),
            "loss_match": float(loss_metrics.get("loss_match", 0.0)),
            "grad_norm_mean": gstats["grad_norm_mean"],
            "grad_norm_max": gstats["grad_norm_max"],
            "nan_count": nan_count,
            "step_time_ms": step_time_ms,
            "imgs_per_sec": imgs_per_sec,
        }
        if device.type == "cuda":
            row["vram_peak_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024**2))
        else:
            row["vram_peak_mb"] = 0.0

        if step % log_every == 0 or step == 1:
            csv_logger.log(row)
            tb_logger.log_scalars(row, step)
            progress.set_postfix({"loss": f"{row['loss_total']:.4f}", "ips": f"{row['imgs_per_sec']:.1f}"})

        if step % ckpt_every == 0 or step == total_steps:
            save_checkpoint(
                run_dir=run_dir,
                step=step,
                model=model,
                optimizer=optimizer,
                ema_state=ema.state_dict(),
                scaler_state=scaler.state_dict() if use_amp else None,
                cfg=cfg,
            )
            keep_last_checkpoints(run_dir, keep_last_k=keep_last_k)

    tb_logger.close()
    return run_dir
