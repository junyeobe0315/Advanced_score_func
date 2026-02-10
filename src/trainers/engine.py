from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import amp
from tqdm import tqdm

from src.data import make_loader, unpack_batch
from src.metrics import grad_norm_stats, model_has_nan_or_inf
from src.models import build_model
from src.utils.checkpoint import eval_checkpoint_path, keep_last_checkpoints, save_checkpoint, save_checkpoint_to_path
from src.utils.config import config_hash, ensure_experiment_defaults, resolve_model_id, save_config
from src.utils.csv_logger import CSVLogger
from src.utils.ema import EMA
from src.utils.feature_encoder import build_feature_encoder
from src.utils.tb_logger import TBLogger

from .model_selection import (
    append_selection_history_csv,
    evaluate_selection_score,
    prune_eval_candidate_checkpoints,
    save_selection_state,
    score_to_record,
    update_topk_records,
)
from .train_step_baseline import train_step_baseline
from .train_step_m3 import train_step_m3
from .train_step_m4 import train_step_m4
from .train_step_reg import train_step_reg
from .train_step_struct import train_step_struct


def _resolve_amp_dtype(cfg: dict, device: torch.device) -> torch.dtype:
    """Resolve autocast dtype for AMP training."""
    token = str(cfg["train"].get("amp_dtype", cfg.get("compute", {}).get("amp_dtype", "auto"))).lower()
    if token in {"fp16", "float16", "half"}:
        return torch.float16
    if token in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if token != "auto":
        raise ValueError(f"unknown amp_dtype: {token}")
    # Keep backward-compatible default behavior for stability/speed tradeoff.
    del device
    return torch.float16


def _build_adamw(cfg: dict, model: torch.nn.Module, device: torch.device) -> torch.optim.Optimizer:
    """Build AdamW optimizer with fast CUDA backends when available."""
    base_kwargs = {
        "lr": float(cfg["train"]["lr"]),
        "betas": tuple(cfg["train"]["betas"]),
        "weight_decay": float(cfg["train"].get("weight_decay", 0.01)),
    }

    if device.type != "cuda":
        return torch.optim.AdamW(model.parameters(), **base_kwargs)

    # Prefer fused CUDA AdamW, then foreach, then plain AdamW fallback.
    candidates = [
        {"fused": True},
        {"foreach": True},
        {},
    ]
    last_error: Exception | None = None
    for extra in candidates:
        try:
            return torch.optim.AdamW(model.parameters(), **base_kwargs, **extra)
        except (TypeError, RuntimeError, ValueError) as err:
            last_error = err
            continue
    assert last_error is not None
    raise last_error


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


def _dataset_size(loader) -> int | None:
    """Best-effort dataset cardinality extraction from DataLoader."""
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None
    try:
        size = int(len(dataset))
    except Exception:
        return None
    return size if size > 0 else None


def _utc_iso(ts: float) -> str:
    """Convert unix timestamp to UTC ISO-8601 string."""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


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
        allow_tf32 = bool(cfg["compute"].get("allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

    run_dir = make_run_dir(cfg, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Training currently starts from scratch (no resume path), so clear stale
    # checkpoints from previous executions in the same run directory.
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        for pattern in ("step_*.pt", "eval_step_*.pt"):
            for stale in ckpt_dir.glob(pattern):
                stale.unlink(missing_ok=True)

    save_config(cfg, run_dir / "config_resolved.yaml")

    model = build_model(cfg).to(device)
    optimizer = _build_adamw(cfg=cfg, model=model, device=device)

    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(cfg, device) if use_amp else torch.float16
    # GradScaler is only needed for fp16. bf16 is typically stable without scaling.
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = amp.GradScaler("cuda", enabled=use_scaler)
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
    ckpt_every_steps = int(cfg["train"].get("ckpt_every_steps", cfg["train"].get("ckpt_every", 1000)))
    ckpt_every_steps = max(1, ckpt_every_steps)
    keep_last_k = int(cfg["train"].get("keep_last_k", 3))
    clip_grad = float(cfg["train"].get("clip_grad_norm", 1.0))

    dataset_name_raw = str(cfg["dataset"]["name"])
    dataset_name = dataset_name_raw.lower()
    micro_batch_size = int(cfg["dataset"]["batch_size"])
    effective_batch_size = int(micro_batch_size * accum_steps)
    n_seen_total = int(total_steps * effective_batch_size)
    dataset_size = _dataset_size(loader)
    approx_epochs_total = (float(n_seen_total) / float(dataset_size)) if dataset_size else None
    train_start_wall_unix = time.time()
    train_start_wall_utc = _utc_iso(train_start_wall_unix)
    train_start_perf = time.perf_counter()

    fairness_stats = {
        "dataset": dataset_name_raw,
        "model_id": model_id,
        "batch_size": micro_batch_size,
        "grad_accum_steps": accum_steps,
        "effective_batch_size": effective_batch_size,
        "total_steps": total_steps,
        "n_seen_images": n_seen_total,
        "dataset_size": dataset_size,
        "approx_epochs": approx_epochs_total,
        "wall_clock_start_unix": train_start_wall_unix,
        "wall_clock_start_utc": train_start_wall_utc,
    }
    metrics_json_path = run_dir / "metrics.json"
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(fairness_stats, f, indent=2)

    approx_epochs_text = "n/a" if approx_epochs_total is None else f"{approx_epochs_total:.4f}"
    print(
        "[fairness] "
        f"dataset={dataset_name_raw} model_id={model_id} "
        f"batch_size={micro_batch_size} grad_accum_steps={accum_steps} "
        f"B_eff={effective_batch_size} total_steps={total_steps} "
        f"N_seen={n_seen_total} approx_epochs={approx_epochs_text} "
        f"wall_clock_start_utc={train_start_wall_utc}"
    )
    tb_logger.log_scalars(
        {
            "fairness/batch_size": float(micro_batch_size),
            "fairness/grad_accum_steps": float(accum_steps),
            "fairness/effective_batch_size": float(effective_batch_size),
            "fairness/total_steps": float(total_steps),
            "fairness/n_seen_images": float(n_seen_total),
            "fairness/approx_epochs": float(approx_epochs_total) if approx_epochs_total is not None else float("nan"),
        },
        step=0,
    )

    supports_selection = dataset_name.startswith("toy") or dataset_name == "mnist" or dataset_name == "cifar10"
    selection_enable = bool(cfg["train"].get("selection_enable", True)) and supports_selection
    selection_eval_every = max(1, int(cfg["train"].get("selection_eval_every", 100)))
    selection_top_k = max(1, int(cfg["train"].get("selection_top_k", 3)))
    selection_nfe = max(1, int(cfg["train"].get("selection_eval_nfe", 100)))
    selection_sampler = str(cfg["train"].get("selection_sampler", cfg.get("sampler", {}).get("main", "heun")))
    selection_use_ema = bool(cfg["train"].get("selection_use_ema", True))
    selection_num_samples = int(
        cfg["train"].get(
            "selection_eval_num_samples",
            min(2048, int(cfg.get("eval", {}).get("num_fid_samples", 2048))),
        )
    )
    selection_batch_size = int(cfg["train"].get("selection_eval_batch_size", cfg.get("eval", {}).get("batch_size", 64)))
    topk_state_path = run_dir / "eval" / "selection_topk.json"
    topk_history_path = run_dir / "eval" / "selection_history.csv"
    topk_records: list[dict] = []
    if selection_enable:
        topk_state_path.unlink(missing_ok=True)
        topk_history_path.unlink(missing_ok=True)
        prune_eval_candidate_checkpoints(run_dir=run_dir, keep_steps=set())

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

        with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
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
        n_seen_at_step = int(step * effective_batch_size)
        approx_epochs_at_step = (float(n_seen_at_step) / float(dataset_size)) if dataset_size else float("nan")
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
            "batch_size": micro_batch_size,
            "grad_accum_steps": accum_steps,
            "effective_batch_size": effective_batch_size,
            "n_seen_images": n_seen_at_step,
            "approx_epochs": approx_epochs_at_step,
        }
        if device.type == "cuda":
            row["vram_peak_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024**2))
        else:
            row["vram_peak_mb"] = 0.0

        if step % log_every == 0 or step == 1:
            csv_logger.log(row)
            tb_logger.log_scalars(row, step)
            progress.set_postfix({"loss": f"{row['loss_total']:.4f}", "ips": f"{row['imgs_per_sec']:.1f}"})

        if step % ckpt_every_steps == 0 or step == total_steps:
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

        if selection_enable and (step % selection_eval_every == 0 or step == total_steps):
            eval_model = ema.shadow if selection_use_ema else model
            restore_mode = bool(eval_model.training)
            eval_model.eval()
            score = evaluate_selection_score(
                cfg=cfg,
                model=eval_model,
                model_id=model_id,
                device=device,
                num_samples=selection_num_samples,
                nfe=selection_nfe,
                batch_size=selection_batch_size,
                sampler_name=selection_sampler,
            )
            if restore_mode:
                eval_model.train()

            candidate_path = eval_checkpoint_path(run_dir=run_dir, step=step)
            save_checkpoint_to_path(
                path=candidate_path,
                step=step,
                model=model,
                optimizer=optimizer,
                ema_state=ema.state_dict(),
                scaler_state=scaler.state_dict() if use_amp else None,
                cfg=cfg,
            )

            record = score_to_record(step=step, checkpoint_name=candidate_path.name, score=score)
            append_selection_history_csv(path=topk_history_path, row=record)
            topk_records = update_topk_records(previous=topk_records, current=record, top_k=selection_top_k)
            keep_steps = {int(item.get("step", -1)) for item in topk_records}
            prune_eval_candidate_checkpoints(run_dir=run_dir, keep_steps=keep_steps)
            save_selection_state(
                path=topk_state_path,
                payload={
                    "topk": topk_records,
                    "metric_policy": {
                        "dataset": dataset_name,
                        "toy": "maximize toy_logpdf, tie-break by lower mmd",
                        "mnist_cifar10": "minimize fid",
                    },
                    "selection_config": {
                        "selection_eval_every": selection_eval_every,
                        "selection_top_k": selection_top_k,
                        "selection_eval_nfe": selection_nfe,
                        "selection_eval_num_samples": selection_num_samples,
                        "selection_eval_batch_size": selection_batch_size,
                        "selection_sampler": selection_sampler,
                        "selection_use_ema": selection_use_ema,
                    },
                },
            )
            tb_logger.log_scalars(
                {
                    "selection/primary_score": float(score.primary_score),
                    "selection/metric_value": float(score.metric_value),
                    "selection/secondary_score": float(score.secondary_score),
                },
                step,
            )

    train_end_wall_unix = time.time()
    train_elapsed_sec = float(time.perf_counter() - train_start_perf)
    fairness_stats.update(
        {
            "wall_clock_end_unix": train_end_wall_unix,
            "wall_clock_end_utc": _utc_iso(train_end_wall_unix),
            "wall_clock_elapsed_sec": train_elapsed_sec,
            "approx_gpu_hours_wall_clock": float(train_elapsed_sec / 3600.0),
        }
    )
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(fairness_stats, f, indent=2)

    tb_logger.close()
    return run_dir
