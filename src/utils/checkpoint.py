from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def checkpoint_path(run_dir: str | Path, step: int) -> Path:
    path = Path(run_dir) / "checkpoints" / f"step_{step:08d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    run_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_state: dict[str, Any] | None,
    scaler_state: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> Path:
    path = checkpoint_path(run_dir, step)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema_state,
        "scaler": scaler_state,
        "config": cfg,
    }
    torch.save(payload, path)
    return path


def keep_last_checkpoints(run_dir: str | Path, keep_last_k: int) -> None:
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return
    all_ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    for old in all_ckpts[:-keep_last_k]:
        old.unlink(missing_ok=True)


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return None
    all_ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    return all_ckpts[-1] if all_ckpts else None


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)
