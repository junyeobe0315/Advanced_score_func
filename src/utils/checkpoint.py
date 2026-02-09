from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def checkpoint_path(run_dir: str | Path, step: int) -> Path:
    """Build checkpoint file path for a training step.

    Args:
        run_dir: Run directory path.
        step: Training step number.

    Returns:
        Absolute checkpoint file path under ``run_dir/checkpoints``.
    """
    path = Path(run_dir) / "checkpoints" / f"step_{step:08d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def named_checkpoint_path(run_dir: str | Path, name: str) -> Path:
    """Build a named checkpoint path under ``run_dir/checkpoints``.

    Args:
        run_dir: Run directory path.
        name: Filename stem without extension.

    Returns:
        Absolute checkpoint path.
    """
    safe_name = str(name).strip().replace("/", "_")
    path = Path(run_dir) / "checkpoints" / f"{safe_name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def eval_checkpoint_path(run_dir: str | Path, step: int) -> Path:
    """Build model-selection candidate checkpoint path for a training step."""
    return named_checkpoint_path(run_dir, f"eval_step_{int(step):08d}")


def _checkpoint_payload(
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_state: dict[str, Any] | None,
    scaler_state: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Build serialized checkpoint payload."""
    return {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema_state,
        "scaler": scaler_state,
        "config": cfg,
    }


def save_checkpoint_to_path(
    path: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_state: dict[str, Any] | None,
    scaler_state: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> Path:
    """Serialize checkpoint payload to an explicit destination path."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(
            step=step,
            model=model,
            optimizer=optimizer,
            ema_state=ema_state,
            scaler_state=scaler_state,
            cfg=cfg,
        ),
        dst,
    )
    return dst


def save_named_checkpoint(
    run_dir: str | Path,
    name: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_state: dict[str, Any] | None,
    scaler_state: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> Path:
    """Serialize checkpoint payload using a custom filename stem."""
    return save_checkpoint_to_path(
        path=named_checkpoint_path(run_dir, name),
        step=step,
        model=model,
        optimizer=optimizer,
        ema_state=ema_state,
        scaler_state=scaler_state,
        cfg=cfg,
    )


def save_checkpoint(
    run_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_state: dict[str, Any] | None,
    scaler_state: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> Path:
    """Serialize model/optimizer training state to checkpoint.

    Args:
        run_dir: Run directory path.
        step: Current training step.
        model: Model to save.
        optimizer: Optimizer to save.
        ema_state: Optional EMA state dictionary.
        scaler_state: Optional AMP scaler state dictionary.
        cfg: Resolved config snapshot.

    Returns:
        Path to written checkpoint file.
    """
    return save_checkpoint_to_path(
        path=checkpoint_path(run_dir, step),
        step=step,
        model=model,
        optimizer=optimizer,
        ema_state=ema_state,
        scaler_state=scaler_state,
        cfg=cfg,
    )


def keep_last_checkpoints(run_dir: str | Path, keep_last_k: int) -> None:
    """Delete old checkpoints and keep only most recent ``k`` files.

    Args:
        run_dir: Run directory path.
        keep_last_k: Number of latest checkpoints to keep.
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return
    all_ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    for old in all_ckpts[:-keep_last_k]:
        old.unlink(missing_ok=True)


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Return latest checkpoint file in run directory.

    Args:
        run_dir: Run directory path.

    Returns:
        Latest checkpoint path or ``None`` when no checkpoint exists.
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return None
    all_ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    return all_ckpts[-1] if all_ckpts else None


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load checkpoint payload from disk.

    Args:
        path: Checkpoint file path.
        map_location: Torch map location argument.

    Returns:
        Deserialized checkpoint dictionary.
    """
    return torch.load(Path(path), map_location=map_location)
