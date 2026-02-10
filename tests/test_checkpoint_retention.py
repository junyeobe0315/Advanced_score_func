from __future__ import annotations

from pathlib import Path

from src.utils.checkpoint import prune_checkpoints_after_training


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ckpt")


def test_prune_checkpoints_after_training_keeps_topk_and_last(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "toy" / "M3" / "seed0"
    ckpt_dir = run_dir / "checkpoints"

    _touch(ckpt_dir / "step_00000100.pt")
    _touch(ckpt_dir / "step_00000200.pt")
    _touch(ckpt_dir / "step_00000300.pt")
    _touch(ckpt_dir / "eval_step_00000100.pt")
    _touch(ckpt_dir / "eval_step_00000200.pt")
    _touch(ckpt_dir / "eval_step_00000300.pt")

    summary = prune_checkpoints_after_training(run_dir=run_dir, keep_eval_steps={200}, keep_last_k=1)

    assert summary["kept_step"] == 2
    assert summary["deleted_step"] == 1
    assert summary["kept_eval"] == 2
    assert summary["deleted_eval"] == 1

    assert (ckpt_dir / "step_00000100.pt").exists() is False
    assert (ckpt_dir / "step_00000200.pt").exists() is True
    assert (ckpt_dir / "step_00000300.pt").exists() is True

    assert (ckpt_dir / "eval_step_00000100.pt").exists() is False
    assert (ckpt_dir / "eval_step_00000200.pt").exists() is True
    assert (ckpt_dir / "eval_step_00000300.pt").exists() is True
