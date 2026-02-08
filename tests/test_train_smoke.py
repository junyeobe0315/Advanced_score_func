from __future__ import annotations

from pathlib import Path

import yaml

from src.trainers import train
from src.utils.seed import seed_everything


def _load_cfg(path: Path) -> dict:
    """Load YAML file for test helper usage."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _prepare_cfg(root: Path, name: str, run_root: Path) -> dict:
    """Build short-run config for smoke testing.

    Args:
        root: Repository root.
        name: Config filename under ``configs``.
        run_root: Temporary run output directory.

    Returns:
        Mutated config with tiny training budget.
    """
    from src.utils.config import load_config

    cfg = load_config(str(root / "configs" / name))
    cfg["project"]["run_root"] = str(run_root)
    cfg["train"]["total_steps"] = 10
    cfg["train"]["ckpt_every"] = 10
    cfg["train"]["log_every"] = 2
    cfg["train"]["keep_last_k"] = 1
    cfg["dataset"]["batch_size"] = 64
    cfg["loss"]["cycle_samples"] = 4
    cfg["loss"]["cycle_subset"] = 32
    return cfg


def test_train_smoke_for_all_models(tmp_path: Path) -> None:
    """Smoke test tiny training runs for M0..M4 variants."""
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "runs"

    cfg_names = [
        "toy/m0.yaml",
        "toy/m1.yaml",
        "toy/m2.yaml",
        "toy/m3.yaml",
        "toy/m4.yaml",
    ]
    for cfg_name in cfg_names:
        cfg = _prepare_cfg(repo_root, cfg_name, run_root)
        seed_everything(0)
        run_dir = train(cfg, seed=0)

        assert run_dir.exists()
        assert (run_dir / "config_resolved.yaml").exists()
        assert (run_dir / "metrics.csv").exists()
        ckpts = list((run_dir / "checkpoints").glob("step_*.pt"))
        assert len(ckpts) >= 1
