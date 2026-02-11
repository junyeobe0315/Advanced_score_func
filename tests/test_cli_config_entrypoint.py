from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_main_dry_run_emits_experiment_entrypoint() -> None:
    root = _repo_root()
    proc = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--dataset=toy",
            "--models=[m0]",
            "--seeds=[0]",
            "--mode=train",
            "--ablation",
            "none",
            "--dry_run",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    out = proc.stdout
    assert "configs/experiment.yaml" in out
    assert "--dataset toy" in out
    assert "--ablation none" in out


def test_main_train_rejects_non_experiment_config() -> None:
    root = _repo_root()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.main_train",
            "--config",
            "configs/dataset.yaml",
            "--model",
            "m0",
            "--seed",
            "0",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "experiment.yaml" in (proc.stderr + proc.stdout)
