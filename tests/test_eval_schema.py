from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from src.trainers import train
from src.utils.config import load_experiment_config
from src.utils.seed import seed_everything


def test_main_eval_writes_extended_integrability_schema(tmp_path: Path) -> None:
    """Evaluation output should include long-format integrability columns."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_experiment_config(str(repo_root / "configs" / "toy" / "experiment.yaml"), model="m0", ablation="none")

    cfg["project"]["run_root"] = str(tmp_path / "runs")
    cfg["train"]["total_steps"] = 4
    cfg["train"]["ckpt_every_steps"] = 4
    cfg["train"]["log_every"] = 1
    cfg["dataset"]["batch_size"] = 64
    cfg["eval"]["num_fid_samples"] = 128
    cfg["eval"]["batch_size"] = 64
    cfg["eval"]["integrability_batch"] = 64
    cfg["eval"]["sigma_bins"] = 4
    cfg["eval"]["enable_cycle_metrics"] = True

    seed_everything(0)
    run_dir = train(cfg, seed=0)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.main_eval",
            "--run_dir",
            str(run_dir),
            "--nfe_list",
            "10",
        ],
        check=True,
    )

    integ_path = run_dir / "eval" / "integrability_vs_sigma.csv"
    assert integ_path.exists()

    with integ_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])

    expected = {"bin", "sigma_lo", "sigma_hi", "count", "metric_name", "scale_delta", "cycle_len", "value"}
    assert expected.issubset(cols)
