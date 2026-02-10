from __future__ import annotations

import json
from pathlib import Path

from src.trainers import train
from src.utils.config import load_config_with_model
from src.utils.seed import seed_everything


def test_train_smoke_for_all_models(tmp_path: Path) -> None:
    """Smoke test tiny training runs for M0..M4 variants."""
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "runs"

    dataset_cfg = "toy/dataset.yaml"
    model_keys = ["m0", "m1", "m2", "m3", "m4"]
    for model_key in model_keys:
        cfg = load_config_with_model(str(repo_root / "configs" / dataset_cfg), model=model_key)
        cfg["project"]["run_root"] = str(run_root)
        cfg["train"]["total_steps"] = 10
        cfg["train"]["ckpt_every_steps"] = 10
        cfg["train"]["log_every"] = 2
        cfg["train"]["keep_last_k"] = 1
        cfg["dataset"]["batch_size"] = 64
        cfg["loss"]["cycle_samples"] = 4
        cfg["loss"]["cycle_subset"] = 32
        seed_everything(0)
        run_dir = train(cfg, seed=0)

        assert run_dir.exists()
        assert (run_dir / "config_resolved.yaml").exists()
        metrics_json = run_dir / "metrics.json"
        assert metrics_json.exists()
        with metrics_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for key in (
            "dataset",
            "model_id",
            "batch_size",
            "grad_accum_steps",
            "effective_batch_size",
            "total_steps",
            "n_seen_images",
            "approx_epochs",
            "wall_clock_start_utc",
        ):
            assert key in payload
        assert (run_dir / "metrics.csv").exists()
        ckpts = list((run_dir / "checkpoints").glob("step_*.pt"))
        assert len(ckpts) >= 1
