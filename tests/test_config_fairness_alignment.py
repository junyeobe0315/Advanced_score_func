from __future__ import annotations

from pathlib import Path

from src.utils.config import load_experiment_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _effective_batch(cfg: dict) -> int:
    return int(cfg["dataset"]["batch_size"]) * int(cfg["train"].get("grad_accum_steps", 1))


def _assert_common_train_fields(m0: dict, mx: dict) -> None:
    keys = ["lr", "betas", "weight_decay", "ema_decay"]
    for key in keys:
        assert mx["train"][key] == m0["train"][key]


def _assert_common_sigma_fields(m0: dict, mx: dict) -> None:
    keys = [
        "objective",
        "sigma_sampling",
        "edm_p_mean",
        "edm_p_std",
        "sigma_sample_clamp",
        "sigma_data",
        "weight_mode",
        "sigma_min",
        "sigma_max",
    ]
    for key in keys:
        assert mx["loss"][key] == m0["loss"][key]


def test_mnist_m0_m1_m2_alignment() -> None:
    root = _repo_root()
    base = root / "configs/mnist/experiment.yaml"
    m0 = load_experiment_config(base, model="m0", ablation="none")
    m1 = load_experiment_config(base, model="m1", ablation="none")
    m2 = load_experiment_config(base, model="m2", ablation="none")

    assert int(m1["dataset"]["batch_size"]) == int(m0["dataset"]["batch_size"])
    assert int(m1["train"]["grad_accum_steps"]) == int(m0["train"]["grad_accum_steps"])
    assert int(m1["train"]["total_steps"]) == int(m0["train"]["total_steps"])
    _assert_common_train_fields(m0, m1)
    _assert_common_sigma_fields(m0, m1)

    assert int(m2["dataset"]["batch_size"]) == int(m0["dataset"]["batch_size"])
    assert int(m2["train"]["grad_accum_steps"]) == int(m0["train"]["grad_accum_steps"])
    assert int(m2["train"]["total_steps"]) == int(m0["train"]["total_steps"])

    assert _effective_batch(m0) == _effective_batch(m1) == _effective_batch(m2)
    _assert_common_train_fields(m0, m2)
    _assert_common_sigma_fields(m0, m2)


def test_cifar10_m0_m1_m2_alignment() -> None:
    root = _repo_root()
    base = root / "configs/cifar10/experiment.yaml"
    m0 = load_experiment_config(base, model="m0", ablation="none")
    m1 = load_experiment_config(base, model="m1", ablation="none")
    m2 = load_experiment_config(base, model="m2", ablation="none")

    assert int(m1["dataset"]["batch_size"]) == int(m0["dataset"]["batch_size"])
    assert int(m1["train"]["grad_accum_steps"]) == int(m0["train"]["grad_accum_steps"])
    assert int(m1["train"]["total_steps"]) == int(m0["train"]["total_steps"])
    _assert_common_train_fields(m0, m1)
    _assert_common_sigma_fields(m0, m1)

    assert int(m2["dataset"]["batch_size"]) == int(m0["dataset"]["batch_size"])
    assert int(m2["train"]["grad_accum_steps"]) == int(m0["train"]["grad_accum_steps"])
    assert int(m2["train"]["total_steps"]) == int(m0["train"]["total_steps"])

    assert _effective_batch(m0) == _effective_batch(m1) == _effective_batch(m2)
    _assert_common_train_fields(m0, m2)
    _assert_common_sigma_fields(m0, m2)


def test_toy_m2_keeps_seen_images_protocol() -> None:
    root = _repo_root()
    base = root / "configs/toy/experiment.yaml"
    m0 = load_experiment_config(base, model="m0", ablation="none")
    m2 = load_experiment_config(base, model="m2", ablation="none")

    assert _effective_batch(m0) == _effective_batch(m2)
    assert int(m0["train"]["total_steps"]) == int(m2["train"]["total_steps"])
    _assert_common_train_fields(m0, m2)
    _assert_common_sigma_fields(m0, m2)
