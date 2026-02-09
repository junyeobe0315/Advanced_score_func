from __future__ import annotations

from pathlib import Path

from src.utils.config import load_config


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


def test_mnist_m0_m1_m2_epoch_matched_alignment() -> None:
    root = _repo_root()
    m0 = load_config(root / "configs/mnist/m0.yaml")
    m1 = load_config(root / "configs/mnist/m1.yaml")
    m2 = load_config(root / "configs/mnist/m2_epoch_matched.yaml")

    assert int(m0["dataset"]["batch_size"]) == 128
    assert int(m0["train"]["grad_accum_steps"]) == 1
    assert int(m0["train"]["total_steps"]) == 120000

    assert int(m1["dataset"]["batch_size"]) == int(m0["dataset"]["batch_size"])
    assert int(m1["train"]["grad_accum_steps"]) == int(m0["train"]["grad_accum_steps"])
    assert int(m1["train"]["total_steps"]) == int(m0["train"]["total_steps"])
    _assert_common_train_fields(m0, m1)
    _assert_common_sigma_fields(m0, m1)

    assert _effective_batch(m0) == 128
    assert _effective_batch(m1) == 128
    assert _effective_batch(m2) == 128
    assert int(m2["train"]["total_steps"]) == 120000
    _assert_common_train_fields(m0, m2)
    _assert_common_sigma_fields(m0, m2)


def test_cifar10_m0_m1_m2_epoch_matched_alignment() -> None:
    root = _repo_root()
    m0 = load_config(root / "configs/cifar10/m0.yaml")
    m1 = load_config(root / "configs/cifar10/m1.yaml")
    m2 = load_config(root / "configs/cifar10/m2_epoch_matched.yaml")

    assert int(m0["dataset"]["batch_size"]) == 64
    assert int(m0["train"]["grad_accum_steps"]) == 1
    assert int(m0["train"]["total_steps"]) == 250000

    assert int(m1["dataset"]["batch_size"]) == int(m0["dataset"]["batch_size"])
    assert int(m1["train"]["grad_accum_steps"]) == int(m0["train"]["grad_accum_steps"])
    assert int(m1["train"]["total_steps"]) == int(m0["train"]["total_steps"])
    _assert_common_train_fields(m0, m1)
    _assert_common_sigma_fields(m0, m1)

    assert _effective_batch(m0) == 64
    assert _effective_batch(m1) == 64
    assert _effective_batch(m2) == 64
    assert int(m2["train"]["total_steps"]) == 250000
    _assert_common_train_fields(m0, m2)
    _assert_common_sigma_fields(m0, m2)


def test_toy_m2_epoch_matched_keeps_seen_images_protocol() -> None:
    root = _repo_root()
    m0 = load_config(root / "configs/toy/m0.yaml")
    m2 = load_config(root / "configs/toy/m2_epoch_matched.yaml")

    assert _effective_batch(m0) == _effective_batch(m2) == 1024
    assert int(m0["train"]["total_steps"]) == int(m2["train"]["total_steps"]) == 30000
    _assert_common_train_fields(m0, m2)
    _assert_common_sigma_fields(m0, m2)
