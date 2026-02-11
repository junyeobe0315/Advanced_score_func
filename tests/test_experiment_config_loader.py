from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import ConfigError, ensure_required_sections, load_experiment_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_experiment_loader_populates_required_fields_for_all_models() -> None:
    root = _repo_root()
    exp = root / "configs" / "toy" / "experiment.yaml"

    for model_key in ["m0", "m1", "m2", "m3", "m4"]:
        cfg = load_experiment_config(exp, model=model_key, ablation="none")
        ensure_required_sections(cfg)
        assert "model" in cfg
        assert "loss" in cfg
        assert "type" in cfg["model"]
        assert "sigma_min" in cfg["loss"]
        assert "sigma_max" in cfg["loss"]


def test_shared_experiment_entrypoint_works_with_dataset_selector() -> None:
    root = _repo_root()
    exp = root / "configs" / "experiment.yaml"
    cfg = load_experiment_config(exp, dataset="toy", model="m0", ablation="none")
    ensure_required_sections(cfg)
    assert str(cfg["dataset"]["name"]) == "toy"


def test_ablation_patch_is_applied_after_model_preset() -> None:
    root = _repo_root()
    exp = root / "configs" / "cifar10" / "experiment.yaml"

    base = load_experiment_config(exp, model="m1", ablation="none")
    patched = load_experiment_config(exp, model="m1", ablation="reg_both")

    assert float(base["loss"]["mu_loop"]) == 0.0
    assert float(patched["loss"]["mu_loop"]) == 0.03
    assert float(base["loss"]["lambda_sym"]) == float(patched["loss"]["lambda_sym"]) == 0.03
    # Non-ablation defaults must remain intact.
    assert float(patched["loss"]["sigma_min"]) == float(base["loss"]["sigma_min"])
    assert str(patched["model"]["type"]) == str(base["model"]["type"])


def test_unknown_ablation_lists_available_names() -> None:
    root = _repo_root()
    exp = root / "configs" / "cifar10" / "experiment.yaml"

    with pytest.raises(ConfigError) as exc:
        load_experiment_config(exp, model="m1", ablation="not_exist")

    message = str(exc.value)
    assert "ablation config not found" in message
    assert "available" in message
    assert "none" in message


def test_non_experiment_entrypoint_is_rejected() -> None:
    root = _repo_root()
    bad = root / "configs" / "dataset.yaml"

    with pytest.raises(ConfigError) as exc:
        load_experiment_config(bad, model="m0", ablation="none")

    assert "experiment.yaml" in str(exc.value)


def test_m3_dynamic_mu2_config_values_by_dataset() -> None:
    root = _repo_root()
    exp = root / "configs" / "experiment.yaml"

    toy = load_experiment_config(exp, dataset="toy", model="m3", ablation="none")
    mnist = load_experiment_config(exp, dataset="mnist", model="m3", ablation="none")
    cifar10 = load_experiment_config(exp, dataset="cifar10", model="m3", ablation="none")
    imagenet128 = load_experiment_config(exp, dataset="imagenet128", model="m3", ablation="none")

    assert float(toy["loss"]["target_r"]) == 0.001
    assert float(mnist["loss"]["target_r"]) == 0.01
    assert float(cifar10["loss"]["target_r"]) == 0.03
    assert float(imagenet128["loss"]["target_r"]) == 0.01

    assert int(toy["loss"]["update_step"]) == 10000
    assert int(mnist["loss"]["update_step"]) == 5000
    assert int(cifar10["loss"]["update_step"]) == 200
    assert int(imagenet128["loss"]["update_step"]) == 200

    assert float(toy["loss"]["start_mu2"]) == 0.0
    assert float(mnist["loss"]["start_mu2"]) == 0.0
    assert float(cifar10["loss"]["start_mu2"]) == 1.0e-4
    assert float(imagenet128["loss"]["start_mu2"]) == 1.0e-5
