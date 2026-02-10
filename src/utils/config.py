from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when config loading or override resolution fails."""

    pass


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two config mappings.

    Args:
        base: Base configuration dictionary.
        override: Overriding configuration dictionary.

    Returns:
        New dictionary with recursive merge result.

    How it works:
        Nested dictionaries are merged recursively; non-dict fields are
        overwritten by ``override`` values. ``base_config`` key is ignored.
    """
    merged = deepcopy(base)
    for key, value in override.items():
        if key == "base_config":
            continue
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_path(path: Path, maybe_relative: str) -> Path:
    """Resolve ``base_config`` path relative to current config file.

    Args:
        path: Absolute path of current config file.
        maybe_relative: Base config path string from YAML.

    Returns:
        Existing absolute path to base config.
    """
    candidate = (path.parent / maybe_relative).resolve()
    if candidate.exists():
        return candidate
    fallback = Path(maybe_relative).resolve()
    if fallback.exists():
        return fallback
    raise ConfigError(f"base_config not found: {maybe_relative} (from {path})")


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and validate mapping root.

    Args:
        path: YAML path.

    Returns:
        Parsed dictionary object.
    """
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    """Load config with recursive ``base_config`` composition.

    Args:
        path: Entry config file path.

    Returns:
        Fully resolved config dictionary.
    """
    return _load_with_base(Path(path).resolve(), stack=set())


def _normalize_model_key(model: str) -> str:
    """Normalize user-facing model selector to ``m0..m4`` token."""
    token = str(model).strip().lower()
    if not token:
        raise ConfigError("empty model key")
    if token.startswith("m") and len(token) == 2 and token[1].isdigit():
        idx = int(token[1])
        if idx in {0, 1, 2, 3, 4}:
            return f"m{idx}"
    if token in {"baseline", "reg", "struct"}:
        legacy_map = {"baseline": "m0", "reg": "m1", "struct": "m2"}
        return legacy_map[token]
    upper = token.upper()
    if upper in {"M0", "M1", "M2", "M3", "M4"}:
        return upper.lower()
    raise ConfigError(f"invalid model key: {model}")


def load_config_with_model(
    config_path: str | Path,
    model: str | None,
    models_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load dataset config and optionally apply one model preset overlay.

    Args:
        config_path: Dataset-level config path.
        model: Optional model key (e.g. ``m0``..``m4``).
        models_path: Optional explicit path to model-presets YAML.

    Returns:
        Resolved config dictionary with selected model overlay applied.

    How it works:
        If ``models.yaml`` defines ``defaults`` (or ``shared_defaults``),
        those values are merged into dataset config first. Then the selected
        ``model_presets[model]`` overlay is merged.
    """
    cfg = load_config(config_path)
    if models_path is None:
        models_path = Path(config_path).resolve().parent / "models.yaml"
    models_cfg = load_config(models_path)

    defaults = models_cfg.get("defaults")
    if defaults is None:
        defaults = models_cfg.get("shared_defaults")
    if defaults is not None:
        if not isinstance(defaults, dict):
            raise ConfigError(f"defaults must be a mapping: {models_path}")
        cfg = _deep_merge(cfg, defaults)

    if model is None:
        return cfg

    model_key = _normalize_model_key(model)

    presets = models_cfg.get("model_presets")
    if presets is None:
        presets = {k: v for k, v in models_cfg.items() if k not in {"defaults", "shared_defaults"}}
    if not isinstance(presets, dict):
        raise ConfigError(f"model_presets must be a mapping: {models_path}")

    preset = presets.get(model_key)
    if preset is None:
        preset = presets.get(model_key.upper())
    if preset is None:
        available = sorted(str(k) for k in presets.keys())
        raise ConfigError(f"model preset not found: {model_key} (available: {available})")
    if not isinstance(preset, dict):
        raise ConfigError(f"model preset must be a mapping: {model_key}")

    return _deep_merge(cfg, preset)


def _load_with_base(path: Path, stack: set[Path]) -> dict[str, Any]:
    """Internal recursive loader for ``base_config`` hierarchy.

    Args:
        path: Current config file path.
        stack: Active recursion stack used for cycle detection.

    Returns:
        Resolved config dictionary at current node.
    """
    if path in stack:
        raise ConfigError(f"cyclic base_config reference detected at: {path}")
    stack.add(path)
    try:
        current = _load_yaml(path)

        base_ref = current.get("base_config")
        if not base_ref:
            return current

        base_path = _resolve_path(path, str(base_ref))
        base_cfg = _load_with_base(base_path, stack)
        return _deep_merge(base_cfg, current)
    finally:
        stack.remove(path)


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Save resolved config dictionary as YAML.

    Args:
        cfg: Config dictionary to write.
        path: Destination YAML path.
    """
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def ensure_required_sections(cfg: dict[str, Any]) -> None:
    """Validate required top-level config sections.

    Args:
        cfg: Config dictionary to validate.

    Raises:
        ConfigError: If any required section is missing.
    """
    required = [
        "dataset",
        "model",
        "train",
        "loss",
        "sampler",
        "eval",
        "logging",
        "compute",
    ]
    missing = [name for name in required if name not in cfg]
    if missing:
        raise ConfigError(f"config missing required sections: {', '.join(missing)}")


def set_by_dotted_path(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set nested config value from dotted key path.

    Args:
        cfg: Config dictionary to mutate.
        dotted_key: Dotted key string such as ``train.total_steps``.
        value: Value to write.

    How it works:
        Traverses/creates nested dictionaries for each path segment and writes
        ``value`` at the final key.
    """
    keys = dotted_key.split(".")
    cursor = cfg
    for part in keys[:-1]:
        nxt = cursor.get(part)
        if nxt is None:
            cursor[part] = {}
            nxt = cursor[part]
        if not isinstance(nxt, dict):
            raise ConfigError(f"cannot set {dotted_key}: {part} is not a mapping")
        cursor = nxt
    cursor[keys[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply dotted-key overrides to a config copy.

    Args:
        cfg: Base config dictionary.
        overrides: Mapping of dotted keys to replacement values.

    Returns:
        New config dictionary with overrides applied.
    """
    out = deepcopy(cfg)
    for key, value in overrides.items():
        set_by_dotted_path(out, key, value)
    return out


def resolve_model_id(cfg: dict[str, Any]) -> str:
    """Resolve canonical 5-way model id (M0..M4) with backward compatibility.

    Args:
        cfg: Resolved experiment config dictionary.

    Returns:
        Canonical model id string in ``{"M0","M1","M2","M3","M4"}``.

    How it works:
        Prefers ``experiment.model_id``. If absent, maps legacy
        ``model.variant`` names to compatible ids:
        ``baseline->M0``, ``reg->M1``, ``struct->M2``.
    """
    exp_cfg = cfg.get("experiment", {})
    model_id = exp_cfg.get("model_id")
    if model_id is not None:
        model_id = str(model_id).upper()
        if model_id in {"M0", "M1", "M2", "M3", "M4"}:
            return model_id
        raise ConfigError(f"invalid experiment.model_id: {model_id}")

    variant = str(cfg.get("model", {}).get("variant", "")).lower()
    legacy_map = {
        "baseline": "M0",
        "reg": "M1",
        "struct": "M2",
    }
    if variant in legacy_map:
        return legacy_map[variant]
    raise ConfigError("missing experiment.model_id and unsupported legacy model.variant")


def ensure_experiment_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Inject experiment section defaults while keeping backward compatibility.

    Args:
        cfg: Resolved config dictionary.

    Returns:
        Mutated config with normalized ``experiment`` section.
    """
    cfg.setdefault("experiment", {})
    cfg["experiment"]["model_id"] = resolve_model_id(cfg)
    cfg["experiment"].setdefault("name", "integrability_5way")
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Compute deterministic hash for resolved configuration dictionary.

    Args:
        cfg: Resolved config dictionary.

    Returns:
        First 12 hex chars of SHA-256 over sorted JSON representation.
    """
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
