from __future__ import annotations

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
    return _load_with_base(Path(path).resolve(), visited=set())


def _load_with_base(path: Path, visited: set[Path]) -> dict[str, Any]:
    """Internal recursive loader for ``base_config`` hierarchy.

    Args:
        path: Current config file path.
        visited: Set used for cycle detection.

    Returns:
        Resolved config dictionary at current node.
    """
    if path in visited:
        raise ConfigError(f"cyclic base_config reference detected at: {path}")
    visited.add(path)
    current = _load_yaml(path)

    base_ref = current.get("base_config")
    if base_ref:
        base_path = _resolve_path(path, str(base_ref))
        base_cfg = _load_with_base(base_path, visited)
        return _deep_merge(base_cfg, current)
    return current


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
