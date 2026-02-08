from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    pass


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
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
    candidate = (path.parent / maybe_relative).resolve()
    if candidate.exists():
        return candidate
    fallback = Path(maybe_relative).resolve()
    if fallback.exists():
        return fallback
    raise ConfigError(f"base_config not found: {maybe_relative} (from {path})")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    return _load_with_base(Path(path).resolve(), visited=set())


def _load_with_base(path: Path, visited: set[Path]) -> dict[str, Any]:
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
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def ensure_required_sections(cfg: dict[str, Any]) -> None:
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
    out = deepcopy(cfg)
    for key, value in overrides.items():
        set_by_dotted_path(out, key, value)
    return out
