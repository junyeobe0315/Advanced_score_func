from __future__ import annotations

from typing import Any

from src.utils.config import resolve_model_id


def make_loader_kwargs(
    cfg: dict,
    *,
    train: bool,
    shuffle: bool,
    default_num_workers: int,
) -> dict[str, Any]:
    """Build shared DataLoader kwargs with memory-aware defaults.

    Notes:
        M3/M4 typically run slower, so the host-side prefetch queue can hold
        more staged batches for longer. We use a smaller default prefetch
        factor for those model ids to reduce CPU RAM pressure.
    """
    ds_cfg = cfg["dataset"]
    num_workers = int(ds_cfg.get("num_workers", default_num_workers))
    pin_memory = bool(ds_cfg.get("pin_memory", True))

    model_id = resolve_model_id(cfg)
    default_prefetch = 1 if model_id in {"M3", "M4"} else 2
    prefetch_factor = int(ds_cfg.get("prefetch_factor", default_prefetch))

    out: dict[str, Any] = {
        "batch_size": int(ds_cfg["batch_size"]),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": bool(train),
    }
    if num_workers > 0:
        out["prefetch_factor"] = max(1, prefetch_factor)
        out["persistent_workers"] = bool(ds_cfg.get("persistent_workers", False))
    return out

