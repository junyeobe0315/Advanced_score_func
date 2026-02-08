from __future__ import annotations

import argparse
import yaml

from src.trainers import train
from src.utils.config import apply_overrides, ensure_required_sections, load_config
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train score model variants")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Dotted key overrides, e.g. train.total_steps=1000 loss.lambda_sym=0.01",
    )
    return p.parse_args()


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must be key=value: {item}")
        key, raw = item.split("=", 1)
        out[key] = yaml.safe_load(raw)
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_required_sections(cfg)

    if args.override:
        cfg = apply_overrides(cfg, parse_overrides(args.override))

    if args.seed is not None:
        cfg["train"]["seed"] = int(args.seed)

    seed_everything(int(cfg["train"]["seed"]))
    run_dir = train(cfg, seed=int(cfg["train"]["seed"]))
    print(str(run_dir))


if __name__ == "__main__":
    main()
