from __future__ import annotations

import argparse

from src.trainers import train
from src.utils.config import ensure_experiment_defaults, ensure_required_sections, load_experiment_config
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for multi-seed sweep runner."""
    p = argparse.ArgumentParser(description="Simple multi-seed sweep runner")
    p.add_argument("--sweep", type=str, required=True)
    p.add_argument("--dataset", type=str, default=None, help="Optional dataset key for shared experiment.yaml")
    p.add_argument("--model", type=str, required=True, help="Model preset key (m0..m4)")
    p.add_argument("--ablation", type=str, default="none", help="Optional ablation patch name")
    p.add_argument("--seeds", type=str, required=True, help="Comma-separated list, e.g. 0,1,2")
    return p.parse_args()


def main() -> None:
    """Run sweep config across requested seeds.

    Returns:
        None. Prints each produced run directory path.

    How it works:
        Loads one experiment config and repeats training while updating seed.
    """
    args = parse_args()
    cfg = load_experiment_config(args.sweep, model=args.model, ablation=args.ablation, dataset=args.dataset)
    ensure_required_sections(cfg)
    cfg = ensure_experiment_defaults(cfg)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    for seed in seeds:
        cfg["train"]["seed"] = seed
        seed_everything(seed)
        run_dir = train(cfg, seed=seed)
        print(str(run_dir))


if __name__ == "__main__":
    main()
