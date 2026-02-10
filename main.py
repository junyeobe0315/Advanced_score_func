from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

DATASET_ORDER = [
    "toy",
    "mnist",
    "cifar10",
    "imagenet128",
    "imagenet256",
    "imagenet512",
    "lsun256",
    "ffhq256",
]
MODEL_ORDER = ["m0", "m1", "m2", "m3", "m4"]
DEFAULT_NFE_BY_DATASET = {
    "toy": "8,18,32,64,128",
    "mnist": "8,18,32,64,128",
    "cifar10": "8,18,32,64,128",
    "imagenet128": "10,20,50,100,200",
    "imagenet256": "10,20,50,100,200",
    "imagenet512": "10,20,50,100,200",
    "lsun256": "10,20,50,100,200",
    "ffhq256": "10,20,50,100,200",
}


def _parse_list_arg(raw: str, fallback: list[str]) -> list[str]:
    text = str(raw).strip()
    if not text:
        return list(fallback)

    values: list[str]
    if text.startswith("["):
        loaded = yaml.safe_load(text)
        if loaded is None:
            values = []
        elif isinstance(loaded, list):
            values = [str(v).strip() for v in loaded]
        else:
            values = [str(loaded).strip()]
    else:
        values = [v.strip() for v in text.split(",")]

    values = [v for v in values if v]
    if not values:
        return list(fallback)
    return values


def _parse_datasets(raw: str) -> list[str]:
    values = [v.lower() for v in _parse_list_arg(raw, fallback=["toy"])]
    if len(values) == 1 and values[0] == "all":
        return list(DATASET_ORDER)
    unknown = sorted(set(values) - set(DATASET_ORDER))
    if unknown:
        raise ValueError(f"unknown dataset(s): {unknown}")
    deduped: list[str] = []
    for v in values:
        if v not in deduped:
            deduped.append(v)
    return deduped


def _parse_models(raw: str) -> list[str]:
    values = [v.lower() for v in _parse_list_arg(raw, fallback=MODEL_ORDER)]
    normalized: list[str] = []
    for v in values:
        if v in {"baseline", "reg", "struct"}:
            mapped = {"baseline": "m0", "reg": "m1", "struct": "m2"}[v]
            normalized.append(mapped)
            continue
        if v.upper() in {"M0", "M1", "M2", "M3", "M4"}:
            normalized.append(v.upper().lower())
            continue
        if v in MODEL_ORDER:
            normalized.append(v)
            continue
        raise ValueError(f"unknown model key: {v}")

    deduped: list[str] = []
    for v in normalized:
        if v not in deduped:
            deduped.append(v)
    return deduped


def _parse_seeds(raw: str) -> list[int]:
    values = _parse_list_arg(raw, fallback=["0", "1", "2"])
    seeds: list[int] = []
    for v in values:
        seeds.append(int(v))
    deduped: list[int] = []
    for v in seeds:
        if v not in deduped:
            deduped.append(v)
    return deduped


def _run(cmd: list[str], dry_run: bool) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified train/eval runner")
    p.add_argument("--dataset", type=str, default="toy", help="toy | mnist | ... | all or list like [toy,mnist]")
    p.add_argument("--seeds", type=str, default="[0,1,2]", help="List format recommended, e.g. [0,1,2]")
    p.add_argument("--models", type=str, default="[m0,m1,m2,m3,m4]", help="List format recommended, e.g. [m0,m3,m4]")
    p.add_argument("--mode", type=str, choices=["train", "eval", "both"], default="both")
    p.add_argument("--run_root", type=str, default="runs")
    p.add_argument("--nfe_list", type=str, default=None, help="Override NFE list for eval")
    p.add_argument("--override", nargs="*", default=[], help="Forwarded to src.main_train")
    p.add_argument("--toy_report", action="store_true", help="Run toy report generation after eval")
    p.add_argument("--report_out", type=str, default="reports/figures")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    datasets = _parse_datasets(args.dataset)
    models = _parse_models(args.models)
    seeds = _parse_seeds(args.seeds)

    for dataset in datasets:
        dataset_cfg = Path("configs") / dataset / "dataset.yaml"
        if not dataset_cfg.exists():
            raise FileNotFoundError(f"missing dataset config: {dataset_cfg}")

        nfe_list = str(args.nfe_list or DEFAULT_NFE_BY_DATASET[dataset])

        for seed in seeds:
            if args.mode in {"train", "both"}:
                for model in models:
                    cmd = [
                        sys.executable,
                        "-m",
                        "src.main_train",
                        "--config",
                        str(dataset_cfg),
                        "--model",
                        model,
                        "--seed",
                        str(seed),
                    ]
                    if args.override:
                        cmd.extend(["--override", *args.override])
                    _run(cmd, dry_run=bool(args.dry_run))

            if args.mode in {"eval", "both"}:
                for model in models:
                    model_id = model.upper()
                    run_dir = Path(args.run_root) / dataset / model_id / f"seed{seed}"
                    if not run_dir.exists():
                        print(f"[skip] missing run dir: {run_dir}")
                        continue
                    cmd = [
                        sys.executable,
                        "-m",
                        "src.main_eval",
                        "--run_dir",
                        str(run_dir),
                        "--nfe_list",
                        nfe_list,
                    ]
                    _run(cmd, dry_run=bool(args.dry_run))

            if dataset == "toy" and bool(args.toy_report) and args.mode in {"eval", "both"}:
                cmd = [
                    sys.executable,
                    "scripts/make_toy_modified_report.py",
                    "--run_root",
                    str(Path(args.run_root) / "toy"),
                    "--out_dir",
                    str(Path(args.report_out) / f"seed{seed}"),
                    "--seed",
                    str(seed),
                ]
                _run(cmd, dry_run=bool(args.dry_run))

        if dataset == "toy" and bool(args.toy_report) and args.mode in {"eval", "both"}:
            cmd = [
                sys.executable,
                "scripts/make_toy_modified_report.py",
                "--run_root",
                str(Path(args.run_root) / "toy"),
                "--out_dir",
                str(Path(args.report_out)),
                "--seed",
                "all",
            ]
            _run(cmd, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
