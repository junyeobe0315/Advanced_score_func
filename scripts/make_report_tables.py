from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def _safe_float(value: str | float | int | None) -> float:
    """Parse numeric value into float and map invalid values to NaN."""
    if value is None:
        return float("nan")
    if isinstance(value, (float, int)):
        out = float(value)
        return out if math.isfinite(out) else float("nan")
    token = str(value).strip().lower()
    if token in {"", "nan", "none", "null", "inf", "+inf", "-inf"}:
        return float("nan")
    try:
        out = float(token)
    except ValueError:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _parse_seed(seed_name: str) -> int:
    """Convert seed directory token like ``seed3`` to integer."""
    token = str(seed_name)
    if token.startswith("seed"):
        token = token[4:]
    try:
        return int(token)
    except ValueError:
        return -1


def _parse_float_list(raw: str) -> list[float]:
    """Parse comma-separated float list string."""
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV file into list of row dictionaries."""
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_table(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Write rows to CSV, creating parent directory automatically.

    When ``rows`` is empty, this still writes a header-only file if explicit
    ``fieldnames`` are provided. This keeps report artifacts predictable for
    downstream automation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(fieldnames) if fieldnames is not None else (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as f:
        if not cols:
            return
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _collect_fid_rows(run_root: Path) -> list[dict]:
    """Collect per-run FID/KID/IS rows with run metadata attached."""
    out: list[dict] = []
    for eval_file in run_root.glob("*/**/seed*/eval/fid_vs_nfe.csv"):
        rel = eval_file.relative_to(run_root)
        if len(rel.parts) < 5:
            continue
        dataset, model_id, seed_name = rel.parts[0], rel.parts[1], rel.parts[2]
        seed = _parse_seed(seed_name)
        for row in read_csv(eval_file):
            out.append(
                {
                    "dataset": dataset,
                    "model_id": model_id,
                    "seed": seed,
                    "sampler": str(row.get("sampler", "heun")),
                    "nfe": int(row.get("nfe", "0")),
                    "fid": _safe_float(row.get("fid")),
                    "kid": _safe_float(row.get("kid")),
                    "is_mean": _safe_float(row.get("is_mean")),
                    "is_std": _safe_float(row.get("is_std")),
                    "latency_sec": _safe_float(row.get("latency_sec")),
                    "latency_per_step_ms": _safe_float(row.get("latency_per_step_ms")),
                }
            )
    return out


def _collect_integrability_rows(run_root: Path) -> list[dict]:
    """Collect long-format integrability rows with run metadata attached."""
    out: list[dict] = []
    for integ_file in run_root.glob("*/**/seed*/eval/integrability_vs_sigma.csv"):
        rel = integ_file.relative_to(run_root)
        if len(rel.parts) < 5:
            continue
        dataset, model_id, seed_name = rel.parts[0], rel.parts[1], rel.parts[2]
        seed = _parse_seed(seed_name)
        for row in read_csv(integ_file):
            out.append(
                {
                    "dataset": dataset,
                    "model_id": model_id,
                    "seed": seed,
                    "sigma_bin": int(row.get("bin", "-1")),
                    "metric_name": str(row.get("metric_name", "")),
                    "scale_delta": str(row.get("scale_delta", "")),
                    "cycle_len": str(row.get("cycle_len", "")),
                    "value": _safe_float(row.get("value")),
                }
            )
    return out


def _collect_compute_hours(run_root: Path) -> dict[tuple[str, str, int], float]:
    """Load ``approx_gpu_hours`` from per-run compute summaries."""
    out: dict[tuple[str, str, int], float] = {}
    for summary_file in run_root.glob("*/**/seed*/eval/compute_summary.json"):
        rel = summary_file.relative_to(run_root)
        if len(rel.parts) < 5:
            continue
        dataset, model_id, seed_name = rel.parts[0], rel.parts[1], rel.parts[2]
        seed = _parse_seed(seed_name)
        with summary_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        out[(dataset, model_id, seed)] = _safe_float(payload.get("approx_gpu_hours"))
    return out


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return mean/std for finite values; NaN when no finite input exists."""
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan"), float("nan")
    arr = np.asarray(finite, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def summarize_fid_rows(fid_rows: list[dict]) -> list[dict]:
    """Aggregate FID rows by dataset/model/sampler/NFE."""
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in fid_rows:
        if not math.isfinite(row["fid"]):
            continue
        key = (row["dataset"], row["model_id"], row["sampler"], int(row["nfe"]))
        grouped[key].append(float(row["fid"]))

    out: list[dict] = []
    for (dataset, model_id, sampler, nfe), values in sorted(grouped.items()):
        mean, std = _mean_std(values)
        out.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "sampler": sampler,
                "nfe": int(nfe),
                "fid_mean": mean,
                "fid_std": std,
                "num_seeds": len(values),
            }
        )
    return out


def summarize_integrability_rows(integ_rows: list[dict]) -> list[dict]:
    """Aggregate integrability long-table rows across seeds."""
    grouped: dict[tuple[str, str, int, str, str, str], list[float]] = defaultdict(list)
    for row in integ_rows:
        if not row["metric_name"] or not math.isfinite(row["value"]):
            continue
        key = (
            row["dataset"],
            row["model_id"],
            int(row["sigma_bin"]),
            row["metric_name"],
            row["scale_delta"],
            row["cycle_len"],
        )
        grouped[key].append(float(row["value"]))

    out: list[dict] = []
    for (dataset, model_id, sigma_bin, metric_name, scale_delta, cycle_len), values in sorted(grouped.items()):
        mean, std = _mean_std(values)
        out.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "sigma_bin": int(sigma_bin),
                "metric_name": metric_name,
                "scale_delta": scale_delta,
                "cycle_len": cycle_len,
                "value_mean": mean,
                "value_std": std,
                "num_points": len(values),
            }
        )
    return out


def steps_to_target_fid_rows(
    fid_rows: list[dict],
    targets: list[float],
    sampler: str,
) -> tuple[list[dict], list[dict]]:
    """Compute minimal NFE needed to reach each target FID threshold."""
    by_run: dict[tuple[str, str, int, str], list[dict]] = defaultdict(list)
    for row in fid_rows:
        if sampler != "all" and row["sampler"] != sampler:
            continue
        by_run[(row["dataset"], row["model_id"], int(row["seed"]), row["sampler"])].append(row)

    raw_rows: list[dict] = []
    for (dataset, model_id, seed, sampler_name), rows in sorted(by_run.items()):
        rows_sorted = sorted(rows, key=lambda r: int(r["nfe"]))
        for target in targets:
            reached = [r for r in rows_sorted if math.isfinite(r["fid"]) and float(r["fid"]) <= float(target)]
            if reached:
                best = reached[0]
                nfe = int(best["nfe"])
                fid_at_target = float(best["fid"])
                latency_sec = float(best["latency_sec"]) if math.isfinite(best["latency_sec"]) else float("nan")
            else:
                nfe = -1
                fid_at_target = float("nan")
                latency_sec = float("nan")
            raw_rows.append(
                {
                    "dataset": dataset,
                    "model_id": model_id,
                    "seed": int(seed),
                    "sampler": sampler_name,
                    "target_fid": float(target),
                    "steps_to_target_fid": int(nfe),
                    "fid_at_target_step": fid_at_target,
                    "latency_sec_at_target_step": latency_sec,
                }
            )

    grouped_steps: dict[tuple[str, str, str, float], list[float]] = defaultdict(list)
    grouped_hits: dict[tuple[str, str, str, float], int] = defaultdict(int)
    for row in raw_rows:
        key = (row["dataset"], row["model_id"], row["sampler"], float(row["target_fid"]))
        if row["steps_to_target_fid"] >= 0:
            grouped_steps[key].append(float(row["steps_to_target_fid"]))
            grouped_hits[key] += 1

    summary_rows: list[dict] = []
    for key in sorted(set(grouped_steps.keys()) | set(grouped_hits.keys())):
        dataset, model_id, sampler_name, target = key
        values = grouped_steps.get(key, [])
        mean, std = _mean_std(values)
        summary_rows.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "sampler": sampler_name,
                "target_fid": float(target),
                "steps_to_target_fid_mean": mean,
                "steps_to_target_fid_std": std,
                "num_success": int(grouped_hits.get(key, 0)),
            }
        )

    return raw_rows, summary_rows


def compute_matched_fid_rows(
    fid_rows: list[dict],
    compute_hours: dict[tuple[str, str, int], float],
    sampler: str,
) -> tuple[list[dict], list[dict]]:
    """Compute FID under shared training-compute budget per dataset."""
    model_hours: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (dataset, model_id, _seed), hours in compute_hours.items():
        if math.isfinite(hours):
            model_hours[(dataset, model_id)].append(hours)

    budgets: dict[str, float] = {}
    budget_rows: list[dict] = []
    dataset_to_models: dict[str, list[str]] = defaultdict(list)
    for (dataset, model_id), hours_list in model_hours.items():
        if hours_list:
            dataset_to_models[dataset].append(model_id)

    for dataset, models in sorted(dataset_to_models.items()):
        per_model_mean = {m: float(np.mean(model_hours[(dataset, m)])) for m in sorted(set(models))}
        budget = min(per_model_mean.values())
        budgets[dataset] = budget
        for model_id, mean_h in per_model_mean.items():
            budget_rows.append(
                {
                    "dataset": dataset,
                    "model_id": model_id,
                    "model_mean_gpu_hours": float(mean_h),
                    "shared_budget_gpu_hours": float(budget),
                }
            )

    by_group: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)
    for row in fid_rows:
        if sampler != "all" and row["sampler"] != sampler:
            continue
        if not math.isfinite(row["fid"]):
            continue
        key = (row["dataset"], row["model_id"], row["sampler"], int(row["nfe"]))
        by_group[key].append(row)

    matched_rows: list[dict] = []
    for (dataset, model_id, sampler_name, nfe), rows in sorted(by_group.items()):
        budget = budgets.get(dataset)
        if budget is None:
            continue

        run_rows = []
        for r in rows:
            hours = compute_hours.get((dataset, model_id, int(r["seed"])), float("nan"))
            if not math.isfinite(hours):
                continue
            run_rows.append({**r, "gpu_hours": hours})
        if not run_rows:
            continue

        within = [r for r in run_rows if r["gpu_hours"] <= budget]
        if within:
            selected = within
            selection_mode = "within_budget"
        else:
            min_over = min(r["gpu_hours"] for r in run_rows)
            selected = [r for r in run_rows if abs(r["gpu_hours"] - min_over) < 1.0e-12]
            selection_mode = "closest_over_budget"

        fid_vals = [float(r["fid"]) for r in selected]
        fid_mean, fid_std = _mean_std(fid_vals)
        matched_rows.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "sampler": sampler_name,
                "nfe": int(nfe),
                "shared_budget_gpu_hours": float(budget),
                "selection_mode": selection_mode,
                "selected_run_count": int(len(selected)),
                "selected_gpu_hours_mean": float(np.mean([r["gpu_hours"] for r in selected])),
                "fid_mean": fid_mean,
                "fid_std": fid_std,
            }
        )

    return budget_rows, matched_rows


def parse_args() -> argparse.Namespace:
    """Parse CLI options for report table generation."""
    p = argparse.ArgumentParser(description="Aggregate run artifacts into report tables")
    p.add_argument("--run_root", type=str, default="runs")
    p.add_argument("--out_dir", type=str, default="reports")
    p.add_argument("--target_fids", type=str, default="5,10,20,50")
    p.add_argument("--steps_sampler", type=str, default="heun", help='Sampler for steps-to-target ("heun" or "all")')
    p.add_argument("--compute_sampler", type=str, default="heun", help='Sampler for compute-matched FID ("heun" or "all")')
    return p.parse_args()


def main() -> None:
    """Generate all report CSV files from experiment run artifacts."""
    args = parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    targets = _parse_float_list(args.target_fids)

    fid_rows = _collect_fid_rows(run_root)
    integ_rows = _collect_integrability_rows(run_root)
    compute_hours = _collect_compute_hours(run_root)

    fid_summary = summarize_fid_rows(fid_rows)
    integ_summary = summarize_integrability_rows(integ_rows)
    steps_raw, steps_summary = steps_to_target_fid_rows(fid_rows, targets=targets, sampler=str(args.steps_sampler))
    compute_budget_rows, compute_matched_rows = compute_matched_fid_rows(
        fid_rows, compute_hours=compute_hours, sampler=str(args.compute_sampler)
    )

    fid_path = out_dir / "fid_summary.csv"
    integ_path = out_dir / "integrability_summary.csv"
    steps_raw_path = out_dir / "steps_to_target_fid.csv"
    steps_summary_path = out_dir / "steps_to_target_fid_summary.csv"
    budget_path = out_dir / "compute_budget_by_dataset.csv"
    compute_matched_path = out_dir / "compute_matched_fid.csv"

    write_table(
        fid_path,
        fid_summary,
        fieldnames=["dataset", "model_id", "sampler", "nfe", "fid_mean", "fid_std", "num_seeds"],
    )
    write_table(
        integ_path,
        integ_summary,
        fieldnames=[
            "dataset",
            "model_id",
            "sigma_bin",
            "metric_name",
            "scale_delta",
            "cycle_len",
            "value_mean",
            "value_std",
            "num_points",
        ],
    )
    write_table(
        steps_raw_path,
        steps_raw,
        fieldnames=[
            "dataset",
            "model_id",
            "seed",
            "sampler",
            "target_fid",
            "steps_to_target_fid",
            "fid_at_target_step",
            "latency_sec_at_target_step",
        ],
    )
    write_table(
        steps_summary_path,
        steps_summary,
        fieldnames=[
            "dataset",
            "model_id",
            "sampler",
            "target_fid",
            "steps_to_target_fid_mean",
            "steps_to_target_fid_std",
            "num_success",
        ],
    )
    write_table(
        budget_path,
        compute_budget_rows,
        fieldnames=["dataset", "model_id", "model_mean_gpu_hours", "shared_budget_gpu_hours"],
    )
    write_table(
        compute_matched_path,
        compute_matched_rows,
        fieldnames=[
            "dataset",
            "model_id",
            "sampler",
            "nfe",
            "shared_budget_gpu_hours",
            "selection_mode",
            "selected_run_count",
            "selected_gpu_hours_mean",
            "fid_mean",
            "fid_std",
        ],
    )

    print(str(fid_path))
    print(str(integ_path))
    print(str(steps_raw_path))
    print(str(steps_summary_path))
    print(str(budget_path))
    print(str(compute_matched_path))


if __name__ == "__main__":
    main()
