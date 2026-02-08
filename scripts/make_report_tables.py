from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_csv(path: Path):
    """Read CSV file into list of row dictionaries."""
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_runs(run_root: Path) -> tuple[list[dict], list[dict]]:
    """Aggregate run-level evaluation CSV files into summary tables.

    Args:
        run_root: Root directory containing run outputs.

    Returns:
        Tuple ``(fid_table, integrability_table)`` with mean/std aggregates.

    How it works:
        Scans run directories, groups metrics by dataset/model id/settings, and
        computes summary statistics across seeds.
    """
    by_group_fid = defaultdict(list)
    by_group_integrability = defaultdict(list)

    for eval_file in run_root.glob("*/**/seed*/eval/fid_vs_nfe.csv"):
        # runs/{dataset}/{model_id}/seed{n}/eval/fid_vs_nfe.csv
        rel = eval_file.relative_to(run_root)
        dataset = rel.parts[0]
        model_id = rel.parts[1]

        rows = read_csv(eval_file)
        for row in rows:
            key = (dataset, model_id, row.get("sampler", "heun"), row.get("nfe", "0"))
            fid = row.get("fid", "nan")
            if fid != "" and fid != "nan":
                by_group_fid[key].append(float(fid))

        integ_path = eval_file.parent / "integrability_vs_sigma.csv"
        if integ_path.exists():
            for row in read_csv(integ_path):
                name = str(row.get("metric_name", ""))
                value = row.get("value", "nan")
                if not name or value in {"", "nan"}:
                    continue
                key = (
                    dataset,
                    model_id,
                    row.get("bin", "-1"),
                    name,
                    row.get("scale_delta", ""),
                    row.get("cycle_len", ""),
                )
                by_group_integrability[key].append(float(value))

    fid_table = []
    for (dataset, model_id, sampler, nfe), values in sorted(by_group_fid.items()):
        fid_table.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "sampler": sampler,
                "nfe": int(nfe),
                "fid_mean": float(np.mean(values)),
                "fid_std": float(np.std(values)),
                "num_seeds": len(values),
            }
        )

    integrability_table = []
    for (dataset, model_id, bin_id, metric_name, scale_delta, cycle_len), values in sorted(
        by_group_integrability.items()
    ):
        integrability_table.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "sigma_bin": int(bin_id),
                "metric_name": metric_name,
                "scale_delta": scale_delta,
                "cycle_len": cycle_len,
                "value_mean": float(np.mean(values)),
                "value_std": float(np.std(values)),
                "num_points": len(values),
            }
        )

    return fid_table, integrability_table


def write_table(path: Path, rows: list[dict]) -> None:
    """Write summary rows to CSV if rows are non-empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Generate report CSV files from experiment run artifacts."""
    run_root = Path("runs")
    fid_table, integ_table = summarize_runs(run_root)
    write_table(Path("reports/fid_summary.csv"), fid_table)
    write_table(Path("reports/integrability_summary.csv"), integ_table)
    print("reports/fid_summary.csv")
    print("reports/integrability_summary.csv")


if __name__ == "__main__":
    main()
