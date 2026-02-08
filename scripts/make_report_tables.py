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
        Scans run directories, groups metrics by dataset/variant/settings, and
        computes summary statistics across seeds.
    """
    by_group = defaultdict(list)
    by_group_integrability = defaultdict(list)

    for eval_file in run_root.glob("*/**/seed*/eval/fid_vs_nfe.csv"):
        parts = eval_file.parts
        # runs/{dataset}/{variant}/seed{n}/eval/fid_vs_nfe.csv
        dataset = parts[-6]
        variant = parts[-5]
        _seed = parts[-4]

        rows = read_csv(eval_file)
        for row in rows:
            key = (dataset, variant, row["sampler"], row["nfe"])
            by_group[key].append(float(row["fid"]))

        integ_path = eval_file.parent / "integrability_vs_sigma.csv"
        if integ_path.exists():
            irows = read_csv(integ_path)
            for row in irows:
                key = (dataset, variant, row["bin"])
                if row.get("r_sym") and row["r_sym"] != "nan":
                    by_group_integrability[key].append((float(row["r_sym"]), float(row["r_loop"])))

    fid_table = []
    for (dataset, variant, sampler, nfe), vals in sorted(by_group.items()):
        fid_table.append(
            {
                "dataset": dataset,
                "variant": variant,
                "sampler": sampler,
                "nfe": int(nfe),
                "fid_mean": float(np.mean(vals)),
                "fid_std": float(np.std(vals)),
                "num_seeds": len(vals),
            }
        )

    integ_table = []
    for (dataset, variant, b), vals in sorted(by_group_integrability.items()):
        sym = [v[0] for v in vals]
        loop = [v[1] for v in vals]
        integ_table.append(
            {
                "dataset": dataset,
                "variant": variant,
                "sigma_bin": int(b),
                "r_sym_mean": float(np.mean(sym)),
                "r_sym_std": float(np.std(sym)),
                "r_loop_mean": float(np.mean(loop)),
                "r_loop_std": float(np.std(loop)),
                "num_points": len(vals),
            }
        )

    return fid_table, integ_table


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
