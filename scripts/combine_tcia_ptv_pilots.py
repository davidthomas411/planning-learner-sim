"""Combine completed TCIA PTV planning pilots into one cohort summary."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_prostate_clinical_dvh_pilot import STATUS_PAGE, write_progress


def read_rows(path: Path) -> list[dict[str, str]]:
    with (path / "trajectory_steps.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def final_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    return [max(values, key=lambda row: int(row["step"])) for values in by_case.values()]


def patient_label(case_id: str) -> str:
    return case_id.replace("tcia-Prostate-", "")


def save_outcomes(rows: list[dict[str, str]], path: Path) -> None:
    labels = [patient_label(row["case_id"]) for row in rows]
    changes = np.asarray([int(row["step"]) for row in rows])
    acceptable = np.asarray([row["acceptable"].lower() == "true" for row in rows])
    colors = np.where(acceptable, "#59a14f", "#e15759")
    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    bars = axis.bar(np.arange(len(rows)), changes, color=colors)
    axis.set_xticks(np.arange(len(rows)), labels, rotation=35, ha="right")
    axis.set_ylabel("Manual priority changes")
    axis.set_title("TCIA margin-only planning cohort")
    axis.grid(axis="y", alpha=0.2)
    for bar, passed in zip(bars, acceptable, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            "Pass" if passed else "Fail",
            ha="center",
            va="bottom",
        )
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def save_metrics(rows: list[dict[str, str]], path: Path) -> None:
    labels = [patient_label(row["case_id"]) for row in rows]
    metrics = (
        (
            "PTV D98 / 57 Gy",
            [float(row["ptv_d98_gy"]) / 57.0 for row in rows],
            "minimum",
            (57.0 - 0.06) / 57.0,
        ),
        (
            "PTV D2 / 63 Gy",
            [float(row["ptv_d02_gy"]) / 63.0 for row in rows],
            "maximum",
            (63.0 + 0.06) / 63.0,
        ),
        (
            "57 Gy volume / PTV / 1.10",
            [float(row["covering_isodose_ratio_57gy"]) / 1.10 for row in rows],
            "maximum",
            1.0,
        ),
        (
            "Worst OAR goal ratio",
            [float(row["worst_oar_goal_ratio"]) for row in rows],
            "maximum",
            1.0,
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    x = np.arange(len(rows))
    for axis, (title, values, direction, cutoff) in zip(axes.flat, metrics, strict=True):
        values_array = np.asarray(values)
        passed = values_array >= cutoff if direction == "minimum" else values_array <= cutoff
        axis.bar(x, values_array, color=np.where(passed, "#59a14f", "#e15759"))
        axis.axhline(1.0, color="#333333", linewidth=1.0)
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.set_ylabel("Observed / limit")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        lower = min(0.75, float(values_array.min()) - 0.03)
        upper = max(1.10, float(values_array.max()) + 0.03)
        axis.set_ylim(lower, upper)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine TCIA PTV planning pilot outputs")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "status.html").write_text(STATUS_PAGE, encoding="utf-8")
    started = time.perf_counter()
    write_progress(args.output_dir, 0, len(args.inputs), started, unit="pilot folders")
    all_rows = []
    for index, input_dir in enumerate(args.inputs, start=1):
        all_rows.extend(read_rows(input_dir))
        write_progress(
            args.output_dir,
            index,
            len(args.inputs),
            started,
            last_case=input_dir.name,
            unit="pilot folders",
        )
    finals = final_rows(all_rows)
    if len({row["case_id"] for row in finals}) != len(finals):
        raise ValueError("Duplicate case identifiers remain after combination")
    finals.sort(key=lambda row: row["case_id"])
    with (args.output_dir / "trajectory_steps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    with (args.output_dir / "case_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(finals[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(finals)
    save_outcomes(finals, args.output_dir / "04_tcia_cohort_outcomes.png")
    save_metrics(finals, args.output_dir / "05_tcia_final_metrics.png")
    acceptable = [row["acceptable"].lower() == "true" for row in finals]
    summary = {
        "status": "complete",
        "cases": len(finals),
        "acceptable_cases": int(sum(acceptable)),
        "acceptable_rate": float(np.mean(acceptable)),
        "failed_case_ids": [row["case_id"] for row, passed in zip(finals, acceptable, strict=True) if not passed],
        "manual_changes_per_case": [int(row["step"]) for row in finals],
        "median_manual_changes": float(np.median([int(row["step"]) for row in finals])),
        "input_folders": [str(path.resolve()) for path in args.inputs],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_progress(
        args.output_dir,
        len(args.inputs),
        len(args.inputs),
        started,
        status="complete",
        last_case=f"{len(finals)} combined cases",
        unit="pilot folders",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
