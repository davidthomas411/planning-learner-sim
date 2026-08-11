"""Select a fixed, unique TCIA cohort after anatomy quality control."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_prostate_clinical_dvh_pilot import STATUS_PAGE


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def uniformly_select(records: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0:
        return []
    if count > len(records):
        raise ValueError(f"Cannot select {count} records from {len(records)} candidates")
    ordered = sorted(
        records,
        key=lambda row: (
            float(row["maximum_ptv_oar_overlap_fraction"]),
            row["patient_id"],
        ),
    )
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int)
    if len(set(indices.tolist())) != count:
        raise RuntimeError("Uniform selection produced duplicate row indices")
    return [ordered[index] for index in indices]


def save_selection_figure(records: list[dict[str, str]], path: Path) -> None:
    margin_only = [row for row in records if parse_bool(row["margin_only_primary_eligible"])]
    interface = [row for row in records if not parse_bool(row["margin_only_primary_eligible"])]
    figure, axis = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
    for values, label, color, marker in (
        (margin_only, "Margin-only", "#4c78a8", "o"),
        (interface, "CTV-OAR interface overlap", "#e15759", "x"),
    ):
        axis.scatter(
            [100.0 * float(row["bladder_ptv_overlap_fraction"]) for row in values],
            [100.0 * float(row["rectum_ptv_overlap_fraction"]) for row in values],
            label=f"{label} (n={len(values)})",
            color=color,
            marker=marker,
            alpha=0.8,
        )
    axis.set_xlabel("PTV overlap with bladder (% of PTV)")
    axis.set_ylabel("PTV overlap with rectum (% of PTV)")
    axis.set_title("Selected TCIA planning cohort")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a TCIA planning cohort after anatomy review")
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("The requested case count must be positive")

    with args.metrics_csv.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    patient_ids = [row["patient_id"] for row in records]
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("The anatomy metrics contain duplicate patient identifiers")
    if len(records) < args.count:
        raise ValueError(
            f"Only {len(records)} valid unique patients are available; {args.count} were requested"
        )

    margin_only = [row for row in records if parse_bool(row["margin_only_primary_eligible"])]
    interface = [row for row in records if not parse_bool(row["margin_only_primary_eligible"])]
    selected_margin = uniformly_select(margin_only, min(len(margin_only), args.count))
    selected_interface = uniformly_select(interface, args.count - len(selected_margin))
    selected = selected_margin + selected_interface
    if len({row["patient_id"] for row in selected}) != args.count:
        raise RuntimeError("The selected cohort does not contain the requested number of unique patients")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "status.html").write_text(STATUS_PAGE, encoding="utf-8")
    selected = sorted(selected, key=lambda row: row["patient_id"])
    fieldnames = ["selection_order", "anatomy_stratum", *selected[0].keys()]
    output_rows = []
    for index, row in enumerate(selected, start=1):
        output_rows.append(
            {
                "selection_order": index,
                "anatomy_stratum": (
                    "margin_only"
                    if parse_bool(row["margin_only_primary_eligible"])
                    else "interface_overlap"
                ),
                **row,
            }
        )
    with (args.output_dir / "cohort_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    (args.output_dir / "selected_subjects.txt").write_text(
        "\n".join(row["patient_id"] for row in output_rows) + "\n",
        encoding="utf-8",
    )
    save_selection_figure(output_rows, args.output_dir / "03_tcia_selected_cohort.png")
    summary = {
        "status": "complete",
        "requested_cases": args.count,
        "available_valid_cases": len(records),
        "selected_unique_cases": len(output_rows),
        "margin_only_cases": len(selected_margin),
        "interface_overlap_cases": len(selected_interface),
        "selection_rule": (
            "all available margin-only cases, then uniform sampling of interface-overlap "
            "cases across maximum PTV-OAR overlap"
        ),
        "metrics_csv": str(args.metrics_csv.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    progress = {
        "status": "complete",
        "completed": len(output_rows),
        "total": len(output_rows),
        "percent_complete": 100.0,
        "elapsed_seconds": 0.0,
        "estimated_seconds_remaining": 0.0,
        "last_case": output_rows[-1]["patient_id"],
        "unit": "selected patients",
    }
    (args.output_dir / "progress.json").write_text(
        json.dumps(progress, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
