"""Import and review a TCIA prostate anatomy cohort before dose planning."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from dosim_sim.clinical3d import load_tcia_prostate_case
from render_tcia_prostate_case import DISPLAY_COLORS, overlap_display
from dosim_sim.visuals3d import add_hfs_orientation_labels
from run_prostate_clinical_dvh_pilot import STATUS_PAGE, should_update_figures, write_progress


def overlap_metrics(case) -> dict[str, float | int | str]:
    ctv = case.clinical_target
    if ctv is None:
        raise ValueError(f"{case.case_id}: clinical target is absent")
    if not np.all(~ctv | case.target):
        raise ValueError(f"{case.case_id}: PTV does not contain the prostate/CTV")
    margin = case.target & ~ctv
    target_count = int(np.count_nonzero(case.target))
    margin_count = int(np.count_nonzero(margin))
    record: dict[str, float | int | str] = {
        "case_id": case.case_id,
        "patient_id": case.case_id.removeprefix("tcia-"),
        "body_voxels": int(np.count_nonzero(case.body)),
        "prostate_ctv_voxels": int(np.count_nonzero(ctv)),
        "ptv_voxels": target_count,
        "ptv_margin_voxels": margin_count,
        "all_required_masks_nonempty": bool(
            case.body.any()
            and ctv.any()
            and case.target.any()
            and all(mask.any() for mask in case.oars)
        ),
    }
    for structure in ("bladder", "rectum"):
        index = case.structure_names.index(structure)
        oar = case.oars[index]
        overlap = case.target & oar
        overlap_count = int(np.count_nonzero(overlap))
        record[f"{structure}_oar_overlap_fraction"] = overlap_count / max(int(np.count_nonzero(oar)), 1)
        record[f"{structure}_ptv_overlap_fraction"] = overlap_count / max(target_count, 1)
        record[f"{structure}_margin_overlap_fraction"] = overlap_count / max(margin_count, 1)
        record[f"{structure}_ctv_overlap_voxels"] = int(np.count_nonzero(ctv & oar))
        record[f"{structure}_ctv_overlap_fraction"] = float(np.count_nonzero(ctv & oar)) / max(
            int(np.count_nonzero(ctv)), 1
        )
        record[f"{structure}_outside_body_voxels"] = int(np.count_nonzero(oar & ~case.body))
    record["maximum_ptv_oar_overlap_fraction"] = max(
        float(record["bladder_ptv_overlap_fraction"]),
        float(record["rectum_ptv_overlap_fraction"]),
    )
    record["margin_only_primary_eligible"] = bool(
        int(record["bladder_ctv_overlap_voxels"]) == 0
        and int(record["rectum_ctv_overlap_voxels"]) == 0
    )
    return record


def overlap_slice(case) -> int:
    overlap = case.target & (case.oars[0] | case.oars[1])
    counts = overlap.sum(axis=(0, 1))
    if counts.max() > 0:
        return int(np.argmax(counts))
    return int(np.rint(np.argwhere(case.target).mean(axis=0))[2])


def save_cohort_montage(cases: list, records: list[dict], path: Path) -> None:
    maximum_panels = 48
    if len(cases) > maximum_panels:
        selected_indices = np.rint(np.linspace(0, len(cases) - 1, maximum_panels)).astype(int)
        displayed_cases = [cases[index] for index in selected_indices]
        displayed_records = [records[index] for index in selected_indices]
    else:
        displayed_cases = cases
        displayed_records = records
    columns = 4
    rows = int(np.ceil(len(displayed_cases) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.8 * rows), squeeze=False)
    for axis, case, record in zip(axes.flat, displayed_cases, displayed_records, strict=False):
        index = overlap_slice(case)
        axis.imshow(overlap_display(case, 2, index), origin="lower")
        add_hfs_orientation_labels(axis, "axial")
        bladder = 100.0 * float(record["bladder_ptv_overlap_fraction"])
        rectum = 100.0 * float(record["rectum_ptv_overlap_fraction"])
        axis.set_title(f"{record['patient_id']}\nPTV overlap: bladder {bladder:.1f}%, rectum {rectum:.1f}%", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(displayed_cases) :]:
        axis.axis("off")
    figure.legend(
        handles=[
            Patch(color=DISPLAY_COLORS["clinical_target"], label="Prostate/CTV"),
            Patch(color=DISPLAY_COLORS["ptv_margin"], label="5 mm PTV margin"),
            Patch(color=DISPLAY_COLORS["bladder"], label="Bladder"),
            Patch(color=DISPLAY_COLORS["rectum"], label="Rectum"),
            Patch(color=DISPLAY_COLORS["bladder_overlap"], label="PTV-bladder overlap"),
            Patch(color=DISPLAY_COLORS["rectum_overlap"], label="PTV-rectum overlap"),
        ],
        loc="outside lower center",
        ncol=3,
    )
    figure.suptitle(
        f"TCIA prostate anatomy cohort before dose planning: "
        f"{len(displayed_cases)} of {len(cases)} valid cases shown"
    )
    figure.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def save_overlap_review(records: list[dict], path: Path) -> None:
    records = sorted(
        records,
        key=lambda record: (
            float(record["maximum_ptv_oar_overlap_fraction"]),
            str(record["patient_id"]),
        ),
    )
    labels = [str(record["patient_id"]).replace("Prostate-", "") for record in records]
    x = np.arange(len(records))
    bladder = 100.0 * np.asarray([record["bladder_ptv_overlap_fraction"] for record in records])
    rectum = 100.0 * np.asarray([record["rectum_ptv_overlap_fraction"] for record in records])
    ctv_overlap = np.asarray(
        [record["bladder_ctv_overlap_voxels"] + record["rectum_ctv_overlap_voxels"] for record in records]
    )
    figure, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, constrained_layout=True)
    axes[0].bar(x - 0.18, bladder, 0.36, label="Bladder", color="#4c78a8")
    axes[0].bar(x + 0.18, rectum, 0.36, label="Rectum", color="#59a14f")
    axes[0].set_ylabel("Overlap / PTV volume (%)")
    axes[0].set_title("PTV overlap is expected at the bladder and rectum interfaces")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(x, ctv_overlap, color="#e15759")
    axes[1].axhline(0, color="#444444", linewidth=0.8)
    axes[1].set_ylabel("Prostate/CTV overlap (voxels)")
    axes[1].set_xlabel("TCIA patient")
    tick_count = min(len(records), 16)
    tick_indices = np.unique(np.rint(np.linspace(0, len(records) - 1, tick_count)).astype(int))
    axes[1].set_xticks(tick_indices, [labels[index] for index in tick_indices], rotation=45, ha="right")
    axes[1].set_title("CTV–OAR overlap should be absent or limited to rasterization interfaces")
    axes[1].grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def save_representative_planes(cases: list, records: list[dict], path: Path) -> None:
    order = np.argsort([record["maximum_ptv_oar_overlap_fraction"] for record in records])
    selected_indices = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    figure, axes = plt.subplots(3, 3, figsize=(11, 10.5), constrained_layout=True)
    views = ((2, "axial"), (1, "coronal"), (0, "sagittal"))
    for row, case_index in enumerate(selected_indices):
        case = cases[case_index]
        record = records[case_index]
        center = np.rint(np.argwhere(case.target).mean(axis=0)).astype(int)
        overlap = case.target & (case.oars[0] | case.oars[1])
        if overlap.any():
            center = np.rint(np.argwhere(overlap).mean(axis=0)).astype(int)
        for column, (axis_index, view_name) in enumerate(views):
            axes[row, column].imshow(
                overlap_display(case, axis_index, int(center[axis_index])),
                origin="lower",
            )
            add_hfs_orientation_labels(axes[row, column], view_name)
            axes[row, column].set_title(
                f"{record['patient_id']} | {view_name}\n"
                f"maximum PTV overlap {100.0 * float(record['maximum_ptv_oar_overlap_fraction']):.1f}%",
                fontsize=9,
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.suptitle("Low, median, and high PTV–OAR overlap cases")
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review TCIA prostate anatomy before planning")
    parser.add_argument("--tcia-root", type=Path, default=Path("data/tcia"))
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--maximum-figure-updates", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.maximum_figure_updates <= 0:
        raise ValueError("Maximum figure updates must be positive")
    subject_dirs = (
        [args.tcia_root / name for name in args.subjects]
        if args.subjects
        else sorted(path for path in args.tcia_root.glob("Prostate-AEC-*") if path.is_dir())[: args.cases]
    )
    if not subject_dirs:
        raise ValueError("No TCIA subject directories were found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "status.html").write_text(STATUS_PAGE, encoding="utf-8")
    started = time.perf_counter()
    write_progress(args.output_dir, 0, len(subject_dirs), started, unit="anatomy cases")
    cases = []
    records = []
    exclusions = []
    for index, subject_dir in enumerate(subject_dirs, start=1):
        try:
            case = load_tcia_prostate_case(subject_dir, args.grid_size)
            record = overlap_metrics(case)
        except (ValueError, RuntimeError) as error:
            exclusions.append({"patient_id": subject_dir.name, "reason": str(error)})
            print(f"[{index}/{len(subject_dirs)}] excluded {subject_dir.name}: {error}", flush=True)
            last_case = f"excluded {subject_dir.name}"
        else:
            cases.append(case)
            records.append(record)
            if should_update_figures(
                index,
                len(subject_dirs),
                maximum_updates=args.maximum_figure_updates,
            ):
                save_cohort_montage(cases, records, args.output_dir / "00_selected_anatomy.png")
            print(f"[{index}/{len(subject_dirs)}] {case.case_id}", flush=True)
            last_case = case.case_id
        write_progress(
            args.output_dir,
            index,
            len(subject_dirs),
            started,
            last_case=last_case,
            unit="anatomy cases",
        )
    if not cases:
        raise ValueError("All requested TCIA cases failed anatomy import")
    save_cohort_montage(cases, records, args.output_dir / "00_selected_anatomy.png")
    save_overlap_review(records, args.output_dir / "01_tcia_overlap_review.png")
    save_representative_planes(cases, records, args.output_dir / "02_tcia_representative_planes.png")
    fieldnames = list(records[0])
    with (args.output_dir / "tcia_anatomy_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "status": "complete",
        "requested_cases": len(subject_dirs),
        "valid_cases": len(records),
        "excluded_cases": len(exclusions),
        "exclusions": exclusions,
        "all_required_masks_nonempty": all(record["all_required_masks_nonempty"] for record in records),
        "patients_with_ctv_oar_interface_voxels": sum(
            int(record["bladder_ctv_overlap_voxels"] + record["rectum_ctv_overlap_voxels"] > 0)
            for record in records
        ),
        "margin_only_primary_cases": sum(
            int(record["margin_only_primary_eligible"]) for record in records
        ),
        "margin_only_primary_patient_ids": [
            record["patient_id"] for record in records if record["margin_only_primary_eligible"]
        ],
        "bladder_ptv_overlap_fraction_range": [
            float(min(record["bladder_ptv_overlap_fraction"] for record in records)),
            float(max(record["bladder_ptv_overlap_fraction"] for record in records)),
        ],
        "rectum_ptv_overlap_fraction_range": [
            float(min(record["rectum_ptv_overlap_fraction"] for record in records)),
            float(max(record["rectum_ptv_overlap_fraction"] for record in records)),
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "exclusions.json").write_text(
        json.dumps(exclusions, indent=2) + "\n", encoding="utf-8"
    )
    write_progress(
        args.output_dir,
        len(subject_dirs),
        len(subject_dirs),
        started,
        status="complete",
        last_case=cases[-1].case_id,
        unit="anatomy cases",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
