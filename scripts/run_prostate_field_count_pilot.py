import argparse
import csv
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dosim_sim.delivery3d import delivery_mode_3d, prostate_delivery_modes_3d
from dosim_sim.objective import PlanningPriorities
from dosim_sim.planning3d import HighLevelSearchConfig3D, is_acceptable_3d
from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D, optimize_fluence_3d_torch
from dosim_sim.volume3d import generate_prostate_case_3d


LABELS = {
    "static_4": "4-field IMRT",
    "static_7": "7-field IMRT",
    "static_9": "9-field IMRT",
    "static_12": "12-field IMRT",
    "arc_like_360": "36-point arc-like",
}


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def conformity_metrics(case, dose: np.ndarray) -> dict[str, float]:
    target_volume = float(case.target.sum())
    prescription = (dose >= 0.95) & case.body
    half_prescription = (dose >= 0.50) & case.body
    covered_target = float((prescription & case.target).sum())
    prescription_volume = float(prescription.sum())
    return {
        "target_v95_percent": 100.0 * covered_target / target_volume,
        "paddick_ci_95": covered_target**2 / max(target_volume * prescription_volume, 1.0),
        "rtog_ci_95": prescription_volume / target_volume,
        "r50": float(half_prescription.sum()) / target_volume,
        "body_mean_dose": float(dose[case.body].mean()),
    }


def cumulative_dvh(dose: np.ndarray, mask: np.ndarray, bins: np.ndarray) -> np.ndarray:
    values = dose[mask]
    return np.asarray([100.0 * np.mean(values >= level) for level in bins])


def save_summary(rows: list[dict], modes, path: Path) -> None:
    names = [mode.name for mode in modes]
    x = np.arange(len(names))
    labels = [LABELS[name] for name in names]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    metrics = (
        ("paddick_ci_95", "Paddick CI95", "Higher is better"),
        ("r50", "R50", "Lower is better"),
        ("target_d95", "PTV D95", "Coverage"),
        ("maximum_oar_ratio", "Maximum OAR mean / limit", "Lower is better"),
    )
    seeds = sorted({int(row["seed"]) for row in rows})
    for axis, (key, ylabel, title) in zip(axes.flat, metrics, strict=True):
        for seed in seeds:
            subset = [row for row in rows if int(row["seed"]) == seed]
            axis.plot(
                x,
                [float(next(row[key] for row in subset if row["mode"] == name)) for name in names],
                color="#888888",
                alpha=0.35,
            )
        medians = [np.median([float(row[key]) for row in rows if row["mode"] == name]) for name in names]
        axis.plot(x, medians, color="#d62728", marker="o", linewidth=2.5, label="cohort median")
        axis.set_xticks(x, labels, rotation=15, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        if key == "maximum_oar_ratio":
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    figure.suptitle("Prostate field-count pilot: identical cases and objective")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_case_review(case, plans: dict[str, object], output_dir: Path) -> None:
    modes = list(plans)
    colors = {
        "PTV": "#d62728",
        "bladder": "#1f77b4",
        "rectum": "#2ca02c",
        "femoral heads": "#9467bd",
        "normal body": "#6b6b6b",
    }
    masks = {
        "PTV": case.target,
        "bladder": case.oars[0],
        "rectum": case.oars[1],
        "femoral heads": case.oars[2],
        "normal body": case.body & ~case.target,
    }
    doses = {name: plan.dose.detach().float().cpu().numpy() for name, plan in plans.items()}
    maximum = max(float(dose.max()) for dose in doses.values())
    bins = np.linspace(0.0, max(1.5, np.ceil(maximum * 20) / 20), 181)
    figure, axes = plt.subplots(1, len(modes), figsize=(19, 4.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for axis, mode in zip(axes, modes, strict=True):
        for structure, mask in masks.items():
            axis.plot(bins, cumulative_dvh(doses[mode], mask, bins), color=colors[structure], linewidth=1.8, label=structure)
        axis.axvline(1.0, color="black", linestyle="--", linewidth=1)
        conformity = conformity_metrics(case, doses[mode])
        axis.text(
            0.97,
            0.96,
            f"CI95 {conformity['paddick_ci_95']:.2f}\nR50 {conformity['r50']:.1f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
        )
        axis.set_title(LABELS[mode])
        axis.set_xlabel("Relative dose")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Volume receiving at least dose (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=5, frameon=False)
    figure.suptitle(f"Field-count DVHs: {case.difficulty} case {case.case_id}", y=0.995)
    figure.subplots_adjust(left=0.05, right=0.995, bottom=0.13, top=0.80, wspace=0.06)
    figure.savefig(output_dir / "02_representative_dvhs.png", dpi=180)
    plt.close(figure)

    axial_index = int(np.argmax(case.target.sum(axis=(0, 1))))
    figure, axes = plt.subplots(1, len(modes), figsize=(19, 4.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, mode in zip(axes, modes, strict=True):
        image = axis.imshow(doses[mode][:, :, axial_index].T, origin="lower", cmap="turbo", vmin=0.0, vmax=maximum)
        axis.contour(case.target[:, :, axial_index].T, levels=[0.5], colors=["white"], linewidths=1.3)
        for color, oar in zip(("cyan", "lime", "magenta"), case.oars, strict=True):
            axis.contour(oar[:, :, axial_index].T, levels=[0.5], colors=[color], linewidths=0.9)
        axis.set_title(LABELS[mode])
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(image, ax=axes, shrink=0.75, label="Relative dose")
    figure.suptitle(f"Field-count dose comparison: {case.difficulty} case {case.case_id}")
    figure.savefig(output_dir / "03_representative_dose.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired prostate field-count and arc-like complexity pilot")
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/prostate300_local/merged"))
    parser.add_argument("--cases-per-stratum", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--fluence-size", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--modes", nargs="+", choices=("static_4", "static_7", "static_9", "static_12", "arc_like_360"))
    parser.add_argument("--normal-tissue-weight", type=float, default=0.0)
    parser.add_argument("--normal-tissue-threshold", type=float, default=0.5)
    parser.add_argument("--integral-dose-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prostate_field_count_pilot"))
    args = parser.parse_args()

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    records = [row for row in load_records(args.dataset_dir / "trajectory_view.jsonl") if row["split"] == "validation"]
    selected = [
        row
        for difficulty in ("easy", "moderate", "hard")
        for row in [value for value in records if value["difficulty"] == difficulty][: args.cases_per_stratum]
    ]
    modes = (
        tuple(delivery_mode_3d(name) for name in args.modes)
        if args.modes
        else prostate_delivery_modes_3d()
    )
    rows = []
    representative_plans = None
    representative_case = None
    total = len(selected) * len(modes)
    completed = 0
    for record in selected:
        case = generate_prostate_case_3d(int(record["seed"]), args.grid_size, difficulty=record["difficulty"])
        case_plans = {}
        for mode in modes:
            torch.cuda.reset_peak_memory_stats(device)
            engine = TorchImplicitDoseEngine3D(case, mode.angles_degrees, args.fluence_size, device=device, dtype=torch.float32)
            started = time.perf_counter()
            plan = optimize_fluence_3d_torch(
                case,
                engine,
                mode.active_beams,
                PlanningPriorities.for_case(case),
                iterations=args.iterations,
                normal_tissue_weight=args.normal_tissue_weight,
                normal_tissue_threshold=args.normal_tissue_threshold,
                integral_dose_weight=args.integral_dose_weight,
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            dose = plan.dose.detach().float().cpu().numpy()
            conformity = conformity_metrics(case, dose)
            maximum_oar_ratio = max(value / limit for value, limit in zip(plan.metrics.oar_mean, case.oar_limits, strict=True))
            rows.append({
                "case_id": case.case_id,
                "seed": case.seed,
                "difficulty": case.difficulty,
                "mode": mode.name,
                "field_count": len(mode.angles_degrees),
                "arc_like": mode.arc_like,
                "angles_degrees": "|".join(f"{value:.3f}" for value in mode.angles_degrees),
                "acceptable_synthetic_rules": is_acceptable_3d(plan.metrics, case, HighLevelSearchConfig3D()),
                "target_d95": plan.metrics.target_d95,
                "target_d02": plan.metrics.target_d02,
                "oar_0_mean": plan.metrics.oar_mean[0],
                "oar_1_mean": plan.metrics.oar_mean[1],
                "oar_2_mean": plan.metrics.oar_mean[2],
                "maximum_oar_ratio": maximum_oar_ratio,
                **conformity,
                "elapsed_seconds": elapsed,
                "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            })
            if case.difficulty == "moderate" and representative_plans is None:
                case_plans[mode.name] = plan
            completed += 1
            print(f"[{completed:02d}/{total}] {case.case_id} {mode.name}", flush=True)
            del engine
        if case.difficulty == "moderate" and representative_plans is None:
            representative_case = case
            representative_plans = case_plans

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    save_summary(rows, modes, args.output_dir / "01_field_count_summary.png")
    if representative_case is not None and representative_plans is not None:
        save_case_review(representative_case, representative_plans, args.output_dir)
    summary = {
        "status": "paired field-count engineering pilot; arc-like mode is not delivery-realistic VMAT",
        "cases": len(selected),
        "grid_size": args.grid_size,
        "fluence_size": args.fluence_size,
        "iterations": args.iterations,
        "normal_tissue_weight": args.normal_tissue_weight,
        "normal_tissue_threshold": args.normal_tissue_threshold,
        "integral_dose_weight": args.integral_dose_weight,
        "modes": {
            mode.name: {
                "field_count": len(mode.angles_degrees),
                "median_paddick_ci_95": float(np.median([row["paddick_ci_95"] for row in rows if row["mode"] == mode.name])),
                "median_r50": float(np.median([row["r50"] for row in rows if row["mode"] == mode.name])),
                "median_target_d95": float(np.median([row["target_d95"] for row in rows if row["mode"] == mode.name])),
                "median_maximum_oar_ratio": float(np.median([row["maximum_oar_ratio"] for row in rows if row["mode"] == mode.name])),
                "synthetic_acceptability_rate": float(np.mean([row["acceptable_synthetic_rules"] for row in rows if row["mode"] == mode.name])),
            }
            for mode in modes
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
