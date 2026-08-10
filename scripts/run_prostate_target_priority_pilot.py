import argparse
import csv
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dosim_sim.delivery3d import delivery_mode_3d
from dosim_sim.objective import PlanningPriorities
from dosim_sim.prostate_protocol import (
    evaluate_prostate_60gy20fx,
    protocol_oar_max_ratios,
)
from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D, optimize_fluence_3d_torch
from dosim_sim.volume3d import generate_prostate_case_3d
from run_prostate_clinical_dvh_pilot import STATUS_PAGE, load_records, write_progress


def save_target_response(rows: list[dict], path: Path) -> None:
    priorities = sorted({row["target_priority"] for row in rows})
    case_ids = list(dict.fromkeys(row["case_id"] for row in rows))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for case_id in case_ids:
        selected = sorted((row for row in rows if row["case_id"] == case_id), key=lambda row: row["target_priority"])
        label = case_id.replace("prostate3d-", "")
        axes[0].plot(priorities, [row["target_d98_gy"] for row in selected], marker="o", label=label)
        axes[1].plot(priorities, [row["maximum_oar_variation_ratio"] for row in selected], marker="o", label=label)
    axes[0].axhline(58.8, color="black", linestyle="--", linewidth=1, label="variation goal")
    axes[0].axhline(60.0, color="black", linestyle=":", linewidth=1, label="per-protocol goal")
    axes[0].set_ylabel("PTV D98 (Gy)")
    axes[0].set_title("Coverage response")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Worst OAR value / variation limit")
    axes[1].set_title("OAR tradeoff")
    for axis in axes:
        axis.set_xlabel("Manual target-priority multiplier")
        axis.set_xticks(priorities, [f"{value:.2f}" for value in priorities])
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=6, frameon=False)
    figure.suptitle("Hard prostate cases: one manual target-priority change", y=1.12)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_pass_summary(rows: list[dict], path: Path) -> None:
    priorities = sorted({row["target_priority"] for row in rows})
    rates = [100.0 * np.mean([row["variation_acceptable"] for row in rows if row["target_priority"] == value]) for value in priorities]
    figure, axis = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    axis.bar(np.arange(len(priorities)), rates)
    axis.set_xticks(np.arange(len(priorities)), [f"{value:.2f}" for value in priorities])
    axis.set_xlabel("Manual target-priority multiplier")
    axis.set_ylabel("Hard plans meeting represented variation goals (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Target-priority calibration")
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test manual target-priority changes on hard prostate cases")
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/prostate300_local/merged"))
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--fluence-size", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--target-priorities", nargs="+", type=float, default=(1.0, 1.75, 3.0625, 5.359375))
    parser.add_argument("--clinical-dvh-weight", type=float, default=5.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prostate_target_priority_pilot"))
    args = parser.parse_args()

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "status.html").write_text(STATUS_PAGE, encoding="utf-8")
    records = [
        row for row in load_records(args.dataset_dir / "trajectory_view.jsonl")
        if row["split"] == "validation" and row["difficulty"] == "hard"
    ][: args.cases]
    mode = delivery_mode_3d("static_7")
    total = len(records) * len(args.target_priorities)
    completed = 0
    started = time.perf_counter()
    write_progress(args.output_dir, completed, total, started)
    rows: list[dict] = []
    for record in records:
        case = generate_prostate_case_3d(int(record["seed"]), args.grid_size, difficulty="hard")
        engine = TorchImplicitDoseEngine3D(case, mode.angles_degrees, args.fluence_size, device=device, dtype=torch.float32)
        for target_priority in args.target_priorities:
            priorities = PlanningPriorities(
                target=target_priority,
                hotspot=1.0,
                oars=(1.0, 1.0, 1.0),
                normal_tissue=1.0,
            )
            plan = optimize_fluence_3d_torch(
                case,
                engine,
                mode.active_beams,
                priorities,
                iterations=args.iterations,
                normal_tissue_weight=50.0,
                normal_tissue_threshold=0.5,
                integral_dose_weight=2.0,
                clinical_dvh_weight=args.clinical_dvh_weight,
            )
            evaluation = evaluate_prostate_60gy20fx(case, plan.dose.detach().float().cpu().numpy())
            oar_ratios = protocol_oar_max_ratios(case, evaluation, "variation_acceptable")
            rows.append({
                "case_id": case.case_id,
                "seed": case.seed,
                "target_priority": target_priority,
                "variation_acceptable": evaluation.variation_acceptable,
                "target_variation_acceptable": evaluation.target_variation_acceptable,
                "oars_variation_acceptable": evaluation.oars_variation_acceptable,
                "target_d98_gy": evaluation.target_d98_gy,
                "target_d99_gy": evaluation.target_d99_gy,
                "target_d02_gy": evaluation.target_d02_gy,
                "maximum_oar_variation_ratio": max(oar_ratios),
                "bladder_variation_ratio": oar_ratios[0],
                "rectum_variation_ratio": oar_ratios[1],
                "femoral_heads_variation_ratio": oar_ratios[2],
                "paddick_ci_95": plan.metrics.paddick_ci_95,
                "r50": plan.metrics.r50,
            })
            completed += 1
            write_progress(
                args.output_dir,
                completed,
                total,
                started,
                last_case=case.case_id,
                last_target_priority=target_priority,
            )
            print(f"[{completed:02d}/{total}] {case.case_id} target={target_priority:.4f}", flush=True)

    with (args.output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    save_target_response(rows, args.output_dir / "01_target_priority_response.png")
    save_pass_summary(rows, args.output_dir / "02_target_priority_pass_rate.png")
    summary = {
        "status": "manual target-priority calibration on fixed seven-field hard cases",
        "cases": len(records),
        "grid_size": args.grid_size,
        "fluence_size": args.fluence_size,
        "iterations": args.iterations,
        "clinical_dvh_weight": args.clinical_dvh_weight,
        "priorities": {
            str(priority): {
                "variation_acceptable_rate": float(np.mean([row["variation_acceptable"] for row in rows if row["target_priority"] == priority])),
                "target_acceptable_rate": float(np.mean([row["target_variation_acceptable"] for row in rows if row["target_priority"] == priority])),
                "oars_acceptable_rate": float(np.mean([row["oars_variation_acceptable"] for row in rows if row["target_priority"] == priority])),
                "median_d98_gy": float(np.median([row["target_d98_gy"] for row in rows if row["target_priority"] == priority])),
                "median_maximum_oar_variation_ratio": float(np.median([row["maximum_oar_variation_ratio"] for row in rows if row["target_priority"] == priority])),
                "median_paddick_ci_95": float(np.median([row["paddick_ci_95"] for row in rows if row["target_priority"] == priority])),
                "median_r50": float(np.median([row["r50"] for row in rows if row["target_priority"] == priority])),
            }
            for priority in args.target_priorities
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_progress(
        args.output_dir,
        completed,
        total,
        started,
        status="complete",
        last_case=rows[-1]["case_id"],
        last_target_priority=float(rows[-1]["target_priority"]),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
