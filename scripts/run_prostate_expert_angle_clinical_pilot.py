"""Test anatomy-guided 10-degree field shifts after priority calibration."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dosim_sim.objective import PlanningPriorities
from dosim_sim.planning3d import (
    HighLevelSearchConfig3D,
    beam_eye_view_avoidance_scores_3d,
    is_acceptable_3d,
)
from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D, optimize_fluence_3d_torch
from dosim_sim.volume3d import generate_prostate_case_3d
from run_prostate_clinical_dvh_pilot import STATUS_PAGE, should_update_figures, write_progress
from run_prostate_expert_angle_pilot import minimum_angle_gap, save_review


def load_final_settings(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    final = {}
    for row in rows:
        if row["case_id"] not in final or int(row["step"]) > int(final[row["case_id"]]["step"]):
            final[row["case_id"]] = row
    return list(final.values())


def select_shift(case, plan, angles: list[float], shift_degrees: float):
    ratios = np.asarray(plan.metrics.protocol_oar_per_protocol_ratios, dtype=float)
    weights = np.maximum(ratios, 0.05) * np.asarray(plan.priorities.oars)
    current = beam_eye_view_avoidance_scores_3d(case, tuple(angles), weights)
    candidates = []
    for field, old_angle in enumerate(angles):
        for direction in (-1.0, 1.0):
            new_angle = float((old_angle + direction * shift_degrees) % 360.0)
            changed = [*angles]
            changed[field] = new_angle
            if minimum_angle_gap(changed) < 25.0:
                continue
            new_score = beam_eye_view_avoidance_scores_3d(case, (new_angle,), weights)[0]
            candidates.append((float(new_score - current[field]), field, new_angle))
    if not candidates:
        return None
    gain, field, new_angle = max(candidates, key=lambda value: (value[0], -value[1], -value[2]))
    return (field, new_angle, gain) if gain > 0.002 else None


def optimize(case, angles, priorities, args, initial_fluence=None):
    engine = TorchImplicitDoseEngine3D(
        case,
        tuple(angles),
        args.fluence_size,
        device=torch.device(args.device),
        dtype=torch.float32,
    )
    plan = optimize_fluence_3d_torch(
        case,
        engine,
        tuple(range(len(angles))),
        priorities,
        args.iterations,
        initial_fluence=initial_fluence,
        normal_tissue_weight=50.0,
        normal_tissue_threshold=0.5,
        integral_dose_weight=2.0,
        clinical_dvh_weight=5.0,
    )
    return plan


def metrics_row(case, initial, final, angles, actions, acceptable_config) -> dict:
    return {
        "case_id": case.case_id,
        "actions": " | ".join(actions),
        "action_count": len(actions),
        "final_angles_degrees": "|".join(f"{value:.1f}" for value in angles),
        "initial_acceptable": is_acceptable_3d(initial.metrics, case, acceptable_config),
        "final_acceptable": is_acceptable_3d(final.metrics, case, acceptable_config),
        "initial_d98_gy": initial.metrics.target_d98_gy,
        "final_d98_gy": final.metrics.target_d98_gy,
        "initial_d02_gy": 60.0 * initial.metrics.target_d02,
        "final_d02_gy": 60.0 * final.metrics.target_d02,
        "initial_maximum_oar_per_protocol_ratio": max(initial.metrics.protocol_oar_per_protocol_ratios),
        "final_maximum_oar_per_protocol_ratio": max(final.metrics.protocol_oar_per_protocol_ratios),
        "initial_maximum_oar_variation_ratio": max(initial.metrics.protocol_oar_variation_ratios),
        "final_maximum_oar_variation_ratio": max(final.metrics.protocol_oar_variation_ratios),
        "initial_ci95": initial.metrics.paddick_ci_95,
        "final_ci95": final.metrics.paddick_ci_95,
        "initial_r50": initial.metrics.r50,
        "final_r50": final.metrics.r50,
    }


def save_summary(rows: list[dict], path: Path) -> None:
    metrics = (
        ("d98_gy", "PTV D98 (Gy)"),
        ("d02_gy", "PTV D02 (Gy)"),
        ("maximum_oar_per_protocol_ratio", "Worst OAR / per-protocol limit"),
        ("ci95", "Paddick CI95"),
        ("r50", "R50"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, metrics, strict=False):
        for row in rows:
            axis.plot([0, 1], [row[f"initial_{metric}"], row[f"final_{metric}"]], color="#777777", alpha=0.55)
        axis.set_xticks([0, 1], ["Equal angles", "Guided angles"])
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    figure.suptitle("Anatomy-guided 10-degree field shifts after priority calibration")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clinical-metric expert angle pilot")
    parser.add_argument("--trajectory-csv", type=Path, default=Path("outputs/prostate_manual_target_hotspot_hard15/trajectory_steps.csv"))
    parser.add_argument("--cases", type=int, default=15)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--fluence-size", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--manual-shifts", type=int, default=2)
    parser.add_argument("--shift-degrees", type=float, default=10.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prostate_expert_angle_clinical_hard15"))
    args = parser.parse_args()

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(torch.device(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "status.html").write_text(STATUS_PAGE, encoding="utf-8")
    settings = load_final_settings(args.trajectory_csv)[: args.cases]
    acceptable_config = HighLevelSearchConfig3D(
        d95_min=0.94,
        d02_max=1.25,
        minimum_field_count=7,
        prostate_protocol_tier="variation_acceptable",
        paddick_ci_95_min=0.40,
        r50_max=15.0,
    )
    started = time.perf_counter()
    write_progress(args.output_dir, 0, len(settings), started, unit="cases")
    rows = []
    representative = None
    best_reduction = -float("inf")
    for index, setting in enumerate(settings, start=1):
        seed = int(setting["seed"])
        case = generate_prostate_case_3d(seed, args.grid_size, difficulty="hard")
        priorities = PlanningPriorities(
            target=float(setting["target_priority"]),
            hotspot=float(setting["hotspot_priority"]),
            oars=(
                float(setting["bladder_priority"]),
                float(setting["rectum_priority"]),
                float(setting["femoral_heads_priority"]),
            ),
            normal_tissue=1.0,
        )
        angles = [field * 360.0 / 7.0 for field in range(7)]
        initial = optimize(case, angles, priorities, args)
        current = initial
        actions = []
        for _ in range(args.manual_shifts):
            decision = select_shift(case, current, angles, args.shift_degrees)
            if decision is None:
                break
            field, new_angle, gain = decision
            old_angle = angles[field]
            angles[field] = new_angle
            current = optimize(case, angles, priorities, args, current.fluence)
            actions.append(f"Field {field + 1}: {old_angle:.1f} to {new_angle:.1f} degrees; BEV gain {gain:.4f}")
        row = metrics_row(case, initial, current, angles, actions, acceptable_config)
        rows.append(row)
        reduction = row["initial_maximum_oar_per_protocol_ratio"] - row["final_maximum_oar_per_protocol_ratio"]
        if reduction > best_reduction:
            representative = (case, initial, current, actions)
            best_reduction = reduction
        if should_update_figures(index, len(settings)):
            save_summary(rows, args.output_dir / "01_angle_shift_summary.png")
            if representative is not None:
                save_review(*representative, args.output_dir / "02_representative_plan.png")
        write_progress(args.output_dir, index, len(settings), started, last_case=case.case_id, unit="cases")
        print(f"[{index}/{len(settings)}] {case.case_id}: {len(actions)} shifts", flush=True)

    with (args.output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    save_summary(rows, args.output_dir / "01_angle_shift_summary.png")
    if representative is not None:
        save_review(*representative, args.output_dir / "02_representative_plan.png")
    summary = {
        "cases": len(rows),
        "shift_degrees": args.shift_degrees,
        "initial_acceptable": sum(row["initial_acceptable"] for row in rows),
        "final_acceptable": sum(row["final_acceptable"] for row in rows),
        "median_action_count": float(np.median([row["action_count"] for row in rows])),
        "median_d98_change_gy": float(np.median([row["final_d98_gy"] - row["initial_d98_gy"] for row in rows])),
        "median_d02_change_gy": float(np.median([row["final_d02_gy"] - row["initial_d02_gy"] for row in rows])),
        "median_oar_per_protocol_ratio_change": float(np.median([row["final_maximum_oar_per_protocol_ratio"] - row["initial_maximum_oar_per_protocol_ratio"] for row in rows])),
        "median_ci95_change": float(np.median([row["final_ci95"] - row["initial_ci95"] for row in rows])),
        "median_r50_change": float(np.median([row["final_r50"] - row["initial_r50"] for row in rows])),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_progress(args.output_dir, len(settings), len(settings), started, status="complete", last_case=rows[-1]["case_id"], unit="cases")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
