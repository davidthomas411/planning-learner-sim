import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dosim_sim.objective import PlanningPriorities
from dosim_sim.planning3d import beam_eye_view_avoidance_scores_3d
from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D, optimize_fluence_3d_torch
from dosim_sim.volume3d import generate_prostate_case_3d


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cumulative_dvh(dose: np.ndarray, mask: np.ndarray, bins: np.ndarray) -> np.ndarray:
    values = dose[mask]
    return np.asarray([100.0 * np.mean(values >= level) for level in bins])


def minimum_angle_gap(angles: list[float]) -> float:
    ordered = np.sort(np.mod(angles, 360.0))
    return float(np.min(np.diff(np.r_[ordered, ordered[0] + 360.0])))


def select_expert_shift(case, plan, angles: list[float], shift: float) -> tuple[int, float, float] | None:
    ratios = np.asarray(plan.metrics.oar_mean) / np.asarray(case.oar_limits)
    weights = np.maximum(ratios - 0.75, 0.05) * np.asarray(plan.priorities.oars)
    current_scores = beam_eye_view_avoidance_scores_3d(case, tuple(angles), weights)
    candidates = []
    for field, old_angle in enumerate(angles):
        for direction in (-1.0, 1.0):
            new_angle = float((old_angle + direction * shift) % 360.0)
            changed = list(angles)
            changed[field] = new_angle
            if minimum_angle_gap(changed) < 25.0:
                continue
            new_score = beam_eye_view_avoidance_scores_3d(case, (new_angle,), weights)[0]
            gain = float(new_score - current_scores[field])
            candidates.append((gain, field, new_angle))
    if not candidates:
        return None
    gain, field, new_angle = max(candidates, key=lambda value: (value[0], -value[1], -value[2]))
    return (field, new_angle, gain) if gain > 0.002 else None


def optimize(case, angles, fluence_size, iterations, device, priorities, initial_fluence=None):
    engine = TorchImplicitDoseEngine3D(
        case, tuple(angles), fluence_size, device=device, dtype=torch.float32
    )
    plan = optimize_fluence_3d_torch(
        case,
        engine,
        tuple(range(len(angles))),
        priorities,
        iterations=iterations,
        initial_fluence=initial_fluence,
        normal_tissue_weight=50.0,
        normal_tissue_threshold=0.5,
        integral_dose_weight=2.0,
    )
    return engine, plan


def save_review(case, initial_plan, final_plan, actions, path: Path) -> None:
    initial_dose = initial_plan.dose.detach().float().cpu().numpy()
    final_dose = final_plan.dose.detach().float().cpu().numpy()
    axial = int(np.argmax(case.target.sum(axis=(0, 1))))
    maximum = max(float(initial_dose.max()), float(final_dose.max()), 1.2)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for axis, dose, plan, label in zip(
        axes[:2],
        (initial_dose, final_dose),
        (initial_plan, final_plan),
        ("Equal separation", "Anatomy-guided angles"),
        strict=True,
    ):
        image = axis.imshow(dose[:, :, axial].T, origin="lower", cmap="turbo", vmin=0.0, vmax=maximum)
        axis.contour(case.target[:, :, axial].T, levels=[0.5], colors="white", linewidths=1.4)
        for color, mask in zip(("cyan", "lime", "magenta"), case.oars, strict=True):
            axis.contour(mask[:, :, axial].T, levels=[0.5], colors=color, linewidths=0.8)
        axis.set_title(f"{label}\nCI95 {plan.metrics.paddick_ci_95:.2f}; R50 {plan.metrics.r50:.1f}")
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03, label="Relative dose")
    bins = np.linspace(0.0, maximum, 161)
    structures = (("PTV", case.target, "#d62728"), ("bladder", case.oars[0], "#1f77b4"), ("rectum", case.oars[1], "#2ca02c"), ("femoral heads", case.oars[2], "#9467bd"))
    for name, mask, color in structures:
        axes[2].plot(bins, cumulative_dvh(initial_dose, mask, bins), color=color, linestyle="--", alpha=0.65)
        axes[2].plot(bins, cumulative_dvh(final_dose, mask, bins), color=color, label=name)
    axes[2].set_xlabel("Relative dose (dashed equal; solid guided)")
    axes[2].set_ylabel("Volume receiving at least dose (%)")
    axes[2].set_ylim(0, 101)
    axes[2].grid(alpha=0.2)
    axes[2].set_title("; ".join(actions) if actions else "No beneficial 10-degree shift")
    figure.legend(*axes[2].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test anatomy-guided 10-degree prostate beam shifts")
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/prostate300_local/merged"))
    parser.add_argument("--cases-per-stratum", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--fluence-size", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--manual-shifts", type=int, default=2)
    parser.add_argument("--shift-degrees", type=float, default=10.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prostate_expert_angle_pilot"))
    args = parser.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.dataset_dir / "trajectory_view.jsonl")
    records = [row for row in records if row.get("split") == "validation"]
    selected = [
        row
        for difficulty in ("easy", "moderate", "hard")
        for row in [value for value in records if value["difficulty"] == difficulty][: args.cases_per_stratum]
    ]
    device = torch.device(args.device)
    rows = []
    representative = None
    for index, record in enumerate(selected, start=1):
        case = generate_prostate_case_3d(int(record["seed"]), args.grid_size, record["difficulty"])
        angles = [field * 360.0 / 7.0 for field in range(7)]
        priorities = PlanningPriorities.for_case(case)
        engine, initial_plan = optimize(case, angles, args.fluence_size, args.iterations, device, priorities)
        current_plan = initial_plan
        actions = []
        for _ in range(args.manual_shifts):
            decision = select_expert_shift(case, current_plan, angles, args.shift_degrees)
            if decision is None:
                break
            field, new_angle, gain = decision
            old_angle = angles[field]
            angles[field] = new_angle
            _, current_plan = optimize(
                case,
                angles,
                args.fluence_size,
                args.iterations,
                device,
                priorities,
                initial_fluence=current_plan.fluence,
            )
            actions.append(f"field {field + 1}: {old_angle:.1f} to {new_angle:.1f} degrees")
        rows.append({
            "case_id": case.case_id,
            "difficulty": case.difficulty,
            "actions": " | ".join(actions),
            "final_angles_degrees": "|".join(f"{value:.1f}" for value in angles),
            "initial_d95": initial_plan.metrics.target_d95,
            "final_d95": current_plan.metrics.target_d95,
            "initial_ci95": initial_plan.metrics.paddick_ci_95,
            "final_ci95": current_plan.metrics.paddick_ci_95,
            "initial_r50": initial_plan.metrics.r50,
            "final_r50": current_plan.metrics.r50,
            "initial_maximum_oar_ratio": max(np.asarray(initial_plan.metrics.oar_mean) / np.asarray(case.oar_limits)),
            "final_maximum_oar_ratio": max(np.asarray(current_plan.metrics.oar_mean) / np.asarray(case.oar_limits)),
        })
        if case.difficulty == "moderate" and representative is None:
            representative = (case, initial_plan, current_plan, actions)
        print(f"[{index}/{len(selected)}] {case.case_id}: {len(actions)} shifts", flush=True)
        del engine
    with (args.output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = ("d95", "ci95", "r50", "maximum_oar_ratio")
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for axis, metric in zip(axes.flat, metrics, strict=True):
        for row in rows:
            axis.plot([0, 1], [row[f"initial_{metric}"], row[f"final_{metric}"]], color="#777777", alpha=0.5)
        axis.set_xticks([0, 1], ["Equal", "Anatomy-guided"])
        axis.set_ylabel(metric.upper())
        axis.grid(alpha=0.2)
    figure.suptitle("Rule-based 10-degree beam-angle refinement")
    figure.savefig(args.output_dir / "01_angle_shift_summary.png", dpi=180)
    plt.close(figure)
    if representative is not None:
        save_review(*representative, args.output_dir / "02_representative_plan.png")
    summary = {
        "cases": len(rows),
        "shift_degrees": args.shift_degrees,
        "median_shifts": float(np.median([0 if not row["actions"] else len(row["actions"].split(" | ")) for row in rows])),
        "median_r50_change": float(np.median([row["final_r50"] - row["initial_r50"] for row in rows])),
        "median_ci95_change": float(np.median([row["final_ci95"] - row["initial_ci95"] for row in rows])),
        "median_maximum_oar_ratio_change": float(np.median([row["final_maximum_oar_ratio"] - row["initial_maximum_oar_ratio"] for row in rows])),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
