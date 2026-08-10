"""Run a reviewable manual target-priority sequence on hard prostate cases."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from dosim_sim.delivery3d import delivery_mode_3d
from dosim_sim.objective import PlanningPriorities
from dosim_sim.planning3d import (
    HighLevelSearchConfig3D,
    PlanningStep3D,
    PlanningTrajectory3D,
    clinical_violation_score_3d,
    is_acceptable_3d,
    optimizer_objective_kwargs_3d,
)
from dosim_sim.prostate_protocol import PRESCRIPTION_GY
from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D, optimize_fluence_3d_torch
from dosim_sim.volume3d import generate_prostate_case_3d
from dosim_sim.manual_planning import ManualAction
from run_prostate_clinical_dvh_pilot import STATUS_PAGE, cumulative_dvh, load_records, write_progress


def step_row(case, step) -> dict:
    metrics = step.plan.metrics
    action = step.action
    return {
        "case_id": case.case_id,
        "seed": case.seed,
        "difficulty": case.difficulty,
        "step": step.step,
        "action_type": "initial_plan" if action is None else action.kind,
        "action": "Create initial seven-field plan" if action is None else action.description,
        "target_priority": step.plan.priorities.target,
        "hotspot_priority": step.plan.priorities.hotspot,
        "bladder_priority": step.plan.priorities.oars[0],
        "rectum_priority": step.plan.priorities.oars[1],
        "femoral_heads_priority": step.plan.priorities.oars[2],
        "target_d98_gy": metrics.target_d98_gy,
        "target_d99_gy": metrics.target_d99_gy,
        "target_d02_gy": metrics.target_d02 * PRESCRIPTION_GY,
        "target_variation_acceptable": metrics.protocol_target_variation_acceptable,
        "oars_variation_acceptable": max(metrics.protocol_oar_variation_ratios) <= 1.0,
        "variation_acceptable": metrics.protocol_variation_acceptable,
        "maximum_oar_variation_ratio": max(metrics.protocol_oar_variation_ratios),
        "paddick_ci_95": metrics.paddick_ci_95,
        "r50": metrics.r50,
        "violation_score": step.violation_score,
    }


def save_trajectory_plot(rows: list[dict], path: Path) -> None:
    case_ids = list(dict.fromkeys(row["case_id"] for row in rows))
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for case_id in case_ids:
        selected = [row for row in rows if row["case_id"] == case_id]
        label = case_id.replace("prostate3d-", "")
        axes[0].plot(
            [row["step"] for row in selected],
            [row["target_d98_gy"] for row in selected],
            marker="o",
            label=label,
        )
        axes[1].plot(
            [row["step"] for row in selected],
            [row["target_d02_gy"] for row in selected],
            marker="o",
            label=label,
        )
        axes[2].plot(
            [row["step"] for row in selected],
            [row["maximum_oar_variation_ratio"] for row in selected],
            marker="o",
            label=label,
        )
    axes[0].axhline(58.8, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("PTV D98 (Gy)")
    axes[0].set_title("Target coverage")
    axes[1].axhline(75.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("PTV D02 (Gy)")
    axes[1].set_title("Engineering hot-spot check")
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Worst OAR value / variation limit")
    axes[2].set_title("OAR limits")
    for axis in axes:
        axis.set_xlabel("Manual planning step")
        axis.set_xticks(sorted({row["step"] for row in rows}))
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=8, frameon=False)
    figure.suptitle("Hard prostate cases: manual target and hot-spot priority sequence", y=1.03)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_representative_review(case, trajectory, path: Path) -> None:
    first = trajectory.steps[0].plan
    final = trajectory.final.plan
    slice_index = int(np.round(np.argwhere(case.target)[:, 2].mean()))
    bins = np.linspace(0.0, 78.0, 235)
    colors = {"PTV": "#d62728", "bladder": "#1f77b4", "rectum": "#2ca02c", "femoral heads": "#9467bd"}
    masks = {"PTV": case.target, **dict(zip(("bladder", "rectum", "femoral heads"), case.oars, strict=True))}
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for column, (name, plan) in enumerate((("Initial plan", first), ("Final plan", final))):
        dose_gy = plan.dose.detach().float().cpu().numpy() * PRESCRIPTION_GY
        image = axes[0, column].imshow(dose_gy[:, :, slice_index].T, origin="lower", cmap="turbo", vmin=0, vmax=72)
        axes[0, column].contour(case.target[:, :, slice_index].T, levels=[0.5], colors=[colors["PTV"]], linewidths=1.5)
        axes[0, column].contour(case.oars[0][:, :, slice_index].T, levels=[0.5], colors=[colors["bladder"]], linewidths=1.2)
        axes[0, column].contour(case.oars[1][:, :, slice_index].T, levels=[0.5], colors=[colors["rectum"]], linewidths=1.2)
        axes[0, column].set_title(
            f"{name}: target {plan.priorities.target:.2f}; hot spot {plan.priorities.hotspot:.2f}"
        )
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
        for structure, mask in masks.items():
            axes[1, column].plot(bins, cumulative_dvh(dose_gy[mask], bins), color=colors[structure], label=structure)
        axes[1, column].axvline(PRESCRIPTION_GY, color="black", linestyle="--", linewidth=1)
        axes[1, column].set_xlabel("Dose (Gy)")
        axes[1, column].set_ylabel("Volume receiving at least dose (%)")
        axes[1, column].grid(alpha=0.2)
        axes[1, column].legend(frameon=False)
    figure.colorbar(image, ax=axes[0, :], label="Dose (Gy)", shrink=0.85)
    figure.suptitle(f"Review plan: {case.case_id}")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_action_map(rows: list[dict], path: Path) -> None:
    case_ids = list(dict.fromkeys(row["case_id"] for row in rows))
    actions = [row for row in rows if int(row["step"]) > 0]
    maximum_step = max(int(row["step"]) for row in actions)
    values = np.zeros((len(case_ids), maximum_step), dtype=int)
    labels = np.full(values.shape, "", dtype=object)
    case_index = {case_id: index for index, case_id in enumerate(case_ids)}
    for row in actions:
        y = case_index[row["case_id"]]
        x = int(row["step"]) - 1
        if row["action_type"] == "increase_target_priority":
            values[y, x] = 1
            labels[y, x] = f"Target\n{float(row['target_priority']):.2f}"
        elif row["action_type"] == "increase_hotspot_priority":
            values[y, x] = 2
            labels[y, x] = f"Hot spot\n{float(row['hotspot_priority']):.2f}"
    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    axis.imshow(values, aspect="auto", cmap=ListedColormap(("#f2f2f2", "#4c78a8", "#f58518")), vmin=0, vmax=2)
    axis.set_xticks(np.arange(maximum_step), [f"Step {value}" for value in range(1, maximum_step + 1)])
    axis.set_yticks(np.arange(len(case_ids)), [case_id.replace("prostate3d-", "") for case_id in case_ids])
    axis.set_xlabel("Manual planning action")
    axis.set_ylabel("Hard validation case")
    axis.set_title("Recorded target and hot-spot priority changes")
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            if labels[y, x]:
                axis.text(x, y, labels[y, x], ha="center", va="center", color="white", fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_manual_target_sequence(case, engine, active_beams, config, action_set) -> PlanningTrajectory3D:
    """Apply review-triggered target and optional hot-spot priority changes."""

    priorities = PlanningPriorities.for_case(case)
    plan = optimize_fluence_3d_torch(
        case,
        engine,
        active_beams,
        priorities,
        config.optimizer_iterations,
        **optimizer_objective_kwargs_3d(config),
    )
    steps = [PlanningStep3D(0, None, plan, clinical_violation_score_3d(plan.metrics, case, config))]
    if is_acceptable_3d(plan.metrics, case, config):
        return PlanningTrajectory3D(case.case_id, tuple(steps), "acceptable")
    for step_index in range(1, config.max_steps + 1):
        target_failed = plan.metrics.protocol_target_variation_acceptable is False
        hotspot_failed = plan.metrics.target_d02 > config.d02_max
        if target_failed:
            priority_name = "target"
            action_kind = "increase_target_priority"
            old = plan.priorities.target
        elif hotspot_failed and action_set == "target_hotspot":
            priority_name = "hotspot"
            action_kind = "increase_hotspot_priority"
            old = plan.priorities.hotspot
        else:
            return PlanningTrajectory3D(case.case_id, tuple(steps), "different_action_required")
        new = min(old * config.priority_factor, config.priority_ceiling)
        if new <= old:
            return PlanningTrajectory3D(case.case_id, tuple(steps), "priority_ceiling")
        priorities = replace(plan.priorities, **{priority_name: new})
        action = ManualAction(
            action_kind,
            f"Increase {priority_name.replace('_', '-')} priority {old:.2f} -> {new:.2f}",
            old_value=old,
            new_value=new,
        )
        plan = optimize_fluence_3d_torch(
            case,
            engine,
            active_beams,
            priorities,
            config.optimizer_iterations,
            initial_fluence=plan.fluence,
            **optimizer_objective_kwargs_3d(config),
        )
        steps.append(
            PlanningStep3D(
                step_index,
                action,
                plan,
                clinical_violation_score_3d(plan.metrics, case, config),
            )
        )
        if is_acceptable_3d(plan.metrics, case, config):
            return PlanningTrajectory3D(case.case_id, tuple(steps), "acceptable")
    return PlanningTrajectory3D(case.case_id, tuple(steps), "manual_step_limit")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the manual target-priority planning sequence")
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/prostate300_local/merged"))
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--fluence-size", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--action-set", choices=("target_only", "target_hotspot"), default="target_only")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prostate_manual_target_trajectory"))
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
    config = HighLevelSearchConfig3D(
        max_steps=args.max_steps,
        beam_width=1,
        add_candidates=0,
        remove_candidates=0,
        shift_candidates=0,
        optimizer_iterations=args.iterations,
        priority_factor=1.75,
        priority_ceiling=5.359375,
        priority_floor=1.0,
        d95_min=0.94,
        d02_max=1.25,
        initial_field_count=7,
        minimum_field_count=7,
        normal_tissue_weight=50.0,
        normal_tissue_threshold=0.5,
        integral_dose_weight=2.0,
        clinical_dvh_weight=5.0,
        prostate_protocol_tier="variation_acceptable",
        paddick_ci_95_min=0.40,
        r50_max=15.0,
    )
    started = time.perf_counter()
    write_progress(args.output_dir, 0, len(records), started, unit="cases")
    rows: list[dict] = []
    trajectories = []
    cases = []
    mode = delivery_mode_3d("static_7")
    for index, record in enumerate(records, start=1):
        case = generate_prostate_case_3d(int(record["seed"]), args.grid_size, difficulty="hard")
        engine = TorchImplicitDoseEngine3D(
            case,
            mode.angles_degrees,
            args.fluence_size,
            device=device,
            dtype=torch.float32,
        )
        trajectory = run_manual_target_sequence(case, engine, mode.active_beams, config, args.action_set)
        rows.extend(step_row(case, step) for step in trajectory.steps)
        trajectories.append(trajectory)
        cases.append(case)
        save_trajectory_plot(rows, args.output_dir / "01_manual_trajectory.png")
        if any(int(row["step"]) > 0 for row in rows):
            save_action_map(rows, args.output_dir / "02_action_map.png")
        longest_so_far = max(range(len(trajectories)), key=lambda value: len(trajectories[value].steps))
        save_representative_review(
            cases[longest_so_far],
            trajectories[longest_so_far],
            args.output_dir / "03_representative_plan_review.png",
        )
        write_progress(
            args.output_dir,
            index,
            len(records),
            started,
            last_case=case.case_id,
            last_target_priority=trajectory.final.plan.priorities.target,
            unit="cases",
        )
        print(f"[{index}/{len(records)}] {case.case_id}: {trajectory.stopping_reason}, {len(trajectory.steps) - 1} changes", flush=True)

    with (args.output_dir / "trajectory_steps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    save_trajectory_plot(rows, args.output_dir / "01_manual_trajectory.png")
    save_action_map(rows, args.output_dir / "02_action_map.png")
    longest = max(range(len(trajectories)), key=lambda value: len(trajectories[value].steps))
    save_representative_review(cases[longest], trajectories[longest], args.output_dir / "03_representative_plan_review.png")
    review_dir = args.output_dir / "review_plans"
    review_dir.mkdir(exist_ok=True)
    for case, trajectory in zip(cases, trajectories, strict=True):
        save_representative_review(case, trajectory, review_dir / f"{case.case_id}.png")
    summary = {
        "status": "manual target and hot-spot priority trajectory calibration",
        "action_set": args.action_set,
        "cases": len(trajectories),
        "acceptable_cases": sum(trajectory.stopping_reason == "acceptable" for trajectory in trajectories),
        "changes_per_case": [len(trajectory.steps) - 1 for trajectory in trajectories],
        "actions": [[step.action.description for step in trajectory.steps if step.action] for trajectory in trajectories],
        "initial_median_d98_gy": float(np.median([trajectory.steps[0].plan.metrics.target_d98_gy for trajectory in trajectories])),
        "final_median_d98_gy": float(np.median([trajectory.final.plan.metrics.target_d98_gy for trajectory in trajectories])),
        "initial_median_maximum_oar_variation_ratio": float(np.median([max(trajectory.steps[0].plan.metrics.protocol_oar_variation_ratios) for trajectory in trajectories])),
        "final_median_maximum_oar_variation_ratio": float(np.median([max(trajectory.final.plan.metrics.protocol_oar_variation_ratios) for trajectory in trajectories])),
        "initial_median_paddick_ci_95": float(np.median([trajectory.steps[0].plan.metrics.paddick_ci_95 for trajectory in trajectories])),
        "final_median_paddick_ci_95": float(np.median([trajectory.final.plan.metrics.paddick_ci_95 for trajectory in trajectories])),
        "initial_median_r50": float(np.median([trajectory.steps[0].plan.metrics.r50 for trajectory in trajectories])),
        "final_median_r50": float(np.median([trajectory.final.plan.metrics.r50 for trajectory in trajectories])),
        "elapsed_seconds": time.perf_counter() - started,
        "beam_angles_degrees": mode.angles_degrees,
        "configuration": vars(config),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_progress(
        args.output_dir,
        len(records),
        len(records),
        started,
        status="complete",
        last_case=cases[-1].case_id,
        last_target_priority=trajectories[-1].final.plan.priorities.target,
        unit="cases",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
