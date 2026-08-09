import argparse
import csv
import json
from pathlib import Path

from dosim_sim import SimulationConfig, build_dose_influence, generate_case, run_manual_planner
from dosim_sim.manual_visuals import save_manual_filmstrip, save_manual_metrics, save_nested_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a high-level manual-planning trajectory")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/manual_demo"))
    args = parser.parse_args()

    cfg = SimulationConfig()
    case = generate_case(args.seed, cfg)
    influence, _, _ = build_dose_influence(case, cfg)
    trajectory = run_manual_planner(case, influence, cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_nested_workflow(args.output_dir / "01_nested_workflow.png")
    save_manual_filmstrip(case, trajectory, args.output_dir / "02_manual_trajectory.png", cfg)
    save_manual_metrics(case, trajectory, args.output_dir / "03_manual_metrics.png", cfg)

    with (args.output_dir / "manual_trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "manual_step",
            "manual_action",
            "active_beam_angles_degrees",
            "target_priority",
            "hotspot_priority",
            "oar_1_priority",
            "oar_2_priority",
            "optimizer_iterations",
            "target_d95",
            "target_d02",
            "oar_1_mean_over_limit",
            "oar_2_mean_over_limit",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in trajectory.steps:
            metrics = step.plan.clinical_metrics
            writer.writerow(
                {
                    "manual_step": step.step,
                    "manual_action": "Initial four beams" if step.action is None else step.action.description,
                    "active_beam_angles_degrees": "|".join(str(beam * 30) for beam in step.plan.active_beams),
                    "target_priority": step.plan.priorities.target,
                    "hotspot_priority": step.plan.priorities.hotspot,
                    "oar_1_priority": step.plan.priorities.oars[0],
                    "oar_2_priority": step.plan.priorities.oars[1],
                    "optimizer_iterations": step.plan.optimizer_iterations,
                    "target_d95": metrics.target_d95,
                    "target_d02": metrics.target_d02,
                    "oar_1_mean_over_limit": metrics.oar_mean[0] / case.oar_limits[0],
                    "oar_2_mean_over_limit": metrics.oar_mean[1] / case.oar_limits[1],
                }
            )

    summary = {
        "environment_version": cfg.environment_version,
        "case_id": case.case_id,
        "manual_actions": len(trajectory.steps) - 1,
        "stopping_reason": trajectory.stopping_reason,
        "actions": [step.action.description for step in trajectory.steps[1:]],
        "final_active_beam_angles_degrees": [beam * 30 for beam in trajectory.final.plan.active_beams],
        "final_target_priority": trajectory.final.plan.priorities.target,
        "final_oar_priorities": trajectory.final.plan.priorities.oars,
        "final_target_d95": trajectory.final.plan.clinical_metrics.target_d95,
        "final_target_d02": trajectory.final.plan.clinical_metrics.target_d02,
        "final_oar_mean_over_limit": [
            value / limit
            for value, limit in zip(trajectory.final.plan.clinical_metrics.oar_mean, case.oar_limits)
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

