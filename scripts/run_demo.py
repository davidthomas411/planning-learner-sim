import argparse
import csv
import json
from pathlib import Path

from dosim_sim import SimulationConfig, build_dose_influence, generate_case, run_greedy_expert
from dosim_sim.visuals import save_action_sequence, save_case_overview, save_trajectory_story


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one explainable synthetic planning trajectory")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()

    cfg = SimulationConfig()
    case = generate_case(args.seed, cfg)
    influence, _, _ = build_dose_influence(case, cfg)
    trajectory = run_greedy_expert(case, influence, cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_case_overview(case, influence, args.output_dir / "01_case_construction.png", cfg)
    save_trajectory_story(case, trajectory, args.output_dir / "02_expert_trajectory.png", cfg)
    save_action_sequence(trajectory, args.output_dir / "03_action_sequence.png", cfg)

    with (args.output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "beam_index",
                "beamlet_across_field",
                "intensity_change",
                "objective_before",
                "objective_after",
                "target_d95",
                "target_d02",
                "oar_1_mean",
                "oar_2_mean",
            ],
        )
        writer.writeheader()
        for step in trajectory.steps[1:]:
            writer.writerow(
                {
                    "step": step.step,
                    "beam_index": int(step.beamlet) // cfg.beamlets_per_beam,
                    "beamlet_across_field": int(step.beamlet) % cfg.beamlets_per_beam,
                    "intensity_change": step.delta,
                    "objective_before": step.objective_before,
                    "objective_after": step.objective_after,
                    "target_d95": step.metrics.target_d95,
                    "target_d02": step.metrics.target_d02,
                    "oar_1_mean": step.metrics.oar_mean[0],
                    "oar_2_mean": step.metrics.oar_mean[1],
                }
            )

    summary = {
        "environment_version": cfg.environment_version,
        "case_id": case.case_id,
        "seed": args.seed,
        "difficulty": case.difficulty,
        "n_actions": len(trajectory.steps) - 1,
        "stopping_reason": trajectory.stopping_reason,
        "initial_objective": trajectory.steps[0].metrics.total,
        "final_objective": trajectory.final.metrics.total,
        "final_target_d95": trajectory.final.metrics.target_d95,
        "final_target_d02": trajectory.final.metrics.target_d02,
        "final_oar_mean": trajectory.final.metrics.oar_mean,
        "oar_limits": case.oar_limits,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
