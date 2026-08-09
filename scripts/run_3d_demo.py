import argparse
import csv
import json
import time
from pathlib import Path

from dosim_sim import (
    ImplicitDoseEngine3D,
    PlanningPriorities,
    generate_case_3d,
    optimize_fluence_3d,
)
from dosim_sim.visuals3d import save_3d_case_slices, save_3d_planning_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the implicit 3D planning prototype")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=45)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/3d_demo"))
    args = parser.parse_args()

    started = time.perf_counter()
    case = generate_case_3d(args.seed, args.grid_size)
    angles = tuple(float(value) for value in range(0, 360, 30))
    engine = ImplicitDoseEngine3D(case, angles, fluence_size=8)
    priorities0 = PlanningPriorities.for_case(case)
    plan0 = optimize_fluence_3d(case, engine, (0, 3, 6, 9), priorities0, args.iterations)

    # These are explicit human-level demonstration edits, not optimizer-selected
    # beamlet changes: raise a visible OAR priority, then add one beam angle.
    worst_oar = max(
        range(len(case.oars)),
        key=lambda index: plan0.metrics.oar_mean[index] / case.oar_limits[index],
    )
    revised_oars = list(priorities0.oars)
    revised_oars[worst_oar] *= 1.75
    priorities1 = PlanningPriorities(target=1.0, hotspot=1.0, oars=tuple(revised_oars))
    plan1 = optimize_fluence_3d(
        case,
        engine,
        plan0.active_beams,
        priorities1,
        args.iterations,
        initial_fluence=plan0.fluence,
    )
    added_beam = 1
    plan2 = optimize_fluence_3d(
        case,
        engine,
        tuple(sorted((*plan1.active_beams, added_beam))),
        priorities1,
        args.iterations,
        initial_fluence=plan1.fluence,
    )
    worst_after_beam = max(
        range(len(case.oars)),
        key=lambda index: plan2.metrics.oar_mean[index] / case.oar_limits[index],
    )
    revised_oars2 = list(priorities1.oars)
    revised_oars2[worst_after_beam] *= 1.75
    priorities2 = PlanningPriorities(target=1.0, hotspot=1.0, oars=tuple(revised_oars2))
    plan3 = optimize_fluence_3d(
        case,
        engine,
        plan2.active_beams,
        priorities2,
        args.iterations,
        initial_fluence=plan2.fluence,
    )
    priorities3 = PlanningPriorities(target=1.75, hotspot=1.0, oars=priorities2.oars)
    plan4 = optimize_fluence_3d(
        case,
        engine,
        plan3.active_beams,
        priorities3,
        args.iterations,
        initial_fluence=plan3.fluence,
    )
    plans = (plan0, plan1, plan2, plan3, plan4)
    labels = (
        "Initial 4 beams",
        f"Increase OAR {worst_oar + 1} priority",
        f"Add {angles[added_beam]:.0f}° beam",
        f"Increase OAR {worst_after_beam + 1} priority",
        "Increase target priority",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_3d_case_slices(case, args.output_dir / "01_3d_anatomy.png")
    save_3d_planning_steps(case, plans, labels, angles, args.output_dir / "02_3d_planning_steps.png")
    with (args.output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "manual_action", "active_angles", "target_d95", "target_d02", "oar_1_ratio", "oar_2_ratio", "loss"],
        )
        writer.writeheader()
        for step, (label, plan) in enumerate(zip(labels, plans, strict=True)):
            writer.writerow(
                {
                    "step": step,
                    "manual_action": label,
                    "active_angles": "|".join(str(int(angles[index])) for index in plan.active_beams),
                    "target_d95": plan.metrics.target_d95,
                    "target_d02": plan.metrics.target_d02,
                    "oar_1_ratio": plan.metrics.oar_mean[0] / case.oar_limits[0],
                    "oar_2_ratio": plan.metrics.oar_mean[1] / case.oar_limits[1],
                    "loss": plan.metrics.loss,
                }
            )

    elapsed = time.perf_counter() - started
    dense_values = args.grid_size**3 * len(angles) * engine.fluence_size**2
    summary = {
        "case_id": case.case_id,
        "grid": [args.grid_size] * 3,
        "voxels": args.grid_size**3,
        "candidate_beams": len(angles),
        "fluence_map_per_beam": [engine.fluence_size, engine.fluence_size],
        "implicit_geometry_cache_mb": engine.cache_bytes / 2**20,
        "avoided_dense_matrix_gb_float32": dense_values * 4 / 2**30,
        "elapsed_seconds_cpu_reference": elapsed,
        "manual_actions": labels[1:],
        "final_metrics": {
            "target_d95": plan4.metrics.target_d95,
            "target_d02": plan4.metrics.target_d02,
            "oar_mean_over_limit": [
                value / limit for value, limit in zip(plan4.metrics.oar_mean, case.oar_limits, strict=True)
            ],
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
