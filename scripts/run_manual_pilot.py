import argparse
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dosim_sim import SimulationConfig, build_dose_influence, generate_case, run_manual_planner
from dosim_sim.objective import is_acceptable


def evaluate_seed(seed: int) -> dict[str, object]:
    cfg = SimulationConfig()
    try:
        case = generate_case(seed, cfg)
    except ValueError as error:
        return {"valid": False, "seed": seed, "error": str(error)}
    influence, _, _ = build_dose_influence(case, cfg)
    trajectory = run_manual_planner(case, influence, cfg)
    initial = trajectory.steps[0].plan.clinical_metrics
    final = trajectory.final.plan.clinical_metrics
    return {
        "valid": True,
        "environment_version": cfg.environment_version,
        "seed": seed,
        "case_id": case.case_id,
        "difficulty": case.difficulty,
        "acceptable": is_acceptable(final, case, cfg),
        "stopping_reason": trajectory.stopping_reason,
        "manual_actions": len(trajectory.steps) - 1,
        "action_types": "|".join(step.action.kind for step in trajectory.steps[1:]),
        "action_descriptions": "|".join(step.action.description for step in trajectory.steps[1:]),
        "initial_target_d95": initial.target_d95,
        "final_target_d95": final.target_d95,
        "initial_target_d02": initial.target_d02,
        "final_target_d02": final.target_d02,
        "initial_worst_oar_ratio": max(
            value / limit for value, limit in zip(initial.oar_mean, case.oar_limits)
        ),
        "final_worst_oar_ratio": max(
            value / limit for value, limit in zip(final.oar_mean, case.oar_limits)
        ),
        "final_beam_angles": "|".join(str(beam * 30) for beam in trajectory.final.plan.active_beams),
    }


def save_figure(rows: list[dict[str, object]], output_path: Path) -> None:
    acceptable = np.array([bool(row["acceptable"]) for row in rows])
    colors = np.where(acceptable, "#4daf4a", "#e41a1c")
    initial_d95 = np.array([float(row["initial_target_d95"]) for row in rows])
    final_d95 = np.array([float(row["final_target_d95"]) for row in rows])
    initial_oar = np.array([float(row["initial_worst_oar_ratio"]) for row in rows])
    final_oar = np.array([float(row["final_worst_oar_ratio"]) for row in rows])
    action_counter: Counter[str] = Counter()
    for row in rows:
        action_counter.update(value for value in str(row["action_types"]).split("|") if value)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for index in range(len(rows)):
        axes[0].plot([0, 1], [initial_d95[index], final_d95[index]], color=colors[index], alpha=0.5)
        axes[0].scatter([0, 1], [initial_d95[index], final_d95[index]], color=colors[index], s=22)
    axes[0].axhline(0.85, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_xticks([0, 1], ["initial optimized plan", "after manual trajectory"])
    axes[0].set_ylabel("target D95")
    axes[0].set_title("Did manual changes improve coverage?")
    axes[0].grid(axis="y", alpha=0.2)

    for index in range(len(rows)):
        axes[1].plot([0, 1], [initial_oar[index], final_oar[index]], color=colors[index], alpha=0.5)
        axes[1].scatter([0, 1], [initial_oar[index], final_oar[index]], color=colors[index], s=22)
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_xticks([0, 1], ["initial optimized plan", "after manual trajectory"])
    axes[1].set_ylabel("worst OAR mean / limit")
    axes[1].set_title("Did manual changes reduce OAR excess?")
    axes[1].grid(axis="y", alpha=0.2)

    labels = [
        "OAR priority",
        "Target priority",
        "Hot-spot priority",
        "Add beam",
        "Remove beam",
    ]
    keys = [
        "increase_oar_priority",
        "increase_target_priority",
        "increase_hotspot_priority",
        "add_beam",
        "remove_beam",
    ]
    values = [action_counter[key] for key in keys]
    axes[2].barh(labels, values, color="#7b3294", alpha=0.8)
    for y, value in enumerate(values):
        axes[2].text(value + 0.3, y, str(value), va="center")
    axes[2].set_xlabel("recorded manual actions")
    axes[2].set_title("What the planner actually changed")
    axes[2].grid(axis="x", alpha=0.2)

    rate = 100 * acceptable.mean()
    fig.suptitle(
        f"High-level manual-planning pilot: {len(rows)} cases | {rate:.1f}% reached provisional goals",
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot the high-level manual trajectory generator")
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/manual_pilot"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    next_seed = args.seed_start
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        while len(rows) < args.cases:
            seeds = list(range(next_seed, next_seed + (args.cases - len(rows)) + 4))
            next_seed = seeds[-1] + 1
            for result in executor.map(evaluate_seed, seeds):
                if bool(result["valid"]):
                    rows.append(result)
                    if len(rows) == args.cases:
                        break

    fieldnames = list(rows[0].keys())
    with (args.output_dir / "manual_pilot_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    save_figure(rows, args.output_dir / "01_manual_pilot.png")

    print(f"valid_cases={len(rows)}")
    print(f"acceptable_rate={np.mean([bool(row['acceptable']) for row in rows]):.3f}")
    print(f"median_manual_actions={np.median([int(row['manual_actions']) for row in rows]):.1f}")


if __name__ == "__main__":
    main()
