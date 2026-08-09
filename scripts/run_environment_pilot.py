import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dosim_sim import SimulationConfig, build_dose_influence, generate_case, run_greedy_expert
from dosim_sim.objective import is_acceptable


def evaluate_seed(payload: tuple[int, float, float, float]) -> dict[str, float | int | str | bool]:
    seed, target_weight, hotspot_weight, oar_weight = payload
    cfg = SimulationConfig(
        target_underdose_weight=target_weight,
        target_hotspot_weight=hotspot_weight,
        oar_weight=oar_weight,
    )
    try:
        case = generate_case(seed, cfg)
    except ValueError as error:
        return {"seed": seed, "valid": False, "error": str(error)}
    influence, _, _ = build_dose_influence(case, cfg)
    trajectory = run_greedy_expert(case, influence, cfg)
    initial = trajectory.steps[0].metrics
    final = trajectory.final.metrics
    return {
        "environment_version": cfg.environment_version,
        "seed": seed,
        "valid": True,
        "case_id": case.case_id,
        "difficulty": case.difficulty,
        "actions": len(trajectory.steps) - 1,
        "stopping_reason": trajectory.stopping_reason,
        "acceptable": is_acceptable(final, case, cfg),
        "initial_objective": initial.total,
        "final_objective": final.total,
        "objective_reduction_fraction": (initial.total - final.total) / initial.total,
        "target_d95": final.target_d95,
        "target_d02": final.target_d02,
        "worst_oar_mean_ratio": max(
            value / limit for value, limit in zip(final.oar_mean, case.oar_limits, strict=True)
        ),
    }


def save_pilot_figure(rows: list[dict[str, float | int | str | bool]], output_path: Path) -> None:
    accepted = np.array([bool(row["acceptable"]) for row in rows])
    d95 = np.array([float(row["target_d95"]) for row in rows])
    oar_ratio = np.array([float(row["worst_oar_mean_ratio"]) for row in rows])
    actions = np.array([int(row["actions"]) for row in rows])
    reductions = np.array([float(row["objective_reduction_fraction"]) for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    colors = np.where(accepted, "#4daf4a", "#e41a1c")
    axes[0].scatter(d95, oar_ratio, c=colors, alpha=0.75, edgecolor="black", linewidth=0.3)
    axes[0].axvline(0.85, color="#4daf4a", linestyle="--", linewidth=1)
    axes[0].axhline(1.0, color="#377eb8", linestyle="--", linewidth=1)
    axes[0].set_xlabel("final target D95")
    axes[0].set_ylabel("worst OAR mean / limit")
    axes[0].set_title("Each dot is one case\ngreen = provisionally acceptable")
    axes[0].grid(alpha=0.2)

    axes[1].hist(actions[accepted], bins=12, alpha=0.75, color="#4daf4a", label="acceptable")
    if (~accepted).any():
        axes[1].hist(actions[~accepted], bins=12, alpha=0.65, color="#e41a1c", label="not acceptable")
    axes[1].set_xlabel("number of expert actions")
    axes[1].set_ylabel("cases")
    axes[1].set_title("How long the expert needed")
    axes[1].legend()

    axes[2].hist(reductions, bins=12, color="#7b3294", alpha=0.8)
    axes[2].set_xlabel("fraction of objective removed")
    axes[2].set_ylabel("cases")
    axes[2].set_title("Did optimization materially work?")
    axes[2].grid(axis="y", alpha=0.2)

    success_rate = 100 * accepted.mean()
    fig.suptitle(
        f"Environment pilot: {len(rows)} valid synthetic cases · {success_rate:.1f}% provisionally acceptable",
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pre-model environment validation pilot")
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target-weight", type=float, default=20.0)
    parser.add_argument("--hotspot-weight", type=float, default=5.0)
    parser.add_argument("--oar-weight", type=float, default=7.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/environment_pilot"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str | bool]] = []
    next_seed = args.seed_start
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        while len(rows) < args.cases:
            needed = args.cases - len(rows)
            seeds = list(range(next_seed, next_seed + needed + 8))
            next_seed = seeds[-1] + 1
            payloads = [
                (seed, args.target_weight, args.hotspot_weight, args.oar_weight) for seed in seeds
            ]
            for result in executor.map(evaluate_seed, payloads):
                if bool(result["valid"]):
                    rows.append(result)
                    if len(rows) == args.cases:
                        break

    fieldnames = list(rows[0].keys())
    with (args.output_dir / "pilot_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    save_pilot_figure(rows, args.output_dir / "01_environment_pilot.png")

    success_rate = sum(bool(row["acceptable"]) for row in rows) / len(rows)
    print(f"valid_cases={len(rows)}")
    print(f"acceptable_rate={success_rate:.3f}")
    print(f"median_actions={np.median([int(row['actions']) for row in rows]):.1f}")
    print(f"median_objective_reduction={np.median([float(row['objective_reduction_fraction']) for row in rows]):.3f}")


if __name__ == "__main__":
    main()
