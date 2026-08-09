import argparse
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dosim_sim import (
    SimulationConfig,
    build_dose_influence,
    clinical_violation_score,
    generate_case,
    run_high_level_oracle,
    run_manual_planner,
)
from dosim_sim.objective import is_acceptable
from dosim_sim.oracle_visuals import save_oracle_case_comparison


def evaluate_seed(seed: int) -> dict[str, object]:
    cfg = SimulationConfig()
    try:
        case = generate_case(seed, cfg)
    except ValueError as error:
        return {"valid": False, "seed": seed, "error": str(error)}
    influence, _, _ = build_dose_influence(case, cfg)
    manual = run_manual_planner(case, influence, cfg)
    oracle = run_high_level_oracle(case, influence, cfg)
    manual_score = clinical_violation_score(manual.final.plan.clinical_metrics, case, cfg)
    oracle_score = oracle.final.violation_score
    manual_ok = is_acceptable(manual.final.plan.clinical_metrics, case, cfg)
    oracle_ok = is_acceptable(oracle.final.plan.clinical_metrics, case, cfg)
    outcome = (
        "both_reach"
        if manual_ok and oracle_ok
        else "oracle_only"
        if oracle_ok
        else "manual_only"
        if manual_ok
        else "neither_reaches"
    )
    return {
        "valid": True,
        "environment_version": cfg.environment_version,
        "seed": seed,
        "case_id": case.case_id,
        "outcome": outcome,
        "initial_violation": clinical_violation_score(
            manual.steps[0].plan.clinical_metrics, case, cfg
        ),
        "manual_violation": manual_score,
        "oracle_violation": oracle_score,
        "manual_acceptable": manual_ok,
        "oracle_reachable": oracle_ok,
        "manual_actions": len(manual.steps) - 1,
        "oracle_actions": len(oracle.steps) - 1,
        "manual_stopping_reason": manual.stopping_reason,
        "oracle_stopping_reason": oracle.stopping_reason,
        "manual_action_types": "|".join(step.action.kind for step in manual.steps[1:]),
        "oracle_action_types": "|".join(step.action.kind for step in oracle.steps[1:]),
    }


def save_cohort_figure(rows: list[dict[str, object]], output_path: Path) -> None:
    outcomes = Counter(str(row["outcome"]) for row in rows)
    labels = ["Both reach", "Oracle only", "Manual only", "Neither reaches"]
    keys = ["both_reach", "oracle_only", "manual_only", "neither_reaches"]
    colors = ["#4daf4a", "#377eb8", "#ff7f00", "#e41a1c"]
    manual_score = np.array([float(row["manual_violation"]) for row in rows])
    oracle_score = np.array([float(row["oracle_violation"]) for row in rows])
    initial_score = np.array([float(row["initial_violation"]) for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    values = [outcomes[key] for key in keys]
    axes[0].bar(labels, values, color=colors, alpha=0.85)
    for index, value in enumerate(values):
        axes[0].text(index, value + 0.15, str(value), ha="center")
    axes[0].set_ylabel("cases")
    axes[0].set_title("Can the goals be reached?")
    axes[0].tick_params(axis="x", rotation=25)

    floor = 1e-7
    axes[1].scatter(
        np.maximum(oracle_score, floor),
        np.maximum(manual_score, floor),
        c=[colors[keys.index(str(row["outcome"]))] for row in rows],
        edgecolor="black",
        linewidth=0.4,
    )
    upper = max(float(np.max(manual_score)), float(np.max(oracle_score)), 1e-3)
    axes[1].plot([floor, upper], [floor, upper], color="#555555", linestyle="--", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("oracle final violation")
    axes[1].set_ylabel("manual final violation")
    axes[1].set_title("How far is the manual policy from the oracle?")
    axes[1].grid(alpha=0.2)

    for index in range(len(rows)):
        axes[2].plot(
            [0, 1, 2],
            [initial_score[index], manual_score[index], oracle_score[index]],
            color="#7b3294",
            alpha=0.35,
        )
    axes[2].set_xticks([0, 1, 2], ["initial", "manual", "oracle"])
    axes[2].set_ylabel("clinical violation score")
    axes[2].set_title("Priority-independent improvement")
    axes[2].grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"Manual planner versus high-level search oracle: {len(rows)} cases",
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare manual planning with a high-level search oracle")
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oracle_pilot"))
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
    with (args.output_dir / "oracle_pilot_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    save_cohort_figure(rows, args.output_dir / "01_oracle_pilot.png")

    comparison = next(
        (row for row in rows if row["outcome"] == "oracle_only"),
        rows[0],
    )
    cfg = SimulationConfig()
    case = generate_case(int(comparison["seed"]), cfg)
    influence, _, _ = build_dose_influence(case, cfg)
    manual = run_manual_planner(case, influence, cfg)
    oracle = run_high_level_oracle(case, influence, cfg)
    save_oracle_case_comparison(
        case,
        manual,
        oracle,
        args.output_dir / "02_case_comparison.png",
        cfg,
    )

    counts = Counter(str(row["outcome"]) for row in rows)
    print(f"valid_cases={len(rows)}")
    for key in ["both_reach", "oracle_only", "manual_only", "neither_reaches"]:
        print(f"{key}={counts[key]}")
    print(f"comparison_seed={comparison['seed']}")


if __name__ == "__main__":
    main()
