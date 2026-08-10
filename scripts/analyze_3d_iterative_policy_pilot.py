import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> str:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def boolean(value: str) -> bool:
    return value.lower() == "true"


def hierarchical_interval(matrix: np.ndarray, iterations: int, rng: np.random.Generator) -> list[float]:
    seed_count, case_count = matrix.shape
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        seed_indices = rng.integers(0, seed_count, seed_count)
        case_indices = rng.integers(0, case_count, case_count)
        estimates[index] = matrix[np.ix_(seed_indices, case_indices)].mean()
    return [float(value) for value in np.quantile(estimates, (0.025, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine and audit deterministic iterative-policy pilot runs")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [row for directory in args.run_dirs for row in read_csv(directory / "case_metrics.csv")]
    history = [row for directory in args.run_dirs for row in read_csv(directory / "training_history.csv")]
    diagnostics = [row for directory in args.run_dirs for row in read_csv(directory / "training_diagnostics.csv")]
    run_summaries = [json.loads((directory / "summary.json").read_text(encoding="utf-8")) for directory in args.run_dirs]
    rows.sort(key=lambda row: (int(row["training_seed"]), row["condition"], row["case_id"]))
    history.sort(key=lambda row: (int(row["seed"]), row["condition"], int(row["update"])))
    seeds = sorted({int(row["training_seed"]) for row in rows})
    cases = sorted({row["case_id"] for row in rows})
    if seeds != list(range(min(seeds), max(seeds) + 1)):
        raise ValueError(f"noncontiguous or duplicate seed coverage: {seeds}")
    expected_keys = {(seed, condition, case_id) for seed in seeds for condition in ("endpoint", "trajectory") for case_id in cases}
    keyed = {(int(row["training_seed"]), row["condition"], row["case_id"]): row for row in rows}
    if len(keyed) != len(rows) or set(keyed) != expected_keys:
        raise ValueError("case metrics do not contain one complete endpoint/trajectory grid")
    update_counts = {
        len([row for row in history if int(row["seed"]) == seed and row["condition"] == condition])
        for seed in seeds for condition in ("endpoint", "trajectory")
    }
    if len(update_counts) != 1:
        raise ValueError(f"seed-condition update counts differ: {sorted(update_counts)}")
    training_cases = {int(summary["training_cases"]) for summary in run_summaries}
    if len(training_cases) != 1:
        raise ValueError(f"training-case counts differ: {sorted(training_cases)}")

    accept_difference = np.empty((len(seeds), len(cases)), dtype=np.float64)
    violation_difference = np.empty_like(accept_difference)
    endpoint_accept = np.empty_like(accept_difference)
    trajectory_accept = np.empty_like(accept_difference)
    endpoint_violation = np.empty_like(accept_difference)
    trajectory_violation = np.empty_like(accept_difference)
    for seed_index, seed in enumerate(seeds):
        for case_index, case_id in enumerate(cases):
            endpoint = keyed[(seed, "endpoint", case_id)]
            trajectory = keyed[(seed, "trajectory", case_id)]
            endpoint_accept[seed_index, case_index] = boolean(endpoint["acceptable"])
            trajectory_accept[seed_index, case_index] = boolean(trajectory["acceptable"])
            endpoint_violation[seed_index, case_index] = float(endpoint["violation_score"])
            trajectory_violation[seed_index, case_index] = float(trajectory["violation_score"])
    accept_difference[:] = trajectory_accept - endpoint_accept
    violation_difference[:] = trajectory_violation - endpoint_violation

    rng = np.random.default_rng(20260809)
    per_seed = []
    for seed_index, seed in enumerate(seeds):
        per_seed.append({
            "training_seed": seed,
            "endpoint_acceptable_rate": float(endpoint_accept[seed_index].mean()),
            "trajectory_acceptable_rate": float(trajectory_accept[seed_index].mean()),
            "acceptable_rate_difference": float(accept_difference[seed_index].mean()),
            "endpoint_mean_violation": float(endpoint_violation[seed_index].mean()),
            "trajectory_mean_violation": float(trajectory_violation[seed_index].mean()),
            "violation_difference": float(violation_difference[seed_index].mean()),
        })
    difficulties = {case_id: keyed[(seeds[0], "endpoint", case_id)]["difficulty"] for case_id in cases}
    by_difficulty = {}
    for difficulty in ("easy", "moderate", "hard"):
        indices = [index for index, case_id in enumerate(cases) if difficulties[case_id] == difficulty]
        by_difficulty[difficulty] = {
            "cases": len(indices),
            "endpoint_acceptable_rate": float(endpoint_accept[:, indices].mean()),
            "trajectory_acceptable_rate": float(trajectory_accept[:, indices].mean()),
            "endpoint_mean_violation": float(endpoint_violation[:, indices].mean()),
            "trajectory_mean_violation": float(trajectory_violation[:, indices].mean()),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_hash = write_csv(args.output_dir / "case_metrics.csv", rows)
    history_hash = write_csv(args.output_dir / "training_history.csv", history)
    write_csv(args.output_dir / "per_seed_metrics.csv", per_seed)
    summary = {
        "status": "completed matched iterative-policy variance pilot; not the primary test-set result",
        "training_cases": next(iter(training_cases)),
        "validation_cases": len(cases),
        "training_seeds": len(seeds),
        "updates_per_seed_condition": next(iter(update_counts)),
        "paired_case_seed_evaluations": int(accept_difference.size),
        "same_case_condition_grid": True,
        "case_metrics_sha256": metrics_hash,
        "training_history_sha256": history_hash,
        "endpoint_acceptable_rate": float(endpoint_accept.mean()),
        "trajectory_acceptable_rate": float(trajectory_accept.mean()),
        "paired_acceptable_rate_difference": float(accept_difference.mean()),
        "paired_acceptable_rate_difference_hierarchical_95ci": hierarchical_interval(accept_difference, args.bootstrap_iterations, rng),
        "endpoint_mean_violation": float(endpoint_violation.mean()),
        "trajectory_mean_violation": float(trajectory_violation.mean()),
        "paired_mean_violation_difference": float(violation_difference.mean()),
        "paired_mean_violation_difference_hierarchical_95ci": hierarchical_interval(violation_difference, args.bootstrap_iterations, rng),
        "seeds_favoring_trajectory_acceptability": sum(row["acceptable_rate_difference"] > 0 for row in per_seed),
        "seeds_tied_on_acceptability": sum(row["acceptable_rate_difference"] == 0 for row in per_seed),
        "seeds_favoring_trajectory_violation": sum(row["violation_difference"] < 0 for row in per_seed),
        "paired_acceptability_outcomes": {
            "both_acceptable": int(np.sum((endpoint_accept == 1) & (trajectory_accept == 1))),
            "trajectory_only": int(np.sum((endpoint_accept == 0) & (trajectory_accept == 1))),
            "endpoint_only": int(np.sum((endpoint_accept == 1) & (trajectory_accept == 0))),
            "neither": int(np.sum((endpoint_accept == 0) & (trajectory_accept == 0))),
        },
        "by_difficulty": by_difficulty,
        "endpoint_training_action_accuracy_diagnostic_only": float(np.mean([
            float(row["action_accuracy"]) for row in diagnostics if row["condition"] == "endpoint"
        ])),
        "trajectory_training_action_accuracy": float(np.mean([
            float(row["action_accuracy"]) for row in diagnostics if row["condition"] == "trajectory"
        ])),
    }
    if "diagnostic_action_accuracy" in history[0]:
        final_history = [row for row in history if int(row["update"]) == 60]
        summary["endpoint_final_training_action_accuracy"] = float(np.mean([
            float(row["diagnostic_action_accuracy"])
            for row in final_history if row["condition"] == "endpoint"
        ]))
        summary["trajectory_final_training_action_accuracy"] = float(np.mean([
            float(row["diagnostic_action_accuracy"])
            for row in final_history if row["condition"] == "trajectory"
        ]))
        summary["endpoint_final_training_action_cross_entropy"] = float(np.mean([
            float(row["diagnostic_action_cross_entropy"])
            for row in final_history if row["condition"] == "endpoint"
        ]))
        summary["trajectory_final_training_action_cross_entropy"] = float(np.mean([
            float(row["diagnostic_action_cross_entropy"])
            for row in final_history if row["condition"] == "trajectory"
        ]))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    colors = {"endpoint": "#4E79A7", "trajectory": "#E15759"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    x = np.arange(2)
    for row in per_seed:
        axes[0, 0].plot(x, [row["endpoint_acceptable_rate"], row["trajectory_acceptable_rate"]], marker="o", color="#777777", alpha=0.65)
    axes[0, 0].set_xticks(x, ("endpoint only", "trajectory supervised"))
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("Acceptable-plan rate")
    axes[0, 0].set_title("Each line is one training seed")
    for row in per_seed:
        axes[0, 1].plot(x, [row["endpoint_mean_violation"], row["trajectory_mean_violation"]], marker="o", color="#777777", alpha=0.65)
    axes[0, 1].set_xticks(x, ("endpoint only", "trajectory supervised"))
    axes[0, 1].set_ylabel("Mean violation score")
    axes[0, 1].set_title("Violation by training seed")
    difficulty_x = np.arange(3)
    axes[1, 0].bar(difficulty_x - 0.18, [by_difficulty[d]["endpoint_acceptable_rate"] for d in ("easy", "moderate", "hard")], 0.36, label="endpoint only", color=colors["endpoint"])
    axes[1, 0].bar(difficulty_x + 0.18, [by_difficulty[d]["trajectory_acceptable_rate"] for d in ("easy", "moderate", "hard")], 0.36, label="trajectory supervised", color=colors["trajectory"])
    axes[1, 0].set_xticks(difficulty_x, ("easy", "moderate", "hard"))
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel("Acceptable-plan rate")
    axes[1, 0].set_title("Validation performance by difficulty")
    axes[1, 0].legend()
    axes[1, 1].hist(violation_difference.ravel(), bins=30, color="#59A14F")
    axes[1, 1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_xlabel("Trajectory minus endpoint violation")
    axes[1, 1].set_ylabel("Case-seed evaluations")
    axes[1, 1].set_title("Negative values favor trajectory supervision")
    fig.suptitle("Matched iterative-policy variance pilot", fontweight="bold")
    fig.savefig(args.output_dir / "01_variance_pilot_summary.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
