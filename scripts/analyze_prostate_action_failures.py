import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def action_counts(rows: list[dict[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(action for action in row.get("action_sequence", "").split("|") if action)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit prostate policy action sequences")
    parser.add_argument("case_metric_csvs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = read_rows(args.case_metric_csvs)
    rows = [row for row in rows if row.get("action_sequence")]
    trajectory = [row for row in rows if row["condition"] == "trajectory"]
    moderate = [row for row in trajectory if row["difficulty"] == "moderate"]
    seeds = sorted({int(row["training_seed"]) for row in moderate})
    accepted_by_seed = []
    for seed in seeds:
        seed_rows = [row for row in moderate if int(row["training_seed"]) == seed]
        accepted_by_seed.append(sum(row["acceptable"] == "True" for row in seed_rows) / len(seed_rows))

    accepted = [row for row in moderate if row["acceptable"] == "True"]
    failed = [row for row in moderate if row["acceptable"] == "False"]
    accepted_counts = action_counts(accepted)
    failed_counts = action_counts(failed)
    all_counts = accepted_counts + failed_counts
    top_actions = [action for action, _ in all_counts.most_common(6)]
    accepted_rate = [accepted_counts[action] / len(accepted) for action in top_actions]
    failed_rate = [failed_counts[action] / max(len(failed), 1) for action in top_actions]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "included_training_seeds": seeds,
        "moderate_case_seed_evaluations": len(moderate),
        "moderate_acceptable": len(accepted),
        "moderate_failed": len(failed),
        "moderate_acceptable_rate_by_seed": dict(zip(map(str, seeds), accepted_by_seed)),
        "failed_at_action_limit": sum(int(row["high_level_actions"]) == 10 for row in failed),
        "dominant_failed_sequence": (
            Counter(row["action_sequence"] for row in failed).most_common(1)[0] if failed else None
        ),
        "mean_action_occurrences_per_rollout": {
            action: {"acceptable": accepted_rate[index], "failed": failed_rate[index]}
            for index, action in enumerate(top_actions)
        },
    }
    (args.output_dir / "action_failure_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    axes[0].bar([str(seed) for seed in seeds], accepted_by_seed, color="#4c78a8")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Training seed")
    axes[0].set_ylabel("Acceptable-plan rate")
    axes[0].set_title("Moderate cases vary by training seed")
    for index, value in enumerate(accepted_by_seed):
        axes[0].text(index, value + 0.025, f"{value:.2f}", ha="center")

    positions = np.arange(len(top_actions))
    width = 0.38
    axes[1].barh(positions - width / 2, accepted_rate, height=width, label="acceptable", color="#59a14f")
    axes[1].barh(positions + width / 2, failed_rate, height=width, label="failed", color="#e15759")
    axes[1].set_yticks(positions, [action.replace("_", " ") for action in top_actions])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean occurrences per rollout")
    axes[1].set_title("Actions selected on moderate cases")
    axes[1].legend(frameon=False)
    figure.suptitle("Prostate trajectory-policy failure audit")
    figure.savefig(args.output_dir / "02_action_failure_summary.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
