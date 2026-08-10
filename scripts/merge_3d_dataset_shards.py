import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def canonical_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)


def save_summary_figure(endpoint_rows: list[dict], trajectory_rows: list[dict], attempt_rows: list[dict], path: Path) -> None:
    difficulties = ("easy", "moderate", "hard")
    attempted = [sum(row["difficulty"] == value for row in attempt_rows) for value in difficulties]
    retained = [sum(row["difficulty"] == value for row in endpoint_rows) for value in difficulties]
    x = np.arange(len(difficulties))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].bar(x - 0.18, attempted, 0.36, label="attempted", color="#4E79A7")
    axes[0, 0].bar(x + 0.18, retained, 0.36, label="retained", color="#F28E2B")
    axes[0, 0].set_xticks(x, difficulties)
    axes[0, 0].set_ylabel("Cases")
    axes[0, 0].set_title("Retention by difficulty")
    axes[0, 0].legend()
    bottoms = np.zeros(2)
    for difficulty, color in zip(difficulties, ("#59A14F", "#EDC948", "#E15759"), strict=True):
        values = [sum(row["split"] == split and row["difficulty"] == difficulty for row in endpoint_rows) for split in ("train", "validation")]
        axes[0, 1].bar(("train", "validation"), values, bottom=bottoms, label=difficulty, color=color)
        bottoms += values
    axes[0, 1].set_ylabel("Retained cases")
    axes[0, 1].set_title("Frozen partition composition")
    axes[0, 1].legend()
    lengths = [len(row["trajectory"]) - 1 for row in trajectory_rows]
    axes[1, 0].hist(lengths, bins=np.arange(0.5, 11.5), color="#4E79A7")
    axes[1, 0].set_xlabel("High-level actions before stop")
    axes[1, 0].set_ylabel("Retained cases")
    axes[1, 0].set_title("Demonstration length")
    actions = Counter(
        transition["action_name"]
        for row in trajectory_rows
        for transition in row["trajectory"]
        if transition["action_name"] != "stop"
    ).most_common(10)
    labels = [name.replace("_", " ") for name, _ in reversed(actions)]
    values = [count for _, count in reversed(actions)]
    axes[1, 1].barh(labels, values, color="#59A14F")
    axes[1, 1].set_xlabel("Recorded actions")
    axes[1, 1].set_title("Ten most frequent actions")
    fig.suptitle("Three-dimensional 300-case train/validation dataset", fontweight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and merge nonoverlapping 3D dataset shards")
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--expected-train", type=int, required=True)
    parser.add_argument("--expected-validation", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    endpoint_rows: list[dict] = []
    trajectory_rows: list[dict] = []
    attempt_rows: list[dict] = []
    for shard in args.shards:
        endpoint_rows.extend(read_jsonl(shard / "endpoint_view.jsonl"))
        trajectory_rows.extend(read_jsonl(shard / "trajectory_view.jsonl"))
        attempt_rows.extend(read_jsonl(shard / "attempt_manifest.jsonl"))

    endpoint_by_id = {row["case_id"]: row for row in endpoint_rows}
    trajectory_by_id = {row["case_id"]: row for row in trajectory_rows}
    if len(endpoint_by_id) != len(endpoint_rows) or len(trajectory_by_id) != len(trajectory_rows):
        raise ValueError("duplicate retained case identifier across shards")
    if set(endpoint_by_id) != set(trajectory_by_id):
        raise ValueError("endpoint and trajectory retained case identifiers differ")
    if len({(row["split"], row["split_ordinal"]) for row in attempt_rows}) != len(attempt_rows):
        raise ValueError("duplicate attempted split ordinal across shards")
    for case_id, endpoint in endpoint_by_id.items():
        trajectory = trajectory_by_id[case_id]
        shared = ("seed", "difficulty", "split", "split_ordinal", "initial_features", "final_features", "final_settings")
        if any(endpoint[field] != trajectory[field] for field in shared):
            raise ValueError(f"endpoint/trajectory mismatch for {case_id}")
        if endpoint.get("anatomy", "generic") != trajectory.get("anatomy", "generic"):
            raise ValueError(f"endpoint/trajectory anatomy mismatch for {case_id}")

    counts = {
        split: sum(row["split"] == split for row in endpoint_rows)
        for split in ("train", "validation")
    }
    expected = {"train": args.expected_train, "validation": args.expected_validation}
    if counts != expected:
        raise ValueError(f"retained split counts {counts} do not equal expected {expected}")
    if any(row["split"] not in expected for row in endpoint_rows):
        raise ValueError("pilot merge may contain only train and validation records")

    order = {"train": 0, "validation": 1}
    endpoint_rows.sort(key=lambda row: (order[row["split"]], int(row["split_ordinal"])))
    trajectory_rows.sort(key=lambda row: (order[row["split"]], int(row["split_ordinal"])))
    attempt_rows.sort(key=lambda row: (order[row["split"]], int(row["split_ordinal"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    endpoint_text = canonical_jsonl(endpoint_rows)
    trajectory_text = canonical_jsonl(trajectory_rows)
    attempts_text = canonical_jsonl(attempt_rows)
    (args.output_dir / "endpoint_view.jsonl").write_text(endpoint_text, encoding="utf-8")
    (args.output_dir / "trajectory_view.jsonl").write_text(trajectory_text, encoding="utf-8")
    (args.output_dir / "attempt_manifest.jsonl").write_text(attempts_text, encoding="utf-8")
    save_summary_figure(endpoint_rows, trajectory_rows, attempt_rows, args.output_dir / "01_dataset_summary.png")
    attempted_by_difficulty = Counter(row["difficulty"] for row in attempt_rows)
    retained_by_difficulty = Counter(row["difficulty"] for row in endpoint_rows)
    trajectory_lengths = [len(row["trajectory"]) - 1 for row in trajectory_rows]
    summary = {
        "status": "validated merged train/validation dataset",
        "anatomies": sorted({row.get("anatomy", "generic") for row in endpoint_rows}),
        "shards": [str(path) for path in args.shards],
        "retained_by_split": counts,
        "attempted": len(attempt_rows),
        "retained_by_difficulty": dict(retained_by_difficulty),
        "attempted_by_difficulty": dict(attempted_by_difficulty),
        "retention_rate_by_difficulty": {
            difficulty: retained_by_difficulty[difficulty] / attempted_by_difficulty[difficulty]
            for difficulty in ("easy", "moderate", "hard")
        },
        "search_misses_among_reference_reachable": sum(
            not row["search_acceptable"] and row["reference_acceptable"] for row in attempt_rows
        ),
        "mean_high_level_actions": float(np.mean(trajectory_lengths)),
        "median_high_level_actions": float(np.median(trajectory_lengths)),
        "endpoint_sha256": hashlib.sha256(endpoint_text.encode("utf-8")).hexdigest(),
        "trajectory_sha256": hashlib.sha256(trajectory_text.encode("utf-8")).hexdigest(),
        "attempt_manifest_sha256": hashlib.sha256(attempts_text.encode("utf-8")).hexdigest(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
