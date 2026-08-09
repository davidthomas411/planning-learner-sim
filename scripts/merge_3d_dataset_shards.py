import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def canonical_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)


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
    summary = {
        "status": "validated merged train/validation dataset",
        "shards": [str(path) for path in args.shards],
        "retained_by_split": counts,
        "attempted": len(attempt_rows),
        "endpoint_sha256": hashlib.sha256(endpoint_text.encode("utf-8")).hexdigest(),
        "trajectory_sha256": hashlib.sha256(trajectory_text.encode("utf-8")).hexdigest(),
        "attempt_manifest_sha256": hashlib.sha256(attempts_text.encode("utf-8")).hexdigest(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
