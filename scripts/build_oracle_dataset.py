import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dosim_sim import SimulationConfig, build_dose_influence, generate_case, run_high_level_oracle
from dosim_sim.dataset import action_record, case_features, plan_state
from dosim_sim.objective import is_acceptable


def build_seed(payload: tuple[int, str]) -> dict[str, object]:
    seed, cases_dir_text = payload
    cases_dir = Path(cases_dir_text)
    cfg = SimulationConfig()
    try:
        case = generate_case(seed, cfg)
    except ValueError as error:
        return {"valid": False, "seed": seed, "reason": str(error), "reachable": False}
    influence, _, _ = build_dose_influence(case, cfg)
    trajectory = run_high_level_oracle(case, influence, cfg)
    reachable = is_acceptable(trajectory.final.plan.clinical_metrics, case, cfg)
    record: dict[str, object] = {
        "valid": True,
        "seed": seed,
        "case_id": case.case_id,
        "environment_version": cfg.environment_version,
        "reachable": reachable,
        "stopping_reason": trajectory.stopping_reason,
        "manual_actions": len(trajectory.steps) - 1,
        "initial_violation": trajectory.steps[0].violation_score,
        "final_violation": trajectory.final.violation_score,
    }
    if not reachable:
        return record

    endpoint = {
        "case_id": case.case_id,
        "seed": seed,
        "environment_version": cfg.environment_version,
        "case_features": case_features(case),
        "initial_state": plan_state(trajectory.steps[0].plan, case, cfg),
        "final_state": plan_state(trajectory.final.plan, case, cfg),
    }
    transitions = []
    for before, after in zip(trajectory.steps, trajectory.steps[1:]):
        transitions.append(
            {
                "manual_step": after.step,
                "state": plan_state(before.plan, case, cfg),
                "action": action_record(after.action),
                "next_state": plan_state(after.plan, case, cfg),
            }
        )
    trajectory_record = {
        **endpoint,
        "trajectory": transitions,
        "stopping_reason": trajectory.stopping_reason,
    }
    doses = np.stack([step.plan.dose for step in trajectory.steps]).astype(np.float32)
    intensities = np.stack([step.plan.intensities for step in trajectory.steps]).astype(np.float32)
    cases_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cases_dir / f"{case.case_id}.npz",
        body=case.body.astype(np.uint8),
        target=case.target.astype(np.uint8),
        oars=np.stack(case.oars).astype(np.uint8),
        dose_influence=influence.astype(np.float32),
        trajectory_doses=doses,
        trajectory_intensities=intensities,
    )
    record["endpoint"] = endpoint
    record["trajectory_record"] = trajectory_record
    record["action_types"] = [step.action.kind for step in trajectory.steps[1:]]
    return record


def save_audit(records: list[dict[str, object]], output_path: Path) -> None:
    reachable = [record for record in records if bool(record.get("reachable"))]
    action_counts: Counter[str] = Counter()
    for record in reachable:
        action_counts.update(record.get("action_types", []))
    lengths = [int(record["manual_actions"]) for record in reachable]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].bar(
        ["Oracle-reachable", "Not reached"],
        [len(reachable), len(records) - len(reachable)],
        color=["#4daf4a", "#e41a1c"],
    )
    axes[0].set_ylabel("valid generated cases")
    axes[0].set_title("Dataset inclusion is explicit")

    bins = np.arange(0.5, max(lengths, default=1) + 1.5, 1)
    axes[1].hist(lengths, bins=bins, color="#377eb8", alpha=0.8)
    axes[1].set_xlabel("high-level actions")
    axes[1].set_ylabel("included cases")
    axes[1].set_title("Successful trajectory lengths")

    keys = [
        "increase_target_priority",
        "increase_hotspot_priority",
        "increase_oar_priority",
        "add_beam",
        "remove_beam",
    ]
    labels = ["Target priority", "Hot-spot priority", "OAR priority", "Add beam", "Remove beam"]
    values = [action_counts[key] for key in keys]
    axes[2].barh(labels, values, color="#7b3294", alpha=0.8)
    for index, value in enumerate(values):
        axes[2].text(value + 0.15, index, str(value), va="center")
    axes[2].set_xlabel("stored action labels")
    axes[2].set_title("Trajectory supervision content")
    fig.suptitle("Oracle-trajectory dataset construction audit", fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build matched endpoint and high-level trajectory datasets")
    parser.add_argument("--reachable-cases", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oracle_dataset_pilot"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = args.output_dir / "cases"

    all_records: list[dict[str, object]] = []
    included: list[dict[str, object]] = []
    next_seed = args.seed_start
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        while len(included) < args.reachable_cases:
            batch_size = max(4, args.reachable_cases - len(included))
            seeds = list(range(next_seed, next_seed + batch_size))
            next_seed = seeds[-1] + 1
            payloads = [(seed, str(cases_dir)) for seed in seeds]
            for record in executor.map(build_seed, payloads):
                all_records.append(record)
                if bool(record.get("reachable")):
                    included.append(record)
                    if len(included) == args.reachable_cases:
                        break

    with (args.output_dir / "endpoints.jsonl").open("w", encoding="utf-8") as handle:
        for record in included:
            handle.write(json.dumps(record["endpoint"], separators=(",", ":")) + "\n")
    with (args.output_dir / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for record in included:
            handle.write(json.dumps(record["trajectory_record"], separators=(",", ":")) + "\n")

    included_ids = {str(record["case_id"]) for record in included}
    for case_file in cases_dir.glob("*.npz"):
        if case_file.stem not in included_ids:
            case_file.unlink()

    manifest_fields = [
        "seed",
        "case_id",
        "environment_version",
        "valid",
        "reachable",
        "stopping_reason",
        "manual_actions",
        "initial_violation",
        "final_violation",
        "reason",
    ]
    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)
    save_audit(all_records, args.output_dir / "01_dataset_audit.png")
    print(f"attempted_cases={len(all_records)}")
    print(f"included_reachable_cases={len(included)}")
    print(f"endpoint_records={len(included)}")
    print(f"trajectory_records={len(included)}")


if __name__ == "__main__":
    main()
