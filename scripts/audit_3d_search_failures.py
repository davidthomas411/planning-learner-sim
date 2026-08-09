import argparse
import csv
import json
import time
from pathlib import Path

import torch

from dosim_sim.planning3d import HighLevelSearchConfig3D, is_acceptable_3d, run_high_level_search_3d
from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D
from dosim_sim.volume3d import generate_case_3d


ANGLES = tuple(float(value) for value in range(0, 360, 30))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a deeper high-level search only to reference-reachable failures")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--fluence-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--add-candidates", type=int, default=3)
    parser.add_argument("--remove-candidates", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/3d_search_failure_audit"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    failures = [row for row in source if row["reference_acceptable"] == "True" and row["search_acceptable"] == "False"]
    device = torch.device(args.device)
    cfg = HighLevelSearchConfig3D(
        max_steps=args.max_steps,
        beam_width=args.beam_width,
        add_candidates=args.add_candidates,
        remove_candidates=args.remove_candidates,
        optimizer_iterations=args.iterations,
    )
    rows = []
    started = time.perf_counter()
    for index, source_row in enumerate(failures, start=1):
        seed = int(source_row["seed"])
        difficulty = source_row["difficulty"]
        case = generate_case_3d(seed, args.grid_size, difficulty=difficulty)
        engine = TorchImplicitDoseEngine3D(case, ANGLES, args.fluence_size, device=device, dtype=torch.float16)
        trajectory = run_high_level_search_3d(case, engine, cfg)
        actions = [step.action.kind for step in trajectory.steps[1:] if step.action is not None]
        rows.append({
            "seed": seed,
            "difficulty": difficulty,
            "shallow_violation": float(source_row["search_violation"]),
            "deep_acceptable": is_acceptable_3d(trajectory.final.plan.metrics, case, cfg),
            "deep_violation": trajectory.final.violation_score,
            "manual_actions": len(actions),
            "actions": "|".join(actions),
            "stopping_reason": trajectory.stopping_reason,
        })
        print(f"audited {index}/{len(failures)}: seed={seed} acceptable={rows[-1]['deep_acceptable']}", flush=True)
    elapsed = time.perf_counter() - started
    if rows:
        with (args.output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    recovered = sum(bool(row["deep_acceptable"]) for row in rows)
    reference_reachable = sum(row["reference_acceptable"] == "True" for row in source)
    shallow_acceptable = sum(row["search_acceptable"] == "True" and row["reference_acceptable"] == "True" for row in source)
    summary = {
        "audited_failures": len(rows),
        "recovered": recovered,
        "elapsed_seconds": elapsed,
        "reference_reachable_cases": reference_reachable,
        "combined_search_coverage_among_reference_reachable": (shallow_acceptable + recovered) / max(reference_reachable, 1),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
