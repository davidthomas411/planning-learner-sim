import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dosim_sim.optimizer3d import objective_value_3d
from dosim_sim.planning3d import (
    HighLevelSearchConfig3D,
    clinical_violation_score_3d,
    is_acceptable_3d,
    run_high_level_search_3d,
    run_reference_optimizer_3d,
)
from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D
from dosim_sim.volume3d import generate_case_3d


ANGLES = tuple(float(value) for value in range(0, 360, 30))


def save_plot(rows: list[dict[str, object]], path: Path) -> None:
    difficulties = ("easy", "moderate", "hard")
    colors = {"easy": "#59A14F", "moderate": "#F28E2B", "hard": "#E15759"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    x = np.arange(3)
    initial = []
    search = []
    reference = []
    for difficulty in difficulties:
        subset = [row for row in rows if row["difficulty"] == difficulty]
        initial.append(np.mean([bool(row["initial_acceptable"]) for row in subset]))
        search.append(np.mean([bool(row["search_acceptable"]) for row in subset]))
        reference.append(np.mean([bool(row["reference_acceptable"]) for row in subset]))
    axes[0, 0].bar(x - 0.25, initial, 0.25, label="initial")
    axes[0, 0].bar(x, search, 0.25, label="bounded high-level search")
    axes[0, 0].bar(x + 0.25, reference, 0.25, label="continuous reference")
    axes[0, 0].set_xticks(x, difficulties)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("Proportion acceptable")
    axes[0, 0].set_title("Reachability by difficulty")
    axes[0, 0].legend(fontsize=8)

    for difficulty in difficulties:
        subset = [row for row in rows if row["difficulty"] == difficulty]
        for row in subset:
            axes[0, 1].plot(
                [0, 1],
                [float(row["initial_violation"]), float(row["search_violation"])],
                color=colors[difficulty],
                alpha=0.65,
            )
    axes[0, 1].set_xticks([0, 1], ["initial", "search result"])
    axes[0, 1].set_ylabel("Clinical violation score")
    axes[0, 1].set_title("Effect of high-level planning")

    for index, difficulty in enumerate(difficulties):
        subset = [row for row in rows if row["difficulty"] == difficulty]
        values = [float(row["canonical_objective_gap"]) for row in subset]
        axes[1, 0].scatter(np.full(len(values), index), values, color=colors[difficulty], s=55)
    axes[1, 0].axhline(0, color="black", ls="--", lw=1)
    axes[1, 0].set_xticks(x, difficulties)
    axes[1, 0].set_ylabel("(search objective - reference) / |reference|")
    axes[1, 0].set_title("Terminal gap to fixed-priority reference")

    action_counts = Counter()
    for row in rows:
        for action in str(row["actions"]).split("|"):
            if action:
                action_counts[action] += 1
    labels = list(action_counts)
    axes[1, 1].barh(labels, [action_counts[label] for label in labels], color="#4E79A7")
    axes[1, 1].set_xlabel("Recorded actions")
    axes[1, 1].set_title("High-level action composition")
    fig.suptitle(
        f"Bounded three-dimensional planning pilot: {len(rows)} cases\n"
        "Search changes beam angles or named priorities; the inner optimizer changes fluence",
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot bounded high-level planning in 3D")
    parser.add_argument("--cases-per-stratum", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=13000)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--fluence-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--reference-iterations", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--add-candidates", type=int, default=2)
    parser.add_argument("--remove-candidates", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/3d_search_pilot"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    dtype = getattr(torch, args.dtype)
    cfg = HighLevelSearchConfig3D(
        max_steps=args.max_steps,
        beam_width=args.beam_width,
        add_candidates=args.add_candidates,
        remove_candidates=args.remove_candidates,
        optimizer_iterations=args.iterations,
    )
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    case_number = 0
    for difficulty in ("easy", "moderate", "hard"):
        for _ in range(args.cases_per_stratum):
            seed = args.seed_start + case_number
            case_number += 1
            case = generate_case_3d(seed, args.grid_size, difficulty=difficulty)
            engine = TorchImplicitDoseEngine3D(case, ANGLES, args.fluence_size, device=device, dtype=dtype)
            trajectory = run_high_level_search_3d(case, engine, cfg)
            reference_plan = run_reference_optimizer_3d(case, engine, args.reference_iterations)
            initial = trajectory.steps[0]
            final = trajectory.final
            reference_violation = clinical_violation_score_3d(reference_plan.metrics, case, cfg)
            final_objective = objective_value_3d(case, final.plan.dose.detach().float().cpu().numpy())
            reference_objective = objective_value_3d(case, reference_plan.dose.detach().float().cpu().numpy())
            actions = [step.action.kind for step in trajectory.steps[1:] if step.action is not None]
            row = {
                "seed": seed,
                "case_id": case.case_id,
                "difficulty": difficulty,
                "n_oars": len(case.oars),
                "target_oar_overlap_fraction": sum(np.count_nonzero(case.target & mask) for mask in case.oars) / max(np.count_nonzero(case.target), 1),
                "initial_acceptable": is_acceptable_3d(initial.plan.metrics, case, cfg),
                "search_acceptable": is_acceptable_3d(final.plan.metrics, case, cfg),
                "reference_acceptable": is_acceptable_3d(reference_plan.metrics, case, cfg),
                "initial_violation": initial.violation_score,
                "search_violation": final.violation_score,
                "reference_violation": reference_violation,
                "manual_actions": len(actions),
                "actions": "|".join(actions),
                "stopping_reason": trajectory.stopping_reason,
                "search_objective": final_objective,
                "reference_objective": reference_objective,
                "canonical_objective_gap": (final_objective - reference_objective) / max(abs(reference_objective), 1e-8),
            }
            rows.append(row)
            print(f"completed {len(rows)}/{3 * args.cases_per_stratum}: {difficulty} seed={seed} search={row['search_acceptable']} reference={row['reference_acceptable']}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    with (args.output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_plot(rows, args.output_dir / "01_3d_search_pilot.png")
    summary = {
        "status": "engineering search pilot; not a learner comparison",
        "gpu": torch.cuda.get_device_name(device),
        "grid_size": args.grid_size,
        "fluence_size": args.fluence_size,
        "cases": len(rows),
        "elapsed_seconds": elapsed,
        "seconds_per_case": elapsed / len(rows),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "search_acceptable_rate": np.mean([bool(row["search_acceptable"]) for row in rows]),
        "reference_acceptable_rate": np.mean([bool(row["reference_acceptable"]) for row in rows]),
        "by_difficulty": {
            difficulty: {
                "search_acceptable_rate": np.mean([bool(row["search_acceptable"]) for row in rows if row["difficulty"] == difficulty]),
                "reference_acceptable_rate": np.mean([bool(row["reference_acceptable"]) for row in rows if row["difficulty"] == difficulty]),
                "median_search_violation": np.median([float(row["search_violation"]) for row in rows if row["difficulty"] == difficulty]),
            }
            for difficulty in ("easy", "moderate", "hard")
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
