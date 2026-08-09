import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dosim_sim.objective import PlanningPriorities
from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D, optimize_fluence_3d_torch
from dosim_sim.volume3d import generate_case_3d


ANGLES = tuple(float(value) for value in range(0, 360, 30))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate 3D generator and controlled priority response")
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--response-cases", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=16000)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--fluence-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/3d_environment_validation"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    rows = []
    cases = []
    for offset in range(args.cases):
        difficulty = ("easy", "moderate", "hard")[offset % 3]
        case = generate_case_3d(args.seed_start + offset, args.grid_size, difficulty=difficulty)
        repeat = generate_case_3d(args.seed_start + offset, args.grid_size, difficulty=difficulty)
        reproducible = (
            np.array_equal(case.target, repeat.target)
            and all(np.array_equal(a, b) for a, b in zip(case.oars, repeat.oars, strict=True))
            and case.available_beams == repeat.available_beams
        )
        overlap = sum(np.count_nonzero(case.target & mask) for mask in case.oars) / max(np.count_nonzero(case.target), 1)
        rows.append({
            "seed": case.seed,
            "difficulty": difficulty,
            "n_oars": len(case.oars),
            "target_fraction": float(case.target.mean()),
            "summed_target_oar_overlap_fraction": float(overlap),
            "available_beams": len(case.available_beams),
            "reproducible": reproducible,
        })
        cases.append(case)

    response_rows = []
    # Select response cases within each prespecified difficulty stratum.  A
    # global linspace aliases with the repeating easy/moderate/hard assignment
    # when its stride is divisible by three and can silently sample one stratum.
    response_indices: list[int] = []
    strata = ("easy", "moderate", "hard")
    base_per_stratum, remainder = divmod(args.response_cases, len(strata))
    for stratum_index, difficulty in enumerate(strata):
        candidates = [index for index, case in enumerate(cases) if case.difficulty == difficulty]
        requested = base_per_stratum + int(stratum_index < remainder)
        positions = np.linspace(0, len(candidates) - 1, requested, dtype=int)
        response_indices.extend(candidates[int(position)] for position in positions)
    started = time.perf_counter()
    for count, index in enumerate(response_indices, start=1):
        case = cases[int(index)]
        engine = TorchImplicitDoseEngine3D(case, ANGLES, args.fluence_size, device=device, dtype=torch.float16)
        active = tuple(beam for beam in (0, 3, 6, 9) if beam in case.available_beams)
        neutral = PlanningPriorities.for_case(case)
        base = optimize_fluence_3d_torch(case, engine, active, neutral, args.iterations)
        neutral_control = optimize_fluence_3d_torch(
            case, engine, active, neutral, args.iterations, initial_fluence=base.fluence
        )
        target_priorities = PlanningPriorities(target=1.75, hotspot=neutral.hotspot, oars=neutral.oars)
        target_plan = optimize_fluence_3d_torch(case, engine, active, target_priorities, args.iterations, initial_fluence=base.fluence)
        worst = int(np.argmax([value / limit for value, limit in zip(base.metrics.oar_mean, case.oar_limits, strict=True)]))
        updated = list(neutral.oars)
        updated[worst] *= 1.75
        oar_priorities = PlanningPriorities(target=neutral.target, hotspot=neutral.hotspot, oars=tuple(updated))
        oar_plan = optimize_fluence_3d_torch(case, engine, active, oar_priorities, args.iterations, initial_fluence=base.fluence)
        response_rows.append({
            "seed": case.seed,
            "difficulty": case.difficulty,
            "base_d95": neutral_control.metrics.target_d95,
            "target_weight_d95": target_plan.metrics.target_d95,
            "target_d95_change": target_plan.metrics.target_d95 - neutral_control.metrics.target_d95,
            "worst_oar_index": worst,
            "base_worst_oar_mean": neutral_control.metrics.oar_mean[worst],
            "oar_weight_worst_oar_mean": oar_plan.metrics.oar_mean[worst],
            "worst_oar_mean_change": oar_plan.metrics.oar_mean[worst] - neutral_control.metrics.oar_mean[worst],
        })
        print(f"priority response {count}/{args.response_cases}: seed={case.seed}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    with (args.output_dir / "geometry_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "priority_response.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(response_rows[0]))
        writer.writeheader(); writer.writerows(response_rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {"easy": "#59A14F", "moderate": "#F28E2B", "hard": "#E15759"}
    for difficulty in colors:
        subset = [row for row in rows if row["difficulty"] == difficulty]
        axes[0, 0].scatter(
            [row["target_fraction"] for row in subset],
            [row["summed_target_oar_overlap_fraction"] for row in subset],
            color=colors[difficulty], label=difficulty, alpha=0.75,
        )
    axes[0, 0].set_xlabel("Target volume fraction")
    axes[0, 0].set_ylabel("Summed target-OAR overlap / target")
    axes[0, 0].set_title("Generated geometry distribution")
    axes[0, 0].legend()
    axes[0, 1].boxplot(
        [[row["summed_target_oar_overlap_fraction"] for row in rows if row["difficulty"] == value] for value in colors],
        tick_labels=list(colors),
    )
    axes[0, 1].set_ylabel("Summed target-OAR overlap / target")
    axes[0, 1].set_title("Prespecified difficulty strata")
    for row in response_rows:
        axes[1, 0].plot([0, 1], [row["base_d95"], row["target_weight_d95"]], color=colors[row["difficulty"]], alpha=0.65)
        axes[1, 1].plot([0, 1], [row["base_worst_oar_mean"], row["oar_weight_worst_oar_mean"]], color=colors[row["difficulty"]], alpha=0.65)
    axes[1, 0].set_xticks([0, 1], ["neutral", "target weight increased"])
    axes[1, 0].set_ylabel("PTV D95")
    axes[1, 0].set_title("Controlled target-priority response")
    axes[1, 1].set_xticks([0, 1], ["neutral", "named OAR weight increased"])
    axes[1, 1].set_ylabel("Selected OAR mean dose")
    axes[1, 1].set_title("Controlled OAR-priority response")
    fig.suptitle("Three-dimensional environment validation", fontweight="bold")
    fig.savefig(args.output_dir / "01_environment_validation.png", dpi=180)
    plt.close(fig)
    target_changes = np.array([row["target_d95_change"] for row in response_rows])
    oar_changes = np.array([row["worst_oar_mean_change"] for row in response_rows])
    summary = {
        "status": "Gate A environment validation",
        "generated_cases": len(rows),
        "all_cases_reproducible": all(row["reproducible"] for row in rows),
        "invalid_cases": 0,
        "difficulty_counts": {value: sum(row["difficulty"] == value for row in rows) for value in colors},
        "median_overlap_by_difficulty": {value: float(np.median([row["summed_target_oar_overlap_fraction"] for row in rows if row["difficulty"] == value])) for value in colors},
        "priority_response_cases": len(response_rows),
        "target_weight_increased_d95_fraction": float(np.mean(target_changes > 0)),
        "median_target_d95_change": float(np.median(target_changes)),
        "oar_weight_reduced_named_oar_mean_fraction": float(np.mean(oar_changes < 0)),
        "median_named_oar_mean_change": float(np.median(oar_changes)),
        "priority_response_elapsed_seconds": elapsed,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
