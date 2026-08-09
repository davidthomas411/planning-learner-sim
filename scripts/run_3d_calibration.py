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
from dosim_sim.volume3d import SyntheticCase3D, generate_case_3d


ANGLES = tuple(float(value) for value in range(0, 360, 30))


def _acceptable(d95: float, d02: float, oar_ratios: list[float]) -> bool:
    return d95 >= 0.85 and d02 <= 1.25 and max(oar_ratios) <= 1.0


def _centroid_xy(mask: np.ndarray, axis: np.ndarray) -> np.ndarray:
    indices = np.argwhere(mask)
    return np.array([axis[indices[:, 0]].mean(), axis[indices[:, 1]].mean()])


def _select_separating_beam(case: SyntheticCase3D, oar_index: int, active: tuple[int, ...]) -> int:
    """Select an inactive angle whose lateral axis best separates target and named OAR."""

    target = _centroid_xy(case.target, case.axis)
    oar = _centroid_xy(case.oars[oar_index], case.axis)
    displacement = oar - target
    inactive = [index for index in range(len(ANGLES)) if index not in active]
    scores = []
    for index in inactive:
        angle = np.deg2rad(ANGLES[index])
        lateral_axis = np.array([-np.sin(angle), np.cos(angle)])
        scores.append(abs(float(displacement @ lateral_axis)))
    return inactive[int(np.argmax(scores))]


def run_case(seed: int, args, dtype: torch.dtype) -> dict[str, object]:
    case = generate_case_3d(seed, args.grid_size)
    engine = TorchImplicitDoseEngine3D(
        case,
        ANGLES,
        fluence_size=args.fluence_size,
        device=args.device,
        dtype=dtype,
    )
    priorities0 = PlanningPriorities.for_case(case)
    active0 = (0, 3, 6, 9)
    plans = [
        optimize_fluence_3d_torch(case, engine, active0, priorities0, args.iterations)
    ]
    action_labels: list[str] = []

    ratios0 = [
        value / limit
        for value, limit in zip(plans[-1].metrics.oar_mean, case.oar_limits, strict=True)
    ]
    worst0 = int(np.argmax(ratios0))
    oar_weights1 = list(priorities0.oars)
    oar_weights1[worst0] *= args.priority_factor
    priorities1 = PlanningPriorities(
        target=priorities0.target,
        hotspot=priorities0.hotspot,
        oars=tuple(oar_weights1),
    )
    plans.append(
        optimize_fluence_3d_torch(
            case, engine, plans[-1].active_beams, priorities1, args.iterations, initial_fluence=plans[-1].fluence
        )
    )
    action_labels.append(f"increase_oar_{worst0 + 1}_priority")

    added = _select_separating_beam(case, worst0, plans[-1].active_beams)
    active2 = tuple(sorted((*plans[-1].active_beams, added)))
    plans.append(
        optimize_fluence_3d_torch(
            case, engine, active2, priorities1, args.iterations, initial_fluence=plans[-1].fluence
        )
    )
    action_labels.append(f"add_beam_{ANGLES[added]:.0f}_degrees")

    ratios2 = [
        value / limit
        for value, limit in zip(plans[-1].metrics.oar_mean, case.oar_limits, strict=True)
    ]
    worst2 = int(np.argmax(ratios2))
    oar_weights2 = list(priorities1.oars)
    oar_weights2[worst2] *= args.priority_factor
    priorities2 = PlanningPriorities(
        target=priorities1.target,
        hotspot=priorities1.hotspot,
        oars=tuple(oar_weights2),
    )
    plans.append(
        optimize_fluence_3d_torch(
            case, engine, plans[-1].active_beams, priorities2, args.iterations, initial_fluence=plans[-1].fluence
        )
    )
    action_labels.append(f"increase_oar_{worst2 + 1}_priority")

    priorities3 = PlanningPriorities(
        target=priorities2.target * args.priority_factor,
        hotspot=priorities2.hotspot,
        oars=priorities2.oars,
    )
    plans.append(
        optimize_fluence_3d_torch(
            case, engine, plans[-1].active_beams, priorities3, args.iterations, initial_fluence=plans[-1].fluence
        )
    )
    action_labels.append("increase_target_priority")

    initial = plans[0].metrics
    final = plans[-1].metrics
    initial_ratios = [
        value / limit for value, limit in zip(initial.oar_mean, case.oar_limits, strict=True)
    ]
    final_ratios = [
        value / limit for value, limit in zip(final.oar_mean, case.oar_limits, strict=True)
    ]
    return {
        "seed": seed,
        "case_id": case.case_id,
        "initial_d95": initial.target_d95,
        "final_d95": final.target_d95,
        "initial_d02": initial.target_d02,
        "final_d02": final.target_d02,
        "initial_max_oar_ratio": max(initial_ratios),
        "final_max_oar_ratio": max(final_ratios),
        "initial_acceptable": _acceptable(initial.target_d95, initial.target_d02, initial_ratios),
        "final_acceptable": _acceptable(final.target_d95, final.target_d02, final_ratios),
        "added_beam_degrees": ANGLES[added],
        "actions": "|".join(action_labels),
        "final_oar_ratios": "|".join(f"{value:.8f}" for value in final_ratios),
        "cache_mib": engine.cache_bytes / 2**20,
    }


def save_plot(rows: list[dict[str, object]], path: Path) -> None:
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fields = (
        ("initial_d95", "final_d95", 0.85, "PTV D95", "higher is preferable"),
        ("initial_d02", "final_d02", 1.25, "PTV D02", "lower is preferable"),
        ("initial_max_oar_ratio", "final_max_oar_ratio", 1.0, "Maximum OAR mean/limit", "lower is preferable"),
    )
    for axis, (initial_field, final_field, threshold, label, direction) in zip(axes.flat[:3], fields, strict=True):
        initial = np.array([float(row[initial_field]) for row in rows])
        final = np.array([float(row[final_field]) for row in rows])
        for index in x:
            axis.plot([0, 1], [initial[index], final[index]], color="#7F8C8D", alpha=0.5, lw=1)
        axis.scatter(np.zeros_like(x), initial, color="#4C78A8", label="initial optimized plan")
        axis.scatter(np.ones_like(x), final, color="#E45756", label="after four manual-level changes")
        axis.axhline(threshold, color="#222222", ls="--", lw=1, label="provisional threshold")
        axis.set_xticks([0, 1], ["initial", "final"])
        axis.set_ylabel(label)
        axis.set_title(f"{label} ({direction})")
    initial_ok = sum(bool(row["initial_acceptable"]) for row in rows)
    final_ok = sum(bool(row["final_acceptable"]) for row in rows)
    axes[1, 1].bar([0, 1], [initial_ok / len(rows), final_ok / len(rows)], color=["#4C78A8", "#E45756"])
    axes[1, 1].set_xticks([0, 1], ["initial", "final"])
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("Proportion acceptable")
    axes[1, 1].set_title("All provisional criteria satisfied")
    fig.suptitle(
        f"Three-dimensional environment calibration: {len(rows)} cases\n"
        "Recorded changes are beam-angle or named-priority edits; fluence is optimized internally",
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the 3D environment with high-level manual changes")
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=11000)
    parser.add_argument("--grid-size", type=int, default=96)
    parser.add_argument("--fluence-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--priority-factor", type=float, default=1.75)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/3d_calibration_local"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.cases < 1:
        raise ValueError("cases must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.cases):
        rows.append(run_case(seed, args, dtype))
        print(f"completed {len(rows)}/{args.cases}: seed={seed} acceptable={rows[-1]['final_acceptable']}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    with (args.output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_plot(rows, args.output_dir / "01_3d_calibration.png")
    summary = {
        "status": "engineering calibration; not a learner comparison",
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "grid_size": args.grid_size,
        "fluence_size": args.fluence_size,
        "iterations_per_state": args.iterations,
        "cases": len(rows),
        "states_per_case": 5,
        "high_level_actions_per_case": 4,
        "elapsed_seconds": elapsed,
        "seconds_per_case": elapsed / len(rows),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "initial_acceptable_rate": float(np.mean([bool(row["initial_acceptable"]) for row in rows])),
        "final_acceptable_rate": float(np.mean([bool(row["final_acceptable"]) for row in rows])),
        "median_initial_d95": float(np.median([float(row["initial_d95"]) for row in rows])),
        "median_final_d95": float(np.median([float(row["final_d95"]) for row in rows])),
        "median_initial_max_oar_ratio": float(np.median([float(row["initial_max_oar_ratio"]) for row in rows])),
        "median_final_max_oar_ratio": float(np.median([float(row["final_max_oar_ratio"]) for row in rows])),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
