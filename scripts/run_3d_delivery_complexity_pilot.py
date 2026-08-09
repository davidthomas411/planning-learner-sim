import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dosim_sim.delivery3d import standard_delivery_modes_3d
from dosim_sim.objective import PlanningPriorities
from dosim_sim.planning3d import HighLevelSearchConfig3D, is_acceptable_3d
from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D, optimize_fluence_3d_torch
from dosim_sim.volume3d import generate_case_3d


def save_plot(rows: list[dict], path: Path) -> None:
    modes = [mode.name for mode in standard_delivery_modes_3d()]
    labels = ["4 static", "12 static", "180° arc-like\n19 control points", "360° arc-like\n36 control points"]
    colors = ["#4E79A7", "#59A14F", "#F28E2B", "#E15759"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    x = np.arange(len(modes))
    axes[0, 0].bar(x, [np.mean([bool(row["acceptable"]) for row in rows if row["mode"] == mode]) for mode in modes], color=colors)
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("Proportion acceptable")
    axes[0, 0].set_title("Plan acceptability")
    for case_seed in sorted({int(row["seed"]) for row in rows}):
        subset = [row for row in rows if int(row["seed"]) == case_seed]
        axes[0, 1].plot(x, [float(next(row["target_d95"] for row in subset if row["mode"] == mode)) for mode in modes], color="#777777", alpha=0.45)
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylabel("PTV D95")
    axes[0, 1].set_title("Paired target coverage")
    for case_seed in sorted({int(row["seed"]) for row in rows}):
        subset = [row for row in rows if int(row["seed"]) == case_seed]
        axes[1, 0].plot(x, [float(next(row["maximum_oar_ratio"] for row in subset if row["mode"] == mode)) for mode in modes], color="#777777", alpha=0.45)
    axes[1, 0].axhline(1.0, color="black", ls="--", lw=1)
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylabel("Maximum OAR mean / limit")
    axes[1, 0].set_title("Paired OAR burden")
    axes[1, 1].bar(x, [np.median([float(row["elapsed_seconds"]) for row in rows if row["mode"] == mode]) for mode in modes], color=colors)
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylabel("Median optimization time (s)")
    axes[1, 1].set_title("Computational cost")
    fig.suptitle(
        "Angular delivery complexity pilot\nArc-like modes use independent fluence maps and are not delivery-realistic VMAT",
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare static-field and arc-like angular sampling")
    parser.add_argument("--cases-per-stratum", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=19000)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--fluence-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/3d_delivery_complexity_pilot"))
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    modes = standard_delivery_modes_3d()
    rows: list[dict] = []
    case_number = 0
    for difficulty in ("easy", "moderate", "hard"):
        for _ in range(args.cases_per_stratum):
            seed = args.seed_start + case_number
            case_number += 1
            case = generate_case_3d(seed, args.grid_size, difficulty=difficulty)
            for mode in modes:
                torch.cuda.reset_peak_memory_stats()
                engine = TorchImplicitDoseEngine3D(
                    case, mode.angles_degrees, args.fluence_size, device=device, dtype=torch.float16
                )
                started = time.perf_counter()
                plan = optimize_fluence_3d_torch(
                    case,
                    engine,
                    mode.active_beams,
                    PlanningPriorities.for_case(case),
                    iterations=args.iterations,
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                maximum_oar_ratio = max(
                    value / limit
                    for value, limit in zip(plan.metrics.oar_mean, case.oar_limits, strict=True)
                )
                rows.append({
                    "seed": seed,
                    "difficulty": difficulty,
                    "mode": mode.name,
                    "arc_like": mode.arc_like,
                    "angular_samples": len(mode.angles_degrees),
                    "angular_span_degrees": max(mode.angles_degrees) - min(mode.angles_degrees),
                    "fluence_pixels": len(mode.angles_degrees) * args.fluence_size**2,
                    "acceptable": is_acceptable_3d(plan.metrics, case, HighLevelSearchConfig3D()),
                    "target_d95": plan.metrics.target_d95,
                    "target_d02": plan.metrics.target_d02,
                    "maximum_oar_ratio": maximum_oar_ratio,
                    "loss": plan.metrics.loss,
                    "elapsed_seconds": elapsed,
                    "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                })
                del plan, engine
            print(f"completed {case_number}/{3 * args.cases_per_stratum}: {difficulty} seed={seed}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_plot(rows, args.output_dir / "01_delivery_complexity.png")
    summary = {
        "status": "angular-sampling engineering pilot; arc-like modes are not delivery-realistic VMAT",
        "cases": 3 * args.cases_per_stratum,
        "grid_size": args.grid_size,
        "fluence_size": args.fluence_size,
        "iterations": args.iterations,
        "modes": {
            mode.name: {
                "angular_samples": len(mode.angles_degrees),
                "acceptable_rate": float(np.mean([bool(row["acceptable"]) for row in rows if row["mode"] == mode.name])),
                "median_target_d95": float(np.median([float(row["target_d95"]) for row in rows if row["mode"] == mode.name])),
                "median_maximum_oar_ratio": float(np.median([float(row["maximum_oar_ratio"]) for row in rows if row["mode"] == mode.name])),
                "median_elapsed_seconds": float(np.median([float(row["elapsed_seconds"]) for row in rows if row["mode"] == mode.name])),
                "maximum_peak_memory_mib": float(max(float(row["peak_memory_mib"]) for row in rows if row["mode"] == mode.name)),
            }
            for mode in modes
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
