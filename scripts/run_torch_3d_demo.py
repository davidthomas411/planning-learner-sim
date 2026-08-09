import argparse
import csv
import json
import time
from pathlib import Path

import torch

from dosim_sim.objective import PlanningPriorities
from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D, optimize_fluence_3d_torch
from dosim_sim.volume3d import generate_case_3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Time a complete high-level 3D trajectory on CUDA")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--grid-size", type=int, default=96)
    parser.add_argument("--fluence-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gpu_demo"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    dtype = getattr(torch, args.dtype)
    case = generate_case_3d(args.seed, args.grid_size)
    angles = tuple(float(value) for value in range(0, 360, 30))
    engine = TorchImplicitDoseEngine3D(
        case,
        angles,
        fluence_size=args.fluence_size,
        device=args.device,
        dtype=dtype,
    )
    torch.cuda.synchronize(args.device)
    started = time.perf_counter()

    priorities0 = PlanningPriorities.for_case(case)
    plan0 = optimize_fluence_3d_torch(
        case, engine, (0, 3, 6, 9), priorities0, args.iterations
    )
    worst_oar = max(
        range(len(case.oars)),
        key=lambda index: plan0.metrics.oar_mean[index] / case.oar_limits[index],
    )
    oars1 = list(priorities0.oars)
    oars1[worst_oar] *= 1.75
    priorities1 = PlanningPriorities(oars=tuple(oars1))
    plan1 = optimize_fluence_3d_torch(
        case,
        engine,
        plan0.active_beams,
        priorities1,
        args.iterations,
        initial_fluence=plan0.fluence,
    )
    added_beam = 1
    plan2 = optimize_fluence_3d_torch(
        case,
        engine,
        tuple(sorted((*plan1.active_beams, added_beam))),
        priorities1,
        args.iterations,
        initial_fluence=plan1.fluence,
    )
    worst_after_beam = max(
        range(len(case.oars)),
        key=lambda index: plan2.metrics.oar_mean[index] / case.oar_limits[index],
    )
    oars2 = list(priorities1.oars)
    oars2[worst_after_beam] *= 1.75
    priorities2 = PlanningPriorities(oars=tuple(oars2))
    plan3 = optimize_fluence_3d_torch(
        case,
        engine,
        plan2.active_beams,
        priorities2,
        args.iterations,
        initial_fluence=plan2.fluence,
    )
    priorities3 = PlanningPriorities(target=1.75, oars=priorities2.oars)
    plan4 = optimize_fluence_3d_torch(
        case,
        engine,
        plan3.active_beams,
        priorities3,
        args.iterations,
        initial_fluence=plan3.fluence,
    )
    torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter() - started
    plans = (plan0, plan1, plan2, plan3, plan4)
    labels = (
        "Initial 4 beams",
        f"Increase OAR {worst_oar + 1} priority",
        f"Add {angles[added_beam]:.0f}-degree beam",
        f"Increase OAR {worst_after_beam + 1} priority",
        "Increase target priority",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "manual_action",
                "target_d95",
                "target_d02",
                "oar_1_ratio",
                "oar_2_ratio",
            ],
        )
        writer.writeheader()
        for step, (label, plan) in enumerate(zip(labels, plans, strict=True)):
            writer.writerow(
                {
                    "step": step,
                    "manual_action": label,
                    "target_d95": plan.metrics.target_d95,
                    "target_d02": plan.metrics.target_d02,
                    "oar_1_ratio": plan.metrics.oar_mean[0] / case.oar_limits[0],
                    "oar_2_ratio": plan.metrics.oar_mean[1] / case.oar_limits[1],
                }
            )
    final = plan4.metrics
    summary = {
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "grid_size": args.grid_size,
        "fluence_size": args.fluence_size,
        "dtype": args.dtype,
        "fluence_master_dtype": "float32",
        "optimizer_iterations_per_state": args.iterations,
        "states": len(plans),
        "elapsed_seconds": elapsed,
        "peak_memory_mib": torch.cuda.max_memory_allocated(args.device) / 2**20,
        "final_target_d95": final.target_d95,
        "final_target_d02": final.target_d02,
        "final_oar_mean_over_limit": [
            value / limit for value, limit in zip(final.oar_mean, case.oar_limits, strict=True)
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
