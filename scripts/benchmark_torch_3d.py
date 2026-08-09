"""Benchmark the differentiable implicit dose operator on one or more GPUs."""

import argparse
import csv
import json
import multiprocessing as mp
import time
from pathlib import Path
from statistics import mean, median


def _worker(payload: tuple[int, list[int], int, int, int, int, str]) -> list[dict[str, float | int | str]]:
    device_index, grids, fluence_size, batch_size, warmup, repeats, dtype_name = payload
    import torch

    from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D
    from dosim_sim.volume3d import generate_case_3d

    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    dtype = getattr(torch, dtype_name)
    angles = tuple(float(value) for value in range(0, 360, 30))
    rows: list[dict[str, float | int | str]] = []
    for grid_size in grids:
        case = generate_case_3d(10000 + device_index, grid_size)
        engine = TorchImplicitDoseEngine3D(
            case,
            angles,
            fluence_size=fluence_size,
            device=device,
            dtype=dtype,
        )
        fluence = torch.full(
            (batch_size, len(angles), fluence_size, fluence_size),
            0.05,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        def iteration() -> None:
            dose = engine.forward(fluence)
            loss = dose.float().square().mean()
            loss.backward()
            fluence.grad = None

        for _ in range(warmup):
            iteration()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        elapsed: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            iteration()
            torch.cuda.synchronize(device)
            elapsed.append(time.perf_counter() - started)
        rows.append(
            {
                "device": device_index,
                "gpu_name": torch.cuda.get_device_name(device),
                "grid_size": grid_size,
                "batch_size": batch_size,
                "fluence_size": fluence_size,
                "dtype": dtype_name,
                "mean_iteration_seconds": mean(elapsed),
                "median_iteration_seconds": median(elapsed),
                "cases_per_second": batch_size / mean(elapsed),
                "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                "geometry_cache_mib": engine.cache_bytes / 2**20,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--grids", type=int, nargs="+", default=[64, 96])
    parser.add_argument("--fluence-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--output", type=Path, default=Path("outputs/gpu_benchmark/torch_operator.csv"))
    args = parser.parse_args()
    payloads = [
        (
            device,
            args.grids,
            args.fluence_size,
            args.batch_size,
            args.warmup,
            args.repeats,
            args.dtype,
        )
        for device in args.devices
    ]
    context = mp.get_context("spawn")
    with context.Pool(processes=len(payloads)) as pool:
        nested_rows = pool.map(_worker, payloads)
    rows = [row for device_rows in nested_rows for row in device_rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
