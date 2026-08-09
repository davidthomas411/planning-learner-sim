import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from dosim_sim import ImplicitDoseEngine3D, generate_case_3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark implicit 3D forward and adjoint kernels")
    parser.add_argument("--grids", type=int, nargs="+", default=[64, 96, 128])
    parser.add_argument("--fluence-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("outputs/3d_benchmark/operator_benchmark.csv"))
    args = parser.parse_args()
    angles = tuple(float(value) for value in range(0, 360, 30))
    rows: list[dict[str, float | int]] = []
    for grid_size in args.grids:
        case = generate_case_3d(10000, grid_size)
        started = time.perf_counter()
        engine = ImplicitDoseEngine3D(case, angles, args.fluence_size)
        build_seconds = time.perf_counter() - started
        fluence = np.full(
            (len(angles), args.fluence_size, args.fluence_size),
            0.05,
            dtype=np.float32,
        )
        started = time.perf_counter()
        dose = engine.forward(fluence)
        forward_seconds = time.perf_counter() - started
        started = time.perf_counter()
        engine.adjoint(dose)
        adjoint_seconds = time.perf_counter() - started
        dense_values = grid_size**3 * len(angles) * args.fluence_size**2
        rows.append(
            {
                "grid_size": grid_size,
                "voxels": grid_size**3,
                "body_voxels": int(case.body.sum()),
                "fluence_size": args.fluence_size,
                "cache_mib": engine.cache_bytes / 2**20,
                "dense_matrix_gib_float32": dense_values * 4 / 2**30,
                "build_seconds": build_seconds,
                "forward_seconds": forward_seconds,
                "adjoint_seconds": adjoint_seconds,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
