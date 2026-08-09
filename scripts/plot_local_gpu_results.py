import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    root = Path("outputs/gpu_demo")
    output_dir = Path("outputs/gpu_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for grid_size in (96, 128, 192, 256):
        path = root / f"rtx4060_{grid_size}" / "summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(payload)
    with (output_dir / "local_rtx4060_full_trajectories.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grids = [row["grid_size"] for row in rows]
    seconds = [row["elapsed_seconds"] for row in rows]
    memory_gib = [row["peak_memory_mib"] / 1024 for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    axes[0].plot(grids, seconds, marker="o", linewidth=2)
    axes[0].set_xlabel("Cubic grid dimension")
    axes[0].set_ylabel("Seconds for five-state trajectory")
    axes[0].set_title("End-to-end time")
    axes[1].plot(grids, memory_gib, marker="s", linewidth=2, color="#e15759")
    axes[1].axhline(8.0, linewidth=1, linestyle="--", color="black", alpha=0.5)
    axes[1].set_xlabel("Cubic grid dimension")
    axes[1].set_ylabel("Peak allocated memory (GiB)")
    axes[1].set_title("Torch memory (8 GiB card)")
    for axis, values in zip(axes, (seconds, memory_gib), strict=True):
        axis.grid(alpha=0.2)
        axis.set_xticks(grids)
        for x, value in zip(grids, values, strict=True):
            label = f"{value:.1f} s" if axis is axes[0] else f"{value:.2f} GiB"
            axis.annotate(label, (x, value), xytext=(0, 8), textcoords="offset points", ha="center")
    fig.suptitle("Local RTX 4060: complete 3D manual-planning trajectories")
    fig.savefig(output_dir / "01_local_gpu_scaling.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
