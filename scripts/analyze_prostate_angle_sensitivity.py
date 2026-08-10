"""Combine the anatomy-guided beam-angle sensitivity pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize prostate angle-shift sensitivity")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prostate_expert_angle_sensitivity"))
    args = parser.parse_args()
    if len(args.run_dirs) != len(args.labels):
        raise ValueError("run_dirs and labels must have the same length")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [json.loads((path / "summary.json").read_text(encoding="utf-8")) for path in args.run_dirs]
    x = np.arange(len(summaries))
    colors = ("#4c78a8", "#f58518", "#54a24b")
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels = (
        ("final_acceptance_percent", "Final acceptable plans (%)", [100.0 * row["final_acceptable"] / row["cases"] for row in summaries], (0, 100)),
        ("median_d98_change_gy", "Median D98 change (Gy)", [row["median_d98_change_gy"] for row in summaries], None),
        ("median_d02_change_gy", "Median D02 change (Gy)", [row["median_d02_change_gy"] for row in summaries], None),
        ("median_oar_ratio_change", "Median worst OAR ratio change", [row["median_oar_per_protocol_ratio_change"] for row in summaries], None),
    )
    consolidated = []
    for axis, (_, title, values, limits) in zip(axes.flat, panels, strict=True):
        bars = axis.bar(x, values, color=colors[: len(values)])
        axis.set_xticks(x, args.labels)
        axis.set_title(title)
        axis.axhline(0.0, color="black", linewidth=1)
        if limits:
            axis.set_ylim(*limits)
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top")
    figure.suptitle("Anatomy-guided field-shift sensitivity: 15 hard prostate cases")
    figure.savefig(args.output_dir / "01_angle_sensitivity.png", dpi=180)
    plt.close(figure)
    for label, run_dir, summary in zip(args.labels, args.run_dirs, summaries, strict=True):
        consolidated.append({"label": label, "run_dir": str(run_dir), **summary})
    (args.output_dir / "summary.json").write_text(json.dumps(consolidated, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
