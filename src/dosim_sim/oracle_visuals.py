from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import SimulationConfig
from .geometry import SyntheticCase
from .manual_planning import ManualTrajectory
from .objective import clinical_violation_score
from .oracle import OracleTrajectory


def save_oracle_case_comparison(
    case: SyntheticCase,
    manual: ManualTrajectory,
    oracle: OracleTrajectory,
    output_path: Path,
    config: SimulationConfig | None = None,
) -> None:
    cfg = config or SimulationConfig()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    initial = manual.steps[0].plan
    plans = [initial, manual.final.plan, oracle.final.plan]
    labels = ["Common initial optimized plan", "Rule-based manual planner", "High-level search oracle"]
    vmax = max(float(plan.dose.max()) for plan in plans)

    fig = plt.figure(figsize=(13.5, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.25, 0.75])
    for index, (plan, label) in enumerate(zip(plans, labels, strict=True)):
        axis = fig.add_subplot(grid[0, index])
        image = axis.imshow(plan.dose, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        axis.contour(case.target, levels=[0.5], colors=["#4daf4a"], linewidths=1.5)
        for oar, color in zip(case.oars, ["#377eb8", "#ff7f00"], strict=False):
            axis.contour(oar, levels=[0.5], colors=[color], linewidths=1.2)
        metrics = plan.clinical_metrics
        violation = clinical_violation_score(metrics, case, cfg)
        axis.set_title(
            f"{label}\nD95 {metrics.target_d95:.2f} | D02 {metrics.target_d02:.2f} | "
            f"violation {violation:.4f}",
            fontsize=10,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(image, ax=[fig.axes[0], fig.axes[1], fig.axes[2]], fraction=0.02, label="relative dose")

    manual_axis = fig.add_subplot(grid[1, :1])
    manual_axis.axis("off")
    manual_axis.set_title(f"Manual actions ({manual.stopping_reason})", loc="left", fontweight="bold")
    manual_lines = [
        f"{step.step}. {fill(step.action.description, width=38)}"
        for step in manual.steps[1:]
    ]
    manual_axis.text(0, 0.96, "\n".join(manual_lines), va="top", family="monospace", fontsize=9)

    oracle_axis = fig.add_subplot(grid[1, 1:])
    oracle_axis.axis("off")
    oracle_axis.set_title(f"Oracle actions ({oracle.stopping_reason})", loc="left", fontweight="bold")
    oracle_lines = [
        f"{step.step}. {fill(step.action.description, width=62)}  [{step.violation_score:.4f}]"
        for step in oracle.steps[1:]
    ]
    oracle_axis.text(0, 0.96, "\n".join(oracle_lines), va="top", family="monospace", fontsize=9)
    fig.suptitle(
        "Same starting plan, different high-level decision sequences",
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

