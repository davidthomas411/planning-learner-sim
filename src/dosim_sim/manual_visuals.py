from math import ceil
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import SimulationConfig
from .geometry import SyntheticCase
from .manual_planning import ManualTrajectory


def save_nested_workflow(output_path: Path) -> None:
    """Explain which decisions are automated and which enter the trajectory."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 6)
    axis.axis("off")

    def box(x: float, y: float, width: float, height: float, text: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=color,
            edgecolor="#333333",
            linewidth=1.2,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=11)

    def arrow(start: tuple[float, float], end: tuple[float, float], label: str = "") -> None:
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4))
        if label:
            axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.18, label, ha="center", fontsize=9)

    axis.text(2.1, 5.55, "Recorded manual trajectory", ha="center", fontsize=14, fontweight="bold")
    box(0.35, 3.65, 3.5, 1.15, "Planner reviews dose, DVH,\nand unmet goals", "#d9edf7")
    box(0.35, 1.65, 3.5, 1.15, "Planner changes a beam angle\nor target/OAR priority", "#dff0d8")
    arrow((2.1, 3.65), (2.1, 2.8))
    axis.text(2.28, 3.24, "high-level\njudgment", ha="left", va="center", fontsize=9)

    axis.text(7.4, 5.55, "Automated inner optimization", ha="center", fontsize=14, fontweight="bold")
    box(5.15, 3.65, 4.5, 1.15, "Optimizer adjusts beamlet intensities\nfor the fixed angles and priorities", "#f2e6ff")
    box(5.15, 1.65, 4.5, 1.15, "Dose is recalculated and\nplan metrics are returned", "#fce8d5")
    arrow((7.4, 3.65), (7.4, 2.8))
    axis.text(7.58, 3.24, "many hidden\nbeamlet iterations", ha="left", va="center", fontsize=9)
    arrow((3.85, 2.22), (5.15, 4.22))
    arrow((5.15, 2.22), (3.85, 4.22))
    axis.text(4.35, 3.85, "rerun", ha="center", fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    axis.text(4.65, 2.55, "review", ha="center", fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})

    box(10.25, 2.65, 1.45, 1.15, "Approved\nplan", "#fff2b2")
    arrow((9.65, 2.22), (10.25, 3.0))
    axis.text(9.92, 3.18, "goals met", ha="center", fontsize=9)
    axis.text(
        6.0,
        0.62,
        "Stored training example: review state -> manual beam/priority change -> reoptimized state",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
    axis.text(
        6.0,
        0.18,
        "Individual beamlet adjustments remain inside the optimizer and are not labeled as manual actions.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _draw_beams(axis: plt.Axes, active_beams: tuple[int, ...], cfg: SimulationConfig) -> None:
    for beam in active_beams:
        angle = 2 * np.pi * beam / cfg.n_beams
        x0 = 31.5 - 38 * np.cos(angle)
        y0 = 31.5 - 38 * np.sin(angle)
        axis.plot([x0, 31.5], [y0, 31.5], color="#ffffff", alpha=0.45, linewidth=0.8)


def save_manual_filmstrip(
    case: SyntheticCase,
    trajectory: ManualTrajectory,
    output_path: Path,
    config: SimulationConfig | None = None,
) -> None:
    cfg = config or SimulationConfig()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ncols = 3
    nrows = ceil(len(trajectory.steps) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.05 * nrows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    vmax = max(float(step.plan.dose.max()) for step in trajectory.steps)

    for axis, step in zip(axes_array, trajectory.steps, strict=False):
        image = axis.imshow(step.plan.dose, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        axis.contour(case.target, levels=[0.5], colors=["#4daf4a"], linewidths=1.5)
        for oar, color in zip(case.oars, ["#377eb8", "#ff7f00"], strict=False):
            axis.contour(oar, levels=[0.5], colors=[color], linewidths=1.2)
        _draw_beams(axis, step.plan.active_beams, cfg)
        action = "Initial four beams" if step.action is None else step.action.description
        wrapped_action = fill(action, width=34)
        metrics = step.plan.clinical_metrics
        axis.set_title(
            f"Manual step {step.step}\n{wrapped_action}\n"
            f"D95 {metrics.target_d95:.2f} | D02 {metrics.target_d02:.2f} | "
            f"worst OAR {max(v/l for v, l in zip(metrics.oar_mean, case.oar_limits)):.2f}x limit",
            fontsize=8.5,
            pad=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes_array[len(trajectory.steps) :]:
        axis.axis("off")
    fig.colorbar(image, ax=list(axes_array[: len(trajectory.steps)]), fraction=0.018, label="relative dose")
    fig.suptitle(
        f"Manual planning trajectory: high-level changes separated by full reoptimization\n"
        f"stop: {trajectory.stopping_reason}",
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_manual_metrics(
    case: SyntheticCase,
    trajectory: ManualTrajectory,
    output_path: Path,
    config: SimulationConfig | None = None,
) -> None:
    cfg = config or SimulationConfig()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.array([step.step for step in trajectory.steps])
    d95 = np.array([step.plan.clinical_metrics.target_d95 for step in trajectory.steps])
    d02 = np.array([step.plan.clinical_metrics.target_d02 for step in trajectory.steps])
    oar_ratios = np.array(
        [
            [value / limit for value, limit in zip(step.plan.clinical_metrics.oar_mean, case.oar_limits)]
            for step in trajectory.steps
        ]
    )
    target_priority = np.array([step.plan.priorities.target for step in trajectory.steps])
    oar_priorities = np.array([step.plan.priorities.oars for step in trajectory.steps])
    beam_count = np.array([len(step.plan.active_beams) for step in trajectory.steps])

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7), constrained_layout=True, sharex=True)
    axes[0].plot(steps, d95, marker="o", linewidth=2, label="target D95")
    axes[0].plot(steps, d02, marker="s", linewidth=2, label="target D02")
    axes[0].plot(steps, oar_ratios[:, 0], marker="^", linewidth=2, label="OAR 1 mean / limit")
    axes[0].plot(steps, oar_ratios[:, 1], marker="v", linewidth=2, label="OAR 2 mean / limit")
    axes[0].axhline(0.85, color="#4daf4a", linestyle="--", linewidth=1, label="D95 minimum")
    axes[0].axhline(1.0, color="#377eb8", linestyle=":", linewidth=1, label="OAR limit ratio")
    axes[0].axhline(1.25, color="#e41a1c", linestyle="--", linewidth=1, label="D02 maximum")
    axes[0].set_ylabel("clinical metric / relative dose")
    axes[0].set_title("What the planner sees after each optimizer run")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].step(steps, target_priority, where="mid", linewidth=2, label="target priority")
    axes[1].step(steps, oar_priorities[:, 0], where="mid", linewidth=2, label="OAR 1 priority")
    axes[1].step(steps, oar_priorities[:, 1], where="mid", linewidth=2, label="OAR 2 priority")
    axes[1].step(steps, beam_count, where="mid", linewidth=2, color="#555555", label="active beam count")
    axes[1].set_xlabel("manual planning step")
    axes[1].set_ylabel("setting value")
    axes[1].set_title("What the planner changed")
    axes[1].legend(ncol=4, fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.suptitle("Manual decisions and resulting plan-quality response", fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
