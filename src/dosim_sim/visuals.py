from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .config import SimulationConfig
from .expert import ExpertTrajectory
from .geometry import SyntheticCase


def _anatomy_labels(case: SyntheticCase) -> np.ndarray:
    labels = np.zeros(case.body.shape, dtype=int)
    labels[case.body] = 1
    for index, oar in enumerate(case.oars, start=2):
        labels[oar] = index
    labels[case.target] = 4
    return labels


def save_case_overview(
    case: SyntheticCase,
    influence: np.ndarray,
    output_path: Path,
    config: SimulationConfig | None = None,
) -> None:
    cfg = config or SimulationConfig()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anatomy = _anatomy_labels(case)
    cmap = ListedColormap(["#ffffff", "#d8d8d8", "#4c78a8", "#f58518", "#54a24b"])
    central_beamlet = cfg.beamlets_per_beam // 2
    beam_kernel = influence[:, central_beamlet].reshape(case.body.shape)
    all_beams = influence.sum(axis=1).reshape(case.body.shape)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    axes[0].imshow(anatomy, origin="lower", cmap=cmap, vmin=0, vmax=4)
    axes[0].set_title("1. Synthetic anatomy")
    axes[0].text(2, 3, "gray: body | green: target\nblue/orange: avoidance structures", fontsize=8)
    image = axes[1].imshow(beam_kernel, origin="lower", cmap="magma", vmin=0)
    axes[1].contour(case.target, levels=[0.5], colors=["white"], linewidths=1)
    axes[1].set_title("2. One beamlet's dose influence")
    fig.colorbar(image, ax=axes[1], fraction=0.046, label="relative dose / unit intensity")
    image = axes[2].imshow(all_beams, origin="lower", cmap="magma", vmin=0)
    axes[2].contour(case.target, levels=[0.5], colors=["white"], linewidths=1)
    axes[2].set_title(f"3. All {cfg.n_beamlets} available beamlets")
    fig.colorbar(image, ax=axes[2], fraction=0.046, label="summed influence")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("How a synthetic planning case is constructed", fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_trajectory_story(
    case: SyntheticCase,
    trajectory: ExpertTrajectory,
    output_path: Path,
    config: SimulationConfig | None = None,
) -> None:
    cfg = config or SimulationConfig()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midpoint = trajectory.steps[len(trajectory.steps) // 2]
    snapshots = [trajectory.steps[0], midpoint, trajectory.steps[-1]]
    vmax = max(float(step.dose.max()) for step in snapshots)

    fig = plt.figure(figsize=(12, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85])
    for index, step in enumerate(snapshots):
        axis = fig.add_subplot(grid[0, index])
        image = axis.imshow(step.dose, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        axis.contour(case.target, levels=[0.5], colors=["#4daf4a"], linewidths=1.4)
        for oar, color in zip(case.oars, ["#377eb8", "#ff7f00"], strict=False):
            axis.contour(oar, levels=[0.5], colors=[color], linewidths=1.2)
        label = "start" if index == 0 else "middle" if index == 1 else "final"
        axis.set_title(f"{label}: step {step.step}\nscore {step.metrics.total:.3f}, target D95 {step.metrics.target_d95:.2f}")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(image, ax=[fig.axes[0], fig.axes[1], fig.axes[2]], fraction=0.02, label="relative dose")

    curve_axis = fig.add_subplot(grid[1, :2])
    step_numbers = [step.step for step in trajectory.steps]
    objectives = [step.metrics.total for step in trajectory.steps]
    d95 = [step.metrics.target_d95 for step in trajectory.steps]
    curve_axis.plot(step_numbers, objectives, color="#7b3294", linewidth=2, label="plan score (lower is better)")
    curve_axis.set_xlabel("expert adjustment number")
    curve_axis.set_ylabel("objective score")
    curve_axis.grid(alpha=0.25)
    second_axis = curve_axis.twinx()
    second_axis.plot(step_numbers, d95, color="#008837", linewidth=2, label="target D95")
    second_axis.axhline(0.85 * cfg.prescription, color="#008837", linestyle="--", linewidth=1)
    second_axis.set_ylabel("target D95")
    lines = curve_axis.lines[:1] + second_axis.lines[:1]
    curve_axis.legend(lines, [line.get_label() for line in lines], loc="center right")

    action_axis = fig.add_subplot(grid[1, 2])
    final_intensities = trajectory.final.intensities.reshape(cfg.n_beams, cfg.beamlets_per_beam)
    action_image = action_axis.imshow(final_intensities, aspect="auto", cmap="viridis", origin="lower")
    action_axis.set_xlabel("beamlet across field")
    action_axis.set_ylabel("beam angle index")
    action_axis.set_title("Final choices made by expert")
    fig.colorbar(action_image, ax=action_axis, fraction=0.06, label="intensity")

    fig.suptitle(
        f"What the expert trajectory records — {len(trajectory.steps) - 1} actions, stop: {trajectory.stopping_reason}",
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_action_sequence(
    trajectory: ExpertTrajectory,
    output_path: Path,
    config: SimulationConfig | None = None,
) -> None:
    """Show the exact action chosen at each expert step and why the score moved."""

    cfg = config or SimulationConfig()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = trajectory.steps[1:]
    x = np.array([step.step for step in steps])
    beamlets = np.array([int(step.beamlet) for step in steps])
    beam_indices = beamlets // cfg.beamlets_per_beam
    across_field = beamlets % cfg.beamlets_per_beam
    deltas = np.array([step.delta for step in steps])

    all_steps = trajectory.steps
    all_x = np.array([step.step for step in all_steps])
    target_term = np.array(
        [
            cfg.target_underdose_weight * step.metrics.target_underdose
            + cfg.target_d95_weight * step.metrics.target_d95_shortfall
            + cfg.target_d02_weight * step.metrics.target_d02_excess
            for step in all_steps
        ]
    )
    hotspot_term = np.array([cfg.target_hotspot_weight * step.metrics.target_hotspot for step in all_steps])
    oar_term = np.array(
        [
            cfg.oar_weight * step.metrics.oar_penalty
            + cfg.oar_mean_excess_weight * step.metrics.oar_mean_excess
            for step in all_steps
        ]
    )
    regularization = np.array(
        [
            cfg.complexity_weight * step.metrics.complexity
            + cfg.smoothness_weight * step.metrics.smoothness
            for step in all_steps
        ]
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), constrained_layout=True, sharex=True)
    axes[0].stackplot(
        all_x,
        target_term,
        hotspot_term,
        oar_term,
        regularization,
        labels=["target underdose", "target hot spot", "avoidance dose", "complexity/smoothness"],
        colors=["#4daf4a", "#e41a1c", "#377eb8", "#999999"],
        alpha=0.85,
    )
    axes[0].set_ylabel("weighted contribution to score")
    axes[0].set_title("Why each expert adjustment changes the plan score")
    axes[0].legend(loc="upper right", ncol=2)
    axes[0].grid(axis="y", alpha=0.2)

    sizes = 70 + 280 * np.abs(deltas) / max(cfg.action_step_sizes)
    scatter = axes[1].scatter(
        x,
        beam_indices,
        c=across_field,
        s=sizes,
        marker="o",
        cmap="viridis",
        vmin=0,
        vmax=cfg.beamlets_per_beam - 1,
        edgecolor="black",
        linewidth=0.4,
    )
    for step, beam_index, delta in zip(x, beam_indices, deltas, strict=True):
        axes[1].text(step, beam_index, "+" if delta > 0 else "−", ha="center", va="center", fontsize=8)
    axes[1].set_xlabel("expert adjustment number")
    axes[1].set_ylabel("beam angle index")
    axes[1].set_yticks(range(cfg.n_beams))
    axes[1].set_title("Exact action: angle (vertical), beamlet (color), and intensity change (circle size)")
    axes[1].grid(alpha=0.2)
    fig.colorbar(scatter, ax=axes[1], label="beamlet index across field")
    fig.suptitle("The trajectory is a sequence of labeled decisions, not just a final dose", fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
