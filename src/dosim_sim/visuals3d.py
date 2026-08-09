from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .optimizer3d import OptimizedPlan3D
from .volume3d import SyntheticCase3D


def _structure_overlay(case: SyntheticCase3D) -> np.ndarray:
    labels = np.zeros(case.body.shape, dtype=np.uint8)
    labels[case.oars[0]] = 2
    labels[case.oars[1]] = 3
    labels[case.target] = 1
    return labels


def save_3d_case_slices(case: SyntheticCase3D, path: Path) -> None:
    """Show axial, coronal, and sagittal anatomy through the target center."""

    path.parent.mkdir(parents=True, exist_ok=True)
    center = np.rint(np.mean(np.argwhere(case.target), axis=0)).astype(int)
    labels = _structure_overlay(case)
    cmap = ListedColormap([(0, 0, 0, 0), "#f2c14e", "#e45756", "#4c78a8"])
    views = (
        ("Axial (x–y)", labels[:, :, center[2]].T, case.body[:, :, center[2]].T),
        ("Coronal (x–z)", labels[:, center[1], :].T, case.body[:, center[1], :].T),
        ("Sagittal (y–z)", labels[center[0], :, :].T, case.body[center[0], :, :].T),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, (title, structure_slice, body_slice) in zip(axes, views, strict=True):
        axis.imshow(body_slice, origin="lower", cmap="Greys", vmin=0, vmax=1, alpha=0.22)
        axis.imshow(structure_slice, origin="lower", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("The same synthetic patient viewed in three orthogonal planes")
    fig.text(0.5, 0.01, "Gold = target   Red = OAR 1   Blue = OAR 2", ha="center")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_3d_planning_steps(
    case: SyntheticCase3D,
    plans: tuple[OptimizedPlan3D, ...],
    action_labels: tuple[str, ...],
    beam_angles_degrees: tuple[float, ...],
    path: Path,
) -> None:
    """Show the actual dose after each human-level edit and full reoptimization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    axial_index = int(np.rint(np.mean(np.argwhere(case.target), axis=0))[2])
    labels = _structure_overlay(case)[:, :, axial_index].T
    structure_cmap = ListedColormap([(0, 0, 0, 0), "#f2c14e", "#e45756", "#4c78a8"])
    fig, axes = plt.subplots(2, len(plans), figsize=(4.2 * len(plans), 7.4), constrained_layout=True)
    if len(plans) == 1:
        axes = axes.reshape(2, 1)
    max_dose = max(float(np.max(plan.dose)) for plan in plans)
    dose_image = None
    for column, (plan, action) in enumerate(zip(plans, action_labels, strict=True)):
        dose_slice = plan.dose[:, :, axial_index].T
        dose_image = axes[0, column].imshow(
            dose_slice,
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=max(1.25, max_dose),
        )
        axes[0, column].imshow(labels, origin="lower", cmap=structure_cmap, vmin=0, vmax=3, alpha=0.28)
        axes[0, column].contour(case.target[:, :, axial_index].T, levels=[0.5], colors=["white"], linewidths=1.2)
        axes[0, column].set_title(f"Step {column}: {action}")
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])

        active_angles = [beam_angles_degrees[index] for index in plan.active_beams]
        ratios = [value / limit for value, limit in zip(plan.metrics.oar_mean, case.oar_limits, strict=True)]
        metric_names = ["Target D95", "Target D02", "OAR 1 / limit", "OAR 2 / limit"]
        metric_values = [plan.metrics.target_d95 / 0.85, plan.metrics.target_d02 / 1.25, *ratios]
        colors = ["#59a14f" if value <= 1.0 else "#e15759" for value in metric_values]
        colors[0] = "#59a14f" if metric_values[0] >= 1.0 else "#e15759"
        axes[1, column].barh(metric_names, metric_values, color=colors)
        axes[1, column].axvline(1.0, color="black", linewidth=1, alpha=0.45)
        axes[1, column].set_xlim(0, max(1.35, max(metric_values) * 1.12))
        axes[1, column].invert_yaxis()
        axes[1, column].set_xlabel("Value / pass threshold")
        axes[1, column].set_title("Active angles: " + ", ".join(f"{value:.0f}°" for value in active_angles))
        for row, value in enumerate(metric_values):
            axes[1, column].text(value + 0.02, row, f"{value:.2f}", va="center")
    if dose_image is not None:
        fig.colorbar(dose_image, ax=axes[0, :].tolist(), shrink=0.8, label="Dose / prescription")
    fig.suptitle("One manual edit per step; fluence is reoptimized automatically between images")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
