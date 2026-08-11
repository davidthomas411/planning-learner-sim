"""Compare the parametric pelvic body with one CT-derived body contour."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dosim_sim.clinical3d import load_tcia_prostate_case
from dosim_sim.volume3d import generate_prostate_case_3d


def plane(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return array[index, :, :].T
    if axis == 1:
        return array[:, index, :].T
    return array[:, :, index].T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject_dir", type=Path)
    parser.add_argument("--seed", type=int, default=210018)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = (
        ("Parametric pelvic torso", generate_prostate_case_3d(args.seed, args.grid_size, "hard")),
        ("CT-derived external contour", load_tcia_prostate_case(args.subject_dir, args.grid_size)),
    )
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    views = ((0, "sagittal"), (1, "coronal"), (2, "axial"))
    for row, (source_name, case) in enumerate(cases):
        center = np.rint(np.argwhere(case.target).mean(axis=0)).astype(int)
        for column, (slice_axis, view_name) in enumerate(views):
            axis = axes[row, column]
            axis.imshow(
                plane(case.body, slice_axis, int(center[slice_axis])),
                origin="lower",
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            axis.contour(
                plane(case.target, slice_axis, int(center[slice_axis])),
                levels=[0.5],
                colors=["#e45756"],
                linewidths=1.5,
            )
            if case.clinical_target is not None:
                axis.contour(
                    plane(case.clinical_target, slice_axis, int(center[slice_axis])),
                    levels=[0.5],
                    colors=["#ff66cc"],
                    linewidths=1.0,
                )
            axis.set_title(f"{source_name}: {view_name}")
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        "Pelvic body contours\nmagenta: prostate/CTV; red: PTV",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
