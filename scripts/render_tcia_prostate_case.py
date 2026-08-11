import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from dosim_sim.clinical3d import load_tcia_prostate_case
from dosim_sim.visuals3d import add_hfs_orientation_labels


DISPLAY_COLORS = {
    "body": np.array((0.82, 0.82, 0.82)),
    "ptv_margin": np.array((0.89, 0.34, 0.34)),
    "clinical_target": np.array((0.70, 0.09, 0.17)),
    "bladder": np.array((0.30, 0.47, 0.66)),
    "rectum": np.array((0.35, 0.67, 0.32)),
    "bladder_overlap": np.array((0.95, 0.81, 0.36)),
    "rectum_overlap": np.array((0.95, 0.55, 0.17)),
}


def plane(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return array[index, :, :].T
    if axis == 1:
        return array[:, index, :].T
    return array[:, :, index].T


def overlap_display(case, axis: int, index: int) -> np.ndarray:
    body = plane(case.body, axis, index)
    ptv = plane(case.target, axis, index)
    clinical_target = (
        np.zeros_like(ptv)
        if case.clinical_target is None
        else plane(case.clinical_target, axis, index)
    )
    bladder = plane(case.oars[0], axis, index)
    rectum = plane(case.oars[1], axis, index)
    image = np.zeros((*body.shape, 3), dtype=np.float32)
    image[body] = DISPLAY_COLORS["body"]
    image[ptv] = DISPLAY_COLORS["ptv_margin"]
    image[clinical_target] = DISPLAY_COLORS["clinical_target"]
    image[bladder] = DISPLAY_COLORS["bladder"]
    image[rectum] = DISPLAY_COLORS["rectum"]
    image[ptv & bladder] = DISPLAY_COLORS["bladder_overlap"]
    image[ptv & rectum] = DISPLAY_COLORS["rectum_overlap"]
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one imported TCIA prostate case")
    parser.add_argument("subject_dir", type=Path)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("outputs/tcia_prostate_preview.png"))
    args = parser.parse_args()
    case = load_tcia_prostate_case(args.subject_dir, args.grid_size)
    center = np.rint(np.argwhere(case.target).mean(axis=0)).astype(int)
    overlap = case.target & (case.oars[0] | case.oars[1])
    overlap_center = np.rint(np.argwhere(overlap).mean(axis=0)).astype(int) if overlap.any() else center
    figure, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
    for axis, (slice_axis, title) in zip(axes, ((2, "axial"), (1, "coronal"), (0, "sagittal")), strict=True):
        axis.imshow(overlap_display(case, slice_axis, int(overlap_center[slice_axis])), origin="lower")
        axis.set_title(title)
        add_hfs_orientation_labels(axis, title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.legend(
        handles=[
            Patch(color=DISPLAY_COLORS["clinical_target"], label="Prostate/CTV"),
            Patch(color=DISPLAY_COLORS["ptv_margin"], label="PTV margin"),
            Patch(color=DISPLAY_COLORS["bladder"], label="Bladder"),
            Patch(color=DISPLAY_COLORS["rectum"], label="Rectum"),
            Patch(color=DISPLAY_COLORS["bladder_overlap"], label="PTV-bladder overlap"),
            Patch(color=DISPLAY_COLORS["rectum_overlap"], label="PTV-rectum overlap"),
        ],
        loc="outside lower center",
        ncol=3,
    )
    figure.suptitle(f"Imported TCIA anatomy: {args.subject_dir.name} | head-first supine")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
