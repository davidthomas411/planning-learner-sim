import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from dosim_sim.clinical3d import load_tcia_prostate_case
from render_prostate_phantom import COLORS, colored_slice


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one imported TCIA prostate case")
    parser.add_argument("subject_dir", type=Path)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("outputs/tcia_prostate_preview.png"))
    args = parser.parse_args()
    case = load_tcia_prostate_case(args.subject_dir, args.grid_size)
    center = np.rint(np.argwhere(case.target).mean(axis=0)).astype(int)
    figure, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
    for axis, (slice_axis, title) in zip(axes, ((2, "axial"), (1, "coronal"), (0, "sagittal")), strict=True):
        axis.imshow(colored_slice(case, slice_axis, int(center[slice_axis])), origin="lower")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.legend(
        handles=[
            Patch(color=COLORS[0], label="Prostate PTV"),
            Patch(color=COLORS[1], label="Bladder"),
            Patch(color=COLORS[2], label="Rectum"),
            Patch(color=COLORS[3], label="Femoral heads"),
            Patch(color=(0.30, 0.30, 0.30), label="CT-derived body"),
        ],
        loc="outside lower center",
        ncol=5,
    )
    figure.suptitle(f"Imported TCIA anatomy: {args.subject_dir.name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
