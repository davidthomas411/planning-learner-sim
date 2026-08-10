import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from dosim_sim.volume3d import generate_prostate_case_3d


COLORS = (
    (0.90, 0.20, 0.18),  # target
    (0.20, 0.55, 0.95),  # bladder
    (0.20, 0.75, 0.35),  # rectum
    (0.95, 0.75, 0.15),  # femoral heads
)


def colored_slice(case, axis: int, index: int) -> np.ndarray:
    masks = (case.target, *case.oars)
    body = np.take(case.body, index, axis=axis).T
    image = np.zeros((*body.shape, 3), dtype=np.float32)
    image[body] = 0.30
    for mask, color in zip(masks, COLORS, strict=True):
        selected = np.take(mask, index, axis=axis).T
        image[selected] = color
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the parametric prostate phantom")
    parser.add_argument("--seed", type=int, default=177)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("outputs/prostate_phantom_preview.png"))
    args = parser.parse_args()

    figure, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
    views = ((2, "axial"), (1, "coronal"), (0, "sagittal"))
    for row, difficulty in enumerate(("easy", "moderate", "hard")):
        case = generate_prostate_case_3d(args.seed, args.grid_size, difficulty)
        target_indices = np.argwhere(case.target)
        center = np.rint(target_indices.mean(axis=0)).astype(int)
        for column, (slice_axis, title) in enumerate(views):
            axes[row, column].imshow(colored_slice(case, slice_axis, int(center[slice_axis])), origin="lower")
            axes[row, column].set_title(f"{difficulty}: {title}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.legend(
        handles=[
            Patch(color=COLORS[0], label="Prostate PTV"),
            Patch(color=COLORS[1], label="Bladder"),
            Patch(color=COLORS[2], label="Rectum"),
            Patch(color=COLORS[3], label="Femoral heads"),
            Patch(color=(0.30, 0.30, 0.30), label="Body"),
        ],
        loc="outside lower center",
        ncol=5,
    )
    figure.suptitle("Parametric prostate planning phantom")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
