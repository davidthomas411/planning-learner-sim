from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import SimulationConfig


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    seed: int
    x_grid: FloatArray
    y_grid: FloatArray
    body: BoolArray
    target: BoolArray
    oars: tuple[BoolArray, ...]
    oar_limits: tuple[float, ...]
    difficulty: str


def _ellipse(
    x: FloatArray,
    y: FloatArray,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    rotation: float,
) -> BoolArray:
    cos_a, sin_a = np.cos(rotation), np.sin(rotation)
    dx, dy = x - center_x, y - center_y
    xr = cos_a * dx + sin_a * dy
    yr = -sin_a * dx + cos_a * dy
    return (xr / radius_x) ** 2 + (yr / radius_y) ** 2 <= 1.0


def generate_case(seed: int, config: SimulationConfig | None = None) -> SyntheticCase:
    """Generate one deterministic moderate-difficulty synthetic anatomy."""

    cfg = config or SimulationConfig()
    rng = np.random.default_rng(seed)
    axis = np.linspace(-1.0, 1.0, cfg.grid_size)
    x, y = np.meshgrid(axis, axis)
    body = (x / 0.92) ** 2 + (y / 0.86) ** 2 <= 1.0

    target_center = rng.uniform(-0.12, 0.12, size=2)
    target_radii = rng.uniform((0.16, 0.13), (0.23, 0.19))
    target = _ellipse(
        x,
        y,
        target_center[0],
        target_center[1],
        target_radii[0],
        target_radii[1],
        rng.uniform(0, np.pi),
    ) & body

    oars: list[BoolArray] = []
    oar_limits: list[float] = []
    base_angles = rng.uniform(0, 2 * np.pi, size=2)
    for index, angle in enumerate(base_angles):
        # Primary-environment OARs are close to the target but do not overlap it.
        # Direct overlap will be added later as a prespecified hard/OOD condition.
        distance = rng.uniform(0.21, 0.34)
        cx = target_center[0] + distance * np.cos(angle)
        cy = target_center[1] + distance * np.sin(angle)
        radii = rng.uniform((0.12, 0.10), (0.20, 0.16))
        mask = _ellipse(x, y, cx, cy, radii[0], radii[1], rng.uniform(0, np.pi)) & body
        mask &= ~target
        if mask.sum() < 12:
            raise ValueError(f"Seed {seed} generated an invalid OAR; choose another seed")
        oars.append(mask)
        oar_limits.append(float(rng.uniform(0.46, 0.60)))

    minimum_gap = min(
        float(np.min(np.hypot(x[oar] - target_center[0], y[oar] - target_center[1]))) for oar in oars
    )
    difficulty = "moderate" if minimum_gap < min(target_radii) * 1.15 else "easy"
    return SyntheticCase(
        case_id=f"synthetic-{seed:06d}",
        seed=seed,
        x_grid=x,
        y_grid=y,
        body=body,
        target=target,
        oars=tuple(oars),
        oar_limits=tuple(oar_limits),
        difficulty=difficulty,
    )
