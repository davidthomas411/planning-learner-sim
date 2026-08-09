from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class SyntheticCase3D:
    """One deterministic synthetic 3D planning anatomy."""

    case_id: str
    seed: int
    axis: FloatArray
    body: BoolArray
    target: BoolArray
    oars: tuple[BoolArray, ...]
    oar_limits: tuple[float, ...]


def _ellipsoid(
    x: FloatArray,
    y: FloatArray,
    z: FloatArray,
    center: NDArray[np.floating],
    radii: NDArray[np.floating],
) -> BoolArray:
    return (
        ((x - center[0]) / radii[0]) ** 2
        + ((y - center[1]) / radii[1]) ** 2
        + ((z - center[2]) / radii[2]) ** 2
        <= 1.0
    )


def generate_case_3d(seed: int, grid_size: int = 64) -> SyntheticCase3D:
    """Generate a modest 3D target and two nearby, non-overlapping OARs."""

    if grid_size < 24:
        raise ValueError("grid_size must be at least 24")
    rng = np.random.default_rng(seed)
    axis = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    body = (x / 0.92) ** 2 + (y / 0.86) ** 2 + (z / 0.82) ** 2 <= 1.0

    target_center = rng.uniform(-0.10, 0.10, size=3)
    target_radii = rng.uniform((0.16, 0.14, 0.14), (0.23, 0.20, 0.19))
    target = _ellipsoid(x, y, z, target_center, target_radii) & body

    oars: list[BoolArray] = []
    limits: list[float] = []
    start_angle = rng.uniform(0.0, 2.0 * np.pi)
    for index in range(2):
        angle = start_angle + index * rng.uniform(1.7, 2.5)
        distance = rng.uniform(0.25, 0.34)
        center = target_center + np.array(
            [distance * np.cos(angle), distance * np.sin(angle), rng.uniform(-0.08, 0.08)]
        )
        radii = rng.uniform((0.13, 0.11, 0.13), (0.20, 0.17, 0.21))
        mask = _ellipsoid(x, y, z, center, radii) & body & ~target
        minimum_oar_voxels = max(8, round(80 * (grid_size / 64) ** 3))
        if int(mask.sum()) < minimum_oar_voxels:
            raise ValueError(f"Seed {seed} generated an invalid 3D OAR")
        oars.append(mask)
        limits.append(float(rng.uniform(0.45, 0.58)))

    return SyntheticCase3D(
        case_id=f"synthetic3d-{seed:06d}",
        seed=seed,
        axis=axis,
        body=body,
        target=target,
        oars=tuple(oars),
        oar_limits=tuple(limits),
    )
