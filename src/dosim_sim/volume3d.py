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
    difficulty: str = "moderate"
    available_beams: tuple[int, ...] = tuple(range(12))


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


def generate_case_3d(
    seed: int,
    grid_size: int = 64,
    difficulty: str = "moderate",
    n_oars: int | None = None,
) -> SyntheticCase3D:
    """Generate a controlled 3D planning case.

    ``easy`` cases contain separated structures, ``moderate`` cases contain
    closer structures with occasional limited target overlap, and ``hard``
    cases contain three OARs, greater overlap, and a restricted beam set.
    """

    if grid_size < 24:
        raise ValueError("grid_size must be at least 24")
    if difficulty not in {"easy", "moderate", "hard"}:
        raise ValueError("difficulty must be easy, moderate, or hard")
    rng = np.random.default_rng(seed)
    axis = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    body = (x / 0.92) ** 2 + (y / 0.86) ** 2 + (z / 0.82) ** 2 <= 1.0

    target_center = rng.uniform(-0.10, 0.10, size=3)
    target_radii = rng.uniform((0.16, 0.14, 0.14), (0.23, 0.20, 0.19))
    target = _ellipsoid(x, y, z, target_center, target_radii) & body

    if n_oars is None:
        n_oars = {"easy": 1, "moderate": 2, "hard": 3}[difficulty]
    if n_oars < 1 or n_oars > 3:
        raise ValueError("n_oars must be between 1 and 3")

    distance_ranges = {
        "easy": (0.34, 0.44),
        "moderate": (0.27, 0.38),
        "hard": (0.24, 0.36),
    }
    limit_ranges = {
        "easy": (0.50, 0.62),
        "moderate": (0.44, 0.58),
        "hard": (0.45, 0.58),
    }
    oars: list[BoolArray] = []
    limits: list[float] = []
    start_angle = rng.uniform(0.0, 2.0 * np.pi)
    for index in range(n_oars):
        angle = start_angle + index * (2.0 * np.pi / n_oars) + rng.uniform(-0.25, 0.25)
        distance = rng.uniform(*distance_ranges[difficulty])
        center = target_center + np.array(
            [distance * np.cos(angle), distance * np.sin(angle), rng.uniform(-0.08, 0.08)]
        )
        radii = rng.uniform((0.13, 0.11, 0.13), (0.20, 0.17, 0.21))
        mask = _ellipsoid(x, y, z, center, radii) & body
        if difficulty == "easy":
            mask &= ~target
        minimum_oar_voxels = max(8, round(80 * (grid_size / 64) ** 3))
        if int(mask.sum()) < minimum_oar_voxels:
            raise ValueError(f"Seed {seed} generated an invalid 3D OAR")
        oars.append(mask)
        limits.append(float(rng.uniform(*limit_ranges[difficulty])))

    if difficulty == "hard":
        # Remove two angles while retaining at least three cardinal starting
        # directions. The restriction is part of the case, not a planner action.
        removable = [beam for beam in range(12) if beam not in (0, 3, 6, 9)]
        removed = set(int(value) for value in rng.choice(removable, size=2, replace=False))
        available_beams = tuple(beam for beam in range(12) if beam not in removed)
    else:
        available_beams = tuple(range(12))

    return SyntheticCase3D(
        case_id=f"synthetic3d-{seed:06d}",
        seed=seed,
        axis=axis,
        body=body,
        target=target,
        oars=tuple(oars),
        oar_limits=tuple(limits),
        difficulty=difficulty,
        available_beams=available_beams,
    )
