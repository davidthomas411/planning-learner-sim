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
    structure_names: tuple[str, ...] = ()
    anatomy: str = "generic"
    difficulty: str = "moderate"
    available_beams: tuple[int, ...] = tuple(range(12))
    clinical_target: BoolArray | None = None
    evaluation_oars: tuple[BoolArray, ...] = ()
    evaluation_structure_names: tuple[str, ...] = ()
    voxel_volume_cc: float | None = None


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


def _pelvic_body(
    x: FloatArray,
    y: FloatArray,
    z: FloatArray,
    rng: np.random.Generator,
) -> BoolArray:
    """Generate a cropped pelvic-torso external contour."""

    lateral_base = rng.uniform(0.82, 0.94)
    depth_base = rng.uniform(0.62, 0.74)
    hip_level = rng.uniform(-0.30, -0.12)
    hip_width = rng.uniform(0.34, 0.48)
    hip_bulge = np.exp(-((z - hip_level) / hip_width) ** 2)
    lateral_radius = lateral_base * (0.94 + 0.09 * hip_bulge - 0.025 * z)
    depth_radius = depth_base * (0.97 + 0.05 * hip_bulge + 0.015 * z)
    center_y = rng.uniform(-0.025, 0.025) + 0.015 * z
    posterior_factor = rng.uniform(1.06, 1.13)
    local_depth_radius = np.where(y < center_y, depth_radius * posterior_factor, depth_radius)
    exponent = rng.uniform(2.25, 2.70)
    vertical_center = rng.uniform(-0.025, 0.025)
    vertical_half_length = rng.uniform(0.74, 0.86)
    vertical_exponent = rng.uniform(6.0, 9.0)
    return (
        (np.abs(x) / lateral_radius) ** exponent
        + (np.abs(y - center_y) / local_depth_radius) ** exponent
        + (np.abs(z - vertical_center) / vertical_half_length) ** vertical_exponent
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
        structure_names=tuple(f"oar_{index}" for index in range(len(oars))),
        difficulty=difficulty,
        available_beams=available_beams,
    )


def generate_prostate_case_3d(
    seed: int,
    grid_size: int = 64,
    difficulty: str = "moderate",
) -> SyntheticCase3D:
    """Generate a prostate phantom with three pelvic OAR priority groups."""

    if grid_size < 24:
        raise ValueError("grid_size must be at least 24")
    if difficulty not in {"easy", "moderate", "hard"}:
        raise ValueError("difficulty must be easy, moderate, or hard")
    rng = np.random.default_rng(seed)
    axis = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    body = _pelvic_body(x, y, z, rng)

    target_center = np.array([
        rng.uniform(-0.035, 0.035),
        rng.uniform(-0.08, -0.01),
        rng.uniform(-0.22, -0.12),
    ])
    target_scale = {"easy": 0.90, "moderate": 1.0, "hard": 1.14}[difficulty]
    target_radii = rng.uniform((0.13, 0.10, 0.10), (0.18, 0.14, 0.15)) * target_scale
    clinical_target = _ellipsoid(x, y, z, target_center, target_radii)

    # The normalized pelvic half-width represents approximately 150 mm.
    # Expand the prostate/CTV by 5 mm. A smaller 3 mm posterior margin is
    # clinically used, but it is below one voxel on the 64-cubed grid and
    # produced unstable overlap counts across grid resolutions.
    scale_mm = 150.0
    general_margin = 5.0 / scale_mm
    posterior_margin = general_margin
    ptv_center = target_center.copy()
    ptv_center[1] += (general_margin - posterior_margin) / 2.0
    ptv_radii = target_radii + np.array([
        general_margin,
        (general_margin + posterior_margin) / 2.0,
        general_margin,
    ])
    target = _ellipsoid(x, y, z, ptv_center, ptv_radii)
    if difficulty == "hard":
        seminal_center = target_center + np.array([0.0, 0.015, target_radii[2] * 0.85])
        seminal_radii = np.array([0.085, 0.055, 0.14])
        expanded_seminal_radii = seminal_radii + np.array([
            general_margin,
            (general_margin + posterior_margin) / 2.0,
            general_margin,
        ])
        for lateral_offset in (-0.075, 0.075):
            lobe_center = seminal_center + np.array([lateral_offset, 0.0, 0.0])
            clinical_target |= _ellipsoid(x, y, z, lobe_center, seminal_radii)
            expanded_lobe_center = lobe_center.copy()
            expanded_lobe_center[1] += (general_margin - posterior_margin) / 2.0
            target |= _ellipsoid(x, y, z, expanded_lobe_center, expanded_seminal_radii)
    clinical_target &= body
    target &= body

    bladder_fill = rng.uniform(0.75, 1.25)
    bladder_radii = np.array([
        rng.uniform(0.19, 0.27) * np.sqrt(bladder_fill),
        rng.uniform(0.15, 0.22) * np.sqrt(bladder_fill),
        rng.uniform(0.20, 0.32) * bladder_fill,
    ])
    bladder_contact_depth = {
        "easy": rng.uniform(0.000, 0.025),
        "moderate": rng.uniform(0.025, 0.060),
        "hard": rng.uniform(0.055, 0.100),
    }[difficulty]
    bladder_center = target_center + np.array([
        rng.uniform(-0.025, 0.025),
        rng.uniform(0.09, 0.15),
        target_radii[2] + bladder_radii[2] - bladder_contact_depth,
    ])
    bladder = (
        _ellipsoid(x, y, z, bladder_center, bladder_radii)
        & ~clinical_target
        & body
    )

    rectum_outer_radii = np.array([
        rng.uniform(0.09, 0.13), rng.uniform(0.065, 0.10), rng.uniform(0.34, 0.48)
    ])
    # Set the distance from the posterior target surface to the anterior
    # rectum surface. A negative value gives limited PTV-rectum overlap. The
    # whole-rectum contour is used because the prostate DVH goals are defined
    # for rectum, not for a generated hollow wall.
    rectum_surface_gap = {
        "easy": rng.uniform(0.045, 0.085),
        "moderate": rng.uniform(0.005, 0.040),
        "hard": rng.uniform(-0.020, 0.005),
    }[difficulty]
    rectum_center = target_center + np.array([
        rng.uniform(-0.025, 0.025),
        -(target_radii[1] + rectum_outer_radii[1] + rectum_surface_gap),
        rng.uniform(0.02, 0.09),
    ])
    rectum = (
        _ellipsoid(x, y, z, rectum_center, rectum_outer_radii)
        & ~clinical_target
        & body
    )

    head_z = rng.uniform(-0.24, -0.10)
    head_y = rng.uniform(-0.01, 0.08)
    head_offset = rng.uniform(0.45, 0.55)
    head_radii = rng.uniform((0.11, 0.11, 0.12), (0.15, 0.15, 0.17))
    left_head = _ellipsoid(x, y, z, np.array([-head_offset, head_y, head_z]), head_radii) & body
    right_head = _ellipsoid(x, y, z, np.array([head_offset, head_y, head_z]), head_radii) & body
    femoral_heads = left_head | right_head

    oars = (bladder, rectum, femoral_heads)
    minimum_voxels = max(6, round(50 * (grid_size / 64) ** 3))
    if int(target.sum()) < minimum_voxels or any(int(mask.sum()) < minimum_voxels for mask in oars):
        raise ValueError(f"Seed {seed} generated an invalid prostate phantom")
    if difficulty == "hard":
        removable = [beam for beam in range(12) if beam not in (0, 3, 6, 9)]
        removed = set(int(value) for value in rng.choice(removable, size=2, replace=False))
        available_beams = tuple(beam for beam in range(12) if beam not in removed)
    else:
        available_beams = tuple(range(12))

    return SyntheticCase3D(
        case_id=f"prostate3d-{seed:06d}",
        seed=seed,
        axis=axis,
        body=body,
        target=target,
        oars=oars,
        oar_limits=(0.48, 0.44, 0.34),
        structure_names=("bladder", "rectum", "femoral_heads"),
        anatomy="prostate",
        difficulty=difficulty,
        available_beams=available_beams,
        clinical_target=clinical_target,
        evaluation_oars=(bladder, rectum, left_head, right_head),
        evaluation_structure_names=(
            "bladder",
            "rectum",
            "femur_head_l",
            "femur_head_r",
        ),
        voxel_volume_cc=(300.0 / grid_size) ** 3 / 1000.0,
    )
