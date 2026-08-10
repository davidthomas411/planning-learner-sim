from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .manual_planning import ManualAction
from .planning3d import PlanningStep3D
from .volume3d import SyntheticCase3D


FloatArray = NDArray[np.float32]
SHIFT_ACTION_NAMES = tuple(
    f"shift_beam_{beam}_to_{(beam + delta) % 12}"
    for beam in range(12)
    for delta in (-1, 1)
)
ACTION_NAMES = tuple(
    [f"add_beam_{beam}" for beam in range(12)]
    + [f"remove_beam_{beam}" for beam in range(12)]
    + ["increase_target_priority", "decrease_target_priority"]
    + ["increase_hotspot_priority", "decrease_hotspot_priority"]
    + [f"increase_oar_{index}_priority" for index in range(3)]
    + [f"decrease_oar_{index}_priority" for index in range(3)]
    + ["stop"]
    + list(SHIFT_ACTION_NAMES)
    + ["increase_normal_tissue_priority", "decrease_normal_tissue_priority"]
)
ACTION_TO_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}


def retention_eligible_3d(search_acceptable: bool, reference_acceptable: bool) -> bool:
    """Require both the demonstration and independent feasibility check to pass."""

    return bool(search_acceptable and reference_acceptable)


def action_name_3d(action: ManualAction | None) -> str:
    if action is None:
        return "stop"
    if action.kind in {"add_beam", "remove_beam"}:
        if action.beam_index is None:
            raise ValueError("Beam action lacks beam_index")
        return f"{action.kind}_{action.beam_index}"
    if action.kind == "shift_beam":
        if action.beam_index is None or action.new_beam_index is None:
            raise ValueError("Beam shift lacks old or new beam_index")
        return f"shift_beam_{action.beam_index}_to_{action.new_beam_index}"
    if action.kind in {"increase_oar_priority", "decrease_oar_priority"}:
        if action.structure_index is None:
            raise ValueError("OAR action lacks structure_index")
        prefix = "increase" if action.kind.startswith("increase") else "decrease"
        return f"{prefix}_oar_{action.structure_index}_priority"
    if action.kind not in ACTION_TO_INDEX:
        raise ValueError(f"Unsupported action: {action.kind}")
    return action.kind


def action_index_3d(action: ManualAction | None) -> int:
    return ACTION_TO_INDEX[action_name_3d(action)]


def _centroid_and_volume(mask: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, float]:
    indices = np.argwhere(mask)
    centroid = np.array(
        [axis[indices[:, dimension]].mean() for dimension in range(3)], dtype=np.float32
    )
    return centroid, float(mask.mean())


def state_features_3d(
    case: SyntheticCase3D,
    step: PlanningStep3D,
    max_steps: int,
) -> FloatArray:
    difficulty = np.array(
        [case.difficulty == value for value in ("easy", "moderate", "hard")], dtype=np.float32
    )
    target_centroid, target_volume = _centroid_and_volume(case.target, case.axis)
    geometry: list[float] = [*target_centroid, target_volume]
    target_voxels = max(int(case.target.sum()), 1)
    for index in range(3):
        if index < len(case.oars):
            centroid, volume = _centroid_and_volume(case.oars[index], case.axis)
            overlap = float(np.count_nonzero(case.target & case.oars[index]) / target_voxels)
            geometry.extend([*centroid, volume, overlap, case.oar_limits[index]])
        else:
            geometry.extend([0.0] * 6)
    active = np.zeros(12, dtype=np.float32)
    active[list(step.plan.active_beams)] = 1.0
    available = np.zeros(12, dtype=np.float32)
    available[list(case.available_beams)] = 1.0
    oar_ratios = np.zeros(3, dtype=np.float32)
    oar_priorities = np.zeros(3, dtype=np.float32)
    for index, (value, limit, priority) in enumerate(
        zip(step.plan.metrics.oar_mean, case.oar_limits, step.plan.priorities.oars, strict=True)
    ):
        oar_ratios[index] = value / limit
        oar_priorities[index] = priority / 25.0
    dynamic = np.array(
        [
            step.plan.metrics.target_d95,
            step.plan.metrics.target_d02,
            *oar_ratios,
            step.plan.priorities.target / 25.0,
            step.plan.priorities.hotspot / 25.0,
            *oar_priorities,
            step.plan.priorities.normal_tissue / 25.0,
            step.step / max(max_steps, 1),
            step.violation_score,
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [difficulty, np.asarray(geometry, dtype=np.float32), available, active, dynamic]
    ).astype(np.float32)


def final_settings_target_3d(step: PlanningStep3D) -> FloatArray:
    active = np.zeros(12, dtype=np.float32)
    active[list(step.plan.active_beams)] = 1.0
    priorities = np.zeros(5, dtype=np.float32)
    priorities[0] = step.plan.priorities.target / 25.0
    priorities[1] = step.plan.priorities.hotspot / 25.0
    priorities[2 : 2 + len(step.plan.priorities.oars)] = np.asarray(
        step.plan.priorities.oars, dtype=np.float32
    ) / 25.0
    return np.concatenate(
        [active, priorities, np.asarray([step.plan.priorities.normal_tissue / 25.0], dtype=np.float32)]
    )
