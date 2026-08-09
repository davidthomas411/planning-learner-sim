from __future__ import annotations

from typing import Any

import numpy as np

from .config import SimulationConfig
from .geometry import SyntheticCase
from .manual_planning import ManualAction
from .objective import clinical_violation_score
from .optimizer import OptimizedPlan


def _centroid(mask: np.ndarray, case: SyntheticCase) -> tuple[float, float]:
    return float(np.mean(case.x_grid[mask])), float(np.mean(case.y_grid[mask]))


def case_features(case: SyntheticCase) -> dict[str, Any]:
    target_x, target_y = _centroid(case.target, case)
    result: dict[str, Any] = {
        "target_area_fraction": float(case.target.mean()),
        "target_centroid_x": target_x,
        "target_centroid_y": target_y,
        "difficulty": case.difficulty,
    }
    for index, (oar, limit) in enumerate(zip(case.oars, case.oar_limits, strict=True), start=1):
        oar_x, oar_y = _centroid(oar, case)
        result.update(
            {
                f"oar_{index}_area_fraction": float(oar.mean()),
                f"oar_{index}_centroid_x": oar_x,
                f"oar_{index}_centroid_y": oar_y,
                f"oar_{index}_target_distance": float(np.hypot(oar_x - target_x, oar_y - target_y)),
                f"oar_{index}_limit": float(limit),
            }
        )
    return result


def plan_state(
    plan: OptimizedPlan,
    case: SyntheticCase,
    config: SimulationConfig,
) -> dict[str, Any]:
    metrics = plan.clinical_metrics
    return {
        "active_beams": list(plan.active_beams),
        "active_beam_angles_degrees": [beam * 360 / config.n_beams for beam in plan.active_beams],
        "target_priority": float(plan.priorities.target),
        "hotspot_priority": float(plan.priorities.hotspot),
        "oar_priorities": [float(value) for value in plan.priorities.oars],
        "target_d95": float(metrics.target_d95),
        "target_d02": float(metrics.target_d02),
        "oar_mean": [float(value) for value in metrics.oar_mean],
        "oar_mean_over_limit": [
            float(value / limit)
            for value, limit in zip(metrics.oar_mean, case.oar_limits, strict=True)
        ],
        "clinical_violation": clinical_violation_score(metrics, case, config),
        "optimizer_iterations": int(plan.optimizer_iterations),
    }


def action_record(action: ManualAction) -> dict[str, Any]:
    return {
        "kind": action.kind,
        "description": action.description,
        "beam_index": action.beam_index,
        "structure_index": action.structure_index,
        "old_value": action.old_value,
        "new_value": action.new_value,
    }

