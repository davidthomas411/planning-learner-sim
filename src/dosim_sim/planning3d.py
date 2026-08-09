from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .manual_planning import ManualAction
from .objective import PlanningPriorities
from .optimizer3d import PlanMetrics3D
from .torch_dose3d import TorchImplicitDoseEngine3D, TorchOptimizedPlan3D, optimize_fluence_3d_torch
from .volume3d import SyntheticCase3D


@dataclass(frozen=True)
class HighLevelSearchConfig3D:
    max_steps: int = 5
    beam_width: int = 3
    add_candidates: int = 2
    remove_candidates: int = 1
    optimizer_iterations: int = 40
    priority_factor: float = 1.75
    priority_ceiling: float = 6.0
    priority_floor: float = 0.5
    d95_min: float = 0.85
    d02_max: float = 1.25


@dataclass(frozen=True)
class PlanningStep3D:
    step: int
    action: ManualAction | None
    plan: TorchOptimizedPlan3D
    violation_score: float


@dataclass(frozen=True)
class PlanningTrajectory3D:
    case_id: str
    steps: tuple[PlanningStep3D, ...]
    stopping_reason: str

    @property
    def final(self) -> PlanningStep3D:
        return self.steps[-1]


def is_acceptable_3d(
    metrics: PlanMetrics3D,
    case: SyntheticCase3D,
    config: HighLevelSearchConfig3D | None = None,
) -> bool:
    cfg = config or HighLevelSearchConfig3D()
    return (
        metrics.target_d95 >= cfg.d95_min
        and metrics.target_d02 <= cfg.d02_max
        and all(value <= limit for value, limit in zip(metrics.oar_mean, case.oar_limits, strict=True))
    )


def clinical_violation_score_3d(
    metrics: PlanMetrics3D,
    case: SyntheticCase3D,
    config: HighLevelSearchConfig3D | None = None,
) -> float:
    cfg = config or HighLevelSearchConfig3D()
    coverage = max(cfg.d95_min - metrics.target_d95, 0.0) / cfg.d95_min
    hotspot = max(metrics.target_d02 - cfg.d02_max, 0.0) / cfg.d02_max
    oars = [
        max(value / limit - 1.0, 0.0)
        for value, limit in zip(metrics.oar_mean, case.oar_limits, strict=True)
    ]
    return float(coverage + hotspot + sum(oars))


def _centroid_xy(mask: np.ndarray, axis: np.ndarray) -> np.ndarray:
    indices = np.argwhere(mask)
    return np.array([axis[indices[:, 0]].mean(), axis[indices[:, 1]].mean()])


def _beam_separation_scores(case: SyntheticCase3D, oar_weights: np.ndarray) -> np.ndarray:
    target = _centroid_xy(case.target, case.axis)
    displacements = np.stack([_centroid_xy(mask, case.axis) - target for mask in case.oars])
    scores = np.zeros(12, dtype=np.float64)
    normalized_weights = oar_weights / max(float(oar_weights.sum()), 1e-8)
    for beam, angle_degrees in enumerate(range(0, 360, 30)):
        angle = np.deg2rad(angle_degrees)
        lateral_axis = np.array([-np.sin(angle), np.cos(angle)])
        scores[beam] = float(np.sum(np.abs(displacements @ lateral_axis) * normalized_weights))
    return scores


def _candidate_settings(
    plan: TorchOptimizedPlan3D,
    case: SyntheticCase3D,
    config: HighLevelSearchConfig3D,
) -> list[tuple[ManualAction, tuple[int, ...], PlanningPriorities]]:
    priorities = plan.priorities
    metrics = plan.metrics
    active = plan.active_beams
    factor = config.priority_factor
    candidates: list[tuple[ManualAction, tuple[int, ...], PlanningPriorities]] = []

    if metrics.target_d95 < config.d95_min and priorities.target < config.priority_ceiling:
        new = min(priorities.target * factor, config.priority_ceiling)
        candidates.append((ManualAction("increase_target_priority", f"Increase target priority {priorities.target:.2f} -> {new:.2f}", old_value=priorities.target, new_value=new), active, replace(priorities, target=new)))
    elif metrics.target_d95 > config.d95_min + 0.04 and priorities.target > 1.0:
        new = max(priorities.target / factor, config.priority_floor)
        candidates.append((ManualAction("decrease_target_priority", f"Decrease target priority {priorities.target:.2f} -> {new:.2f}", old_value=priorities.target, new_value=new), active, replace(priorities, target=new)))
    if metrics.target_d02 > config.d02_max and priorities.hotspot < config.priority_ceiling:
        new = min(priorities.hotspot * factor, config.priority_ceiling)
        candidates.append((ManualAction("increase_hotspot_priority", f"Increase target hot-spot priority {priorities.hotspot:.2f} -> {new:.2f}", old_value=priorities.hotspot, new_value=new), active, replace(priorities, hotspot=new)))
    elif metrics.target_d02 < config.d02_max - 0.08 and priorities.hotspot > 1.0:
        new = max(priorities.hotspot / factor, config.priority_floor)
        candidates.append((ManualAction("decrease_hotspot_priority", f"Decrease target hot-spot priority {priorities.hotspot:.2f} -> {new:.2f}", old_value=priorities.hotspot, new_value=new), active, replace(priorities, hotspot=new)))
    oar_violation = np.array([
        max(value / limit - 1.0, 0.0)
        for value, limit in zip(metrics.oar_mean, case.oar_limits, strict=True)
    ])
    for index, (violation, old) in enumerate(zip(oar_violation, priorities.oars, strict=True)):
        if violation > 0 and old < config.priority_ceiling:
            updated = list(priorities.oars)
            updated[index] = min(old * factor, config.priority_ceiling)
            candidates.append((ManualAction("increase_oar_priority", f"Increase OAR {index + 1} priority {old:.2f} -> {updated[index]:.2f}", structure_index=index, old_value=old, new_value=updated[index]), active, replace(priorities, oars=tuple(updated))))
        elif metrics.oar_mean[index] < 0.8 * case.oar_limits[index] and old > 1.0:
            updated = list(priorities.oars)
            updated[index] = max(old / factor, config.priority_floor)
            candidates.append((ManualAction("decrease_oar_priority", f"Decrease OAR {index + 1} priority {old:.2f} -> {updated[index]:.2f}", structure_index=index, old_value=old, new_value=updated[index]), active, replace(priorities, oars=tuple(updated))))

    weights = np.maximum(oar_violation, 0.05) * np.asarray(priorities.oars)
    separation = _beam_separation_scores(case, weights)
    inactive = [beam for beam in case.available_beams if beam not in active]
    for beam in sorted(inactive, key=lambda value: (-separation[value], value))[: config.add_candidates]:
        candidates.append((ManualAction("add_beam", f"Add {beam * 30} degree beam", beam_index=beam), tuple(sorted((*active, beam))), priorities))
    if len(active) > 3:
        for beam in sorted(active, key=lambda value: (separation[value], value))[: config.remove_candidates]:
            candidates.append((ManualAction("remove_beam", f"Remove {beam * 30} degree beam", beam_index=beam), tuple(value for value in active if value != beam), priorities))
    return candidates


def run_high_level_search_3d(
    case: SyntheticCase3D,
    engine: TorchImplicitDoseEngine3D,
    config: HighLevelSearchConfig3D | None = None,
) -> PlanningTrajectory3D:
    """Bounded search over manual beam-angle and named-priority changes."""

    cfg = config or HighLevelSearchConfig3D()
    active = tuple(beam for beam in (0, 3, 6, 9) if beam in case.available_beams)
    if len(active) < 3:
        active = case.available_beams[:3]
    priorities = PlanningPriorities.for_case(case)
    initial = optimize_fluence_3d_torch(case, engine, active, priorities, cfg.optimizer_iterations)
    score = clinical_violation_score_3d(initial.metrics, case, cfg)
    initial_steps = (PlanningStep3D(0, None, initial, score),)
    if is_acceptable_3d(initial.metrics, case, cfg):
        return PlanningTrajectory3D(case.case_id, initial_steps, "acceptable")

    frontier = [(initial, score, initial_steps)]
    best_score, best_steps = score, initial_steps
    visited = {(initial.active_beams, priorities.target, priorities.hotspot, priorities.oars)}
    stopping_reason = "search_step_limit"
    for step_index in range(1, cfg.max_steps + 1):
        expanded = []
        for current, _, steps in frontier:
            for action, beams, candidate_priorities in _candidate_settings(current, case, cfg):
                key = (beams, candidate_priorities.target, candidate_priorities.hotspot, candidate_priorities.oars)
                if key in visited:
                    continue
                visited.add(key)
                candidate = optimize_fluence_3d_torch(case, engine, beams, candidate_priorities, cfg.optimizer_iterations, initial_fluence=current.fluence)
                violation = clinical_violation_score_3d(candidate.metrics, case, cfg)
                candidate_steps = (*steps, PlanningStep3D(step_index, action, candidate, violation))
                if violation < best_score:
                    best_score, best_steps = violation, candidate_steps
                if is_acceptable_3d(candidate.metrics, case, cfg):
                    return PlanningTrajectory3D(case.case_id, candidate_steps, "acceptable")
                expanded.append((violation, candidate.metrics.loss, action.description, candidate, candidate_steps))
        if not expanded:
            stopping_reason = "search_exhausted"
            break
        expanded.sort(key=lambda value: value[:3])
        frontier = [(plan, violation, steps) for violation, _, _, plan, steps in expanded[: cfg.beam_width]]
    return PlanningTrajectory3D(case.case_id, best_steps, stopping_reason)


def run_reference_optimizer_3d(
    case: SyntheticCase3D,
    engine: TorchImplicitDoseEngine3D,
    iterations: int = 400,
    penalty_rounds: int = 8,
) -> TorchOptimizedPlan3D:
    """Independent adaptive-penalty feasibility reference.

    Penalty adaptation is internal to the evaluation solver. It is not stored as
    a manual trajectory and is never supplied to a learner.
    """

    if penalty_rounds < 1:
        raise ValueError("penalty_rounds must be positive")
    priorities = PlanningPriorities.for_case(case)
    fluence = None
    best: TorchOptimizedPlan3D | None = None
    best_key = (float("inf"), float("inf"))
    iterations_per_round = max(1, iterations // penalty_rounds)
    cfg = HighLevelSearchConfig3D()
    for _ in range(penalty_rounds):
        plan = optimize_fluence_3d_torch(
            case,
            engine,
            case.available_beams,
            priorities,
            iterations=iterations_per_round,
            learning_rate=0.05,
            initial_fluence=fluence,
        )
        fluence = plan.fluence
        key = (clinical_violation_score_3d(plan.metrics, case, cfg), plan.metrics.loss)
        if key < best_key:
            best, best_key = plan, key
        updated_oars = list(priorities.oars)
        for index, (value, limit) in enumerate(zip(plan.metrics.oar_mean, case.oar_limits, strict=True)):
            if value > limit:
                updated_oars[index] = min(updated_oars[index] * 2.5, 25.0)
        priorities = PlanningPriorities(
            target=min(priorities.target * 2.5, 25.0) if plan.metrics.target_d95 < cfg.d95_min else priorities.target,
            hotspot=min(priorities.hotspot * 2.5, 25.0) if plan.metrics.target_d02 > cfg.d02_max else priorities.hotspot,
            oars=tuple(updated_oars),
        )
    assert best is not None
    return best
