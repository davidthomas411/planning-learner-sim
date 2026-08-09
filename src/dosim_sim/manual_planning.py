from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from .config import SimulationConfig
from .geometry import SyntheticCase
from .objective import PlanningPriorities, is_acceptable
from .optimizer import OptimizedPlan, optimize_beamlets


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ManualAction:
    kind: str
    description: str
    beam_index: int | None = None
    structure_index: int | None = None
    old_value: float | None = None
    new_value: float | None = None


@dataclass(frozen=True)
class ManualStep:
    step: int
    action: ManualAction | None
    plan: OptimizedPlan


@dataclass(frozen=True)
class ManualTrajectory:
    case_id: str
    steps: tuple[ManualStep, ...]
    stopping_reason: str

    @property
    def final(self) -> ManualStep:
        return self.steps[-1]


def _beam_target_oar_scores(
    case: SyntheticCase, influence: FloatArray, cfg: SimulationConfig
) -> tuple[np.ndarray, np.ndarray]:
    target_scores = np.zeros(cfg.n_beams)
    oar_scores = np.zeros((len(case.oars), cfg.n_beams))
    for beam in range(cfg.n_beams):
        columns = slice(
            beam * cfg.beamlets_per_beam,
            (beam + 1) * cfg.beamlets_per_beam,
        )
        beam_map = influence[:, columns].sum(axis=1).reshape(case.body.shape)
        target_scores[beam] = float(np.mean(beam_map[case.target]))
        for index, oar in enumerate(case.oars):
            oar_scores[index, beam] = float(np.mean(beam_map[oar]))
    return target_scores, oar_scores


def _best_beam_to_add(
    active_beams: tuple[int, ...],
    target_scores: np.ndarray,
    oar_scores: np.ndarray,
    oar_priorities: tuple[float, ...],
) -> int | None:
    inactive = [beam for beam in range(len(target_scores)) if beam not in active_beams]
    if not inactive:
        return None
    weighted_harm = np.average(oar_scores, axis=0, weights=np.asarray(oar_priorities))
    utility = target_scores / (0.05 + weighted_harm)
    return max(inactive, key=lambda beam: (utility[beam], -beam))


def _worst_active_beam_for_oar(
    active_beams: tuple[int, ...],
    oar_index: int,
    target_scores: np.ndarray,
    oar_scores: np.ndarray,
) -> int | None:
    if len(active_beams) <= 3:
        return None
    harm = oar_scores[oar_index] / (0.05 + target_scores)
    return max(active_beams, key=lambda beam: (harm[beam], -beam))


def _choose_manual_action(
    plan: OptimizedPlan,
    case: SyntheticCase,
    influence: FloatArray,
    config: SimulationConfig,
    history: tuple[ManualAction, ...],
) -> tuple[ManualAction, tuple[int, ...], PlanningPriorities] | None:
    metrics = plan.clinical_metrics
    priorities = plan.priorities
    active = plan.active_beams
    target_scores, oar_scores = _beam_target_oar_scores(case, influence, config)

    coverage_gap = max(0.85 * config.prescription - metrics.target_d95, 0.0) / 0.85
    hotspot_gap = max(metrics.target_d02 - 1.25 * config.prescription, 0.0) / 1.25
    oar_gaps = [
        max(value / limit - 1.0, 0.0)
        for value, limit in zip(metrics.oar_mean, case.oar_limits, strict=True)
    ]
    worst_oar = int(np.argmax(oar_gaps))
    worst_oar_gap = oar_gaps[worst_oar]

    if worst_oar_gap >= max(coverage_gap, hotspot_gap) and worst_oar_gap > 0:
        old = priorities.oars[worst_oar]
        has_priority_edit = any(
            action.kind == "increase_oar_priority" and action.structure_index == worst_oar
            for action in history
        )
        has_beam_removal = any(
            action.kind == "remove_beam" and action.structure_index == worst_oar
            for action in history
        )
        has_beam_addition = any(action.kind == "add_beam" for action in history)
        if not has_priority_edit:
            updated = list(priorities.oars)
            updated[worst_oar] = old * config.manual_priority_factor
            new_priorities = replace(priorities, oars=tuple(updated))
            action = ManualAction(
                kind="increase_oar_priority",
                description=f"Increase OAR {worst_oar + 1} priority {old:.2f} -> {updated[worst_oar]:.2f}",
                structure_index=worst_oar,
                old_value=old,
                new_value=updated[worst_oar],
            )
            return action, active, new_priorities
        beam = _worst_active_beam_for_oar(active, worst_oar, target_scores, oar_scores)
        if not has_beam_removal and beam is not None:
            new_active = tuple(value for value in active if value != beam)
            return (
                ManualAction(
                    kind="remove_beam",
                    description=f"Remove {beam * 30}° beam to reduce OAR {worst_oar + 1} exposure",
                    beam_index=beam,
                    structure_index=worst_oar,
                ),
                new_active,
                priorities,
            )
        replacement = _best_beam_to_add(active, target_scores, oar_scores, priorities.oars)
        if has_beam_removal and not has_beam_addition and replacement is not None:
            return (
                ManualAction(
                    kind="add_beam",
                    description=f"Add {replacement * 30}° beam as an alternate OAR-sparing approach",
                    beam_index=replacement,
                    structure_index=worst_oar,
                ),
                tuple(sorted((*active, replacement))),
                priorities,
            )
        if old < 6.0:
            updated = list(priorities.oars)
            updated[worst_oar] = min(old * config.manual_priority_factor, 6.0)
            return (
                ManualAction(
                    kind="increase_oar_priority",
                    description=f"Increase OAR {worst_oar + 1} priority {old:.2f} -> {updated[worst_oar]:.2f}",
                    structure_index=worst_oar,
                    old_value=old,
                    new_value=updated[worst_oar],
                ),
                active,
                replace(priorities, oars=tuple(updated)),
            )

    if coverage_gap >= hotspot_gap and coverage_gap > 0:
        old = priorities.target
        has_target_edit = any(action.kind == "increase_target_priority" for action in history)
        has_beam_addition = any(action.kind == "add_beam" for action in history)
        if not has_target_edit:
            new = old * config.manual_priority_factor
            return (
                ManualAction(
                    kind="increase_target_priority",
                    description=f"Increase target priority {old:.2f} -> {new:.2f}",
                    old_value=old,
                    new_value=new,
                ),
                active,
                replace(priorities, target=new),
            )
        beam = _best_beam_to_add(active, target_scores, oar_scores, priorities.oars)
        if not has_beam_addition and beam is not None:
            return (
                ManualAction(
                    kind="add_beam",
                    description=f"Add {beam * 30}° beam for another target approach",
                    beam_index=beam,
                ),
                tuple(sorted((*active, beam))),
                priorities,
            )
        if old < 6.0:
            new = old * config.manual_priority_factor
            return (
                ManualAction(
                    kind="increase_target_priority",
                    description=f"Increase target priority {old:.2f} -> {new:.2f}",
                    old_value=old,
                    new_value=new,
                ),
                active,
                replace(priorities, target=new),
            )

    if hotspot_gap > 0:
        old = priorities.hotspot
        has_hotspot_edit = any(action.kind == "increase_hotspot_priority" for action in history)
        has_beam_addition = any(action.kind == "add_beam" for action in history)
        if not has_hotspot_edit:
            new = old * config.manual_priority_factor
            return (
                ManualAction(
                    kind="increase_hotspot_priority",
                    description=f"Increase target hot-spot priority {old:.2f} -> {new:.2f}",
                    old_value=old,
                    new_value=new,
                ),
                active,
                replace(priorities, hotspot=new),
            )
        beam = _best_beam_to_add(active, target_scores, oar_scores, priorities.oars)
        if not has_beam_addition and beam is not None:
            return (
                ManualAction(
                    kind="add_beam",
                    description=f"Add {beam * 30}° beam to spread target dose",
                    beam_index=beam,
                ),
                tuple(sorted((*active, beam))),
                priorities,
            )
        if old < 6.0:
            new = old * config.manual_priority_factor
            return (
                ManualAction(
                    kind="increase_hotspot_priority",
                    description=f"Increase target hot-spot priority {old:.2f} -> {new:.2f}",
                    old_value=old,
                    new_value=new,
                ),
                active,
                replace(priorities, hotspot=new),
            )
    return None


def run_manual_planner(
    case: SyntheticCase,
    influence: FloatArray,
    config: SimulationConfig | None = None,
) -> ManualTrajectory:
    """Record only human-scale changes; rerun the optimizer after each one."""

    cfg = config or SimulationConfig()
    active_beams = (0, 3, 6, 9)
    priorities = PlanningPriorities.for_case(case)
    plan = optimize_beamlets(case, influence, active_beams, priorities, cfg)
    steps: list[ManualStep] = [ManualStep(0, None, plan)]
    stopping_reason = "manual_step_limit"

    for step_index in range(1, cfg.max_manual_steps + 1):
        if is_acceptable(plan.clinical_metrics, case, cfg):
            stopping_reason = "acceptable"
            break
        history = tuple(step.action for step in steps[1:] if step.action is not None)
        decision = _choose_manual_action(plan, case, influence, cfg, history)
        if decision is None:
            stopping_reason = "no_manual_action"
            break
        action, active_beams, priorities = decision
        plan = optimize_beamlets(
            case,
            influence,
            active_beams,
            priorities,
            cfg,
            initial_intensities=plan.intensities,
        )
        steps.append(ManualStep(step_index, action, plan))
    else:
        if is_acceptable(plan.clinical_metrics, case, cfg):
            stopping_reason = "acceptable"

    return ManualTrajectory(case.case_id, tuple(steps), stopping_reason)
