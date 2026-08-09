from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np

from .dataset3d import ACTION_NAMES, ACTION_TO_INDEX, state_features_3d
from .manual_planning import ManualAction
from .objective import PlanningPriorities
from .planning3d import (
    HighLevelSearchConfig3D,
    PlanningStep3D,
    PlanningTrajectory3D,
    clinical_violation_score_3d,
    is_acceptable_3d,
)
from .torch_dose3d import TorchImplicitDoseEngine3D, optimize_fluence_3d_torch
from .volume3d import SyntheticCase3D


def legal_action_mask_3d(
    case: SyntheticCase3D,
    step: PlanningStep3D,
    config: HighLevelSearchConfig3D,
) -> np.ndarray:
    """Return the shared legal-action mask for learned policy rollouts."""

    mask = np.zeros(len(ACTION_NAMES), dtype=bool)
    active = set(step.plan.active_beams)
    for beam in range(12):
        mask[ACTION_TO_INDEX[f"add_beam_{beam}"]] = beam in case.available_beams and beam not in active
        mask[ACTION_TO_INDEX[f"remove_beam_{beam}"]] = beam in active and len(active) > 3
    priorities = step.plan.priorities
    mask[ACTION_TO_INDEX["increase_target_priority"]] = priorities.target < config.priority_ceiling
    mask[ACTION_TO_INDEX["decrease_target_priority"]] = priorities.target > config.priority_floor
    mask[ACTION_TO_INDEX["increase_hotspot_priority"]] = priorities.hotspot < config.priority_ceiling
    mask[ACTION_TO_INDEX["decrease_hotspot_priority"]] = priorities.hotspot > config.priority_floor
    for index in range(3):
        present = index < len(priorities.oars)
        mask[ACTION_TO_INDEX[f"increase_oar_{index}_priority"]] = present and priorities.oars[index] < config.priority_ceiling
        mask[ACTION_TO_INDEX[f"decrease_oar_{index}_priority"]] = present and priorities.oars[index] > config.priority_floor
    # Demonstrations define stop as acceptance of the current plan; accepting
    # a state that violates the visible rules is therefore not a legal action.
    mask[ACTION_TO_INDEX["stop"]] = is_acceptable_3d(step.plan.metrics, case, config)
    return mask


def action_settings_3d(
    action_index: int,
    step: PlanningStep3D,
    config: HighLevelSearchConfig3D,
) -> tuple[ManualAction | None, tuple[int, ...], PlanningPriorities]:
    """Translate one legal categorical action into high-level settings."""

    name = ACTION_NAMES[action_index]
    active = step.plan.active_beams
    priorities = step.plan.priorities
    factor = config.priority_factor
    if name == "stop":
        return None, active, priorities
    if name.startswith("add_beam_"):
        beam = int(name.rsplit("_", 1)[1])
        return ManualAction("add_beam", f"Add {beam * 30} degree beam", beam_index=beam), tuple(sorted((*active, beam))), priorities
    if name.startswith("remove_beam_"):
        beam = int(name.rsplit("_", 1)[1])
        return ManualAction("remove_beam", f"Remove {beam * 30} degree beam", beam_index=beam), tuple(value for value in active if value != beam), priorities
    direction = "increase" if name.startswith("increase") else "decrease"
    multiplier = factor if direction == "increase" else 1.0 / factor
    if "target_priority" in name:
        value = float(np.clip(priorities.target * multiplier, config.priority_floor, config.priority_ceiling))
        action = ManualAction(f"{direction}_target_priority", f"{direction.title()} target priority", old_value=priorities.target, new_value=value)
        return action, active, replace(priorities, target=value)
    if "hotspot_priority" in name:
        value = float(np.clip(priorities.hotspot * multiplier, config.priority_floor, config.priority_ceiling))
        action = ManualAction(f"{direction}_hotspot_priority", f"{direction.title()} hot-spot priority", old_value=priorities.hotspot, new_value=value)
        return action, active, replace(priorities, hotspot=value)
    index = int(name.split("_")[2])
    updated = list(priorities.oars)
    value = float(np.clip(updated[index] * multiplier, config.priority_floor, config.priority_ceiling))
    action = ManualAction(f"{direction}_oar_priority", f"{direction.title()} OAR {index + 1} priority", structure_index=index, old_value=updated[index], new_value=value)
    updated[index] = value
    return action, active, replace(priorities, oars=tuple(updated))


def initial_policy_step_3d(
    case: SyntheticCase3D,
    engine: TorchImplicitDoseEngine3D,
    config: HighLevelSearchConfig3D,
) -> PlanningStep3D:
    active = tuple(beam for beam in (0, 3, 6, 9) if beam in case.available_beams)
    if len(active) < 3:
        active = case.available_beams[:3]
    plan = optimize_fluence_3d_torch(
        case, engine, active, PlanningPriorities.for_case(case), config.optimizer_iterations
    )
    return PlanningStep3D(0, None, plan, clinical_violation_score_3d(plan.metrics, case, config))


def rollout_policy_3d(
    case: SyntheticCase3D,
    engine: TorchImplicitDoseEngine3D,
    logits_function: Callable[[np.ndarray], np.ndarray],
    config: HighLevelSearchConfig3D,
) -> PlanningTrajectory3D:
    """Execute a learned high-level policy with masking and reoptimization."""

    initial = initial_policy_step_3d(case, engine, config)
    steps = [initial]
    if is_acceptable_3d(initial.plan.metrics, case, config):
        return PlanningTrajectory3D(case.case_id, tuple(steps), "acceptable_initial")
    for step_index in range(1, config.max_steps + 1):
        current = steps[-1]
        features = state_features_3d(case, current, config.max_steps)
        logits = np.asarray(logits_function(features), dtype=np.float64)
        if logits.shape != (len(ACTION_NAMES),):
            raise ValueError(f"Expected {len(ACTION_NAMES)} action logits, received {logits.shape}")
        legal = legal_action_mask_3d(case, current, config)
        action_index = int(np.argmax(np.where(legal, logits, -np.inf)))
        action, beams, priorities = action_settings_3d(action_index, current, config)
        if action is None:
            return PlanningTrajectory3D(case.case_id, tuple(steps), "policy_stop")
        plan = optimize_fluence_3d_torch(
            case,
            engine,
            beams,
            priorities,
            config.optimizer_iterations,
            initial_fluence=current.plan.fluence,
        )
        next_step = PlanningStep3D(
            step_index,
            action,
            plan,
            clinical_violation_score_3d(plan.metrics, case, config),
        )
        steps.append(next_step)
        if is_acceptable_3d(plan.metrics, case, config):
            return PlanningTrajectory3D(case.case_id, tuple(steps), "acceptable")
    return PlanningTrajectory3D(case.case_id, tuple(steps), "policy_step_limit")
