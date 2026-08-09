from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from .config import SimulationConfig
from .geometry import SyntheticCase
from .manual_planning import ManualAction
from .objective import PlanningPriorities, clinical_violation_score, is_acceptable
from .optimizer import OptimizedPlan, optimize_beamlets


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OracleStep:
    step: int
    action: ManualAction | None
    plan: OptimizedPlan
    violation_score: float


@dataclass(frozen=True)
class OracleTrajectory:
    case_id: str
    steps: tuple[OracleStep, ...]
    stopping_reason: str

    @property
    def final(self) -> OracleStep:
        return self.steps[-1]


def _beam_scores(
    case: SyntheticCase,
    influence: FloatArray,
    cfg: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros(cfg.n_beams)
    oars = np.zeros((len(case.oars), cfg.n_beams))
    for beam in range(cfg.n_beams):
        columns = slice(beam * cfg.beamlets_per_beam, (beam + 1) * cfg.beamlets_per_beam)
        dose = influence[:, columns].sum(axis=1).reshape(case.body.shape)
        target[beam] = float(np.mean(dose[case.target]))
        for index, mask in enumerate(case.oars):
            oars[index, beam] = float(np.mean(dose[mask]))
    return target, oars


def _candidate_settings(
    plan: OptimizedPlan,
    case: SyntheticCase,
    influence: FloatArray,
    cfg: SimulationConfig,
) -> list[tuple[ManualAction, tuple[int, ...], PlanningPriorities]]:
    priorities = plan.priorities
    active = plan.active_beams
    metrics = plan.clinical_metrics
    factor = cfg.manual_priority_factor
    candidates: list[tuple[ManualAction, tuple[int, ...], PlanningPriorities]] = []

    if metrics.target_d95 < 0.85 * cfg.prescription and priorities.target < 6.0:
        new = priorities.target * factor
        candidates.append(
            (
                ManualAction(
                    "increase_target_priority",
                    f"Increase target priority {priorities.target:.2f} -> {new:.2f}",
                    old_value=priorities.target,
                    new_value=new,
                ),
                active,
                replace(priorities, target=new),
            )
        )
    if metrics.target_d02 > 1.25 * cfg.prescription and priorities.hotspot < 6.0:
        new = priorities.hotspot * factor
        candidates.append(
            (
                ManualAction(
                    "increase_hotspot_priority",
                    f"Increase target hot-spot priority {priorities.hotspot:.2f} -> {new:.2f}",
                    old_value=priorities.hotspot,
                    new_value=new,
                ),
                active,
                replace(priorities, hotspot=new),
            )
        )
    for index, (value, limit, old) in enumerate(
        zip(metrics.oar_mean, case.oar_limits, priorities.oars, strict=True)
    ):
        if value > limit and old < 6.0:
            updated = list(priorities.oars)
            updated[index] = old * factor
            candidates.append(
                (
                    ManualAction(
                        "increase_oar_priority",
                        f"Increase OAR {index + 1} priority {old:.2f} -> {updated[index]:.2f}",
                        structure_index=index,
                        old_value=old,
                        new_value=updated[index],
                    ),
                    active,
                    replace(priorities, oars=tuple(updated)),
                )
            )

    target_scores, oar_scores = _beam_scores(case, influence, cfg)
    weighted_harm = np.average(oar_scores, axis=0, weights=np.asarray(priorities.oars))
    add_utility = target_scores / (0.05 + weighted_harm)
    inactive = [beam for beam in range(cfg.n_beams) if beam not in active]
    add_beams = sorted(inactive, key=lambda beam: (-add_utility[beam], beam))[
        : cfg.oracle_beam_add_candidates
    ]
    for beam in add_beams:
        candidates.append(
            (
                ManualAction(
                    "add_beam",
                    f"Add {beam * 30} degree beam",
                    beam_index=beam,
                ),
                tuple(sorted((*active, beam))),
                priorities,
            )
        )

    if len(active) > 3:
        violation_weights = np.array(
            [max(value / limit - 1.0, 0.05) for value, limit in zip(metrics.oar_mean, case.oar_limits)]
        )
        weighted_oar_harm = np.average(oar_scores, axis=0, weights=violation_weights)
        removal_harm = weighted_oar_harm / (0.05 + target_scores)
        remove_beams = sorted(active, key=lambda beam: (-removal_harm[beam], beam))[
            : cfg.oracle_beam_remove_candidates
        ]
        for beam in remove_beams:
            candidates.append(
                (
                    ManualAction(
                        "remove_beam",
                        f"Remove {beam * 30} degree beam",
                        beam_index=beam,
                    ),
                    tuple(value for value in active if value != beam),
                    priorities,
                )
            )
    return candidates


def run_high_level_oracle(
    case: SyntheticCase,
    influence: FloatArray,
    config: SimulationConfig | None = None,
) -> OracleTrajectory:
    """Search high-level settings exhaustively enough to benchmark the heuristic.

    This is an action-space reachability oracle, not proof of physical
    infeasibility when it fails.
    """

    cfg = config or SimulationConfig()
    oracle_cfg = replace(cfg, optimizer_max_steps=cfg.oracle_optimizer_steps)
    active = (0, 3, 6, 9)
    priorities = PlanningPriorities.for_case(case)
    plan = optimize_beamlets(case, influence, active, priorities, oracle_cfg)
    score = clinical_violation_score(plan.clinical_metrics, case, cfg)
    initial_steps = (OracleStep(0, None, plan, score),)
    if is_acceptable(plan.clinical_metrics, case, cfg):
        return OracleTrajectory(case.case_id, initial_steps, "acceptable")

    frontier: list[tuple[OptimizedPlan, float, tuple[OracleStep, ...]]] = [
        (plan, score, initial_steps)
    ]
    best_plan, best_score, best_steps = plan, score, initial_steps
    visited: set[tuple[tuple[int, ...], float, float, tuple[float, ...]]] = {
        (plan.active_beams, priorities.target, priorities.hotspot, priorities.oars)
    }
    stopping_reason = "oracle_step_limit"

    for step_index in range(1, cfg.max_oracle_steps + 1):
        expanded: list[
            tuple[float, float, str, OptimizedPlan, tuple[OracleStep, ...]]
        ] = []
        for current_plan, _, current_steps in frontier:
            for action, candidate_active, candidate_priorities in _candidate_settings(
                current_plan, case, influence, cfg
            ):
                key = (
                    candidate_active,
                    candidate_priorities.target,
                    candidate_priorities.hotspot,
                    candidate_priorities.oars,
                )
                if key in visited:
                    continue
                visited.add(key)
                candidate_plan = optimize_beamlets(
                    case,
                    influence,
                    candidate_active,
                    candidate_priorities,
                    oracle_cfg,
                    initial_intensities=current_plan.intensities,
                )
                violation = clinical_violation_score(candidate_plan.clinical_metrics, case, cfg)
                candidate_steps = (
                    *current_steps,
                    OracleStep(step_index, action, candidate_plan, violation),
                )
                if violation < best_score:
                    best_plan, best_score, best_steps = candidate_plan, violation, candidate_steps
                if is_acceptable(candidate_plan.clinical_metrics, case, cfg):
                    return OracleTrajectory(case.case_id, candidate_steps, "acceptable")
                expanded.append(
                    (
                        violation,
                        candidate_plan.clinical_metrics.total,
                        action.description,
                        candidate_plan,
                        candidate_steps,
                    )
                )
        if not expanded:
            stopping_reason = "oracle_search_exhausted"
            break
        expanded.sort(key=lambda item: item[:3])
        frontier = [
            (candidate_plan, violation, candidate_steps)
            for violation, _, _, candidate_plan, candidate_steps in expanded[
                : cfg.oracle_beam_width
            ]
        ]

    return OracleTrajectory(case.case_id, best_steps, stopping_reason)
