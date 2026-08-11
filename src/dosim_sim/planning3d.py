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
    optimizer_learning_rate: float = 0.08
    priority_factor: float = 1.75
    priority_ceiling: float = 6.0
    priority_floor: float = 0.5
    d95_min: float = 0.85
    d98_min: float = 0.0
    d99_min: float = 0.0
    d50_min: float = 0.0
    d50_max: float = float("inf")
    d02_max: float = 1.25
    d1cc_max: float = float("inf")
    clinical_target_v100_min: float = 0.0
    target_hotspot_threshold: float = 1.10
    target_hotspot_weight: float = 5.0
    initial_field_count: int = 4
    normal_tissue_weight: float = 0.0
    normal_tissue_threshold: float = 0.5
    integral_dose_weight: float = 0.0
    high_dose_normal_tissue_weight: float = 0.0
    high_dose_normal_tissue_threshold: float = 0.95
    clinical_dvh_weight: float = 0.0
    target_normalization_d98: float | None = None
    target_normalization_d50: float | None = None
    clinical_target_normalization_d99: float | None = None
    target_normalization_interval: int = 0
    prostate_protocol_tier: str = "off"
    paddick_ci_95_min: float = 0.0
    covering_isodose_ratio_95_max: float = float("inf")
    r50_max: float = float("inf")
    minimum_field_count: int = 0
    # The legacy 30-degree shift is disabled. Prostate angle refinement uses
    # the separate 10-degree expert-rule pilot until its representation is frozen.
    shift_candidates: int = 0
    manual_ptv_oar_crop: bool = False
    ptv_oar_overlap_minimum: float = 0.90
    overlap_floor_is_acceptance: bool = False


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


def initial_beams_3d(case: SyntheticCase3D, field_count: int) -> tuple[int, ...]:
    """Select a stable, near-even subset from the 30-degree beam-angle grid."""

    if field_count < 3:
        raise ValueError("initial_field_count must be at least 3")
    available = tuple(sorted(case.available_beams))
    if field_count >= len(available):
        return available
    if field_count == 4:
        cardinal = tuple(beam for beam in (0, 3, 6, 9) if beam in available)
        if len(cardinal) == 4:
            return cardinal
    selected: list[int] = []
    for target in np.arange(field_count, dtype=np.float64) * 12.0 / field_count:
        candidates = [beam for beam in available if beam not in selected]
        beam = min(
            candidates,
            key=lambda value: (min(abs(value - target), 12.0 - abs(value - target)), value),
        )
        selected.append(beam)
    return tuple(sorted(selected))


def optimizer_objective_kwargs_3d(config: HighLevelSearchConfig3D) -> dict[str, float | None]:
    return {
        "learning_rate": config.optimizer_learning_rate,
        "target_hotspot_threshold": config.target_hotspot_threshold,
        "target_hotspot_weight": config.target_hotspot_weight,
        "normal_tissue_weight": config.normal_tissue_weight,
        "normal_tissue_threshold": config.normal_tissue_threshold,
        "integral_dose_weight": config.integral_dose_weight,
        "high_dose_normal_tissue_weight": config.high_dose_normal_tissue_weight,
        "high_dose_normal_tissue_threshold": config.high_dose_normal_tissue_threshold,
        "clinical_dvh_weight": config.clinical_dvh_weight,
        "target_normalization_d98": config.target_normalization_d98,
        "target_normalization_d50": config.target_normalization_d50,
        "clinical_target_normalization_d99": config.clinical_target_normalization_d99,
        "target_normalization_interval": config.target_normalization_interval,
    }


def coverage_d95_3d(metrics: PlanMetrics3D) -> float:
    """Return D95 for the active full-dose optimization target."""

    return (
        metrics.target_d95
        if metrics.optimization_target_d95 is None
        else metrics.optimization_target_d95
    )


def coverage_d98_3d(metrics: PlanMetrics3D) -> float:
    """Return D98 for the active full-dose optimization target."""

    return (
        metrics.target_d98
        if metrics.optimization_target_d98 is None
        else metrics.optimization_target_d98
    )


def is_acceptable_3d(
    metrics: PlanMetrics3D,
    case: SyntheticCase3D,
    config: HighLevelSearchConfig3D | None = None,
) -> bool:
    cfg = config or HighLevelSearchConfig3D()
    ratio_tolerance = 1e-6
    dose_tolerance = 1e-3
    protocol_ok = True
    mean_oars_ok = all(
        value <= limit for value, limit in zip(metrics.oar_mean, case.oar_limits, strict=True)
    )
    if cfg.prostate_protocol_tier != "off" and case.anatomy in {"prostate", "tcia_prostate"}:
        if cfg.prostate_protocol_tier == "per_protocol":
            protocol_ok = metrics.protocol_per_protocol is True
        elif cfg.prostate_protocol_tier == "variation_acceptable":
            protocol_ok = metrics.protocol_variation_acceptable is True
        elif cfg.prostate_protocol_tier == "oar_per_protocol":
            protocol_ok = bool(metrics.protocol_oar_per_protocol_ratios) and all(
                ratio <= 1.0 for ratio in metrics.protocol_oar_per_protocol_ratios
            )
        elif cfg.prostate_protocol_tier == "oar_variation_acceptable":
            protocol_ok = bool(metrics.protocol_oar_variation_ratios) and all(
                ratio <= 1.0 for ratio in metrics.protocol_oar_variation_ratios
            )
        else:
            raise ValueError(
                "prostate_protocol_tier must be off, per_protocol, variation_acceptable, "
                "oar_per_protocol, or oar_variation_acceptable"
            )
        # The protocol DVH tier replaces the older synthetic mean-dose OAR
        # gate. Retaining both would impose an undocumented extra constraint.
        mean_oars_ok = True
    coverage_d95 = coverage_d95_3d(metrics)
    coverage_d98 = coverage_d98_3d(metrics)
    split_target_active = metrics.relaxed_overlap_minimum is not None
    clinical_target_ok = (
        not split_target_active
        or metrics.clinical_target_d98 is None
        or metrics.clinical_target_d98 + dose_tolerance >= cfg.d98_min
    )
    relaxed_overlap_ok = (
        not cfg.overlap_floor_is_acceptance
        or not split_target_active
        or metrics.relaxed_overlap_d98 is None
        or metrics.relaxed_overlap_d98 + dose_tolerance
        >= float(metrics.relaxed_overlap_minimum)
    )
    d1cc_ok = (
        metrics.target_d1cc is not None
        and metrics.target_d1cc <= cfg.d1cc_max + dose_tolerance
        if np.isfinite(cfg.d1cc_max)
        else True
    )
    clinical_v100_ok = (
        metrics.clinical_target_v100 is not None
        and metrics.clinical_target_v100 + ratio_tolerance
        >= cfg.clinical_target_v100_min
        if cfg.clinical_target_v100_min > 0.0
        else True
    )
    return (
        coverage_d95 + dose_tolerance >= cfg.d95_min
        and coverage_d98 + dose_tolerance >= cfg.d98_min
        and metrics.target_d99 + dose_tolerance >= cfg.d99_min
        and clinical_target_ok
        and relaxed_overlap_ok
        and d1cc_ok
        and clinical_v100_ok
        and metrics.target_d50 + dose_tolerance >= cfg.d50_min
        and metrics.target_d50 <= cfg.d50_max + dose_tolerance
        and metrics.target_d02 <= cfg.d02_max + dose_tolerance
        and mean_oars_ok
        and metrics.paddick_ci_95 + ratio_tolerance >= cfg.paddick_ci_95_min
        and metrics.covering_isodose_ratio_95 <= cfg.covering_isodose_ratio_95_max + ratio_tolerance
        and metrics.r50 <= cfg.r50_max + ratio_tolerance
        and metrics.field_count >= cfg.minimum_field_count
        and protocol_ok
    )


def clinical_violation_score_3d(
    metrics: PlanMetrics3D,
    case: SyntheticCase3D,
    config: HighLevelSearchConfig3D | None = None,
) -> float:
    cfg = config or HighLevelSearchConfig3D()
    active_d95 = coverage_d95_3d(metrics)
    active_d98 = coverage_d98_3d(metrics)
    coverage = max(cfg.d95_min - active_d95, 0.0) / max(cfg.d95_min, 1e-8)
    coverage_d98 = max(cfg.d98_min - active_d98, 0.0) / max(cfg.d98_min, 1e-8)
    coverage_d99 = max(cfg.d99_min - metrics.target_d99, 0.0) / max(cfg.d99_min, 1e-8)
    clinical_target = (
        max(cfg.d98_min - metrics.clinical_target_d98, 0.0) / max(cfg.d98_min, 1e-8)
        if cfg.overlap_floor_is_acceptance
        and metrics.relaxed_overlap_minimum is not None
        and metrics.clinical_target_d98 is not None
        else 0.0
    )
    d1cc = (
        max(metrics.target_d1cc - cfg.d1cc_max, 0.0) / cfg.d1cc_max
        if np.isfinite(cfg.d1cc_max)
        and cfg.d1cc_max > 0.0
        and metrics.target_d1cc is not None
        else 0.0
    )
    clinical_v100 = (
        max(cfg.clinical_target_v100_min - metrics.clinical_target_v100, 0.0)
        / cfg.clinical_target_v100_min
        if cfg.clinical_target_v100_min > 0.0
        and metrics.clinical_target_v100 is not None
        else 0.0
    )
    relaxed_overlap = (
        max(metrics.relaxed_overlap_minimum - metrics.relaxed_overlap_d98, 0.0)
        / max(metrics.relaxed_overlap_minimum, 1e-8)
        if cfg.overlap_floor_is_acceptance
        and metrics.relaxed_overlap_minimum is not None
        and metrics.relaxed_overlap_d98 is not None
        else 0.0
    )
    coverage_d50 = max(cfg.d50_min - metrics.target_d50, 0.0) / max(cfg.d50_min, 1e-8)
    overdose_d50 = (
        max(metrics.target_d50 / cfg.d50_max - 1.0, 0.0)
        if np.isfinite(cfg.d50_max) and cfg.d50_max > 0.0
        else 0.0
    )
    hotspot = (
        max(metrics.target_d02 - cfg.d02_max, 0.0) / cfg.d02_max
        if np.isfinite(cfg.d02_max) and cfg.d02_max > 0.0
        else 0.0
    )
    oars = (
        []
        if cfg.prostate_protocol_tier != "off" and case.anatomy in {"prostate", "tcia_prostate"}
        else [
            max(value / limit - 1.0, 0.0)
            for value, limit in zip(metrics.oar_mean, case.oar_limits, strict=True)
        ]
    )
    conformity = max(cfg.paddick_ci_95_min - metrics.paddick_ci_95, 0.0) / max(
        cfg.paddick_ci_95_min, 1e-8
    )
    covering_isodose = (
        max(metrics.covering_isodose_ratio_95 / cfg.covering_isodose_ratio_95_max - 1.0, 0.0)
        if np.isfinite(cfg.covering_isodose_ratio_95_max)
        and cfg.covering_isodose_ratio_95_max > 0.0
        else 0.0
    )
    dose_spill = (
        max(metrics.r50 / cfg.r50_max - 1.0, 0.0)
        if np.isfinite(cfg.r50_max) and cfg.r50_max > 0.0
        else 0.0
    )
    field_count = max(cfg.minimum_field_count - metrics.field_count, 0) / max(
        cfg.minimum_field_count, 1
    )
    protocol = 0.0
    if cfg.prostate_protocol_tier == "per_protocol" and metrics.protocol_violation_per_protocol is not None:
        protocol = metrics.protocol_violation_per_protocol
    elif cfg.prostate_protocol_tier == "variation_acceptable" and metrics.protocol_violation_variation is not None:
        protocol = metrics.protocol_violation_variation
    elif cfg.prostate_protocol_tier == "oar_per_protocol":
        protocol = float(sum(max(ratio - 1.0, 0.0) for ratio in metrics.protocol_oar_per_protocol_ratios))
    elif cfg.prostate_protocol_tier == "oar_variation_acceptable":
        protocol = float(sum(max(ratio - 1.0, 0.0) for ratio in metrics.protocol_oar_variation_ratios))
    return float(
        coverage
        + coverage_d98
        + coverage_d99
        + clinical_target
        + relaxed_overlap
        + d1cc
        + clinical_v100
        + coverage_d50
        + overdose_d50
        + hotspot
        + sum(oars)
        + conformity
        + covering_isodose
        + dose_spill
        + field_count
        + protocol
    )


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


def beam_eye_view_avoidance_scores_3d(
    case: SyntheticCase3D,
    angles_degrees: tuple[float, ...],
    oar_weights: np.ndarray,
) -> np.ndarray:
    """Score target-OAR ray separation for each coplanar beam angle."""

    target_indices = np.argwhere(case.target)
    oar_indices = [np.argwhere(mask) for mask in case.oars]
    normalized_weights = oar_weights / max(float(oar_weights.sum()), 1e-8)
    bins = np.linspace(-1.5, 1.5, case.body.shape[0] + 1)
    scores = np.zeros(len(angles_degrees), dtype=np.float64)
    for beam, angle_degrees in enumerate(angles_degrees):
        angle = np.deg2rad(angle_degrees)

        def ray_codes(indices: np.ndarray) -> np.ndarray:
            x = case.axis[indices[:, 0]]
            y = case.axis[indices[:, 1]]
            lateral = -np.sin(angle) * x + np.cos(angle) * y
            lateral_bin = np.clip(np.digitize(lateral, bins) - 1, 0, len(bins) - 2)
            return np.unique(lateral_bin * case.body.shape[2] + indices[:, 2])

        target_rays = ray_codes(target_indices)
        overlaps = [
            np.intersect1d(target_rays, ray_codes(indices), assume_unique=True).size
            / max(target_rays.size, 1)
            for indices in oar_indices
        ]
        scores[beam] = 1.0 - float(np.dot(normalized_weights, overlaps))
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

    protocol_target_failed = (
        config.prostate_protocol_tier == "per_protocol"
        and metrics.protocol_target_per_protocol is False
    ) or (
        config.prostate_protocol_tier == "variation_acceptable"
        and metrics.protocol_target_variation_acceptable is False
    )
    coverage_failed = (
        metrics.target_d95 < config.d95_min
        or metrics.target_d98 < config.d98_min
        or metrics.target_d50 < config.d50_min
        or metrics.target_d50 > config.d50_max
        or protocol_target_failed
    )
    if coverage_failed and priorities.target < config.priority_ceiling:
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
    if config.prostate_protocol_tier in {"per_protocol", "oar_per_protocol"} and metrics.protocol_oar_per_protocol_ratios:
        oar_violation = np.maximum(np.asarray(metrics.protocol_oar_per_protocol_ratios) - 1.0, 0.0)
    elif config.prostate_protocol_tier in {"variation_acceptable", "oar_variation_acceptable"} and metrics.protocol_oar_variation_ratios:
        oar_violation = np.maximum(np.asarray(metrics.protocol_oar_variation_ratios) - 1.0, 0.0)
    else:
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

    spatial_violation = (
        metrics.paddick_ci_95 < config.paddick_ci_95_min
        or metrics.covering_isodose_ratio_95 > config.covering_isodose_ratio_95_max
        or metrics.r50 > config.r50_max
    )
    if spatial_violation and priorities.normal_tissue < config.priority_ceiling:
        new = min(priorities.normal_tissue * factor, config.priority_ceiling)
        candidates.append(
            (
                ManualAction(
                    "increase_normal_tissue_priority",
                    f"Increase normal-tissue priority {priorities.normal_tissue:.2f} -> {new:.2f}",
                    old_value=priorities.normal_tissue,
                    new_value=new,
                ),
                active,
                replace(priorities, normal_tissue=new),
            )
        )
    if coverage_failed and priorities.normal_tissue > config.priority_floor:
        new = max(priorities.normal_tissue / factor, config.priority_floor)
        candidates.append(
            (
                ManualAction(
                    "decrease_normal_tissue_priority",
                    f"Decrease normal-tissue priority {priorities.normal_tissue:.2f} -> {new:.2f}",
                    old_value=priorities.normal_tissue,
                    new_value=new,
                ),
                active,
                replace(priorities, normal_tissue=new),
            )
        )

    weights = np.maximum(oar_violation, 0.05) * np.asarray(priorities.oars)
    separation = _beam_separation_scores(case, weights) + beam_eye_view_avoidance_scores_3d(
        case, tuple(float(value) for value in range(0, 360, 30)), weights
    )
    inactive = [beam for beam in case.available_beams if beam not in active]
    for beam in sorted(inactive, key=lambda value: (-separation[value], value))[: config.add_candidates]:
        candidates.append((ManualAction("add_beam", f"Add {beam * 30} degree beam", beam_index=beam), tuple(sorted((*active, beam))), priorities))
    if len(active) > 3:
        for beam in sorted(active, key=lambda value: (separation[value], value))[: config.remove_candidates]:
            candidates.append((ManualAction("remove_beam", f"Remove {beam * 30} degree beam", beam_index=beam), tuple(value for value in active if value != beam), priorities))
    shifts = []
    for old_beam in active:
        for delta in (-1, 1):
            new_beam = (old_beam + delta) % 12
            if new_beam not in case.available_beams or new_beam in active:
                continue
            improvement = separation[new_beam] - separation[old_beam]
            if improvement > 1e-8:
                shifts.append((improvement, old_beam, new_beam))
    for _, old_beam, new_beam in sorted(
        shifts, key=lambda value: (-value[0], value[1], value[2])
    )[: config.shift_candidates]:
        shifted = tuple(sorted(new_beam if beam == old_beam else beam for beam in active))
        candidates.append(
            (
                ManualAction(
                    "shift_beam",
                    f"Shift {old_beam * 30} degree beam to {new_beam * 30} degrees for improved OAR separation",
                    beam_index=old_beam,
                    new_beam_index=new_beam,
                ),
                shifted,
                priorities,
            )
        )
    return candidates


def run_high_level_search_3d(
    case: SyntheticCase3D,
    engine: TorchImplicitDoseEngine3D,
    config: HighLevelSearchConfig3D | None = None,
) -> PlanningTrajectory3D:
    """Bounded search over manual beam-angle and named-priority changes."""

    cfg = config or HighLevelSearchConfig3D()
    active = initial_beams_3d(case, cfg.initial_field_count)
    priorities = PlanningPriorities.for_case(case)
    initial = optimize_fluence_3d_torch(
        case,
        engine,
        active,
        priorities,
        cfg.optimizer_iterations,
        **optimizer_objective_kwargs_3d(cfg),
    )
    score = clinical_violation_score_3d(initial.metrics, case, cfg)
    initial_steps = (PlanningStep3D(0, None, initial, score),)
    if is_acceptable_3d(initial.metrics, case, cfg):
        return PlanningTrajectory3D(case.case_id, initial_steps, "acceptable")

    frontier = [(initial, score, initial_steps)]
    best_score, best_steps = score, initial_steps
    visited = {(
        initial.active_beams,
        priorities.target,
        priorities.hotspot,
        priorities.oars,
        priorities.normal_tissue,
    )}
    stopping_reason = "search_step_limit"
    for step_index in range(1, cfg.max_steps + 1):
        expanded = []
        for current, _, steps in frontier:
            for action, beams, candidate_priorities in _candidate_settings(current, case, cfg):
                key = (
                    beams,
                    candidate_priorities.target,
                    candidate_priorities.hotspot,
                    candidate_priorities.oars,
                    candidate_priorities.normal_tissue,
                )
                if key in visited:
                    continue
                visited.add(key)
                candidate = optimize_fluence_3d_torch(
                    case,
                    engine,
                    beams,
                    candidate_priorities,
                    cfg.optimizer_iterations,
                    initial_fluence=current.fluence,
                    **optimizer_objective_kwargs_3d(cfg),
                )
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
    config: HighLevelSearchConfig3D | None = None,
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
    cfg = config or HighLevelSearchConfig3D()
    for _ in range(penalty_rounds):
        plan = optimize_fluence_3d_torch(
            case,
            engine,
            case.available_beams,
            priorities,
            iterations=iterations_per_round,
            learning_rate=0.05,
            initial_fluence=fluence,
            **optimizer_objective_kwargs_3d(cfg),
        )
        fluence = plan.fluence
        key = (clinical_violation_score_3d(plan.metrics, case, cfg), plan.metrics.loss)
        if key < best_key:
            best, best_key = plan, key
        updated_oars = list(priorities.oars)
        if cfg.prostate_protocol_tier in {"per_protocol", "oar_per_protocol"} and plan.metrics.protocol_oar_per_protocol_ratios:
            oar_ratios = plan.metrics.protocol_oar_per_protocol_ratios
        elif cfg.prostate_protocol_tier in {"variation_acceptable", "oar_variation_acceptable"} and plan.metrics.protocol_oar_variation_ratios:
            oar_ratios = plan.metrics.protocol_oar_variation_ratios
        else:
            oar_ratios = tuple(
                value / limit for value, limit in zip(plan.metrics.oar_mean, case.oar_limits, strict=True)
            )
        for index, ratio in enumerate(oar_ratios):
            if ratio > 1.0:
                updated_oars[index] = min(updated_oars[index] * 2.5, 25.0)
        protocol_target_failed = (
            cfg.prostate_protocol_tier == "per_protocol"
            and plan.metrics.protocol_target_per_protocol is False
        ) or (
            cfg.prostate_protocol_tier == "variation_acceptable"
            and plan.metrics.protocol_target_variation_acceptable is False
        )
        priorities = PlanningPriorities(
            target=min(priorities.target * 2.5, 25.0)
            if (
                plan.metrics.target_d95 < cfg.d95_min
                or plan.metrics.target_d98 < cfg.d98_min
                or plan.metrics.target_d50 < cfg.d50_min
                or plan.metrics.target_d50 > cfg.d50_max
                or protocol_target_failed
            )
            else priorities.target,
            hotspot=min(priorities.hotspot * 2.5, 25.0) if plan.metrics.target_d02 > cfg.d02_max else priorities.hotspot,
            oars=tuple(updated_oars),
            normal_tissue=(
                min(priorities.normal_tissue * 2.5, 25.0)
                if plan.metrics.covering_isodose_ratio_95 > cfg.covering_isodose_ratio_95_max
                else priorities.normal_tissue
            ),
        )
    assert best is not None
    return best
