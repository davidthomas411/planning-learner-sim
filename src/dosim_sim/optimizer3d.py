from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .dose3d import ImplicitDoseEngine3D
from .objective import PlanningPriorities
from .prostate_protocol import (
    PROSTATE_60GY_20FX_OAR_GOALS,
    evaluate_prostate_60gy20fx,
    prostate_evaluation_masks,
    protocol_oar_max_ratios,
    protocol_violation_score,
)
from .volume3d import SyntheticCase3D


FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class OptimizationTarget3D:
    """Manual target structure used by the fluence optimizer.

    ``coverage_mask`` receives the full prescription objective. The disjoint
    ``relaxed_overlap_mask`` is removed from the full-dose optimization
    target. A value of zero permits local undercoverage. The full PTV still
    has the trial-informed V57 Gy coverage objective. The original PTV in
    ``SyntheticCase3D.target`` remains unchanged and is used for all reported
    full-PTV and conformity metrics.
    """

    coverage_mask: BoolArray
    relaxed_overlap_mask: BoolArray
    relaxed_overlap_minimum: float
    cropped_oar_indices: tuple[int, ...] = ()

    def validate(self, case: SyntheticCase3D) -> None:
        if self.coverage_mask.shape != case.target.shape:
            raise ValueError("coverage_mask must have the same shape as the PTV")
        if self.relaxed_overlap_mask.shape != case.target.shape:
            raise ValueError("relaxed_overlap_mask must have the same shape as the PTV")
        if not np.any(self.coverage_mask):
            raise ValueError("coverage_mask must contain at least one voxel")
        if np.any(self.coverage_mask & self.relaxed_overlap_mask):
            raise ValueError("coverage and relaxed-overlap masks must be disjoint")
        if np.any((self.coverage_mask | self.relaxed_overlap_mask) & ~case.target):
            raise ValueError("optimization target masks must remain inside the original PTV")
        if not np.array_equal(
            self.coverage_mask | self.relaxed_overlap_mask,
            case.target,
        ):
            raise ValueError("optimization target masks must partition the complete original PTV")
        if not 0.0 <= self.relaxed_overlap_minimum <= 1.0:
            raise ValueError("relaxed_overlap_minimum must be in [0, 1]")
        if any(index < 0 or index >= len(case.oars) for index in self.cropped_oar_indices):
            raise ValueError("cropped_oar_indices contains an invalid structure index")


def full_optimization_target_3d(case: SyntheticCase3D) -> OptimizationTarget3D:
    """Return the unmodified full-PTV optimization target."""

    return OptimizationTarget3D(
        coverage_mask=case.target,
        relaxed_overlap_mask=np.zeros_like(case.target, dtype=bool),
        relaxed_overlap_minimum=1.0,
    )


def ptv_minus_oars_optimization_target_3d(
    case: SyntheticCase3D,
    cropped_oar_indices: tuple[int, ...],
    overlap_minimum: float,
) -> OptimizationTarget3D:
    """Create a PTV-minus-OAR target while retaining the full clinical target."""

    selected_oars = np.zeros_like(case.target, dtype=bool)
    for index in cropped_oar_indices:
        if index < 0 or index >= len(case.oars):
            raise ValueError("cropped_oar_indices contains an invalid structure index")
        selected_oars |= case.oars[index]
    clinical_target = (
        np.zeros_like(case.target, dtype=bool)
        if case.clinical_target is None
        else case.clinical_target
    )
    relaxed_overlap = case.target & selected_oars & ~clinical_target
    target = OptimizationTarget3D(
        coverage_mask=case.target & ~relaxed_overlap,
        relaxed_overlap_mask=relaxed_overlap,
        relaxed_overlap_minimum=overlap_minimum,
        cropped_oar_indices=tuple(sorted(cropped_oar_indices)),
    )
    target.validate(case)
    return target


@dataclass(frozen=True)
class PlanMetrics3D:
    loss: float
    target_d95: float
    target_d98: float
    target_d99: float
    target_d50: float
    target_d02: float
    oar_mean: tuple[float, ...]
    target_v95: float
    target_v100: float
    paddick_ci_95: float
    covering_isodose_ratio_95: float
    outside_target_ratio_95: float
    prescription_isodose_ratio: float
    outside_target_prescription_ratio: float
    r50: float
    body_mean_dose: float
    field_count: int
    protocol_per_protocol: bool | None = None
    protocol_variation_acceptable: bool | None = None
    protocol_target_per_protocol: bool | None = None
    protocol_target_variation_acceptable: bool | None = None
    protocol_violation_per_protocol: float | None = None
    protocol_violation_variation: float | None = None
    protocol_oar_per_protocol_ratios: tuple[float, ...] = ()
    protocol_oar_variation_ratios: tuple[float, ...] = ()
    target_d98_gy: float | None = None
    target_d99_gy: float | None = None
    target_d1cc: float | None = None
    clinical_target_v100: float | None = None
    optimization_target_d95: float | None = None
    optimization_target_d98: float | None = None
    clinical_target_d98: float | None = None
    relaxed_overlap_d98: float | None = None
    relaxed_overlap_minimum: float | None = None
    relaxed_overlap_fraction: float = 0.0


@dataclass(frozen=True)
class OptimizedPlan3D:
    active_beams: tuple[int, ...]
    priorities: PlanningPriorities
    fluence: FloatArray
    dose: FloatArray
    metrics: PlanMetrics3D
    iterations: int


def _loss_and_dose_gradient(
    case: SyntheticCase3D,
    dose: FloatArray,
    priorities: PlanningPriorities,
    target_hotspot_threshold: float = 1.10,
    target_hotspot_weight: float = 5.0,
    normal_tissue_weight: float = 0.0,
    normal_tissue_threshold: float = 0.5,
    integral_dose_weight: float = 0.0,
    high_dose_normal_tissue_weight: float = 0.0,
    high_dose_normal_tissue_threshold: float = 0.95,
    clinical_dvh_weight: float = 0.0,
    optimization_target: OptimizationTarget3D | None = None,
) -> tuple[float, FloatArray]:
    target_definition = optimization_target or full_optimization_target_3d(case)
    target_definition.validate(case)
    gradient = np.zeros_like(dose, dtype=np.float32)
    coverage_mask = target_definition.coverage_mask
    target_values = dose[coverage_mask]
    target_under = np.maximum(1.0 - target_values, 0.0)
    loss = priorities.target * 20.0 * float(np.mean(target_under**2))
    gradient[coverage_mask] -= priorities.target * 40.0 * target_under / target_values.size

    # A lower-tail coverage term aligns the numerical objective with D95.
    # It acts on the 10% most underdosed target voxels and remains a property
    # of the automated inner optimizer, not a manual planning action.
    tail_count = max(1, int(np.ceil(0.10 * target_values.size)))
    tail_indices = np.argpartition(target_under, -tail_count)[-tail_count:]
    tail_under = target_under[tail_indices]
    loss += priorities.target * 30.0 * float(np.mean(tail_under**2))
    target_gradient = gradient[coverage_mask]
    target_gradient[tail_indices] -= priorities.target * 60.0 * tail_under / tail_count
    gradient[coverage_mask] = target_gradient

    relaxed_mask = target_definition.relaxed_overlap_mask
    if np.any(relaxed_mask):
        relaxed_values = dose[relaxed_mask]
        relaxed_under = np.maximum(
            target_definition.relaxed_overlap_minimum - relaxed_values,
            0.0,
        )
        loss += priorities.target * 20.0 * float(np.mean(relaxed_under**2))
        relaxed_gradient = gradient[relaxed_mask]
        relaxed_gradient -= priorities.target * 40.0 * relaxed_under / relaxed_values.size
        relaxed_tail_count = max(1, int(np.ceil(0.02 * relaxed_values.size)))
        relaxed_tail_indices = np.argpartition(relaxed_under, -relaxed_tail_count)[
            -relaxed_tail_count:
        ]
        relaxed_tail = relaxed_under[relaxed_tail_indices]
        loss += priorities.target * 30.0 * float(np.mean(relaxed_tail**2))
        relaxed_gradient[relaxed_tail_indices] -= (
            priorities.target * 60.0 * relaxed_tail / relaxed_tail_count
        )
        gradient[relaxed_mask] = relaxed_gradient

    original_target_values = dose[case.target]
    if case.anatomy in {"prostate", "tcia_prostate"}:
        if case.voxel_volume_cc is None:
            raise ValueError("physical voxel volume is required for PTV D1cc")
        # Permit less than 1 cc above the limit. Penalize every other PTV
        # voxel above the limit and scale by the physical 1-cc voxel count.
        hot_count = min(
            original_target_values.size,
            max(1, int(np.ceil(1.0 / case.voxel_volume_cc))),
        )
        allowed_hot = min(original_target_values.size - 1, hot_count - 1)
        hot_indices = np.argsort(original_target_values)[
            : original_target_values.size - allowed_hot
        ]
        hot_denominator = hot_count
    else:
        hot_count = max(1, int(np.ceil(0.02 * original_target_values.size)))
        hot_indices = np.argpartition(original_target_values, -hot_count)[-hot_count:]
        hot_denominator = hot_count
    hot = np.maximum(original_target_values[hot_indices] - target_hotspot_threshold, 0.0)
    loss += priorities.hotspot * target_hotspot_weight * float(
        np.sum(hot**2) / hot_denominator
    )
    target_gradient = gradient[case.target]
    target_gradient[hot_indices] += (
        priorities.hotspot * 2.0 * target_hotspot_weight * hot / hot_denominator
    )
    gradient[case.target] = target_gradient

    for mask, limit, priority in zip(case.oars, case.oar_limits, priorities.oars, strict=True):
        values = dose[mask]
        excess = np.maximum(values - limit, 0.0)
        loss += priority * 5.0 * float(np.mean(excess**2))
        gradient[mask] += priority * 10.0 * excess / values.size

    if clinical_dvh_weight > 0.0 and case.anatomy in {"prostate", "tcia_prostate"}:
        if case.clinical_target is None or case.voxel_volume_cc is None:
            raise ValueError("clinical target and physical voxel volume are required")
        # Prostate V60 Gy >= 99%.
        prostate_values = dose[case.clinical_target]
        prostate_cold = int(np.floor(0.01 * prostate_values.size))
        prostate_ordered = np.argsort(prostate_values)
        prostate_constrained = prostate_ordered[prostate_cold:]
        prostate_under = np.maximum(1.0 - prostate_values[prostate_constrained], 0.0)
        coefficient = 20.0
        loss += clinical_dvh_weight * priorities.target * coefficient * float(
            np.mean(prostate_under**2)
        )
        prostate_gradient = gradient[case.clinical_target]
        prostate_gradient[prostate_constrained] -= (
            clinical_dvh_weight
            * priorities.target
            * 2.0
            * coefficient
            * prostate_under
            / prostate_constrained.size
        )
        gradient[case.clinical_target] = prostate_gradient

        # The standard plan uses PTV V57 Gy >=99%. A manual PTV-minus-OAR
        # structure activates the prespecified acceptable-variation floor of
        # PTV V57 Gy >=95% while prostate V60 Gy remains unchanged.
        ptv_cold_fraction = 0.05 if np.any(relaxed_mask) else 0.01
        ptv_cold = int(np.floor(ptv_cold_fraction * original_target_values.size))
        ptv_ordered = np.argsort(original_target_values)
        ptv_constrained = ptv_ordered[ptv_cold:]
        ptv_under = np.maximum(0.95 - original_target_values[ptv_constrained], 0.0)
        ptv_coefficient = 100.0 if np.any(relaxed_mask) else 20.0
        loss += clinical_dvh_weight * priorities.target * ptv_coefficient * float(
            np.mean(ptv_under**2)
        )
        ptv_gradient = gradient[case.target]
        ptv_gradient[ptv_constrained] -= (
            clinical_dvh_weight
            * priorities.target
            * 2.0
            * ptv_coefficient
            * ptv_under
            / ptv_constrained.size
        )
        gradient[case.target] = ptv_gradient

        evaluation_masks = prostate_evaluation_masks(case)
        name_to_index = {name: index for index, name in enumerate(case.structure_names)}
        for goal in PROSTATE_60GY_20FX_OAR_GOALS:
            structure_index = name_to_index[goal.planning_structure_name]
            goal_mask = evaluation_masks[goal.structure]
            values = dose[goal_mask]
            allowed = int(np.floor(goal.per_protocol_volume_percent * values.size / 100.0))
            ordered = np.argsort(values)[::-1]
            constrained = ordered[allowed:]
            excess = np.maximum(values[constrained] - goal.relative_dose, 0.0)
            coefficient = 2.0
            priority = priorities.oars[structure_index]
            loss += clinical_dvh_weight * priority * coefficient * float(np.mean(excess**2))
            structure_gradient = gradient[goal_mask]
            structure_gradient[constrained] += (
                clinical_dvh_weight * priority * 2.0 * coefficient * excess / constrained.size
            )
            gradient[goal_mask] = structure_gradient

    if (
        normal_tissue_weight > 0.0
        or integral_dose_weight > 0.0
        or high_dose_normal_tissue_weight > 0.0
    ):
        normal_mask = case.body & ~case.target
        normal_values = dose[normal_mask]
        normal_gradient = gradient[normal_mask]
        if normal_tissue_weight > 0.0:
            excess = np.maximum(normal_values - normal_tissue_threshold, 0.0)
            loss += priorities.normal_tissue * normal_tissue_weight * float(np.mean(excess**2))
            normal_gradient += (
                priorities.normal_tissue * 2.0 * normal_tissue_weight * excess / normal_values.size
            )
        if integral_dose_weight > 0.0:
            loss += priorities.normal_tissue * integral_dose_weight * float(
                np.mean(normal_values**2)
            )
            normal_gradient += (
                priorities.normal_tissue
                * 2.0
                * integral_dose_weight
                * normal_values
                / normal_values.size
            )
        if high_dose_normal_tissue_weight > 0.0:
            high_excess = np.maximum(normal_values - high_dose_normal_tissue_threshold, 0.0)
            loss += (
                priorities.normal_tissue
                * high_dose_normal_tissue_weight
                * float(np.sum(high_excess**2))
                / target_values.size
            )
            normal_gradient += (
                priorities.normal_tissue
                * 2.0
                * high_dose_normal_tissue_weight
                * high_excess
                / target_values.size
            )
        gradient[normal_mask] = normal_gradient
    return loss, gradient


def evaluate_plan_3d(
    case: SyntheticCase3D,
    dose: FloatArray,
    loss: float,
    field_count: int = 0,
    optimization_target: OptimizationTarget3D | None = None,
) -> PlanMetrics3D:
    target_definition = optimization_target or full_optimization_target_3d(case)
    target_definition.validate(case)
    target_values = dose[case.target]
    optimization_values = dose[target_definition.coverage_mask]
    clinical_target_values = (
        None
        if case.clinical_target is None or not np.any(case.clinical_target)
        else dose[case.clinical_target]
    )
    relaxed_values = (
        dose[target_definition.relaxed_overlap_mask]
        if np.any(target_definition.relaxed_overlap_mask)
        else None
    )
    target_volume = float(case.target.sum())
    covering_isodose = (dose >= 0.95) & case.body
    covered_target = float((covering_isodose & case.target).sum())
    covering_volume = float(covering_isodose.sum())
    outside_covering = float((covering_isodose & ~case.target).sum())
    prescription_isodose = (dose >= 1.0) & case.body
    prescription_covered_target = float((prescription_isodose & case.target).sum())
    prescription_volume = float(prescription_isodose.sum())
    outside_prescription = float((prescription_isodose & ~case.target).sum())
    protocol = (
        evaluate_prostate_60gy20fx(case, dose)
        if case.anatomy in {"prostate", "tcia_prostate"}
        else None
    )
    return PlanMetrics3D(
        loss=float(loss),
        target_d95=float(np.percentile(target_values, 5)),
        target_d98=float(np.percentile(target_values, 2)),
        target_d99=float(np.percentile(target_values, 1)),
        target_d50=float(np.percentile(target_values, 50)),
        target_d02=float(np.percentile(target_values, 98)),
        oar_mean=tuple(float(np.mean(dose[mask])) for mask in case.oars),
        target_v95=covered_target / target_volume,
        target_v100=prescription_covered_target / target_volume,
        paddick_ci_95=covered_target**2 / max(target_volume * covering_volume, 1.0),
        covering_isodose_ratio_95=covering_volume / target_volume,
        outside_target_ratio_95=outside_covering / target_volume,
        prescription_isodose_ratio=prescription_volume / target_volume,
        outside_target_prescription_ratio=outside_prescription / target_volume,
        r50=float(((dose >= 0.50) & case.body).sum()) / target_volume,
        body_mean_dose=float(np.mean(dose[case.body])),
        field_count=field_count,
        protocol_per_protocol=protocol.per_protocol if protocol else None,
        protocol_variation_acceptable=protocol.variation_acceptable if protocol else None,
        protocol_target_per_protocol=protocol.target_per_protocol if protocol else None,
        protocol_target_variation_acceptable=protocol.target_variation_acceptable if protocol else None,
        protocol_violation_per_protocol=protocol_violation_score(protocol, "per_protocol") if protocol else None,
        protocol_violation_variation=protocol_violation_score(protocol, "variation_acceptable") if protocol else None,
        protocol_oar_per_protocol_ratios=protocol_oar_max_ratios(case, protocol, "per_protocol") if protocol else (),
        protocol_oar_variation_ratios=protocol_oar_max_ratios(case, protocol, "variation_acceptable") if protocol else (),
        target_d98_gy=protocol.target_d98_gy if protocol else None,
        target_d99_gy=protocol.target_d99_gy if protocol else None,
        target_d1cc=(protocol.target_d1cc_gy / 60.0 if protocol else None),
        clinical_target_v100=(
            protocol.prostate_v60_percent / 100.0 if protocol else None
        ),
        optimization_target_d95=float(np.percentile(optimization_values, 5)),
        optimization_target_d98=float(np.percentile(optimization_values, 2)),
        clinical_target_d98=(
            None
            if clinical_target_values is None
            else float(np.percentile(clinical_target_values, 2))
        ),
        relaxed_overlap_d98=(
            None if relaxed_values is None else float(np.percentile(relaxed_values, 2))
        ),
        relaxed_overlap_minimum=(
            None if relaxed_values is None else target_definition.relaxed_overlap_minimum
        ),
        relaxed_overlap_fraction=float(target_definition.relaxed_overlap_mask.sum())
        / target_volume,
    )


def objective_value_3d(case: SyntheticCase3D, dose: FloatArray) -> float:
    """Evaluate every plan with one fixed, priority-independent objective."""

    neutral = PlanningPriorities.for_case(case)
    value, _ = _loss_and_dose_gradient(case, np.asarray(dose, dtype=np.float32), neutral)
    return float(value)


def optimize_fluence_3d(
    case: SyntheticCase3D,
    engine: ImplicitDoseEngine3D,
    active_beams: tuple[int, ...],
    priorities: PlanningPriorities,
    iterations: int = 60,
    learning_rate: float = 0.08,
    initial_fluence: FloatArray | None = None,
    target_hotspot_threshold: float = 1.10,
    target_hotspot_weight: float = 5.0,
    normal_tissue_weight: float = 0.0,
    normal_tissue_threshold: float = 0.5,
    integral_dose_weight: float = 0.0,
    high_dose_normal_tissue_weight: float = 0.0,
    high_dose_normal_tissue_threshold: float = 0.95,
    clinical_dvh_weight: float = 0.0,
    target_normalization_d98: float | None = None,
    target_normalization_d50: float | None = None,
    clinical_target_normalization_d99: float | None = None,
    target_normalization_interval: int = 0,
    optimization_target: OptimizationTarget3D | None = None,
) -> OptimizedPlan3D:
    """Automated inner loop for fixed beams and fixed human-set priorities."""

    normalizations = (
        target_normalization_d98,
        target_normalization_d50,
        clinical_target_normalization_d99,
    )
    if sum(value is not None for value in normalizations) > 1:
        raise ValueError("Only one target normalization may be active")
    if target_normalization_d98 is not None and target_normalization_d98 <= 0.0:
        raise ValueError("target_normalization_d98 must be positive")
    if target_normalization_d50 is not None and target_normalization_d50 <= 0.0:
        raise ValueError("target_normalization_d50 must be positive")
    if (
        clinical_target_normalization_d99 is not None
        and clinical_target_normalization_d99 <= 0.0
    ):
        raise ValueError("clinical_target_normalization_d99 must be positive")
    if clinical_target_normalization_d99 is not None and case.clinical_target is None:
        raise ValueError("clinical target is required for clinical-target normalization")
    if target_normalization_interval < 0:
        raise ValueError("target_normalization_interval must be nonnegative")
    active = np.zeros(engine.n_beams, dtype=bool)
    active[list(active_beams)] = True
    if not np.any(active):
        raise ValueError("At least one beam must be active")
    shape = (engine.n_beams, engine.fluence_size, engine.fluence_size)
    fluence = (
        np.full(shape, 0.20 / int(active.sum()), dtype=np.float32)
        if initial_fluence is None
        else np.maximum(np.asarray(initial_fluence, dtype=np.float32).copy(), 0.0)
    )
    fluence[~active] = 0.0

    def project_target_normalization() -> None:
        if all(value is None for value in normalizations):
            return
        projected_dose = engine.forward(fluence)
        if target_normalization_d98 is not None:
            current = float(np.percentile(projected_dose[case.target], 2))
            requested = target_normalization_d98
        elif target_normalization_d50 is not None:
            current = float(np.percentile(projected_dose[case.target], 50))
            requested = target_normalization_d50
        else:
            current = float(
                np.percentile(
                    projected_dose[case.clinical_target],
                    1,
                    method="lower",
                )
            )
            requested = clinical_target_normalization_d99
        fluence.__imul__(requested / max(current, 1e-8))

    periodic_normalization = (
        target_normalization_interval > 0
        and any(value is not None for value in normalizations)
    )
    if periodic_normalization:
        project_target_normalization()
    completed = 0

    while completed < iterations:
        stage_iterations = min(
            target_normalization_interval if periodic_normalization else iterations,
            iterations - completed,
        )
        first_moment = np.zeros_like(fluence)
        second_moment = np.zeros_like(fluence)
        for stage_step in range(1, stage_iterations + 1):
            dose = engine.forward(fluence)
            loss, dose_gradient = _loss_and_dose_gradient(
                case,
                dose,
                priorities,
                target_hotspot_threshold=target_hotspot_threshold,
                target_hotspot_weight=target_hotspot_weight,
                normal_tissue_weight=normal_tissue_weight,
                normal_tissue_threshold=normal_tissue_threshold,
                integral_dose_weight=integral_dose_weight,
                high_dose_normal_tissue_weight=high_dose_normal_tissue_weight,
                high_dose_normal_tissue_threshold=high_dose_normal_tissue_threshold,
                clinical_dvh_weight=clinical_dvh_weight,
                optimization_target=optimization_target,
            )
            fluence_gradient = engine.adjoint(dose_gradient)
            fluence_gradient += 0.001 * fluence
            fluence_gradient[~active] = 0.0
            first_moment = 0.9 * first_moment + 0.1 * fluence_gradient
            second_moment = 0.999 * second_moment + 0.001 * fluence_gradient**2
            corrected_first = first_moment / (1.0 - 0.9**stage_step)
            corrected_second = second_moment / (1.0 - 0.999**stage_step)
            fluence -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-7)
            np.maximum(fluence, 0.0, out=fluence)
            fluence[~active] = 0.0
        completed += stage_iterations
        if periodic_normalization:
            project_target_normalization()

    dose = engine.forward(fluence)
    if target_normalization_d98 is not None:
        current_d98 = float(np.percentile(dose[case.target], 2))
        scale = target_normalization_d98 / max(current_d98, 1e-8)
        fluence *= scale
        dose *= scale
    elif target_normalization_d50 is not None:
        current_d50 = float(np.percentile(dose[case.target], 50))
        scale = target_normalization_d50 / max(current_d50, 1e-8)
        fluence *= scale
        dose *= scale
    elif clinical_target_normalization_d99 is not None:
        current_d99 = float(
            np.percentile(dose[case.clinical_target], 1, method="lower")
        )
        scale = clinical_target_normalization_d99 / max(current_d99, 1e-8)
        fluence *= scale
        dose *= scale
    loss, _ = _loss_and_dose_gradient(
        case,
        dose,
        priorities,
        target_hotspot_threshold=target_hotspot_threshold,
        target_hotspot_weight=target_hotspot_weight,
        normal_tissue_weight=normal_tissue_weight,
        normal_tissue_threshold=normal_tissue_threshold,
        integral_dose_weight=integral_dose_weight,
        high_dose_normal_tissue_weight=high_dose_normal_tissue_weight,
        high_dose_normal_tissue_threshold=high_dose_normal_tissue_threshold,
        clinical_dvh_weight=clinical_dvh_weight,
        optimization_target=optimization_target,
    )
    return OptimizedPlan3D(
        active_beams=tuple(sorted(active_beams)),
        priorities=priorities,
        fluence=fluence,
        dose=dose,
        metrics=evaluate_plan_3d(
            case,
            dose,
            loss,
            field_count=len(active_beams),
            optimization_target=optimization_target,
        ),
        iterations=completed,
    )
