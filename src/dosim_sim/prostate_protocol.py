"""Institutional prostate DVH objectives for 60 Gy in 20 fractions.

The objective set was approved for clinical use in December 2023 and was
supplied for this experiment. The implementation does not make the surrogate
dose calculation suitable for patient care.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .volume3d import SyntheticCase3D


PRESCRIPTION_GY = 60.0
FRACTIONS = 20
DOSE_PER_FRACTION_GY = PRESCRIPTION_GY / FRACTIONS
OBJECTIVE_SET_NAME = "Institutional prostate 60 Gy in 20 fractions"
OBJECTIVE_SET_APPROVAL = "December 2023"
PTV_V57_PER_PROTOCOL_PERCENT = 99.0
PTV_V57_ACCEPTABLE_VARIATION_PERCENT = 95.0
OAR_VOLUME_ACCEPTABLE_VARIATION_POINTS = 5.0


@dataclass(frozen=True)
class DoseVolumeGoal:
    structure: str
    dose_gy: float
    per_protocol_volume_percent: float
    variation_volume_percent: float
    planning_structure: str | None = None
    priority: int = 1
    source_info: str = "CHHiP"

    @property
    def relative_dose(self) -> float:
        return self.dose_gy / PRESCRIPTION_GY

    @property
    def planning_structure_name(self) -> str:
        return self.structure if self.planning_structure is None else self.planning_structure


# Approved institutional evaluator. The source table has no local variation
# limits. The separate study variation limits below are adapted from the
# PERYTON 60 Gy in 20 fractions protocol and are never reported as local limits.
PROSTATE_60GY_20FX_OAR_GOALS = (
    DoseVolumeGoal("rectum", 37.0, 50.0, 55.0, priority=1),
    DoseVolumeGoal("rectum", 46.0, 30.0, 35.0, priority=1),
    DoseVolumeGoal("bladder", 37.0, 50.0, 55.0, priority=1),
    DoseVolumeGoal("bladder", 46.0, 30.0, 35.0, priority=1),
    DoseVolumeGoal(
        "femur_head_l",
        43.0,
        5.0,
        5.0,
        planning_structure="femoral_heads",
        priority=3,
    ),
    DoseVolumeGoal(
        "femur_head_r",
        43.0,
        5.0,
        5.0,
        planning_structure="femoral_heads",
        priority=3,
    ),
)


def clinical_objective_set_record() -> dict[str, object]:
    """Return the supplied evaluator and the separate conformity rule."""

    objectives: list[dict[str, object]] = [
        {
            "structure": "Prostate",
            "metric": "V60Gy[%]",
            "evaluator": ">=99",
            "priority": 1,
            "source_info": "CHHiP, PROFIT",
        },
        {
            "structure": "PTV",
            "metric": "D1cc[Gy]",
            "evaluator": "<=63",
            "priority": 2,
            "source_info": "CHHiP, PROFIT",
        },
        {
            "structure": "PTV",
            "metric": "D99%[Gy]",
            "evaluator": ">=57",
            "priority": 2,
            "source_info": "CHHiP, PROFIT",
        },
    ]
    objectives.extend(
        {
            "structure": goal.structure,
            "metric": f"V{goal.dose_gy:g}Gy[%]",
            "evaluator": f"<={goal.per_protocol_volume_percent:g}",
            "priority": goal.priority,
            "source_info": goal.source_info,
        }
        for goal in PROSTATE_60GY_20FX_OAR_GOALS
    )
    return {
        "name": OBJECTIVE_SET_NAME,
        "approved_for_clinical_use": OBJECTIVE_SET_APPROVAL,
        "prescription_gy": PRESCRIPTION_GY,
        "fractions": FRACTIONS,
        "objectives": objectives,
        "conformity_review": {
            "structure": "PTV",
            "metric": "V57Gy / PTV volume",
            "evaluator": "<=1.10",
            "source_info": "Retained study review criterion",
        },
        "variation_limits": None,
        "institutional_variation_limits": None,
        "study_acceptable_variation_policy": {
            "target": "PTV V57Gy >=95% while all institutional OAR limits pass",
            "oar": (
                "Rectum or bladder volume may exceed an institutional limit by "
                "no more than 5 percentage points while standard target coverage passes"
            ),
            "simultaneous_target_and_oar_variation": "requires physician review",
            "clinical_target": "Prostate V60Gy >=99% remains required",
            "ptv_hotspot": "PTV D1cc <=63Gy remains required",
            "source_info": (
                "PERYTON 60 Gy in 20 fractions acceptable-variation rules; "
                "PACE limited PTV undercoverage at rectal overlap"
            ),
            "sources": (
                "https://doi.org/10.1186/s12885-022-09493-5",
                "https://www.icr.ac.uk/docs/default-source/clinical-trials/"
                "trial-documents/pace/pace_rtqaguidelines_v2-2-%2817-august-2020%29.pdf",
            ),
        },
    }


@dataclass(frozen=True)
class DoseVolumeResult:
    goal: DoseVolumeGoal
    observed_volume_percent: float

    @property
    def per_protocol(self) -> bool:
        return self.observed_volume_percent <= self.goal.per_protocol_volume_percent

    @property
    def variation_acceptable(self) -> bool:
        return self.observed_volume_percent <= self.goal.variation_volume_percent


@dataclass(frozen=True)
class ProstateProtocolEvaluation:
    prescription_gy: float
    fractions: int
    target_d98_gy: float
    target_d99_gy: float
    target_d02_gy: float
    target_d1cc_gy: float
    target_v57_percent: float
    prostate_v60_percent: float
    target_per_protocol: bool
    target_variation_acceptable: bool
    oar_results: tuple[DoseVolumeResult, ...]

    @property
    def oars_per_protocol(self) -> bool:
        return all(item.per_protocol for item in self.oar_results)

    @property
    def oars_variation_acceptable(self) -> bool:
        return all(item.variation_acceptable for item in self.oar_results)

    @property
    def per_protocol(self) -> bool:
        return self.target_per_protocol and self.oars_per_protocol

    @property
    def variation_acceptable(self) -> bool:
        return (
            self.target_per_protocol and self.oars_variation_acceptable
        ) or (
            self.target_variation_acceptable and self.oars_per_protocol
        )

    @property
    def acceptance_class(self) -> str:
        if self.per_protocol:
            return "per_protocol"
        if self.target_per_protocol and self.oars_variation_acceptable:
            return "acceptable_oar_variation"
        if self.target_variation_acceptable and self.oars_per_protocol:
            return "acceptable_target_coverage_variation"
        return "major_variation"


@dataclass(frozen=True)
class AnatomicalObjectiveConflict:
    goal: DoseVolumeGoal
    minimum_volume_percent: float
    minimum_volume_percent_with_standard_target: float


def anatomical_objective_conflicts(
    case: SyntheticCase3D,
) -> tuple[AnatomicalObjectiveConflict, ...]:
    """Return OAR goals that cannot coexist with the target coverage counts."""

    evaluation_masks = prostate_evaluation_masks(case)
    ptv_count = int(np.count_nonzero(case.target))
    prostate_count = (
        0 if case.clinical_target is None else int(np.count_nonzero(case.clinical_target))
    )
    # The exact target rule allows 1% of the PTV below 57 Gy. The trial-informed
    # variation allows 5%. Prostate V60 remains fixed in both paths.
    allowed_ptv_cold_standard = max(0, int(np.floor(0.01 * max(ptv_count - 1, 0))))
    allowed_ptv_cold_variation = max(0, int(np.floor(0.05 * ptv_count)))
    allowed_prostate_cold = int(np.floor(0.01 * prostate_count))
    conflicts = []
    for goal in PROSTATE_60GY_20FX_OAR_GOALS:
        oar = evaluation_masks[goal.structure]
        oar_count = int(np.count_nonzero(oar))
        ptv_overlap = int(np.count_nonzero(oar & case.target))
        prostate_overlap = (
            0
            if case.clinical_target is None
            else int(np.count_nonzero(oar & case.clinical_target))
        )
        forced_high_standard = max(
            ptv_overlap - allowed_ptv_cold_standard,
            prostate_overlap - allowed_prostate_cold,
            0,
        )
        forced_high_variation = max(
            ptv_overlap - allowed_ptv_cold_variation,
            prostate_overlap - allowed_prostate_cold,
            0,
        )
        minimum_standard = 100.0 * forced_high_standard / max(oar_count, 1)
        minimum_variation = 100.0 * forced_high_variation / max(oar_count, 1)
        target_variation_cannot_meet_exact_oar = (
            minimum_variation > goal.per_protocol_volume_percent
        )
        oar_variation_cannot_keep_standard_target = (
            minimum_standard > goal.variation_volume_percent
        )
        if target_variation_cannot_meet_exact_oar and oar_variation_cannot_keep_standard_target:
            conflicts.append(
                AnatomicalObjectiveConflict(
                    goal,
                    minimum_variation,
                    minimum_standard,
                )
            )
    return tuple(conflicts)


def volume_at_least_percent(values: np.ndarray, relative_dose: float) -> float:
    """Return the percentage of a structure receiving at least a dose."""

    if values.size == 0:
        raise ValueError("a DVH structure must contain at least one voxel")
    return 100.0 * float(np.mean(values >= relative_dose))


def dose_to_hottest_volume_gy(
    values: np.ndarray,
    voxel_volume_cc: float,
    volume_cc: float = 1.0,
) -> float:
    """Return the dose to the hottest stated absolute structure volume."""

    if values.size == 0:
        raise ValueError("a DVH structure must contain at least one voxel")
    if voxel_volume_cc <= 0.0 or volume_cc <= 0.0:
        raise ValueError("voxel and requested volumes must be positive")
    voxel_count = min(values.size, max(1, int(np.ceil(volume_cc / voxel_volume_cc))))
    ordered = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    return float(ordered[voxel_count - 1]) * PRESCRIPTION_GY


def prostate_evaluation_masks(case: SyntheticCase3D) -> dict[str, np.ndarray]:
    """Return named clinical-evaluation masks without changing priority groups."""

    masks = {
        name: mask
        for name, mask in zip(
            case.evaluation_structure_names,
            case.evaluation_oars,
            strict=True,
        )
    }
    if masks:
        return masks
    planning_masks = {
        name: mask for name, mask in zip(case.structure_names, case.oars, strict=True)
    }
    masks.update(planning_masks)
    if "femoral_heads" in planning_masks:
        lateral = case.axis[:, None, None]
        masks["femur_head_l"] = planning_masks["femoral_heads"] & (lateral < 0.0)
        masks["femur_head_r"] = planning_masks["femoral_heads"] & (lateral >= 0.0)
    return masks


def evaluate_prostate_60gy20fx(
    case: SyntheticCase3D,
    dose: np.ndarray,
) -> ProstateProtocolEvaluation:
    """Evaluate clinically recognizable DVH goals in absolute dose units."""

    names = prostate_evaluation_masks(case)
    missing = sorted({goal.structure for goal in PROSTATE_60GY_20FX_OAR_GOALS} - names.keys())
    if missing:
        raise ValueError(f"required prostate DVH structures are missing: {missing}")

    target_values = np.asarray(dose[case.target], dtype=np.float64)
    if case.clinical_target is None or not np.any(case.clinical_target):
        raise ValueError("prostate/CTV is required for the 60 Gy evaluator")
    if case.voxel_volume_cc is None:
        raise ValueError("physical voxel volume is required for PTV D1cc")
    prostate_values = np.asarray(dose[case.clinical_target], dtype=np.float64)
    d98 = float(np.percentile(target_values, 2)) * PRESCRIPTION_GY
    d99 = float(np.percentile(target_values, 1)) * PRESCRIPTION_GY
    d02 = float(np.percentile(target_values, 98)) * PRESCRIPTION_GY
    d1cc = dose_to_hottest_volume_gy(target_values, case.voxel_volume_cc)
    target_v57 = volume_at_least_percent(target_values, 0.95)
    prostate_v60 = volume_at_least_percent(prostate_values, 1.0)
    results = tuple(
        DoseVolumeResult(
            goal=goal,
            observed_volume_percent=volume_at_least_percent(dose[names[goal.structure]], goal.relative_dose),
        )
        for goal in PROSTATE_60GY_20FX_OAR_GOALS
    )
    return ProstateProtocolEvaluation(
        prescription_gy=PRESCRIPTION_GY,
        fractions=FRACTIONS,
        target_d98_gy=d98,
        target_d99_gy=d99,
        target_d02_gy=d02,
        target_d1cc_gy=d1cc,
        target_v57_percent=target_v57,
        prostate_v60_percent=prostate_v60,
        target_per_protocol=(
            prostate_v60 >= 99.0
            and target_v57 >= PTV_V57_PER_PROTOCOL_PERCENT
            and d1cc <= 63.0
        ),
        target_variation_acceptable=(
            prostate_v60 >= 99.0
            and target_v57 >= PTV_V57_ACCEPTABLE_VARIATION_PERCENT
            and d1cc <= 63.0
        ),
        oar_results=results,
    )


def protocol_summary_rows(evaluation: ProstateProtocolEvaluation) -> list[dict[str, object]]:
    """Return stable rows for CSV reports and DVH annotations."""

    rows: list[dict[str, object]] = [
        {
            "structure": "Prostate",
            "metric": "V60Gy",
            "observed": evaluation.prostate_v60_percent,
            "unit": "%",
            "per_protocol_goal": ">=99",
            "variation_goal": ">=99",
            "per_protocol": evaluation.prostate_v60_percent >= 99.0,
            "variation_acceptable": evaluation.prostate_v60_percent >= 99.0,
        },
        {
            "structure": "PTV",
            "metric": "D99",
            "observed": evaluation.target_d99_gy,
            "unit": "Gy",
            "per_protocol_goal": ">=57.0",
            "variation_goal": "use V57>=95%",
            "per_protocol": evaluation.target_d99_gy >= 57.0,
            "variation_acceptable": evaluation.target_v57_percent >= 95.0,
        },
        {
            "structure": "PTV",
            "metric": "V57Gy",
            "observed": evaluation.target_v57_percent,
            "unit": "%",
            "per_protocol_goal": ">=99",
            "variation_goal": ">=95",
            "per_protocol": evaluation.target_v57_percent >= 99.0,
            "variation_acceptable": evaluation.target_v57_percent >= 95.0,
        },
        {
            "structure": "PTV",
            "metric": "D1cc",
            "observed": evaluation.target_d1cc_gy,
            "unit": "Gy",
            "per_protocol_goal": "<=63.0",
            "variation_goal": "<=63.0",
            "per_protocol": evaluation.target_d1cc_gy <= 63.0,
            "variation_acceptable": evaluation.target_d1cc_gy <= 63.0,
        },
    ]
    rows.extend(
        {
            "structure": item.goal.structure,
            "metric": f"V{item.goal.dose_gy:g}Gy",
            "observed": item.observed_volume_percent,
            "unit": "%",
            "per_protocol_goal": f"<={item.goal.per_protocol_volume_percent:g}",
            "variation_goal": f"<={item.goal.variation_volume_percent:g}",
            "per_protocol": item.per_protocol,
            "variation_acceptable": item.variation_acceptable,
        }
        for item in evaluation.oar_results
    )
    return rows


def protocol_violation_score(
    evaluation: ProstateProtocolEvaluation,
    tier: str = "per_protocol",
) -> float:
    """Return a dimensionless distance from the represented protocol goals."""

    if tier not in {"per_protocol", "variation_acceptable"}:
        raise ValueError("tier must be per_protocol or variation_acceptable")
    exact_target_gaps = [
        max(99.0 - evaluation.prostate_v60_percent, 0.0) / 99.0,
        max(99.0 - evaluation.target_v57_percent, 0.0) / 99.0,
        max(evaluation.target_d1cc_gy - 63.0, 0.0) / 63.0,
    ]
    variation_target_gaps = [
        max(99.0 - evaluation.prostate_v60_percent, 0.0) / 99.0,
        max(95.0 - evaluation.target_v57_percent, 0.0) / 95.0,
        max(evaluation.target_d1cc_gy - 63.0, 0.0) / 63.0,
    ]
    per_protocol_oar_gaps = []
    variation_oar_gaps = []
    for item in evaluation.oar_results:
        per_protocol_oar_gaps.append(
            max(
                item.observed_volume_percent - item.goal.per_protocol_volume_percent,
                0.0,
            )
            / item.goal.per_protocol_volume_percent
        )
        variation_oar_gaps.append(
            max(
                item.observed_volume_percent - item.goal.variation_volume_percent,
                0.0,
            )
            / item.goal.variation_volume_percent
        )
    if tier == "per_protocol":
        return float(sum(exact_target_gaps) + sum(per_protocol_oar_gaps))
    target_variation_path = sum(variation_target_gaps) + sum(per_protocol_oar_gaps)
    oar_variation_path = sum(exact_target_gaps) + sum(variation_oar_gaps)
    return float(min(target_variation_path, oar_variation_path))


def protocol_oar_max_ratios(
    case: SyntheticCase3D,
    evaluation: ProstateProtocolEvaluation,
    tier: str = "per_protocol",
) -> tuple[float, ...]:
    """Return the worst represented DVH ratio for each planning OAR group."""

    if tier not in {"per_protocol", "variation_acceptable"}:
        raise ValueError("tier must be per_protocol or variation_acceptable")
    ratios: list[float] = []
    for name in case.structure_names:
        items = [
            item
            for item in evaluation.oar_results
            if item.goal.planning_structure_name == name
        ]
        if not items:
            ratios.append(0.0)
            continue
        ratios.append(
            max(
                item.observed_volume_percent
                / (
                    item.goal.per_protocol_volume_percent
                    if tier == "per_protocol"
                    else item.goal.variation_volume_percent
                )
                for item in items
            )
        )
    return tuple(ratios)
