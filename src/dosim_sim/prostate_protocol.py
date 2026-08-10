"""Protocol-inspired prostate DVH goals for the 60 Gy in 20 fraction regimen.

The numerical goals are taken from the 2025 NRG prostate radiotherapy
template.  They make the synthetic planning task clinically recognizable.
They do not make the surrogate dose calculation suitable for patient care.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .volume3d import SyntheticCase3D


PRESCRIPTION_GY = 60.0
FRACTIONS = 20
DOSE_PER_FRACTION_GY = PRESCRIPTION_GY / FRACTIONS


@dataclass(frozen=True)
class DoseVolumeGoal:
    structure: str
    dose_gy: float
    per_protocol_volume_percent: float
    variation_volume_percent: float

    @property
    def relative_dose(self) -> float:
        return self.dose_gy / PRESCRIPTION_GY


# Table 10: radiation delivered at 2.72 Gy or 3.0-3.1 Gy per fraction.
PROSTATE_60GY_20FX_OAR_GOALS = (
    DoseVolumeGoal("rectum", 60.0, 3.0, 8.0),
    DoseVolumeGoal("rectum", 56.0, 15.0, 25.0),
    DoseVolumeGoal("rectum", 52.0, 30.0, 35.0),
    DoseVolumeGoal("rectum", 48.0, 35.0, 50.0),
    DoseVolumeGoal("rectum", 40.0, 50.0, 60.0),
    DoseVolumeGoal("bladder", 60.0, 5.0, 15.0),
    DoseVolumeGoal("bladder", 48.0, 25.0, 30.0),
    DoseVolumeGoal("bladder", 40.0, 50.0, 55.0),
    # The phantom stores both femoral heads as one planning-priority group.
    # This aggregate metric is a temporary proxy for the separate left and
    # right constraints used in clinical plan review.
    DoseVolumeGoal("femoral_heads", 40.0, 5.0, 10.0),
)


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
        return self.target_variation_acceptable and self.oars_variation_acceptable


def volume_at_least_percent(values: np.ndarray, relative_dose: float) -> float:
    """Return the percentage of a structure receiving at least a dose."""

    if values.size == 0:
        raise ValueError("a DVH structure must contain at least one voxel")
    return 100.0 * float(np.mean(values >= relative_dose))


def evaluate_prostate_60gy20fx(
    case: SyntheticCase3D,
    dose: np.ndarray,
) -> ProstateProtocolEvaluation:
    """Evaluate clinically recognizable DVH goals in absolute dose units."""

    names = {name: mask for name, mask in zip(case.structure_names, case.oars, strict=True)}
    missing = sorted({goal.structure for goal in PROSTATE_60GY_20FX_OAR_GOALS} - names.keys())
    if missing:
        raise ValueError(f"required prostate DVH structures are missing: {missing}")

    target_values = np.asarray(dose[case.target], dtype=np.float64)
    d98 = float(np.percentile(target_values, 2)) * PRESCRIPTION_GY
    d99 = float(np.percentile(target_values, 1)) * PRESCRIPTION_GY
    d02 = float(np.percentile(target_values, 98)) * PRESCRIPTION_GY
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
        # NRG generic photon target criteria: D98 >= 100% and D99 >= 95%.
        # D0.03 cc cannot be represented on the present dimensionless grid;
        # D02 remains a visible, conservative engineering hot-spot metric.
        target_per_protocol=d98 >= PRESCRIPTION_GY and d99 >= 0.95 * PRESCRIPTION_GY and d02 <= 1.07 * PRESCRIPTION_GY,
        target_variation_acceptable=d98 >= 0.98 * PRESCRIPTION_GY and d99 >= 0.93 * PRESCRIPTION_GY and d02 <= 1.10 * PRESCRIPTION_GY,
        oar_results=results,
    )


def protocol_summary_rows(evaluation: ProstateProtocolEvaluation) -> list[dict[str, object]]:
    """Return stable rows for CSV reports and DVH annotations."""

    rows: list[dict[str, object]] = [
        {
            "structure": "PTV",
            "metric": "D98",
            "observed": evaluation.target_d98_gy,
            "unit": "Gy",
            "per_protocol_goal": ">=60.0",
            "variation_goal": ">=58.8",
            "per_protocol": evaluation.target_d98_gy >= 60.0,
            "variation_acceptable": evaluation.target_d98_gy >= 58.8,
        },
        {
            "structure": "PTV",
            "metric": "D99",
            "observed": evaluation.target_d99_gy,
            "unit": "Gy",
            "per_protocol_goal": ">=57.0",
            "variation_goal": ">=55.8",
            "per_protocol": evaluation.target_d99_gy >= 57.0,
            "variation_acceptable": evaluation.target_d99_gy >= 55.8,
        },
        {
            "structure": "PTV",
            "metric": "D02 proxy",
            "observed": evaluation.target_d02_gy,
            "unit": "Gy",
            "per_protocol_goal": "<=64.2",
            "variation_goal": "<=66.0",
            "per_protocol": evaluation.target_d02_gy <= 64.2,
            "variation_acceptable": evaluation.target_d02_gy <= 66.0,
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
    if tier == "per_protocol":
        d98_min, d99_min, d02_max = 60.0, 57.0, 64.2
    else:
        d98_min, d99_min, d02_max = 58.8, 55.8, 66.0
    gaps = [
        max(d98_min - evaluation.target_d98_gy, 0.0) / d98_min,
        max(d99_min - evaluation.target_d99_gy, 0.0) / d99_min,
        max(evaluation.target_d02_gy - d02_max, 0.0) / d02_max,
    ]
    for item in evaluation.oar_results:
        limit = (
            item.goal.per_protocol_volume_percent
            if tier == "per_protocol"
            else item.goal.variation_volume_percent
        )
        gaps.append(max(item.observed_volume_percent - limit, 0.0) / limit)
    return float(sum(gaps))


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
        items = [item for item in evaluation.oar_results if item.goal.structure == name]
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
