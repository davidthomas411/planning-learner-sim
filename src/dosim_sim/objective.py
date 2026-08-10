from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import SimulationConfig
from .geometry import SyntheticCase


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PlanningPriorities:
    """High-level settings a planner changes before rerunning optimization."""

    target: float = 1.0
    hotspot: float = 1.0
    oars: tuple[float, ...] = (1.0, 1.0)
    normal_tissue: float = 1.0

    @classmethod
    def for_case(cls, case: SyntheticCase) -> "PlanningPriorities":
        return cls(oars=tuple(1.0 for _ in case.oars))


@dataclass(frozen=True)
class PlanMetrics:
    total: float
    target_underdose: float
    target_d95_shortfall: float
    target_d02_excess: float
    target_hotspot: float
    oar_penalty: float
    oar_mean_excess: float
    complexity: float
    smoothness: float
    target_d95: float
    target_d02: float
    oar_mean: tuple[float, ...]
    oar_near_max: tuple[float, ...]


def evaluate_plan(
    case: SyntheticCase,
    dose: FloatArray,
    intensities: FloatArray,
    config: SimulationConfig | None = None,
    priorities: PlanningPriorities | None = None,
) -> PlanMetrics:
    cfg = config or SimulationConfig()
    planning_priorities = priorities or PlanningPriorities.for_case(case)
    target_dose = dose[case.target]
    target_d95 = float(np.percentile(target_dose, 5))
    target_d02 = float(np.percentile(target_dose, 98))
    target_under = float(np.mean(np.maximum(cfg.prescription - target_dose, 0.0) ** 2))
    target_d95_shortfall = float(max(0.85 * cfg.prescription - target_d95, 0.0) ** 2)
    target_d02_excess = float(max(target_d02 - 1.25 * cfg.prescription, 0.0) ** 2)
    target_hot = float(np.mean(np.maximum(target_dose - 1.07 * cfg.prescription, 0.0) ** 2))

    oar_penalty = 0.0
    oar_mean_excess = 0.0
    oar_mean: list[float] = []
    oar_near_max: list[float] = []
    weighted_oar_terms = 0.0
    for mask, limit, priority in zip(
        case.oars, case.oar_limits, planning_priorities.oars, strict=True
    ):
        values = dose[mask]
        structure_penalty = float(np.mean(np.maximum(values - limit, 0.0) ** 2))
        oar_penalty += structure_penalty
        mean_value = float(np.mean(values))
        oar_mean.append(mean_value)
        structure_mean_excess = max(mean_value - limit, 0.0) ** 2
        oar_mean_excess += structure_mean_excess
        weighted_oar_terms += priority * (
            cfg.oar_weight * structure_penalty
            + cfg.oar_mean_excess_weight * structure_mean_excess
        )
        oar_near_max.append(float(np.percentile(values, 98)))

    complexity = float(np.mean(intensities))
    reshaped = intensities.reshape(cfg.n_beams, cfg.beamlets_per_beam)
    smoothness = float(np.mean(np.diff(reshaped, axis=1) ** 2))
    total = (
        planning_priorities.target * cfg.target_underdose_weight * target_under
        + planning_priorities.target * cfg.target_d95_weight * target_d95_shortfall
        + cfg.target_d02_weight * target_d02_excess
        + planning_priorities.hotspot * cfg.target_hotspot_weight * target_hot
        + weighted_oar_terms
        + cfg.complexity_weight * complexity
        + cfg.smoothness_weight * smoothness
    )
    return PlanMetrics(
        total=total,
        target_underdose=target_under,
        target_d95_shortfall=target_d95_shortfall,
        target_d02_excess=target_d02_excess,
        target_hotspot=target_hot,
        oar_penalty=oar_penalty,
        oar_mean_excess=oar_mean_excess,
        complexity=complexity,
        smoothness=smoothness,
        target_d95=target_d95,
        target_d02=target_d02,
        oar_mean=tuple(oar_mean),
        oar_near_max=tuple(oar_near_max),
    )


def is_acceptable(metrics: PlanMetrics, case: SyntheticCase, config: SimulationConfig | None = None) -> bool:
    cfg = config or SimulationConfig()
    # Provisional feasibility thresholds for the synthetic environment. They are not
    # clinical constraints and must be calibrated/frozen before model training.
    coverage_ok = metrics.target_d95 >= 0.85 * cfg.prescription
    hotspot_ok = metrics.target_d02 <= 1.25 * cfg.prescription
    # The first environment version uses mean-dose constraints. Near-max values
    # remain visible diagnostics rather than hard pass/fail criteria.
    oars_ok = all(value <= limit for value, limit in zip(metrics.oar_mean, case.oar_limits, strict=True))
    return coverage_ok and hotspot_ok and oars_ok


def clinical_violation_score(
    metrics: PlanMetrics,
    case: SyntheticCase,
    config: SimulationConfig | None = None,
) -> float:
    """Priority-independent distance from the visible acceptability rules."""

    cfg = config or SimulationConfig()
    coverage_gap = max(0.85 * cfg.prescription - metrics.target_d95, 0.0) / (0.85 * cfg.prescription)
    hotspot_gap = max(metrics.target_d02 - 1.25 * cfg.prescription, 0.0) / (1.25 * cfg.prescription)
    oar_gaps = [
        max(value - limit, 0.0) / limit
        for value, limit in zip(metrics.oar_mean, case.oar_limits, strict=True)
    ]
    return float(coverage_gap**2 + hotspot_gap**2 + sum(gap**2 for gap in oar_gaps))
