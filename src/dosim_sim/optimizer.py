from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import SimulationConfig
from .dose import calculate_dose
from .geometry import SyntheticCase
from .objective import PlanMetrics, PlanningPriorities, evaluate_plan


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OptimizedPlan:
    active_beams: tuple[int, ...]
    priorities: PlanningPriorities
    intensities: FloatArray
    dose: FloatArray
    optimizer_metrics: PlanMetrics
    clinical_metrics: PlanMetrics
    optimizer_iterations: int


def optimize_beamlets(
    case: SyntheticCase,
    influence: FloatArray,
    active_beams: tuple[int, ...],
    priorities: PlanningPriorities,
    config: SimulationConfig | None = None,
    initial_intensities: FloatArray | None = None,
) -> OptimizedPlan:
    """Automated inner loop: optimize beamlets for fixed angles and priorities.

    Beamlet changes are intentionally not exposed as manual trajectory actions.
    """

    cfg = config or SimulationConfig()
    shape = case.body.shape
    active_set = set(active_beams)
    active_columns = np.array(
        [
            beam * cfg.beamlets_per_beam + offset
            for beam in sorted(active_set)
            for offset in range(cfg.beamlets_per_beam)
        ],
        dtype=int,
    )
    if active_columns.size == 0:
        raise ValueError("At least one beam angle must be active")

    intensities = (
        np.zeros(cfg.n_beamlets, dtype=float)
        if initial_intensities is None
        else np.maximum(np.asarray(initial_intensities, dtype=float).copy(), 0.0)
    )
    inactive = np.ones(cfg.n_beamlets, dtype=bool)
    inactive[active_columns] = False
    intensities[inactive] = 0.0
    dose = calculate_dose(influence, intensities, shape)
    metrics = evaluate_plan(case, dose, intensities, cfg, priorities)
    completed_iterations = 0

    for iteration in range(1, cfg.optimizer_max_steps + 1):
        best: tuple[float, int, float, FloatArray, FloatArray, PlanMetrics] | None = None
        for beamlet in active_columns:
            for magnitude in cfg.action_step_sizes:
                for sign in (1.0, -1.0):
                    delta = sign * magnitude
                    candidate_value = intensities[beamlet] + delta
                    if candidate_value < 0:
                        continue
                    candidate_intensities = intensities.copy()
                    candidate_intensities[beamlet] = candidate_value
                    candidate_dose = dose + delta * influence[:, beamlet].reshape(shape)
                    candidate_metrics = evaluate_plan(
                        case, candidate_dose, candidate_intensities, cfg, priorities
                    )
                    if best is None or candidate_metrics.total < best[0]:
                        best = (
                            candidate_metrics.total,
                            int(beamlet),
                            delta,
                            candidate_intensities,
                            candidate_dose,
                            candidate_metrics,
                        )

        if best is None or metrics.total - best[0] <= cfg.improvement_tolerance:
            break
        _, _, _, intensities, dose, metrics = best
        completed_iterations = iteration

    clinical_metrics = evaluate_plan(case, dose, intensities, cfg)
    return OptimizedPlan(
        active_beams=tuple(sorted(active_set)),
        priorities=priorities,
        intensities=intensities.copy(),
        dose=dose.copy(),
        optimizer_metrics=metrics,
        clinical_metrics=clinical_metrics,
        optimizer_iterations=completed_iterations,
    )

