from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import SimulationConfig
from .dose import calculate_dose
from .geometry import SyntheticCase
from .objective import PlanMetrics, evaluate_plan, is_acceptable


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ExpertStep:
    step: int
    beamlet: int | None
    delta: float
    objective_before: float
    objective_after: float
    intensities: FloatArray
    dose: FloatArray
    metrics: PlanMetrics


@dataclass(frozen=True)
class ExpertTrajectory:
    case_id: str
    steps: tuple[ExpertStep, ...]
    stopping_reason: str

    @property
    def final(self) -> ExpertStep:
        return self.steps[-1]


def run_greedy_expert(
    case: SyntheticCase,
    influence: FloatArray,
    config: SimulationConfig | None = None,
) -> ExpertTrajectory:
    """Choose the single legal adjustment with the best immediate score."""

    cfg = config or SimulationConfig()
    shape = case.body.shape
    intensities = np.zeros(cfg.n_beamlets, dtype=float)
    dose = calculate_dose(influence, intensities, shape)
    metrics = evaluate_plan(case, dose, intensities, cfg)
    steps: list[ExpertStep] = [
        ExpertStep(0, None, 0.0, metrics.total, metrics.total, intensities.copy(), dose.copy(), metrics)
    ]
    stopping_reason = "step_limit"

    for step_index in range(1, cfg.max_expert_steps + 1):
        best: tuple[float, int, float, FloatArray, FloatArray, PlanMetrics] | None = None
        for beamlet in range(cfg.n_beamlets):
            for magnitude in cfg.action_step_sizes:
                for sign in (1.0, -1.0):
                    delta = sign * magnitude
                    candidate_value = intensities[beamlet] + delta
                    if candidate_value < 0:
                        continue
                    candidate_intensities = intensities.copy()
                    candidate_intensities[beamlet] = candidate_value
                    candidate_dose = dose + delta * influence[:, beamlet].reshape(shape)
                    candidate_metrics = evaluate_plan(case, candidate_dose, candidate_intensities, cfg)
                    if best is None or candidate_metrics.total < best[0]:
                        best = (
                            candidate_metrics.total,
                            beamlet,
                            delta,
                            candidate_intensities,
                            candidate_dose,
                            candidate_metrics,
                        )

        if best is None or metrics.total - best[0] <= cfg.improvement_tolerance:
            stopping_reason = "converged"
            break

        prior_objective = metrics.total
        _, beamlet, delta, intensities, dose, metrics = best
        steps.append(
            ExpertStep(
                step_index,
                beamlet,
                delta,
                prior_objective,
                metrics.total,
                intensities.copy(),
                dose.copy(),
                metrics,
            )
        )
        if is_acceptable(metrics, case, cfg):
            stopping_reason = "acceptable"
            break

    return ExpertTrajectory(case.case_id, tuple(steps), stopping_reason)

