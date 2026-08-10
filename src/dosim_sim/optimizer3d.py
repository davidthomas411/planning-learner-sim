from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .dose3d import ImplicitDoseEngine3D
from .objective import PlanningPriorities
from .volume3d import SyntheticCase3D


FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class PlanMetrics3D:
    loss: float
    target_d95: float
    target_d02: float
    oar_mean: tuple[float, ...]
    target_v95: float
    paddick_ci_95: float
    r50: float
    body_mean_dose: float
    field_count: int


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
) -> tuple[float, FloatArray]:
    gradient = np.zeros_like(dose, dtype=np.float32)
    target_values = dose[case.target]
    target_under = np.maximum(1.0 - target_values, 0.0)
    loss = priorities.target * 20.0 * float(np.mean(target_under**2))
    gradient[case.target] -= priorities.target * 40.0 * target_under / target_values.size

    # A lower-tail coverage term aligns the numerical objective with D95.
    # It acts on the 10% most underdosed target voxels and remains a property
    # of the automated inner optimizer, not a manual planning action.
    tail_count = max(1, int(np.ceil(0.10 * target_values.size)))
    tail_indices = np.argpartition(target_under, -tail_count)[-tail_count:]
    tail_under = target_under[tail_indices]
    loss += priorities.target * 30.0 * float(np.mean(tail_under**2))
    target_gradient = gradient[case.target]
    target_gradient[tail_indices] -= priorities.target * 60.0 * tail_under / tail_count
    gradient[case.target] = target_gradient

    hot = np.maximum(target_values - 1.10, 0.0)
    loss += priorities.hotspot * 5.0 * float(np.mean(hot**2))
    gradient[case.target] += priorities.hotspot * 10.0 * hot / target_values.size

    for mask, limit, priority in zip(case.oars, case.oar_limits, priorities.oars, strict=True):
        values = dose[mask]
        excess = np.maximum(values - limit, 0.0)
        loss += priority * 5.0 * float(np.mean(excess**2))
        gradient[mask] += priority * 10.0 * excess / values.size
    return loss, gradient


def evaluate_plan_3d(
    case: SyntheticCase3D,
    dose: FloatArray,
    loss: float,
    field_count: int = 0,
) -> PlanMetrics3D:
    target_values = dose[case.target]
    target_volume = float(case.target.sum())
    prescription = (dose >= 0.95) & case.body
    covered_target = float((prescription & case.target).sum())
    prescription_volume = float(prescription.sum())
    return PlanMetrics3D(
        loss=float(loss),
        target_d95=float(np.percentile(target_values, 5)),
        target_d02=float(np.percentile(target_values, 98)),
        oar_mean=tuple(float(np.mean(dose[mask])) for mask in case.oars),
        target_v95=covered_target / target_volume,
        paddick_ci_95=covered_target**2 / max(target_volume * prescription_volume, 1.0),
        r50=float(((dose >= 0.50) & case.body).sum()) / target_volume,
        body_mean_dose=float(np.mean(dose[case.body])),
        field_count=field_count,
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
) -> OptimizedPlan3D:
    """Automated inner loop for fixed beams and fixed human-set priorities."""

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
    first_moment = np.zeros_like(fluence)
    second_moment = np.zeros_like(fluence)
    completed = 0

    for step in range(1, iterations + 1):
        dose = engine.forward(fluence)
        loss, dose_gradient = _loss_and_dose_gradient(case, dose, priorities)
        fluence_gradient = engine.adjoint(dose_gradient)
        fluence_gradient += 0.001 * fluence
        fluence_gradient[~active] = 0.0
        first_moment = 0.9 * first_moment + 0.1 * fluence_gradient
        second_moment = 0.999 * second_moment + 0.001 * fluence_gradient**2
        corrected_first = first_moment / (1.0 - 0.9**step)
        corrected_second = second_moment / (1.0 - 0.999**step)
        fluence -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-7)
        np.maximum(fluence, 0.0, out=fluence)
        fluence[~active] = 0.0
        completed = step

    dose = engine.forward(fluence)
    loss, _ = _loss_and_dose_gradient(case, dose, priorities)
    return OptimizedPlan3D(
        active_beams=tuple(sorted(active_beams)),
        priorities=priorities,
        fluence=fluence,
        dose=dose,
        metrics=evaluate_plan_3d(case, dose, loss, field_count=len(active_beams)),
        iterations=completed,
    )
