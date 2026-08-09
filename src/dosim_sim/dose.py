import numpy as np
from numpy.typing import NDArray

from .config import SimulationConfig
from .geometry import SyntheticCase


FloatArray = NDArray[np.float64]


def build_dose_influence(
    case: SyntheticCase, config: SimulationConfig | None = None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build a simple ray-like dose kernel for every angle and beamlet.

    This is deliberately not a clinical dose engine. It creates spatially
    structured, competing dose effects suitable for testing learning signals.
    """

    cfg = config or SimulationConfig()
    offsets = np.linspace(-0.68, 0.68, cfg.beamlets_per_beam)
    angles = np.linspace(0.0, 2.0 * np.pi, cfg.n_beams, endpoint=False)
    columns: list[FloatArray] = []

    for angle in angles:
        direction = case.x_grid * np.cos(angle) + case.y_grid * np.sin(angle)
        lateral = -case.x_grid * np.sin(angle) + case.y_grid * np.cos(angle)
        depth = np.clip(direction + 1.0, 0.0, 2.0)
        attenuation = np.exp(-cfg.attenuation * depth)
        for offset in offsets:
            profile = np.exp(-0.5 * ((lateral - offset) / cfg.lateral_sigma) ** 2)
            kernel = profile * attenuation * case.body
            max_value = float(kernel.max())
            if max_value > 0:
                kernel = kernel / max_value
            columns.append(kernel.ravel())

    matrix = np.stack(columns, axis=1)
    return matrix, angles, offsets


def calculate_dose(influence: FloatArray, intensities: FloatArray, shape: tuple[int, int]) -> FloatArray:
    return (influence @ intensities).reshape(shape)

