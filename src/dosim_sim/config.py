from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """All parameters needed to reproduce one environment version."""

    environment_version: str = "0.6-manual"
    grid_size: int = 64
    n_beams: int = 12
    beamlets_per_beam: int = 8
    prescription: float = 1.0
    lateral_sigma: float = 0.105
    attenuation: float = 0.28
    max_expert_steps: int = 70
    optimizer_max_steps: int = 60
    max_manual_steps: int = 8
    oracle_optimizer_steps: int = 35
    max_oracle_steps: int = 8
    oracle_beam_add_candidates: int = 3
    oracle_beam_remove_candidates: int = 2
    oracle_beam_width: int = 2
    manual_priority_factor: float = 1.75
    action_step_sizes: tuple[float, ...] = (0.05, 0.15, 0.30)
    improvement_tolerance: float = 1e-7
    target_underdose_weight: float = 20.0
    target_d95_weight: float = 40.0
    target_d02_weight: float = 40.0
    target_hotspot_weight: float = 5.0
    oar_weight: float = 7.0
    oar_mean_excess_weight: float = 40.0
    complexity_weight: float = 0.002
    smoothness_weight: float = 0.015

    @property
    def n_beamlets(self) -> int:
        return self.n_beams * self.beamlets_per_beam
