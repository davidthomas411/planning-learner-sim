"""Synthetic treatment-planning trajectory experiment."""

from .config import SimulationConfig
from .dose import build_dose_influence
from .expert import ExpertTrajectory, run_greedy_expert
from .geometry import SyntheticCase, generate_case
from .manual_planning import ManualTrajectory, run_manual_planner
from .objective import PlanningPriorities, clinical_violation_score, evaluate_plan
from .oracle import OracleTrajectory, run_high_level_oracle
from .optimizer import OptimizedPlan, optimize_beamlets
from .dose3d import ImplicitDoseEngine3D
from .optimizer3d import (
    OptimizedPlan3D,
    PlanMetrics3D,
    evaluate_plan_3d,
    objective_value_3d,
    optimize_fluence_3d,
)
from .volume3d import SyntheticCase3D, generate_case_3d
from .planning3d import (
    HighLevelSearchConfig3D,
    PlanningStep3D,
    PlanningTrajectory3D,
    clinical_violation_score_3d,
    is_acceptable_3d,
    run_high_level_search_3d,
    run_reference_optimizer_3d,
)
from .policy3d import legal_action_mask_3d, rollout_policy_3d
from .splits import SplitConfig, SplitRow, build_split_rows, split_manifest_sha256

try:
    from .torch_dose3d import (
        TorchImplicitDoseEngine3D,
        TorchOptimizedPlan3D,
        optimize_fluence_3d_torch,
    )
except ImportError:
    # The default lightweight installation intentionally does not require torch.
    TorchImplicitDoseEngine3D = None  # type: ignore[assignment]
    TorchOptimizedPlan3D = None  # type: ignore[assignment]
    optimize_fluence_3d_torch = None  # type: ignore[assignment]

__all__ = [
    "ExpertTrajectory",
    "SimulationConfig",
    "SyntheticCase",
    "ManualTrajectory",
    "OptimizedPlan",
    "OracleTrajectory",
    "PlanningPriorities",
    "ImplicitDoseEngine3D",
    "OptimizedPlan3D",
    "PlanMetrics3D",
    "SyntheticCase3D",
    "HighLevelSearchConfig3D",
    "PlanningStep3D",
    "PlanningTrajectory3D",
    "clinical_violation_score_3d",
    "is_acceptable_3d",
    "run_high_level_search_3d",
    "run_reference_optimizer_3d",
    "legal_action_mask_3d",
    "rollout_policy_3d",
    "SplitConfig",
    "SplitRow",
    "TorchImplicitDoseEngine3D",
    "TorchOptimizedPlan3D",
    "build_dose_influence",
    "build_split_rows",
    "clinical_violation_score",
    "evaluate_plan",
    "generate_case",
    "generate_case_3d",
    "optimize_beamlets",
    "optimize_fluence_3d",
    "objective_value_3d",
    "optimize_fluence_3d_torch",
    "evaluate_plan_3d",
    "run_greedy_expert",
    "run_high_level_oracle",
    "run_manual_planner",
    "split_manifest_sha256",
]
