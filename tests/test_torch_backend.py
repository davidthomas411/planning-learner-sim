import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dosim_sim.dose3d import ImplicitDoseEngine3D
from dosim_sim.objective import PlanningPriorities
from dosim_sim.torch_dose3d import (
    TorchImplicitDoseEngine3D,
    optimize_fluence_3d_torch,
)
from dosim_sim.volume3d import generate_case_3d
from dosim_sim.planning3d import (
    HighLevelSearchConfig3D,
    clinical_violation_score_3d,
    is_acceptable_3d,
    run_high_level_search_3d,
)
from dosim_sim.dataset3d import ACTION_TO_INDEX
from dosim_sim.policy3d import action_settings_3d, initial_policy_step_3d, legal_action_mask_3d


def test_torch_forward_and_adjoint_match_numpy_reference() -> None:
    case = generate_case_3d(11, grid_size=24)
    angles = (0.0, 90.0)
    numpy_engine = ImplicitDoseEngine3D(case, angles, fluence_size=4)
    torch_engine = TorchImplicitDoseEngine3D(case, angles, fluence_size=4)
    rng = np.random.default_rng(12)
    fluence = rng.random((2, 4, 4), dtype=np.float32)
    voxel_probe = rng.random(case.body.shape, dtype=np.float32)

    torch_dose = torch_engine.forward(torch.from_numpy(fluence)).numpy()
    torch_adjoint = torch_engine.adjoint(torch.from_numpy(voxel_probe)).numpy()
    assert np.allclose(torch_dose, numpy_engine.forward(fluence), rtol=2e-5, atol=2e-5)
    assert np.allclose(torch_adjoint, numpy_engine.adjoint(voxel_probe), rtol=2e-5, atol=2e-4)


def test_torch_forward_batches_candidate_fluence_states() -> None:
    case = generate_case_3d(13, grid_size=24)
    engine = TorchImplicitDoseEngine3D(case, (0.0, 120.0, 240.0), fluence_size=4)
    fluence = torch.rand(3, 3, 4, 4)
    batched = engine.forward(fluence)
    individual = torch.stack([engine.forward(item) for item in fluence])
    assert torch.allclose(batched, individual)


def test_torch_optimizer_keeps_inactive_beams_zero() -> None:
    case = generate_case_3d(14, grid_size=24)
    engine = TorchImplicitDoseEngine3D(case, (0.0, 90.0, 180.0, 270.0), fluence_size=4)
    plan = optimize_fluence_3d_torch(
        case,
        engine,
        (0, 2),
        PlanningPriorities.for_case(case),
        iterations=3,
    )
    assert torch.any(plan.fluence[[0, 2]] > 0)
    assert torch.all(plan.fluence[[1, 3]] == 0)


def test_3d_search_records_only_high_level_actions() -> None:
    case = generate_case_3d(81, grid_size=24, difficulty="hard")
    angles = tuple(float(value) for value in range(0, 360, 30))
    engine = TorchImplicitDoseEngine3D(case, angles, fluence_size=4)
    cfg = HighLevelSearchConfig3D(max_steps=1, beam_width=1, optimizer_iterations=2)
    trajectory = run_high_level_search_3d(case, engine, cfg)
    allowed = {
        "add_beam",
        "remove_beam",
        "increase_target_priority",
        "increase_hotspot_priority",
        "increase_oar_priority",
        "decrease_target_priority",
        "decrease_hotspot_priority",
        "decrease_oar_priority",
    }
    assert trajectory.final.violation_score <= trajectory.steps[0].violation_score
    assert all(step.action is None or step.action.kind in allowed for step in trajectory.steps)
    assert all("beamlet" not in step.action.kind for step in trajectory.steps if step.action)
    assert clinical_violation_score_3d(trajectory.final.plan.metrics, case, cfg) >= 0


def test_policy_mask_and_action_translation_enforce_manual_bounds() -> None:
    case = generate_case_3d(91, grid_size=24, difficulty="easy")
    angles = tuple(float(value) for value in range(0, 360, 30))
    engine = TorchImplicitDoseEngine3D(case, angles, fluence_size=4)
    cfg = HighLevelSearchConfig3D(max_steps=2, optimizer_iterations=2)
    step = initial_policy_step_3d(case, engine, cfg)
    mask = legal_action_mask_3d(case, step, cfg)
    assert mask[ACTION_TO_INDEX["stop"]] == is_acceptable_3d(step.plan.metrics, case, cfg)
    assert not mask[ACTION_TO_INDEX["add_beam_0"]]
    assert mask[ACTION_TO_INDEX["remove_beam_0"]]
    assert not mask[ACTION_TO_INDEX["increase_oar_1_priority"]]
    index = ACTION_TO_INDEX["increase_target_priority"]
    action, beams, priorities = action_settings_3d(index, step, cfg)
    assert action is not None and action.kind == "increase_target_priority"
    assert beams == step.plan.active_beams
    assert priorities.target > step.plan.priorities.target
