import numpy as np
import pytest
from dataclasses import replace

torch = pytest.importorskip("torch")

from dosim_sim.dose3d import ImplicitDoseEngine3D
from dosim_sim.objective import PlanningPriorities
from dosim_sim.torch_dose3d import (
    TorchImplicitDoseEngine3D,
    _torch_loss,
    optimize_fluence_3d_torch,
)
from dosim_sim.volume3d import generate_case_3d, generate_prostate_case_3d
from dosim_sim.planning3d import (
    HighLevelSearchConfig3D,
    clinical_violation_score_3d,
    initial_beams_3d,
    is_acceptable_3d,
    run_high_level_search_3d,
)
from dosim_sim.dataset3d import ACTION_TO_INDEX
from dosim_sim.policy3d import action_settings_3d, initial_policy_step_3d, legal_action_mask_3d
from dosim_sim.representation3d import VOLUME_CHANNEL_NAMES, state_volume_3d


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


def test_torch_optimizer_can_normalize_to_prostate_v60() -> None:
    case = generate_prostate_case_3d(140, grid_size=24, difficulty="easy")
    engine = TorchImplicitDoseEngine3D(
        case,
        (0.0, 90.0, 180.0, 270.0),
        fluence_size=4,
    )
    plan = optimize_fluence_3d_torch(
        case,
        engine,
        (0, 1, 2, 3),
        PlanningPriorities.for_case(case),
        iterations=2,
        clinical_target_normalization_d99=1.0,
        target_normalization_interval=1,
    )
    assert plan.metrics.clinical_target_v100 is not None
    assert plan.metrics.clinical_target_v100 >= 0.99


def test_seven_field_start_is_near_even_and_uses_manual_angle_grid() -> None:
    case = generate_case_3d(15, grid_size=24)
    beams = initial_beams_3d(case, 7)
    gaps = np.diff((*beams, beams[0] + 12))
    assert len(beams) == 7
    assert set(beams).issubset(case.available_beams)
    assert gaps.min() >= 1
    assert gaps.max() <= 2


def test_normal_tissue_terms_add_to_inner_optimizer_loss() -> None:
    case = generate_case_3d(16, grid_size=24)
    engine = TorchImplicitDoseEngine3D(case, (0.0, 90.0), fluence_size=4)
    dose = torch.full(case.body.shape, 0.8)
    priorities = PlanningPriorities.for_case(case)
    base = _torch_loss(engine, dose, priorities)
    constrained = _torch_loss(
        engine,
        dose,
        priorities,
        normal_tissue_weight=50.0,
        normal_tissue_threshold=0.5,
        integral_dose_weight=2.0,
    )
    assert constrained > base


def test_high_dose_normal_tissue_term_penalizes_covering_isodose_spill() -> None:
    case = generate_case_3d(160, grid_size=24)
    engine = TorchImplicitDoseEngine3D(case, (0.0, 90.0), fluence_size=4)
    dose = torch.full(case.body.shape, 1.0)
    priorities = PlanningPriorities.for_case(case)
    base = _torch_loss(engine, dose, priorities)
    constrained = _torch_loss(
        engine,
        dose,
        priorities,
        high_dose_normal_tissue_weight=10.0,
        high_dose_normal_tissue_threshold=0.95,
    )
    assert constrained > base


def test_target_d98_normalization_sets_standard_ptv_coverage() -> None:
    case = generate_case_3d(1601, grid_size=24, difficulty="easy")
    angles = (0.0, 90.0, 180.0, 270.0)
    engine = TorchImplicitDoseEngine3D(case, angles, fluence_size=4)
    plan = optimize_fluence_3d_torch(
        case,
        engine,
        tuple(range(len(angles))),
        PlanningPriorities.for_case(case),
        iterations=2,
        target_normalization_d98=0.95,
    )

    assert plan.metrics.target_d98 == pytest.approx(0.95, abs=1e-5)


def test_target_d50_normalization_sets_prescription_median() -> None:
    case = generate_case_3d(1602, grid_size=24, difficulty="easy")
    angles = (0.0, 90.0, 180.0, 270.0)
    engine = TorchImplicitDoseEngine3D(case, angles, fluence_size=4)
    plan = optimize_fluence_3d_torch(
        case,
        engine,
        tuple(range(len(angles))),
        PlanningPriorities.for_case(case),
        iterations=2,
        target_normalization_d50=1.0,
        target_normalization_interval=1,
    )

    assert plan.metrics.target_d50 == pytest.approx(1.0, abs=1e-5)


def test_prostate_clinical_dvh_terms_add_to_inner_optimizer_loss() -> None:
    case = generate_prostate_case_3d(161, grid_size=24, difficulty="moderate")
    engine = TorchImplicitDoseEngine3D(case, (0.0, 90.0), fluence_size=4)
    dose = torch.full(case.body.shape, 0.9)
    priorities = PlanningPriorities.for_case(case)
    base = _torch_loss(engine, dose, priorities)
    constrained = _torch_loss(engine, dose, priorities, clinical_dvh_weight=1.0)
    assert constrained > base


def test_prostate_protocol_tier_is_part_of_acceptance_and_violation() -> None:
    case = generate_prostate_case_3d(162, grid_size=24, difficulty="moderate")
    engine = TorchImplicitDoseEngine3D(case, (0.0, 90.0, 180.0, 270.0), fluence_size=4)
    plan = optimize_fluence_3d_torch(
        case,
        engine,
        (0, 1, 2, 3),
        PlanningPriorities.for_case(case),
        iterations=3,
        clinical_dvh_weight=1.0,
    )
    base = HighLevelSearchConfig3D(d95_min=0.1, d02_max=10.0)
    protocol = replace(base, prostate_protocol_tier="per_protocol")
    assert plan.metrics.protocol_per_protocol is not None
    assert clinical_violation_score_3d(plan.metrics, case, protocol) >= clinical_violation_score_3d(
        plan.metrics, case, base
    )
    if plan.metrics.protocol_per_protocol is False:
        assert not is_acceptable_3d(plan.metrics, case, protocol)


def test_spatial_metrics_and_minimum_field_rule_are_reported() -> None:
    case = generate_case_3d(17, grid_size=24)
    engine = TorchImplicitDoseEngine3D(case, tuple(float(value) for value in range(0, 360, 30)), fluence_size=4)
    base_config = HighLevelSearchConfig3D(optimizer_iterations=2)
    step = initial_policy_step_3d(case, engine, base_config)
    metrics = step.plan.metrics
    assert 0.0 <= metrics.target_v95 <= 1.0
    assert 0.0 <= metrics.target_v100 <= 1.0
    assert metrics.target_d98 <= metrics.target_d50 <= metrics.target_d02
    assert 0.0 <= metrics.paddick_ci_95 <= 1.0
    assert metrics.covering_isodose_ratio_95 >= metrics.target_v95
    assert metrics.outside_target_ratio_95 >= 0.0
    assert metrics.prescription_isodose_ratio >= metrics.target_v100
    assert metrics.outside_target_prescription_ratio >= 0.0
    assert metrics.r50 >= 0.0
    assert metrics.field_count == 4
    required = replace(base_config, minimum_field_count=7)
    assert not is_acceptable_3d(metrics, case, required)
    assert clinical_violation_score_3d(metrics, case, required) > clinical_violation_score_3d(
        metrics, case, base_config
    )
    clinical = replace(
        base_config,
        d98_min=metrics.target_d98 + 0.01,
        covering_isodose_ratio_95_max=max(metrics.covering_isodose_ratio_95 - 0.01, 0.01),
    )
    assert not is_acceptable_3d(metrics, case, clinical)
    assert clinical_violation_score_3d(metrics, case, clinical) > clinical_violation_score_3d(
        metrics, case, base_config
    )
    loose_case = replace(case, oar_limits=tuple(10.0 for _ in case.oar_limits))
    within_reporting_tolerance = replace(
        base_config,
        d95_min=0.0,
        d02_max=metrics.target_d02 - 0.0005,
    )
    outside_reporting_tolerance = replace(
        within_reporting_tolerance,
        d02_max=metrics.target_d02 - 0.002,
    )
    assert is_acceptable_3d(metrics, loose_case, within_reporting_tolerance)
    assert not is_acceptable_3d(metrics, loose_case, outside_reporting_tolerance)


def test_3d_search_records_only_high_level_actions() -> None:
    case = generate_case_3d(81, grid_size=24, difficulty="hard")
    angles = tuple(float(value) for value in range(0, 360, 30))
    engine = TorchImplicitDoseEngine3D(case, angles, fluence_size=4)
    cfg = HighLevelSearchConfig3D(max_steps=1, beam_width=1, optimizer_iterations=2)
    trajectory = run_high_level_search_3d(case, engine, cfg)
    allowed = {
        "add_beam",
        "remove_beam",
        "shift_beam",
        "increase_target_priority",
        "increase_hotspot_priority",
        "increase_oar_priority",
        "decrease_target_priority",
        "decrease_hotspot_priority",
        "decrease_oar_priority",
        "increase_normal_tissue_priority",
        "decrease_normal_tissue_priority",
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


def test_policy_can_shift_one_beam_angle_without_changing_field_count() -> None:
    case = generate_case_3d(94, grid_size=24, difficulty="easy")
    angles = tuple(float(value) for value in range(0, 360, 30))
    engine = TorchImplicitDoseEngine3D(case, angles, fluence_size=4)
    cfg = HighLevelSearchConfig3D(max_steps=2, optimizer_iterations=2, shift_candidates=2)
    step = initial_policy_step_3d(case, engine, cfg)
    action_index = ACTION_TO_INDEX["shift_beam_0_to_1"]
    mask = legal_action_mask_3d(case, step, cfg)
    assert mask[action_index]
    action, beams, priorities = action_settings_3d(action_index, step, cfg)
    assert action is not None and action.kind == "shift_beam"
    assert action.beam_index == 0 and action.new_beam_index == 1
    assert len(beams) == len(step.plan.active_beams)
    assert 0 not in beams and 1 in beams
    assert priorities == step.plan.priorities


def test_policy_can_change_normal_tissue_priority() -> None:
    case = generate_case_3d(95, grid_size=24, difficulty="moderate")
    engine = TorchImplicitDoseEngine3D(
        case, tuple(float(value) for value in range(0, 360, 30)), fluence_size=4
    )
    cfg = HighLevelSearchConfig3D(max_steps=2, optimizer_iterations=2)
    step = initial_policy_step_3d(case, engine, cfg)
    index = ACTION_TO_INDEX["increase_normal_tissue_priority"]
    assert legal_action_mask_3d(case, step, cfg)[index]
    action, beams, priorities = action_settings_3d(index, step, cfg)
    assert action is not None and action.kind == "increase_normal_tissue_priority"
    assert beams == step.plan.active_beams
    assert priorities.normal_tissue > step.plan.priorities.normal_tissue


def test_3d_policy_volume_contains_current_clinical_information() -> None:
    case = generate_case_3d(92, grid_size=24, difficulty="easy")
    angles = tuple(float(value) for value in range(0, 360, 30))
    engine = TorchImplicitDoseEngine3D(case, angles, fluence_size=4)
    cfg = HighLevelSearchConfig3D(max_steps=2, optimizer_iterations=2)
    step = initial_policy_step_3d(case, engine, cfg)
    volume = state_volume_3d(case, step, cfg, output_size=16)
    assert volume.shape == (len(VOLUME_CHANNEL_NAMES), 16, 16, 16)
    assert torch.isfinite(volume).all()
    assert torch.any(volume[VOLUME_CHANNEL_NAMES.index("body")] > 0)
    assert torch.any(volume[VOLUME_CHANNEL_NAMES.index("target")] > 0)
    assert torch.all(volume[VOLUME_CHANNEL_NAMES.index("clinical_target")] == 0)
    assert torch.any(volume[VOLUME_CHANNEL_NAMES.index("oar_0")] > 0)
    assert torch.all(volume[VOLUME_CHANNEL_NAMES.index("oar_1")] == 0)
    assert torch.all(volume[VOLUME_CHANNEL_NAMES.index("oar_2")] == 0)


def test_prostate_policy_volume_contains_separate_clinical_target() -> None:
    case = generate_prostate_case_3d(192, grid_size=24, difficulty="moderate")
    angles = tuple(float(value) for value in range(0, 360, 30))
    engine = TorchImplicitDoseEngine3D(case, angles, fluence_size=4)
    cfg = HighLevelSearchConfig3D(max_steps=1, optimizer_iterations=1)
    step = initial_policy_step_3d(case, engine, cfg)
    volume = state_volume_3d(case, step, cfg, output_size=16)
    target = volume[VOLUME_CHANNEL_NAMES.index("target")]
    clinical_target = volume[VOLUME_CHANNEL_NAMES.index("clinical_target")]
    assert torch.any(clinical_target > 0)
    assert torch.sum(clinical_target) < torch.sum(target)


def test_3d_policy_volume_changes_with_current_dose_but_not_structure_masks() -> None:
    case = generate_case_3d(93, grid_size=24, difficulty="moderate")
    angles = tuple(float(value) for value in range(0, 360, 30))
    engine = TorchImplicitDoseEngine3D(case, angles, fluence_size=4)
    cfg = HighLevelSearchConfig3D(max_steps=2, optimizer_iterations=2)
    step = initial_policy_step_3d(case, engine, cfg)
    changed_plan = replace(step.plan, dose=step.plan.dose + 0.2)
    changed_step = replace(step, plan=changed_plan)
    first = state_volume_3d(case, step, cfg, output_size=16)
    changed = state_volume_3d(case, changed_step, cfg, output_size=16)
    for name in ("body", "target", "clinical_target", "oar_0", "oar_1", "oar_2"):
        index = VOLUME_CHANNEL_NAMES.index(name)
        assert torch.equal(first[index], changed[index])
    assert not torch.equal(
        first[VOLUME_CHANNEL_NAMES.index("dose")],
        changed[VOLUME_CHANNEL_NAMES.index("dose")],
    )
