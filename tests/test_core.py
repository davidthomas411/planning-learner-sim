from dataclasses import replace

import numpy as np

from dosim_sim.clinical3d import _pad_to_physical_cube, _to_hfs_planner_coordinates
from dosim_sim.prostate_protocol import (
    PRESCRIPTION_GY,
    PROSTATE_60GY_20FX_OAR_GOALS,
    anatomical_objective_conflicts,
    dose_to_hottest_volume_gy,
    evaluate_prostate_60gy20fx,
)

from dosim_sim import (
    PlanningPriorities,
    SimulationConfig,
    build_dose_influence,
    clinical_violation_score,
    generate_case,
    optimize_beamlets,
    run_greedy_expert,
    run_manual_planner,
    run_high_level_oracle,
    generate_case_3d,
    generate_prostate_case_3d,
    ImplicitDoseEngine3D,
    optimize_fluence_3d,
)
from dosim_sim.dose import calculate_dose
from dosim_sim.dataset import action_record, case_features, plan_state
from dosim_sim.dataset3d import retention_eligible_3d
from dosim_sim.delivery3d import prostate_delivery_modes_3d
from dosim_sim.objective import evaluate_plan
from dosim_sim.optimizer3d import (
    evaluate_plan_3d,
    ptv_minus_oars_optimization_target_3d,
)
from dosim_sim.planning3d import HighLevelSearchConfig3D, is_acceptable_3d


def test_case_generation_is_reproducible() -> None:
    first = generate_case(17)
    second = generate_case(17)
    assert np.array_equal(first.target, second.target)
    assert all(np.array_equal(a, b) for a, b in zip(first.oars, second.oars, strict=True))


def test_3d_retention_requires_demonstration_and_reference_acceptance() -> None:
    assert retention_eligible_3d(True, True)
    assert not retention_eligible_3d(True, False)
    assert not retention_eligible_3d(False, True)
    assert not retention_eligible_3d(False, False)


def test_prostate_delivery_modes_have_requested_field_counts() -> None:
    modes = prostate_delivery_modes_3d()
    assert [len(mode.angles_degrees) for mode in modes] == [4, 7, 9, 12, 36]
    assert [mode.arc_like for mode in modes] == [False, False, False, False, True]
    for mode in modes[1:4]:
        spacings = np.diff((*mode.angles_degrees, mode.angles_degrees[0] + 360.0))
        assert np.allclose(spacings, spacings[0])


def test_dose_is_linear_and_nonnegative() -> None:
    cfg = SimulationConfig()
    case = generate_case(21, cfg)
    influence, _, _ = build_dose_influence(case, cfg)
    x1 = np.linspace(0.0, 0.2, cfg.n_beamlets)
    x2 = np.linspace(0.1, 0.0, cfg.n_beamlets)
    d1 = calculate_dose(influence, x1, case.body.shape)
    d2 = calculate_dose(influence, x2, case.body.shape)
    combined = calculate_dose(influence, x1 + x2, case.body.shape)
    assert np.all(influence >= 0)
    assert np.allclose(combined, d1 + d2)


def test_expert_actions_match_saved_dose_and_reduce_objective() -> None:
    cfg = SimulationConfig(max_expert_steps=8)
    case = generate_case(31, cfg)
    influence, _, _ = build_dose_influence(case, cfg)
    trajectory = run_greedy_expert(case, influence, cfg)
    objectives = [step.metrics.total for step in trajectory.steps]
    assert len(trajectory.steps) > 1
    assert all(after < before for before, after in zip(objectives, objectives[1:]))
    for step in trajectory.steps:
        reconstructed = calculate_dose(influence, step.intensities, case.body.shape)
        assert np.allclose(reconstructed, step.dose)
        assert np.isclose(evaluate_plan(case, step.dose, step.intensities, cfg).total, step.metrics.total)


def test_inner_optimizer_uses_only_active_beams() -> None:
    cfg = SimulationConfig(optimizer_max_steps=5)
    case = generate_case(42, cfg)
    influence, _, _ = build_dose_influence(case, cfg)
    active = (0, 3, 6, 9)
    plan = optimize_beamlets(case, influence, active, PlanningPriorities.for_case(case), cfg)
    intensities = plan.intensities.reshape(cfg.n_beams, cfg.beamlets_per_beam)
    assert np.all(intensities[[beam for beam in range(cfg.n_beams) if beam not in active]] == 0)
    assert np.any(intensities[list(active)] > 0)


def test_manual_trajectory_contains_only_high_level_actions() -> None:
    cfg = SimulationConfig(optimizer_max_steps=8, max_manual_steps=3)
    case = generate_case(10000, cfg)
    influence, _, _ = build_dose_influence(case, cfg)
    trajectory = run_manual_planner(case, influence, cfg)
    allowed = {
        "add_beam",
        "remove_beam",
        "increase_target_priority",
        "increase_hotspot_priority",
        "increase_oar_priority",
    }
    assert trajectory.steps
    assert all(step.action is None or step.action.kind in allowed for step in trajectory.steps)
    assert all("beamlet" not in step.action.kind for step in trajectory.steps if step.action is not None)


def test_violation_score_is_zero_exactly_for_acceptable_demo_plan() -> None:
    cfg = SimulationConfig(optimizer_max_steps=15)
    case = generate_case(20260809, cfg)
    influence, _, _ = build_dose_influence(case, cfg)
    plan = optimize_beamlets(case, influence, (0, 3, 6, 9), PlanningPriorities.for_case(case), cfg)
    score = clinical_violation_score(plan.clinical_metrics, case, cfg)
    assert score >= 0
    if score == 0:
        from dosim_sim.objective import is_acceptable

        assert is_acceptable(plan.clinical_metrics, case, cfg)


def test_oracle_uses_only_high_level_actions_and_improves_its_best_result() -> None:
    cfg = SimulationConfig(optimizer_max_steps=8, oracle_optimizer_steps=8, max_oracle_steps=2)
    case = generate_case(10000, cfg)
    influence, _, _ = build_dose_influence(case, cfg)
    trajectory = run_high_level_oracle(case, influence, cfg)
    assert trajectory.final.violation_score <= trajectory.steps[0].violation_score
    assert all(
        step.action is None or step.action.kind in {
            "add_beam",
            "remove_beam",
            "increase_target_priority",
            "increase_hotspot_priority",
            "increase_oar_priority",
        }
        for step in trajectory.steps
    )


def test_dataset_records_keep_manual_action_separate_from_optimizer_state() -> None:
    cfg = SimulationConfig(optimizer_max_steps=6, max_manual_steps=1)
    case = generate_case(10000, cfg)
    influence, _, _ = build_dose_influence(case, cfg)
    trajectory = run_manual_planner(case, influence, cfg)
    assert case_features(case)["target_area_fraction"] > 0
    state = plan_state(trajectory.steps[0].plan, case, cfg)
    assert "active_beams" in state and "optimizer_iterations" in state
    if len(trajectory.steps) > 1:
        action = action_record(trajectory.steps[1].action)
        assert action["kind"] != "beamlet_adjustment"


def test_implicit_3d_operator_is_linear_and_has_matching_adjoint() -> None:
    case = generate_case_3d(7, grid_size=24)
    engine = ImplicitDoseEngine3D(case, (0.0, 90.0), fluence_size=4)
    rng = np.random.default_rng(8)
    first = rng.random((2, 4, 4), dtype=np.float32)
    second = rng.random((2, 4, 4), dtype=np.float32)
    assert np.allclose(engine.forward(first + second), engine.forward(first) + engine.forward(second), atol=2e-5)
    voxel_probe = rng.random(case.body.shape, dtype=np.float32)
    forward_inner = float(np.sum(engine.forward(first) * voxel_probe))
    adjoint_inner = float(np.sum(first * engine.adjoint(voxel_probe)))
    assert np.isclose(forward_inner, adjoint_inner, rtol=2e-5, atol=2e-3)


def test_3d_inner_optimizer_respects_active_beams() -> None:
    case = generate_case_3d(9, grid_size=24)
    engine = ImplicitDoseEngine3D(case, (0.0, 90.0, 180.0, 270.0), fluence_size=4)
    plan = optimize_fluence_3d(
        case,
        engine,
        (0, 2),
        PlanningPriorities.for_case(case),
        iterations=3,
    )
    assert np.any(plan.fluence[[0, 2]] > 0)
    assert np.all(plan.fluence[[1, 3]] == 0)


def test_3d_inner_optimizer_can_normalize_between_stages() -> None:
    case = generate_case_3d(10, grid_size=24)
    engine = ImplicitDoseEngine3D(case, (0.0, 90.0, 180.0, 270.0), fluence_size=4)
    plan = optimize_fluence_3d(
        case,
        engine,
        (0, 1, 2, 3),
        PlanningPriorities.for_case(case),
        iterations=2,
        target_normalization_d50=1.0,
        target_normalization_interval=1,
    )
    assert np.isclose(plan.metrics.target_d50, 1.0, atol=1e-5)


def test_3d_difficulty_generator_is_reproducible_and_stratified() -> None:
    easy = generate_case_3d(77, grid_size=32, difficulty="easy")
    hard = generate_case_3d(77, grid_size=32, difficulty="hard")
    hard_repeat = generate_case_3d(77, grid_size=32, difficulty="hard")
    assert len(easy.oars) == 1
    assert len(hard.oars) == 3
    assert len(hard.available_beams) == 10
    assert np.array_equal(hard.target, hard_repeat.target)
    assert hard.available_beams == hard_repeat.available_beams
    easy_overlap = sum(np.count_nonzero(easy.target & mask) for mask in easy.oars)
    hard_overlap = sum(np.count_nonzero(hard.target & mask) for mask in hard.oars)
    assert easy_overlap == 0
    assert hard_overlap > 0


def test_prostate_phantom_has_named_pelvic_structures() -> None:
    case = generate_prostate_case_3d(177, grid_size=32, difficulty="moderate")
    repeat = generate_prostate_case_3d(177, grid_size=32, difficulty="moderate")
    assert case.anatomy == "prostate"
    assert case.structure_names == ("bladder", "rectum", "femoral_heads")
    assert len(case.oars) == 3
    assert np.array_equal(case.target, repeat.target)
    assert case.clinical_target is not None
    assert repeat.clinical_target is not None
    assert np.array_equal(case.clinical_target, repeat.clinical_target)
    assert all(np.count_nonzero(mask) > 0 for mask in (case.target, *case.oars))
    assert not np.any(case.target & ~case.body)
    assert not np.any(case.clinical_target & ~case.target)
    assert not any(np.any(case.clinical_target & oar) for oar in case.oars)
    margin_shell = case.target & ~case.clinical_target
    assert np.any(margin_shell)
    assert all(not np.any((case.target & oar) & ~margin_shell) for oar in case.oars)

    coordinates = np.meshgrid(case.axis, case.axis, case.axis, indexing="ij")
    target_center = np.array([values[case.target].mean() for values in coordinates])
    bladder_center = np.array([values[case.oars[0]].mean() for values in coordinates])
    rectum_center = np.array([values[case.oars[1]].mean() for values in coordinates])
    assert bladder_center[1] > target_center[1]
    assert bladder_center[2] > target_center[2]
    assert rectum_center[1] < target_center[1]
    assert np.any(case.oars[2][: case.axis.size // 2])
    assert np.any(case.oars[2][case.axis.size // 2 :])

    body_area_by_slice = case.body.sum(axis=(0, 1))
    occupied_slices = np.flatnonzero(body_area_by_slice)
    assert occupied_slices.size > 0.65 * case.axis.size
    central_occupied = body_area_by_slice[case.axis.size // 3 : 2 * case.axis.size // 3]
    assert central_occupied.min() > 0.70 * central_occupied.max()
    assert body_area_by_slice[0] == 0
    assert body_area_by_slice[-1] == 0
    central_body = case.body[:, :, case.axis.size // 2]
    occupied = np.argwhere(central_body)
    lateral_extent = occupied[:, 0].max() - occupied[:, 0].min()
    depth_extent = occupied[:, 1].max() - occupied[:, 1].min()
    assert lateral_extent > depth_extent


def test_ptv_minus_oar_target_relaxes_only_the_ptv_margin_overlap() -> None:
    case = generate_prostate_case_3d(177, grid_size=32, difficulty="moderate")
    assert case.clinical_target is not None
    bladder = case.oars[0].copy()
    margin_voxel = tuple(np.argwhere(case.target & ~case.clinical_target)[0])
    bladder[margin_voxel] = True
    case = replace(case, oars=(bladder, *case.oars[1:]))

    target = ptv_minus_oars_optimization_target_3d(case, (0,), 0.90)

    assert np.array_equal(
        target.relaxed_overlap_mask,
        case.target & case.oars[0] & ~case.clinical_target,
    )
    assert np.all(case.clinical_target <= target.coverage_mask)
    assert not np.any(target.coverage_mask & target.relaxed_overlap_mask)
    assert np.array_equal(
        target.coverage_mask | target.relaxed_overlap_mask,
        case.target,
    )


def test_split_target_gate_keeps_ctv_and_overlap_floor_explicit() -> None:
    case = generate_prostate_case_3d(177, grid_size=32, difficulty="moderate")
    assert case.clinical_target is not None
    bladder = case.oars[0].copy()
    margin_voxel = tuple(np.argwhere(case.target & ~case.clinical_target)[0])
    bladder[margin_voxel] = True
    case = replace(case, oars=(bladder, *case.oars[1:]))
    target = ptv_minus_oars_optimization_target_3d(case, (0,), 0.90)
    config = HighLevelSearchConfig3D(
        d95_min=0.95,
        d98_min=0.95,
        d50_min=0.99,
        d50_max=1.01,
        d02_max=1.05,
        covering_isodose_ratio_95_max=1.10,
        minimum_field_count=7,
        overlap_floor_is_acceptance=True,
    )

    dose = np.zeros(case.body.shape, dtype=np.float32)
    dose[target.coverage_mask] = 1.0
    dose[target.relaxed_overlap_mask] = 0.91
    metrics = evaluate_plan_3d(
        case,
        dose,
        loss=0.0,
        field_count=7,
        optimization_target=target,
    )
    assert metrics.target_d98 < metrics.optimization_target_d98
    assert metrics.clinical_target_d98 == 1.0
    assert np.isclose(metrics.relaxed_overlap_d98, 0.91)
    assert is_acceptable_3d(metrics, case, config)

    dose[target.relaxed_overlap_mask] = 0.89
    below_floor = evaluate_plan_3d(
        case,
        dose,
        loss=0.0,
        field_count=7,
        optimization_target=target,
    )
    assert not is_acceptable_3d(below_floor, case, config)


def test_hard_prostate_phantoms_include_limited_rectum_overlap() -> None:
    overlap_fractions = []
    for seed in range(220, 260):
        case = generate_prostate_case_3d(seed, grid_size=48, difficulty="hard")
        overlap = np.count_nonzero(case.target & case.oars[1])
        overlap_fractions.append(overlap / np.count_nonzero(case.target))

    assert max(overlap_fractions) >= 0.01
    assert max(overlap_fractions) <= 0.20


def test_prostate_60gy20fx_dvh_metrics_use_absolute_dose_and_named_structures() -> None:
    case = generate_prostate_case_3d(178, grid_size=32, difficulty="moderate")
    dose = np.ones(case.body.shape, dtype=np.float32)
    evaluation = evaluate_prostate_60gy20fx(case, dose)
    assert evaluation.prescription_gy == 60.0
    assert evaluation.fractions == 20
    assert evaluation.target_d98_gy == PRESCRIPTION_GY
    assert evaluation.target_d99_gy == PRESCRIPTION_GY
    assert evaluation.target_d02_gy == PRESCRIPTION_GY
    assert evaluation.target_d1cc_gy == PRESCRIPTION_GY
    assert evaluation.prostate_v60_percent == 100.0
    assert [
        (goal.structure, goal.dose_gy, goal.per_protocol_volume_percent)
        for goal in PROSTATE_60GY_20FX_OAR_GOALS
    ] == [
        ("rectum", 37.0, 50.0),
        ("rectum", 46.0, 30.0),
        ("bladder", 37.0, 50.0),
        ("bladder", 46.0, 30.0),
        ("femur_head_l", 43.0, 5.0),
        ("femur_head_r", 43.0, 5.0),
    ]
    assert len(evaluation.oar_results) == len(PROSTATE_60GY_20FX_OAR_GOALS)
    assert all(item.observed_volume_percent == 100.0 for item in evaluation.oar_results)
    assert not evaluation.oars_variation_acceptable


def test_d1cc_uses_physical_voxel_volume() -> None:
    relative_dose = np.asarray([1.20, 1.10, 1.00, 0.90], dtype=np.float32)
    assert np.isclose(
        dose_to_hottest_volume_gy(relative_dose, voxel_volume_cc=0.50),
        66.0,
    )


def test_institutional_prostate_gate_uses_v60_d99_d1cc_and_conformity() -> None:
    case = generate_prostate_case_3d(180, grid_size=32, difficulty="easy")
    config = HighLevelSearchConfig3D(
        d99_min=0.95,
        d1cc_max=1.05,
        clinical_target_v100_min=0.99,
        covering_isodose_ratio_95_max=1.10,
        minimum_field_count=7,
        prostate_protocol_tier="oar_per_protocol",
    )
    dose = np.zeros(case.body.shape, dtype=np.float32)
    dose[case.target] = 1.0
    metrics = evaluate_plan_3d(case, dose, loss=0.0, field_count=7)
    assert is_acceptable_3d(metrics, case, config)

    cold = dose.copy()
    prostate_indices = np.argwhere(case.clinical_target)
    cold_count = max(1, int(np.ceil(0.02 * len(prostate_indices))))
    for index in prostate_indices[:cold_count]:
        cold[tuple(index)] = 0.99
    cold_metrics = evaluate_plan_3d(case, cold, loss=0.0, field_count=7)
    assert cold_metrics.clinical_target_v100 < 0.99
    assert not is_acceptable_3d(cold_metrics, case, config)

    hot = dose.copy()
    hottest_voxels = int(np.ceil(1.0 / float(case.voxel_volume_cc)))
    for index in np.argwhere(case.target)[:hottest_voxels]:
        hot[tuple(index)] = 1.06
    hot_metrics = evaluate_plan_3d(case, hot, loss=0.0, field_count=7)
    assert hot_metrics.target_d1cc > 1.05
    assert not is_acceptable_3d(hot_metrics, case, config)


def test_trial_variation_accepts_one_target_or_oar_variation_but_not_both() -> None:
    case = generate_prostate_case_3d(182, grid_size=32, difficulty="easy")
    baseline = np.zeros(case.body.shape, dtype=np.float32)
    baseline[case.target] = 1.0

    target_variation = baseline.copy()
    margin_indices = np.argwhere(case.target & ~case.clinical_target)
    cold_count = int(np.ceil(0.03 * np.count_nonzero(case.target)))
    assert len(margin_indices) >= cold_count
    for index in margin_indices[:cold_count]:
        target_variation[tuple(index)] = 0.90
    target_evaluation = evaluate_prostate_60gy20fx(case, target_variation)
    assert not target_evaluation.target_per_protocol
    assert target_evaluation.target_variation_acceptable
    assert target_evaluation.oars_per_protocol
    assert target_evaluation.acceptance_class == "acceptable_target_coverage_variation"

    oar_variation = baseline.copy()
    bladder = case.evaluation_oars[0]
    available = np.argwhere(bladder & ~case.target)
    warm_count = int(np.ceil(0.32 * np.count_nonzero(bladder)))
    assert len(available) >= warm_count
    for index in available[:warm_count]:
        oar_variation[tuple(index)] = 0.80
    oar_evaluation = evaluate_prostate_60gy20fx(case, oar_variation)
    assert oar_evaluation.target_per_protocol
    assert not oar_evaluation.oars_per_protocol
    assert oar_evaluation.oars_variation_acceptable
    assert oar_evaluation.acceptance_class == "acceptable_oar_variation"

    combined = target_variation.copy()
    for index in available[:warm_count]:
        combined[tuple(index)] = 0.80
    combined_evaluation = evaluate_prostate_60gy20fx(case, combined)
    assert not combined_evaluation.variation_acceptable
    assert combined_evaluation.acceptance_class == "major_variation"


def test_anatomical_conflict_detects_unavoidable_oar_volume() -> None:
    case = generate_prostate_case_3d(181, grid_size=32, difficulty="easy")
    bladder = case.target.copy()
    evaluation_oars = list(case.evaluation_oars)
    evaluation_oars[0] = bladder
    planning_oars = list(case.oars)
    planning_oars[0] = bladder
    conflict_case = replace(
        case,
        oars=tuple(planning_oars),
        evaluation_oars=tuple(evaluation_oars),
    )
    conflicts = anatomical_objective_conflicts(conflict_case)
    bladder_conflicts = [value for value in conflicts if value.goal.structure == "bladder"]
    assert {value.goal.dose_gy for value in bladder_conflicts} == {37.0, 46.0}
    assert all(value.minimum_volume_percent >= 95.0 for value in bladder_conflicts)
    assert all(
        value.minimum_volume_percent_with_standard_target > 99.0
        for value in bladder_conflicts
    )


def test_clinical_arrays_are_padded_to_a_physical_cube() -> None:
    array = np.ones((5, 10, 20), dtype=bool)
    padded = _pad_to_physical_cube([array], (4.0, 2.0, 1.0))[0]
    assert padded.shape == (5, 10, 20)

    narrow = np.ones((5, 5, 5), dtype=bool)
    padded_narrow = _pad_to_physical_cube([narrow], (4.0, 2.0, 1.0))[0]
    assert padded_narrow.shape == (5, 10, 20)
    assert padded_narrow[:, 2:7, 7:12].all()


def test_clinical_hfs_conversion_places_anterior_at_positive_y() -> None:
    raw = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    converted = _to_hfs_planner_coordinates((raw,))[0]
    assert np.array_equal(converted[:, 0, :], raw[:, -1, :])
    assert np.array_equal(converted[:, -1, :], raw[:, 0, :])
    assert converted.strides[1] > 0


def test_delivery_complexity_modes_have_expected_angular_sampling() -> None:
    from dosim_sim.delivery3d import delivery_mode_3d

    assert len(delivery_mode_3d("static_4").angles_degrees) == 4
    assert len(delivery_mode_3d("static_12").angles_degrees) == 12
    assert len(delivery_mode_3d("arc_like_180").angles_degrees) == 19
    assert len(delivery_mode_3d("arc_like_360").angles_degrees) == 36
    assert delivery_mode_3d("arc_like_180").angles_degrees[0] == 90.0
    assert delivery_mode_3d("arc_like_180").angles_degrees[-1] == 270.0
