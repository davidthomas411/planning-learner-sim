import numpy as np

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
    ImplicitDoseEngine3D,
    optimize_fluence_3d,
)
from dosim_sim.dose import calculate_dose
from dosim_sim.dataset import action_record, case_features, plan_state
from dosim_sim.objective import evaluate_plan


def test_case_generation_is_reproducible() -> None:
    first = generate_case(17)
    second = generate_case(17)
    assert np.array_equal(first.target, second.target)
    assert all(np.array_equal(a, b) for a, b in zip(first.oars, second.oars, strict=True))


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
