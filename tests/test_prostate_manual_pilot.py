from types import SimpleNamespace

import csv
import torch

from dosim_sim.delivery3d import delivery_mode_3d
from dosim_sim.manual_planning import ManualAction
from dosim_sim.objective import PlanningPriorities
from dosim_sim.optimizer3d import full_optimization_target_3d
from dosim_sim.planning3d import PlanningStep3D, PlanningTrajectory3D
from dosim_sim.volume3d import generate_prostate_case_3d
from scripts.build_tcia_locked_profile_manifest import (
    PROFILE_QUOTAS,
    build_locked_manifest,
)
from scripts.run_prostate_ptv_manual_pilot import (
    ActionResponse,
    CLINICAL_CONFIG,
    append_delivery_replan,
    at_target_oar_tradeoff_boundary,
    field_template_escalation_allowed,
    increased_priority,
    load_tcia_episode_manifest,
    repeated_unproductive_steps,
    select_manual_action,
    starting_priorities,
    unresolved_hard_failure_reason,
)


def test_fixed_starting_profiles_change_only_the_named_priority_class() -> None:
    case = generate_prostate_case_3d(901, grid_size=24, difficulty="hard")
    custom = (1.0, 1.0, 1.0, 1.0)

    balanced = starting_priorities(case, "balanced_reference", custom)
    oar = starting_priorities(case, "oar_omitted", custom)
    hotspot = starting_priorities(case, "hotspot_low", custom)
    conformity = starting_priorities(case, "conformity_low", custom)
    oar_low = starting_priorities(case, "oar_low", custom)
    oar_guarded = starting_priorities(case, "oar_guarded", custom)
    hotspot_stress = starting_priorities(case, "hotspot_stress", custom)
    conformity_stress = starting_priorities(case, "conformity_stress", custom)

    assert balanced.target == balanced.hotspot == balanced.normal_tissue == 1.0
    assert all(value == 1.0 for value in balanced.oars)
    assert all(value == 0.0 for value in oar.oars)
    assert oar.target == 3.0 and oar.hotspot == oar.normal_tissue == 1.0
    assert hotspot.hotspot == 0.06 and hotspot.target == 3.0
    assert hotspot.normal_tissue == 1.0
    assert conformity.normal_tissue == 0.05 and conformity.target == 3.0
    assert conformity.hotspot == 1.0
    assert all(value == 0.05 for value in oar_low.oars)
    assert oar_low.target == 3.0 and oar_low.normal_tissue == 1.0
    assert all(value == 0.10 for value in oar_guarded.oars)
    assert oar_guarded.target == 3.0 and oar_guarded.normal_tissue == 1.0
    assert hotspot_stress.hotspot == 0.04 and hotspot_stress.target == 3.0
    assert conformity_stress.normal_tissue == 0.02
    assert conformity_stress.target == 3.0


def test_material_response_uses_the_prespecified_threshold() -> None:
    assert ActionResponse("PTV V57 Gy", 94.0, 95.0, 1.0, 1.0, "percentage points").material
    assert not ActionResponse("PTV V57 Gy", 94.0, 94.9, 0.9, 1.0, "percentage points").material
    assert increased_priority(0.0, CLINICAL_CONFIG) == 1.0
    assert increased_priority(1.0, CLINICAL_CONFIG) == 3.0


def test_second_repeated_nonresponsive_action_is_not_an_expert_label() -> None:
    case = generate_prostate_case_3d(902, grid_size=24, difficulty="hard")
    dose = torch.ones(case.body.shape, dtype=torch.float32)
    metrics = SimpleNamespace(covering_isodose_ratio_95=1.0)
    plan = SimpleNamespace(dose=dose, metrics=metrics)
    action_one = ManualAction(
        "increase_target_priority",
        "Increase target priority.",
        old_value=0.5,
        new_value=0.75,
    )
    action_two = ManualAction(
        "increase_target_priority",
        "Increase target priority again.",
        old_value=0.75,
        new_value=1.125,
    )
    trajectory = SimpleNamespace(
        steps=(
            SimpleNamespace(step=0, action=None, plan=plan),
            SimpleNamespace(step=1, action=action_one, plan=plan),
            SimpleNamespace(step=2, action=action_two, plan=plan),
        )
    )

    assert repeated_unproductive_steps(case, trajectory) == {2}


def test_locked_manifest_assigns_one_profile_per_remaining_patient() -> None:
    records = []
    for index in range(116):
        patient_id = f"Prostate-AEC-{index + 1:03d}"
        records.append(
            {
                "patient_id": patient_id,
                "case_id": f"tcia-{patient_id}",
                "margin_only_primary_eligible": str(15 <= index < 46),
                "maximum_ptv_oar_overlap_fraction": str(index / 1000.0),
            }
        )
    development_ids = {row["patient_id"] for row in records[:15]}

    assignments = build_locked_manifest(records, development_ids)

    assert len(assignments) == 101
    assert len({row["patient_id"] for row in assignments}) == 101
    assert not development_ids & {row["patient_id"] for row in assignments}
    assert {
        profile: sum(row["starting_profile"] == profile for row in assignments)
        for profile in PROFILE_QUOTAS
    } == dict(PROFILE_QUOTAS)
    assert sum(row["anatomy_stratum"] == "margin_only" for row in assignments) == 31
    assert sum(row["anatomy_stratum"] == "interface_overlap" for row in assignments) == 70


def test_tcia_episode_manifest_rejects_duplicate_patients(tmp_path) -> None:
    path = tmp_path / "episodes.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("patient_id", "starting_profile"))
        writer.writeheader()
        writer.writerows(
            (
                {
                    "patient_id": "Prostate-AEC-001",
                    "starting_profile": "balanced_reference",
                },
                {
                    "patient_id": "Prostate-AEC-001",
                    "starting_profile": "hotspot_stress",
                },
            )
        )

    try:
        load_tcia_episode_manifest(path)
    except ValueError as error:
        assert "one row per patient" in str(error)
    else:
        raise AssertionError("A duplicate patient was accepted")


def test_expert_review_configuration_has_a_nonzero_ptv_floor_and_safety_cap() -> None:
    assert CLINICAL_CONFIG.ptv_oar_overlap_minimum == 0.90
    assert CLINICAL_CONFIG.dmin_min == 0.90
    assert CLINICAL_CONFIG.max_steps == 32
    assert CLINICAL_CONFIG.priority_ceiling == 22.78125


def test_field_template_escalation_is_limited_to_unresolved_hard_failures() -> None:
    assert field_template_escalation_allowed(
        "technical_planning_failure_unresolved_hotspot"
    )
    assert field_template_escalation_allowed(
        "technical_planning_failure_unresolved_ptv_coverage"
    )
    assert not field_template_escalation_allowed(
        "requires_physician_review_target_oar_boundary"
    )
    assert not field_template_escalation_allowed("technical_failure_invalid_dose")


def test_field_template_replan_is_one_recorded_manual_action() -> None:
    plan_7 = SimpleNamespace(active_beams=tuple(range(7)))
    plan_9 = SimpleNamespace(active_beams=tuple(range(9)))
    first = PlanningTrajectory3D(
        "case",
        (
            PlanningStep3D(0, None, plan_7, 1.0),
            PlanningStep3D(
                1,
                ManualAction("increase_hotspot_priority", "Increase hotspot"),
                plan_7,
                0.8,
            ),
        ),
        "technical_planning_failure_unresolved_hotspot",
    )
    second = PlanningTrajectory3D(
        "case",
        (PlanningStep3D(0, None, plan_9, 0.4),),
        "requires_physician_review_target_oar_boundary",
    )

    combined = append_delivery_replan(
        first,
        second,
        delivery_mode_3d("static_7"),
        delivery_mode_3d("static_9"),
    )

    assert [step.step for step in combined.steps] == [0, 1, 2]
    assert combined.steps[2].action is not None
    assert combined.steps[2].action.kind == "replace_delivery_template"
    assert combined.steps[2].action.old_value == 7.0
    assert combined.steps[2].action.new_value == 9.0
    assert combined.stopping_reason == "requires_physician_review_target_oar_boundary"


def test_ptv_minimum_and_hotspot_failures_are_not_physician_tradeoffs() -> None:
    case = generate_prostate_case_3d(903, grid_size=24, difficulty="hard")
    cold_dose = torch.zeros(case.body.shape, dtype=torch.float32)
    cold_dose[torch.from_numpy(case.target)] = 1.0
    margin_indices = torch.nonzero(
        torch.from_numpy(case.target & ~case.clinical_target),
        as_tuple=False,
    )
    cold_dose[tuple(margin_indices[0])] = 0.89
    cold_plan = SimpleNamespace(dose=cold_dose)
    assert unresolved_hard_failure_reason(case, cold_plan) == (
        "technical_planning_failure_unresolved_target_minimum"
    )

    hot_dose = torch.zeros(case.body.shape, dtype=torch.float32)
    hot_dose[torch.from_numpy(case.target)] = 1.10
    hot_plan = SimpleNamespace(dose=hot_dose)
    assert unresolved_hard_failure_reason(case, hot_plan) == (
        "technical_planning_failure_unresolved_hotspot"
    )


def test_hard_hotspot_cannot_enter_physician_review(monkeypatch) -> None:
    plan = SimpleNamespace(
        dose=None,
        metrics=SimpleNamespace(covering_isodose_ratio_95=1.0),
        optimization_target=SimpleNamespace(cropped_oar_indices=(0,)),
    )
    monkeypatch.setattr(
        "scripts.run_prostate_ptv_manual_pilot.clinical_constraint_record",
        lambda _case, _dose: {
            "prostate_v60gy_pass": True,
            "ptv_d1cc_pass": False,
            "ptv_dmin_expert_floor_pass": True,
            "ptv_v57gy_percent": 96.0,
        },
    )
    monkeypatch.setattr(
        "scripts.run_prostate_ptv_manual_pilot.worst_oar_ratio",
        lambda _metrics, _config: (0, 1.2),
    )

    assert not at_target_oar_tradeoff_boundary(None, plan, CLINICAL_CONFIG)


def test_failed_bladder_goal_creates_split_target_before_weight_changes() -> None:
    case = generate_prostate_case_3d(904, grid_size=24, difficulty="hard")
    assert case.structure_names[0] == "bladder"
    assert bool((case.target & case.oars[0] & ~case.clinical_target).any())
    metrics = SimpleNamespace(
        target_d99=1.0,
        target_dmin=1.0,
        clinical_target_v100=1.0,
        target_d50=1.0,
        target_d1cc=1.04,
        target_v95=1.0,
        protocol_oar_per_protocol_ratios=(1.20, 0.50, 0.10),
        covering_isodose_ratio_95=1.0,
    )
    plan = SimpleNamespace(
        metrics=metrics,
        priorities=PlanningPriorities.for_case(case),
        optimization_target=full_optimization_target_3d(case),
    )

    action, priorities, target = select_manual_action(case, plan, CLINICAL_CONFIG)

    assert action is not None
    assert action.kind == "create_ptv_minus_bladder"
    assert priorities == plan.priorities
    assert target is not None
    assert target.cropped_oar_indices == (0,)
    assert target.relaxed_overlap_minimum == 0.90
