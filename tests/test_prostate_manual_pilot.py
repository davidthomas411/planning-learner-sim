from types import SimpleNamespace

import csv
import torch

from dosim_sim.manual_planning import ManualAction
from dosim_sim.volume3d import generate_prostate_case_3d
from scripts.build_tcia_locked_profile_manifest import (
    PROFILE_QUOTAS,
    build_locked_manifest,
)
from scripts.run_prostate_ptv_manual_pilot import (
    ActionResponse,
    CLINICAL_CONFIG,
    increased_priority,
    load_tcia_episode_manifest,
    repeated_unproductive_steps,
    starting_priorities,
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
