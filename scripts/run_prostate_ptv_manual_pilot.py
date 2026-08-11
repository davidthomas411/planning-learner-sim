"""Run a reviewable clinical manual-planning pilot for prostate cases."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import textwrap
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from dosim_sim.clinical3d import load_tcia_prostate_case
from dosim_sim.delivery3d import delivery_mode_3d
from dosim_sim.manual_planning import ManualAction
from dosim_sim.objective import PlanningPriorities
from dosim_sim.planning3d import (
    HighLevelSearchConfig3D,
    PlanningStep3D,
    PlanningTrajectory3D,
    clinical_violation_score_3d,
    coverage_d98_3d,
    is_acceptable_3d,
    optimizer_objective_kwargs_3d,
)
from dosim_sim.optimizer3d import (
    OptimizationTarget3D,
    ptv_minus_oars_optimization_target_3d,
)
from dosim_sim.prostate_protocol import (
    PRESCRIPTION_GY,
    anatomical_objective_conflicts,
    clinical_objective_set_record,
    evaluate_prostate_60gy20fx,
    prostate_evaluation_masks,
)
from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D, optimize_fluence_3d_torch
from dosim_sim.volume3d import generate_prostate_case_3d
from dosim_sim.visuals3d import add_hfs_orientation_labels
try:
    from run_prostate_clinical_dvh_pilot import (
        STATUS_PAGE,
        cumulative_dvh,
        load_records,
        should_update_figures,
        write_progress,
    )
except ModuleNotFoundError:
    from scripts.run_prostate_clinical_dvh_pilot import (
        STATUS_PAGE,
        cumulative_dvh,
        load_records,
        should_update_figures,
        write_progress,
    )


STARTING_PROFILE_VALUES = {
    "balanced_reference": {
        "target": 1.00,
        "hotspot": 1.00,
        "oar": 1.00,
        "normal_tissue": 1.00,
    },
    "oar_omitted": {
        "target": 3.00,
        "hotspot": 1.00,
        "oar": 0.00,
        "normal_tissue": 1.00,
    },
    "hotspot_low": {
        "target": 3.00,
        "hotspot": 0.06,
        "oar": 1.00,
        "normal_tissue": 1.00,
    },
    "conformity_low": {
        "target": 3.00,
        "hotspot": 1.00,
        "oar": 1.00,
        "normal_tissue": 0.05,
    },
    "oar_low": {
        "target": 3.00,
        "hotspot": 1.00,
        "oar": 0.05,
        "normal_tissue": 1.00,
    },
    "oar_guarded": {
        "target": 3.00,
        "hotspot": 1.00,
        "oar": 0.10,
        "normal_tissue": 1.00,
    },
    "hotspot_stress": {
        "target": 3.00,
        "hotspot": 0.04,
        "oar": 1.00,
        "normal_tissue": 1.00,
    },
    "conformity_stress": {
        "target": 3.00,
        "hotspot": 1.00,
        "oar": 1.00,
        "normal_tissue": 0.02,
    },
}


def load_tcia_episode_manifest(path: Path) -> list[dict[str, str]]:
    """Load a locked one-profile-per-patient TCIA episode manifest."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = {"patient_id", "starting_profile"}
    missing = required - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"TCIA episode manifest is missing columns: {sorted(missing)}")
    if not rows:
        raise ValueError("TCIA episode manifest is empty")
    patient_ids = [row["patient_id"].strip() for row in rows]
    if any(not patient_id for patient_id in patient_ids):
        raise ValueError("TCIA episode manifest contains an empty patient identifier")
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("TCIA episode manifest must contain one row per patient")
    unknown_profiles = sorted(
        {
            row["starting_profile"].strip()
            for row in rows
            if row["starting_profile"].strip() not in STARTING_PROFILE_VALUES
        }
    )
    if unknown_profiles:
        raise ValueError(
            f"TCIA episode manifest contains unknown starting profiles: {unknown_profiles}"
        )
    return [
        {
            **row,
            "patient_id": row["patient_id"].strip(),
            "starting_profile": row["starting_profile"].strip(),
        }
        for row in rows
    ]


@dataclass(frozen=True)
class ActionResponse:
    """Measured response to one high-level manual planning action."""

    metric: str
    previous_value: float
    current_value: float
    improvement: float
    material_threshold: float
    units: str

    @property
    def material(self) -> bool:
        return self.improvement >= self.material_threshold - 1e-9


def action_signature(action: ManualAction) -> tuple[str, int | None]:
    """Return the action class used by the repeated-nonresponse rule."""

    return action.kind, action.structure_index


def increased_priority(old: float, config) -> float:
    """Add an omitted objective at standard priority or scale an active one."""

    if old <= 0.0:
        return min(1.0, config.priority_ceiling)
    return min(old * config.priority_factor, config.priority_ceiling)


def starting_priorities(
    case,
    profile_name: str,
    custom_values: tuple[float, float, float, float],
) -> PlanningPriorities:
    """Return a fixed, preassigned starting profile for one anatomy."""

    base = PlanningPriorities.for_case(case)
    if profile_name == "custom":
        target, hotspot, oar, normal_tissue = custom_values
    else:
        values = STARTING_PROFILE_VALUES[profile_name]
        target = values["target"]
        hotspot = values["hotspot"]
        oar = values["oar"]
        normal_tissue = values["normal_tissue"]
    return replace(
        base,
        target=target,
        hotspot=hotspot,
        oars=tuple(oar for _ in base.oars),
        normal_tissue=normal_tissue,
    )


def worst_oar_ratio(metrics, config) -> tuple[int, float]:
    # Manual planning continues to aim for the institutional OAR limits. The
    # acceptable-variation limit is a review classification, not a new
    # optimizer target.
    ratios = metrics.protocol_oar_per_protocol_ratios
    if not ratios:
        return 0, 0.0
    index = int(np.argmax(ratios))
    return index, float(ratios[index])


def worst_oar_goal_details(case, dose, config) -> dict[str, float | str]:
    """Return the named DVH goal with the largest observed-to-limit ratio."""

    evaluation = evaluate_prostate_60gy20fx(
        case,
        dose.detach().float().cpu().numpy(),
    )
    per_protocol = True
    candidates = []
    for result in evaluation.oar_results:
        limit = (
            result.goal.per_protocol_volume_percent
            if per_protocol
            else result.goal.variation_volume_percent
        )
        candidates.append((result.observed_volume_percent / limit, result, limit))
    ratio, result, limit = max(candidates, key=lambda value: value[0])
    return {
        "structure": result.goal.structure,
        "metric": f"V{result.goal.dose_gy:g}Gy",
        "observed_percent": float(result.observed_volume_percent),
        "limit_percent": float(limit),
        "ratio": float(ratio),
    }


def structure_oar_goal_details(case, dose, structure_index: int) -> dict[str, float | str]:
    """Return the worst normalized DVH result for one planning OAR."""

    structure_name = case.structure_names[structure_index]
    protocol_names = {
        "bladder": {"bladder"},
        "rectum": {"rectum"},
        "femoral_heads": {"femur_head_l", "femur_head_r"},
    }.get(structure_name, {structure_name})
    evaluation = evaluate_prostate_60gy20fx(
        case,
        dose.detach().float().cpu().numpy(),
    )
    candidates = [
        result
        for result in evaluation.oar_results
        if result.goal.structure in protocol_names
    ]
    if not candidates:
        raise ValueError(f"No protocol OAR result for {structure_name}")
    result = max(
        candidates,
        key=lambda value: (
            value.observed_volume_percent / value.goal.per_protocol_volume_percent
        ),
    )
    return {
        "structure": result.goal.structure,
        "metric": f"V{result.goal.dose_gy:g}Gy",
        "observed_percent": float(result.observed_volume_percent),
        "limit_percent": float(result.goal.per_protocol_volume_percent),
    }


def measure_action_response(case, previous_plan, current_plan, action: ManualAction) -> ActionResponse:
    """Measure whether the intended clinical metric changed enough to matter.

    The thresholds are development rules. They are fixed before the locked
    clinical-anatomy evaluation. They prevent repeated weight changes when the
    recalculated plan does not respond in the intended direction.
    """

    previous = clinical_constraint_record(case, previous_plan.dose)
    current = clinical_constraint_record(case, current_plan.dose)
    if action.kind == "increase_target_priority":
        previous_value = float(previous["ptv_v57gy_percent"])
        current_value = float(current["ptv_v57gy_percent"])
        return ActionResponse(
            "PTV V57 Gy",
            previous_value,
            current_value,
            current_value - previous_value,
            1.0,
            "percentage points",
        )
    if action.kind == "increase_hotspot_priority":
        previous_value = float(previous["ptv_d1cc_gy"])
        current_value = float(current["ptv_d1cc_gy"])
        return ActionResponse(
            "PTV D1cc",
            previous_value,
            current_value,
            previous_value - current_value,
            0.3,
            "Gy",
        )
    if action.kind in {
        "increase_oar_priority",
        "create_ptv_minus_bladder",
        "create_ptv_minus_rectum",
    }:
        if action.structure_index is None:
            raise ValueError(f"{action.kind} requires a structure index")
        previous_oar = structure_oar_goal_details(
            case,
            previous_plan.dose,
            action.structure_index,
        )
        current_oar = structure_oar_goal_details(
            case,
            current_plan.dose,
            action.structure_index,
        )
        previous_value = float(previous_oar["observed_percent"])
        current_value = float(current_oar["observed_percent"])
        violation = max(previous_value - float(previous_oar["limit_percent"]), 0.0)
        threshold = max(1.0, 0.10 * violation)
        return ActionResponse(
            f"{str(previous_oar['structure']).replace('_', ' ')} {previous_oar['metric']}",
            previous_value,
            current_value,
            previous_value - current_value,
            threshold,
            "percentage points",
        )
    if action.kind == "increase_normal_tissue_priority":
        previous_value = float(previous_plan.metrics.covering_isodose_ratio_95)
        current_value = float(current_plan.metrics.covering_isodose_ratio_95)
        return ActionResponse(
            "57 Gy covering-isodose ratio",
            previous_value,
            current_value,
            previous_value - current_value,
            0.01,
            "ratio",
        )
    return ActionResponse(action.kind, 0.0, 0.0, 0.0, float("inf"), "")


def trajectory_action_responses(case, trajectory) -> dict[int, ActionResponse]:
    """Return measured responses keyed by the action step number."""

    responses = {}
    for previous_step, current_step in zip(
        trajectory.steps,
        trajectory.steps[1:],
    ):
        if current_step.action is not None:
            responses[current_step.step] = measure_action_response(
                case,
                previous_step.plan,
                current_step.plan,
                current_step.action,
            )
    return responses


def repeated_unproductive_steps(case, trajectory) -> set[int]:
    """Identify the second action in each repeated nonresponsive action pair."""

    responses = trajectory_action_responses(case, trajectory)
    excluded = set()
    for first, second in zip(trajectory.steps[1:], trajectory.steps[2:]):
        if (
            first.action is not None
            and second.action is not None
            and action_signature(first.action) == action_signature(second.action)
            and not responses[first.step].material
            and not responses[second.step].material
        ):
            excluded.add(second.step)
    return excluded


def clinical_constraint_record(case, dose) -> dict[str, float | bool | str]:
    """Return every supplied clinical evaluator as explicit saved columns."""

    evaluation = evaluate_prostate_60gy20fx(
        case,
        dose.detach().float().cpu().numpy(),
    )
    record: dict[str, float | bool] = {
        "prostate_v60gy_percent": evaluation.prostate_v60_percent,
        "prostate_v60gy_pass": evaluation.prostate_v60_percent >= 99.0,
        "ptv_d99_gy": evaluation.target_d99_gy,
        "ptv_d99_pass": evaluation.target_d99_gy >= 57.0,
        "ptv_v57gy_percent": evaluation.target_v57_percent,
        "ptv_v57gy_per_protocol_pass": evaluation.target_v57_percent >= 99.0,
        "ptv_v57gy_acceptable_variation_pass": evaluation.target_v57_percent >= 95.0,
        "ptv_d1cc_gy": evaluation.target_d1cc_gy,
        "ptv_d1cc_pass": evaluation.target_d1cc_gy <= 63.0,
        "protocol_acceptance_class": evaluation.acceptance_class,
    }
    for result in evaluation.oar_results:
        key = f"{result.goal.structure}_v{result.goal.dose_gy:g}gy".lower()
        record[f"{key}_percent"] = result.observed_volume_percent
        record[f"{key}_pass"] = result.per_protocol
    return record


def active_anatomical_conflicts(case, dose) -> tuple:
    """Return geometric target-OAR conflicts for objectives that also fail."""

    evaluation = evaluate_prostate_60gy20fx(
        case,
        dose.detach().float().cpu().numpy(),
    )
    failed_goals = {
        (result.goal.structure, result.goal.dose_gy)
        for result in evaluation.oar_results
        if not result.per_protocol
    }
    return tuple(
        conflict
        for conflict in anatomical_objective_conflicts(case)
        if (conflict.goal.structure, conflict.goal.dose_gy) in failed_goals
    )


def clinical_target_d98_gy(case, dose) -> float:
    """Return prostate/CTV D98 as a diagnostic when that contour exists."""

    if case.clinical_target is None or not np.any(case.clinical_target):
        return float("nan")
    dose_gy = dose.detach().float().cpu().numpy() * PRESCRIPTION_GY
    return float(np.quantile(dose_gy[case.clinical_target], 0.02))


def anatomy_overlap_record(case, structure_name: str | None = None) -> dict[str, float | str]:
    """Return target-OAR overlap measures without using a dose result."""

    target_volume = int(np.count_nonzero(case.target))
    values = []
    for index, oar in enumerate(case.oars):
        if structure_name is not None and case.structure_names[index] != structure_name:
            continue
        overlap_volume = int(np.count_nonzero(case.target & oar))
        values.append(
            (
                overlap_volume / max(int(np.count_nonzero(oar)), 1),
                overlap_volume / max(target_volume, 1),
                index,
            )
        )
    if not values:
        raise ValueError(f"Unknown stress structure: {structure_name}")
    oar_fraction, target_fraction, index = max(values)
    overlap_volume = int(np.count_nonzero(case.target & case.oars[index]))
    clinical_target_overlap = (
        0
        if case.clinical_target is None
        else int(np.count_nonzero(case.clinical_target & case.oars[index]))
    )
    margin_volume = (
        int(np.count_nonzero(case.target))
        if case.clinical_target is None
        else int(np.count_nonzero(case.target & ~case.clinical_target))
    )
    return {
        "overlap_structure": case.structure_names[index],
        "oar_overlap_fraction": float(oar_fraction),
        "ptv_overlap_fraction": float(target_fraction),
        "ptv_margin_overlap_fraction": float(overlap_volume / max(margin_volume, 1)),
        "clinical_target_oar_overlap_voxels": clinical_target_overlap,
    }


def tcia_anatomy_record(case, subject_dir: Path) -> dict[str, float | str | int]:
    """Return a review record for one imported clinical contour set."""

    bladder = anatomy_overlap_record(case, "bladder")
    rectum = anatomy_overlap_record(case, "rectum")
    selected = max(
        (bladder, rectum),
        key=lambda value: float(value["ptv_overlap_fraction"]),
    )
    bladder_ctv_overlap = int(np.count_nonzero(case.clinical_target & case.oars[case.structure_names.index("bladder")]))
    rectum_ctv_overlap = int(np.count_nonzero(case.clinical_target & case.oars[case.structure_names.index("rectum")]))
    return {
        "patient_id": subject_dir.name,
        "seed": case.seed,
        "difficulty": "external",
        "selection_mode": "tcia_clinical_anatomy",
        "anatomy_stratum": (
            "margin_only"
            if bladder_ctv_overlap == 0 and rectum_ctv_overlap == 0
            else "interface_overlap"
        ),
        "bladder_ctv_overlap_voxels": bladder_ctv_overlap,
        "rectum_ctv_overlap_voxels": rectum_ctv_overlap,
        **selected,
        "bladder_oar_overlap_fraction": float(bladder["oar_overlap_fraction"]),
        "bladder_ptv_overlap_fraction": float(bladder["ptv_overlap_fraction"]),
        "rectum_oar_overlap_fraction": float(rectum["oar_overlap_fraction"]),
        "rectum_ptv_overlap_fraction": float(rectum["ptv_overlap_fraction"]),
        "prostate_voxels": int(np.count_nonzero(case.clinical_target)),
        "ptv_voxels": int(np.count_nonzero(case.target)),
        "body_voxels": int(np.count_nonzero(case.body)),
    }


def write_trajectory_rows(rows: list[dict], path: Path) -> None:
    """Write a recoverable trajectory checkpoint after each completed case."""

    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_figure_atomic(figure, path: Path, **kwargs) -> None:
    """Replace a complete image so the live page cannot read a partial file."""

    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary, **kwargs)
    temporary.replace(path)


def summarize_anatomy_strata(
    outcomes: list[dict[str, float | int | str | bool | None]],
) -> dict[str, dict[str, float | int]]:
    """Summarize outcomes without pooling the two TCIA anatomy strata."""

    labels = sorted(
        {
            str(outcome["anatomy_stratum"])
            for outcome in outcomes
            if outcome.get("anatomy_stratum") is not None
        }
    )
    summaries: dict[str, dict[str, float | int]] = {}
    for label in labels:
        selected = [
            outcome for outcome in outcomes if outcome.get("anatomy_stratum") == label
        ]
        accepted = [bool(outcome["acceptable"]) for outcome in selected]
        summaries[label] = {
            "cases": len(selected),
            "acceptable_cases": int(sum(accepted)),
            "acceptable_rate": float(np.mean(accepted)),
            "median_manual_changes": float(
                np.median([int(outcome["manual_changes"]) for outcome in selected])
            ),
        }
    return summaries


def select_oar_stress_records(
    count: int,
    seed_start: int,
    maximum_attempts: int,
    grid_size: int,
    coarse_grid_size: int,
    stress_structure: str,
    minimum_bladder_overlap_fraction: float,
    minimum_rectum_overlap_fraction: float,
    minimum_bladder_ptv_overlap_fraction: float,
    minimum_rectum_ptv_overlap_fraction: float,
    maximum_ptv_overlap_fraction: float,
) -> tuple[list[dict], int]:
    """Select a named and optionally balanced OAR-stress cohort from anatomy."""

    if stress_structure not in {"balanced", "bladder", "rectum"}:
        raise ValueError("stress_structure must be balanced, bladder, or rectum")
    selected: list[dict] = []
    attempted = 0
    if stress_structure == "balanced":
        requested_structures = [
            ("bladder", "rectum")[index % 2]
            for index in range(count)
        ]
    else:
        requested_structures = [stress_structure] * count
    minimum_oar_fraction = {
        "bladder": minimum_bladder_overlap_fraction,
        "rectum": minimum_rectum_overlap_fraction,
    }
    minimum_ptv_fraction = {
        "bladder": minimum_bladder_ptv_overlap_fraction,
        "rectum": minimum_rectum_ptv_overlap_fraction,
    }
    for seed in range(seed_start, seed_start + maximum_attempts):
        attempted += 1
        requested_structure = requested_structures[len(selected)]
        coarse_case = generate_prostate_case_3d(seed, coarse_grid_size, difficulty="hard")
        coarse = anatomy_overlap_record(coarse_case, requested_structure)
        if coarse["oar_overlap_fraction"] < 0.80 * minimum_oar_fraction[requested_structure]:
            continue
        if not (
            0.80 * minimum_ptv_fraction[requested_structure]
            <= coarse["ptv_overlap_fraction"]
            <= 1.10 * maximum_ptv_overlap_fraction
        ):
            continue
        case = generate_prostate_case_3d(seed, grid_size, difficulty="hard")
        exact = anatomy_overlap_record(case, requested_structure)
        if exact["oar_overlap_fraction"] < minimum_oar_fraction[requested_structure]:
            continue
        if not (
            minimum_ptv_fraction[requested_structure]
            <= exact["ptv_overlap_fraction"]
            <= maximum_ptv_overlap_fraction
        ):
            continue
        selected.append(
            {
                "seed": seed,
                "difficulty": "hard",
                "selection_mode": "oar_stress",
                **exact,
            }
        )
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise ValueError(
            f"Selected {len(selected)} of {count} requested OAR-stress cases after "
            f"{attempted} anatomy attempts"
        )
    return selected, attempted


def select_manual_action(
    case,
    plan,
    config,
    excluded_action_signatures: frozenset[tuple[str, int | None]] = frozenset(),
) -> tuple[ManualAction | None, PlanningPriorities, OptimizationTarget3D | None]:
    """Select one review-based change that a dosimetrist can make.

    The reviewer acts on the largest relative clinical-limit violation. A
    priority that has reached its allowed ceiling is skipped. This permits the
    reviewer to address a second failing objective without changing the
    clinical limits or searching over fluence settings.
    """

    metrics = plan.metrics
    priorities = plan.priorities
    optimization_target = plan.optimization_target
    ratio_tolerance = 1e-6
    # Plan metrics are relative to prescription. Use the evaluator's 0.001 Gy
    # numerical tolerance in the same relative units.
    dose_tolerance = 1e-3 / PRESCRIPTION_GY
    target_gaps = [0.0]
    if config.d99_min > 0.0:
        target_gaps.append(
            (config.d99_min - metrics.target_d99 - dose_tolerance) / config.d99_min
        )
    if config.clinical_target_v100_min > 0.0:
        target_gaps.append(
            (
                config.clinical_target_v100_min
                - float(metrics.clinical_target_v100 or 0.0)
                - ratio_tolerance
            )
            / config.clinical_target_v100_min
        )
    if config.d50_min > 0.0:
        target_gaps.append(
            (config.d50_min - metrics.target_d50 - dose_tolerance) / config.d50_min
        )
    if np.isfinite(config.d50_max):
        target_gaps.append(
            (metrics.target_d50 - config.d50_max - dose_tolerance) / config.d50_max
        )
    target_violation = max(target_gaps)
    if config.prostate_protocol_tier == "variation_acceptable":
        target_goal = (
            0.95
            if optimization_target is not None
            and optimization_target.cropped_oar_indices
            else 0.99
        )
        target_violation = max(
            target_violation,
            (target_goal - metrics.target_v95 - ratio_tolerance) / target_goal,
        )
    hotspot_violation = (
        max(
            (
                float(metrics.target_d1cc or 0.0)
                - config.d1cc_max
                - dose_tolerance
            )
            / config.d1cc_max,
            0.0,
        )
        if np.isfinite(config.d1cc_max)
        else max(
            (float(metrics.target_d1cc or 0.0) - 1.05 - dose_tolerance) / 1.05,
            0.0,
        )
    )
    oar_index, oar_ratio = worst_oar_ratio(metrics, config)
    oar_violation = max(oar_ratio - 1.0, 0.0)

    cropped_indices = (
        () if optimization_target is None else optimization_target.cropped_oar_indices
    )
    oar_variation_ratios = metrics.protocol_oar_variation_ratios
    oar_variation_ratio = (
        float(oar_variation_ratios[oar_index])
        if oar_index < len(oar_variation_ratios)
        else float("inf")
    )
    try_oar_weight_first = (
        metrics.protocol_target_per_protocol is True
        and oar_variation_ratio > 1.0 + ratio_tolerance
        and priorities.oars[oar_index]
        < min(config.priority_factor**2, config.priority_ceiling) - ratio_tolerance
        and ("increase_oar_priority", oar_index) not in excluded_action_signatures
    )
    if (
        config.manual_ptv_oar_crop
        and oar_violation > ratio_tolerance
        and case.structure_names[oar_index] in {"bladder", "rectum"}
        and oar_index not in cropped_indices
        and not try_oar_weight_first
    ):
        clinical_target = (
            np.zeros_like(case.target, dtype=bool)
            if case.clinical_target is None
            else case.clinical_target
        )
        margin_overlap = case.target & case.oars[oar_index] & ~clinical_target
        if np.any(margin_overlap):
            new_indices = tuple(sorted((*cropped_indices, oar_index)))
            updated_target = ptv_minus_oars_optimization_target_3d(
                case,
                new_indices,
                config.ptv_oar_overlap_minimum,
            )
            structure_name = case.structure_names[oar_index].replace("_", " ")
            action = ManualAction(
                f"create_ptv_minus_{case.structure_names[oar_index]}",
                f"Create a PTV-minus-{structure_name} optimization target. Keep the "
                "prostate/CTV in the full-dose target. Permit limited undercoverage "
                f"in the PTV-{structure_name} overlap while keeping PTV V57 Gy at "
                "least 95%.",
                structure_index=oar_index,
                old_value=1.0,
                new_value=config.ptv_oar_overlap_minimum,
            )
            if action_signature(action) not in excluded_action_signatures:
                return action, priorities, updated_target
    conformity_violation = max(
        metrics.covering_isodose_ratio_95 / config.covering_isodose_ratio_95_max - 1.0,
        0.0,
    )
    candidates = sorted(
        (
            (target_violation, "target"),
            (hotspot_violation, "hotspot"),
            (oar_violation, "oar"),
            (conformity_violation, "normal_tissue"),
        ),
        reverse=True,
    )
    for violation, name in candidates:
        if violation <= ratio_tolerance:
            continue
        if name == "target":
            old = priorities.target
            new = increased_priority(old, config)
            updated = replace(priorities, target=new)
            action = ManualAction(
                "increase_target_priority",
                (
                    f"Add the PTV objective at priority {new:.2f}."
                    if old <= 0.0
                    else f"Increase the PTV priority from {old:.2f} to {new:.2f}."
                ),
                old_value=old,
                new_value=new,
            )
        elif name == "hotspot":
            old = priorities.hotspot
            new = increased_priority(old, config)
            updated = replace(priorities, hotspot=new)
            action = ManualAction(
                "increase_hotspot_priority",
                (
                    f"Add the PTV hotspot objective at priority {new:.2f}."
                    if old <= 0.0
                    else f"Increase the PTV hotspot priority from {old:.2f} to {new:.2f}."
                ),
                old_value=old,
                new_value=new,
            )
        elif name == "oar":
            structure_name = case.structure_names[oar_index]
            old = priorities.oars[oar_index]
            new = increased_priority(old, config)
            oars = list(priorities.oars)
            oars[oar_index] = new
            updated = replace(priorities, oars=tuple(oars))
            action = ManualAction(
                "increase_oar_priority",
                (
                    f"Add the {structure_name.replace('_', ' ')} objective at "
                    f"priority {new:.2f}."
                    if old <= 0.0
                    else f"Increase the {structure_name.replace('_', ' ')} priority "
                    f"from {old:.2f} to {new:.2f}."
                ),
                structure_index=oar_index,
                old_value=old,
                new_value=new,
            )
        else:
            old = priorities.normal_tissue
            new = increased_priority(old, config)
            updated = replace(priorities, normal_tissue=new)
            action = ManualAction(
                "increase_normal_tissue_priority",
                (
                    f"Add the normal-tissue objective at priority {new:.2f}."
                    if old <= 0.0
                    else f"Increase the normal-tissue priority from {old:.2f} to {new:.2f}."
                ),
                old_value=old,
                new_value=new,
            )
        if new > old and action_signature(action) not in excluded_action_signatures:
            return action, updated, optimization_target
    return None, priorities, optimization_target


def short_review_record(case, plan, config) -> dict[str, str]:
    """Return a concise clinical review and the next high-level decision."""

    metrics = plan.metrics
    oar = worst_oar_goal_details(case, plan.dose, config)
    clinical = clinical_constraint_record(case, plan.dose)
    dose_tolerance = 1e-3
    ratio_tolerance = 1e-6
    prostate_pass = bool(clinical["prostate_v60gy_pass"])
    d99_pass = bool(clinical["ptv_d99_pass"])
    d1cc_pass = bool(clinical["ptv_d1cc_pass"])
    conformity_pass = (
        metrics.covering_isodose_ratio_95
        <= config.covering_isodose_ratio_95_max + ratio_tolerance
    )
    oar_pass = float(oar["ratio"]) <= 1.0 + ratio_tolerance
    findings = (
        f"Prostate V60 Gy is {float(clinical['prostate_v60gy_percent']):.1f}% "
        f"({'pass' if prostate_pass else 'fail'}). "
        f"PTV D99 is {float(clinical['ptv_d99_gy']):.2f} Gy "
        f"({'pass' if d99_pass else 'fail'}). "
        f"PTV V57 Gy is {float(clinical['ptv_v57gy_percent']):.1f}%. "
        f"PTV D1cc is {float(clinical['ptv_d1cc_gy']):.2f} Gy "
        f"({'pass' if d1cc_pass else 'fail'}). "
        f"The 57 Gy covering-isodose ratio is {metrics.covering_isodose_ratio_95:.3f} "
        f"({'pass' if conformity_pass else 'fail'}). "
        f"The worst OAR objective is {str(oar['structure']).replace('_', ' ')} "
        f"{oar['metric']} at {float(oar['observed_percent']):.1f}% with a "
        f"{float(oar['limit_percent']):.1f}% limit ({'pass' if oar_pass else 'fail'})."
    )
    if is_acceptable_3d(metrics, case, config):
        acceptance_class = str(clinical["protocol_acceptance_class"])
        decision = {
            "per_protocol": "Accept. All institutional objectives pass.",
            "acceptable_oar_variation": (
                "Accept as an OAR variation. Standard target coverage passes and the "
                "rectum or bladder excess is no more than 5 percentage points."
            ),
            "acceptable_target_coverage_variation": (
                "Accept as a target-coverage variation. Prostate coverage and all OAR "
                "limits pass, and PTV V57 Gy is at least 95%."
            ),
        }.get(acceptance_class, "Accept under the declared study policy.")
        return {
            "review_findings": findings,
            "review_decision": decision,
            "recommended_action_type": "stop",
        }
    conflicts = active_anatomical_conflicts(case, plan.dose)
    if conflicts:
        conflict = max(
            conflicts,
            key=lambda value: (
                value.minimum_volume_percent
                / value.goal.per_protocol_volume_percent
            ),
        )
        structure = conflict.goal.structure.replace("_", " ")
        return {
            "review_findings": findings,
            "review_decision": (
                "Do not accept. Stop manual weight changes. Even the allowed target or "
                "OAR variation cannot resolve this overlap. Target variation forces at "
                f"least {conflict.minimum_volume_percent:.1f}% of the {structure} above "
                f"{conflict.goal.dose_gy:g} Gy, but the institutional limit is "
                f"{conflict.goal.per_protocol_volume_percent:g}%. Record a major "
                "anatomical objective conflict."
            ),
            "recommended_action_type": "anatomical_objective_conflict",
        }
    action, _, _ = select_manual_action(case, plan, config)
    if action is None:
        return {
            "review_findings": findings,
            "review_decision": "Do not accept. No permitted priority change remains.",
            "recommended_action_type": "none",
        }
    return {
        "review_findings": findings,
        "review_decision": f"Do not accept. {action.description} Recalculate and review again.",
        "recommended_action_type": action.kind,
    }


def run_manual_sequence(
    case,
    engine,
    active_beams,
    config,
    initial_priorities: PlanningPriorities | None = None,
    on_step: Callable[[PlanningTrajectory3D], None] | None = None,
) -> PlanningTrajectory3D:
    priorities = initial_priorities or PlanningPriorities.for_case(case)
    optimization_target = None
    plan = optimize_fluence_3d_torch(
        case,
        engine,
        active_beams,
        priorities,
        config.optimizer_iterations,
        **optimizer_objective_kwargs_3d(config),
        optimization_target=optimization_target,
    )
    steps = [PlanningStep3D(0, None, plan, clinical_violation_score_3d(plan.metrics, case, config))]
    if on_step is not None:
        on_step(PlanningTrajectory3D(case.case_id, tuple(steps), "running"))
    if is_acceptable_3d(plan.metrics, case, config):
        return PlanningTrajectory3D(case.case_id, tuple(steps), "acceptable")
    if not plan_dose_is_numerically_valid(plan):
        return PlanningTrajectory3D(
            case.case_id,
            tuple(steps),
            "technical_failure_invalid_dose",
        )
    if active_anatomical_conflicts(case, plan.dose):
        return PlanningTrajectory3D(
            case.case_id,
            tuple(steps),
            "requires_physician_review_anatomical_conflict",
        )
    excluded_signatures: set[tuple[str, int | None]] = set()
    consecutive_nonresponse_signature: tuple[str, int | None] | None = None
    consecutive_nonresponse_count = 0
    for step_index in range(1, config.max_steps + 1):
        action, priorities, optimization_target = select_manual_action(
            case,
            plan,
            config,
            frozenset(excluded_signatures),
        )
        if action is None:
            reason = (
                "requires_physician_review_nonresponse"
                if excluded_signatures
                else "requires_physician_review_no_allowed_change"
            )
            return PlanningTrajectory3D(case.case_id, tuple(steps), reason)
        previous_plan = plan
        try:
            plan = optimize_fluence_3d_torch(
                case,
                engine,
                active_beams,
                priorities,
                config.optimizer_iterations,
                **optimizer_objective_kwargs_3d(config),
                optimization_target=optimization_target,
            )
        except FloatingPointError:
            return PlanningTrajectory3D(
                case.case_id,
                tuple(steps),
                "technical_failure_nonfinite_optimizer",
            )
        steps.append(
            PlanningStep3D(
                step_index,
                action,
                plan,
                clinical_violation_score_3d(plan.metrics, case, config),
            )
        )
        response = measure_action_response(case, previous_plan, plan, action)
        signature = action_signature(action)
        if not response.material and signature == consecutive_nonresponse_signature:
            consecutive_nonresponse_count += 1
        elif not response.material:
            consecutive_nonresponse_signature = signature
            consecutive_nonresponse_count = 1
        else:
            consecutive_nonresponse_signature = None
            consecutive_nonresponse_count = 0
        if consecutive_nonresponse_count >= 2:
            excluded_signatures.add(signature)
            consecutive_nonresponse_signature = None
            consecutive_nonresponse_count = 0
        if on_step is not None:
            on_step(PlanningTrajectory3D(case.case_id, tuple(steps), "running"))
        if not plan_dose_is_numerically_valid(plan):
            return PlanningTrajectory3D(
                case.case_id,
                tuple(steps),
                "technical_failure_invalid_dose",
            )
        if is_acceptable_3d(plan.metrics, case, config):
            return PlanningTrajectory3D(case.case_id, tuple(steps), "acceptable")
        if active_anatomical_conflicts(case, plan.dose):
            return PlanningTrajectory3D(
                case.case_id,
                tuple(steps),
                "requires_physician_review_anatomical_conflict",
            )
    return PlanningTrajectory3D(
        case.case_id,
        tuple(steps),
        "requires_physician_review_step_limit",
    )


def plan_dose_is_numerically_valid(plan) -> bool:
    """Reject nonfinite or grossly scaled surrogate dose states."""

    dose = plan.dose.detach()
    return bool(torch.all(torch.isfinite(dose))) and float(torch.max(dose)) <= 2.0


def terminal_disposition(trajectory, case, config) -> str:
    """Return the safe terminal label used for learning and clinical review."""

    if is_acceptable_3d(trajectory.final.plan.metrics, case, config):
        acceptance_class = clinical_constraint_record(
            case,
            trajectory.final.plan.dose,
        )["protocol_acceptance_class"]
        return f"accept_{acceptance_class}"
    if trajectory.stopping_reason.startswith("requires_physician_review"):
        return "requires_physician_review"
    return "technical_or_unclassified_failure"


def tradeoff_state_record(case, trajectory, config) -> dict[str, float | int | str]:
    """Compare target-preserved and OAR-preserved states for a major variation."""

    records = []
    for step in trajectory.steps:
        clinical = clinical_constraint_record(case, step.plan.dose)
        common_hard_passes = int(
            bool(clinical["prostate_v60gy_pass"])
            and bool(clinical["ptv_d1cc_pass"])
            and step.plan.metrics.covering_isodose_ratio_95
            <= config.covering_isodose_ratio_95_max + 1e-6
        )
        records.append(
            {
                "step": step.step,
                "ptv_v57gy_percent": float(clinical["ptv_v57gy_percent"]),
                "worst_oar_goal_ratio": worst_oar_ratio(step.plan.metrics, config)[1],
                "common_hard_passes": common_hard_passes,
            }
        )
    target_state = max(
        records,
        key=lambda value: (
            value["common_hard_passes"],
            value["ptv_v57gy_percent"],
            -value["worst_oar_goal_ratio"],
        ),
    )
    oar_state = max(
        records,
        key=lambda value: (
            value["common_hard_passes"],
            -value["worst_oar_goal_ratio"],
            value["ptv_v57gy_percent"],
        ),
    )
    return {
        "target_preserved_step": int(target_state["step"]),
        "target_preserved_ptv_v57gy_percent": float(target_state["ptv_v57gy_percent"]),
        "target_preserved_worst_oar_goal_ratio": float(target_state["worst_oar_goal_ratio"]),
        "oar_preserved_step": int(oar_state["step"]),
        "oar_preserved_ptv_v57gy_percent": float(oar_state["ptv_v57gy_percent"]),
        "oar_preserved_worst_oar_goal_ratio": float(oar_state["worst_oar_goal_ratio"]),
        "expert_disposition": "requires_physician_review",
        "expert_reason": (
            "No state meets an automatic acceptance class. Compare preserved target "
            "coverage with preserved OAR dose; do not create an automatic training label."
        ),
    }


def step_row(
    case,
    trajectory,
    step,
    config,
    episode_id: str | None = None,
    starting_profile: str = "custom",
) -> dict:
    metrics = step.plan.metrics
    action = step.action
    oar_details = worst_oar_goal_details(case, step.plan.dose, config)
    review = short_review_record(case, step.plan, config)
    responses = trajectory_action_responses(case, trajectory)
    response = responses.get(step.step)
    unproductive_steps = repeated_unproductive_steps(case, trajectory)
    return {
        "case_id": case.case_id,
        "episode_id": episode_id or case.case_id,
        "starting_profile": starting_profile,
        "seed": case.seed,
        "difficulty": case.difficulty,
        "step": step.step,
        "action_type": "initial_plan" if action is None else action.kind,
        "action": (
            f"Create the initial {len(step.plan.active_beams)}-field plan."
            if action is None
            else action.description
        ),
        "action_structure": (
            ""
            if action is None or action.structure_index is None
            else case.structure_names[action.structure_index]
        ),
        "stopping_reason": trajectory.stopping_reason if step is trajectory.final else "",
        "target_priority": step.plan.priorities.target,
        "hotspot_priority": step.plan.priorities.hotspot,
        "bladder_priority": step.plan.priorities.oars[0],
        "rectum_priority": step.plan.priorities.oars[1],
        "femoral_heads_priority": step.plan.priorities.oars[2],
        "normal_tissue_priority": step.plan.priorities.normal_tissue,
        **clinical_constraint_record(case, step.plan.dose),
        "ptv_d98_gy": metrics.target_d98 * PRESCRIPTION_GY,
        "optimization_target_d98_gy": coverage_d98_3d(metrics) * PRESCRIPTION_GY,
        "clinical_target_d98_gy": clinical_target_d98_gy(case, step.plan.dose),
        "relaxed_overlap_d98_gy": (
            ""
            if metrics.relaxed_overlap_d98 is None
            else metrics.relaxed_overlap_d98 * PRESCRIPTION_GY
        ),
        "relaxed_overlap_minimum_gy": (
            ""
            if metrics.relaxed_overlap_minimum is None
            else metrics.relaxed_overlap_minimum * PRESCRIPTION_GY
        ),
        "relaxed_overlap_fraction": metrics.relaxed_overlap_fraction,
        "cropped_oar_structures": (
            ""
            if step.plan.optimization_target is None
            else ";".join(
                case.structure_names[index]
                for index in step.plan.optimization_target.cropped_oar_indices
            )
        ),
        "ptv_d50_gy": metrics.target_d50 * PRESCRIPTION_GY,
        "ptv_d02_gy": metrics.target_d02 * PRESCRIPTION_GY,
        "covering_isodose_ratio_57gy": metrics.covering_isodose_ratio_95,
        "outside_ptv_57gy_percent": 100.0 * metrics.outside_target_ratio_95,
        "paddick_ci_57gy": metrics.paddick_ci_95,
        "worst_oar_structure": oar_details["structure"],
        "worst_oar_metric": oar_details["metric"],
        "worst_oar_observed_percent": oar_details["observed_percent"],
        "worst_oar_limit_percent": oar_details["limit_percent"],
        "worst_oar_goal_ratio": oar_details["ratio"],
        "acceptable": is_acceptable_3d(metrics, case, config),
        "violation_score": step.violation_score,
        "response_metric": "" if response is None else response.metric,
        "response_previous_value": "" if response is None else response.previous_value,
        "response_current_value": "" if response is None else response.current_value,
        "response_improvement": "" if response is None else response.improvement,
        "response_material_threshold": "" if response is None else response.material_threshold,
        "response_units": "" if response is None else response.units,
        "response_material": "" if response is None else response.material,
        "repeated_unproductive_action": step.step in unproductive_steps,
        "expert_action_label_eligible": (
            action is not None and step.step not in unproductive_steps
        ),
        "terminal_disposition": (
            terminal_disposition(trajectory, case, config)
            if step is trajectory.final
            else ""
        ),
        **review,
    }


def save_trajectory_plot(rows: list[dict], path: Path) -> None:
    case_ids = list(dict.fromkeys(row.get("episode_id", row["case_id"]) for row in rows))
    figure, axes = plt.subplots(1, 5, figsize=(22, 4.6), constrained_layout=True)
    fields = (
        ("prostate_v60gy_percent", "Prostate V60 Gy (%)", 99.0, "Prostate coverage"),
        ("ptv_d99_gy", "PTV D99 (Gy)", 57.0, "PTV coverage"),
        ("ptv_d1cc_gy", "PTV D1cc (Gy)", 63.0, "PTV hotspot"),
        ("covering_isodose_ratio_57gy", "57 Gy volume / PTV volume", 1.10, "Conformity"),
        ("worst_oar_goal_ratio", "Worst OAR value / limit", 1.0, "OAR goals"),
    )
    for case_id in case_ids:
        selected = [
            row for row in rows
            if row.get("episode_id", row["case_id"]) == case_id
        ]
        label = case_id.replace("prostate3d-", "").replace("tcia-Prostate-", "")
        for axis, (field, _, _, _) in zip(axes, fields, strict=True):
            axis.plot(
                [row["step"] for row in selected],
                [row[field] for row in selected],
                marker="o",
                linewidth=1.2,
                label=label,
            )
    for axis, (_, ylabel, limit, title) in zip(axes, fields, strict=True):
        axis.axhline(limit, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("Manual planning step")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=6, frameon=False)
    figure.suptitle("Prostate planning: approved clinical metrics after each manual change")
    save_figure_atomic(figure, path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_action_map(rows: list[dict], path: Path) -> None:
    case_ids = list(dict.fromkeys(row.get("episode_id", row["case_id"]) for row in rows))
    actions = [row for row in rows if int(row["step"]) > 0]
    maximum_step = max((int(row["step"]) for row in actions), default=1)
    values = np.zeros((len(case_ids), maximum_step), dtype=int)
    labels = np.full(values.shape, "", dtype=object)
    code = {
        "increase_target_priority": 1,
        "increase_hotspot_priority": 2,
        "increase_oar_priority": 3,
        "increase_normal_tissue_priority": 4,
        "create_ptv_minus_bladder": 5,
        "create_ptv_minus_rectum": 5,
    }
    names = {
        1: "PTV",
        2: "Hotspot",
        3: "OAR",
        4: "Normal tissue",
        5: "PTV-minus-OAR",
    }
    case_index = {case_id: index for index, case_id in enumerate(case_ids)}
    for row in actions:
        y = case_index[row.get("episode_id", row["case_id"])]
        x = int(row["step"]) - 1
        value = code[row["action_type"]]
        values[y, x] = value
        labels[y, x] = (
            str(row["action_structure"]).replace("_", " ").title()
            if row["action_type"] == "increase_oar_priority"
            else names[value]
        )
    figure, axis = plt.subplots(figsize=(9, max(4.0, 0.42 * len(case_ids))), constrained_layout=True)
    axis.imshow(
        values,
        aspect="auto",
        cmap=ListedColormap(
            ("#f2f2f2", "#4c78a8", "#f58518", "#54a24b", "#b279a2", "#eeca3b")
        ),
        vmin=0,
        vmax=5,
    )
    axis.set_xticks(np.arange(maximum_step), [f"Step {value}" for value in range(1, maximum_step + 1)])
    axis.set_yticks(np.arange(len(case_ids)), [value.replace("prostate3d-", "") for value in case_ids])
    axis.set_xlabel("Manual planning action")
    axis.set_ylabel("Planning case")
    axis.set_title("Allowed manual changes: target structure and clinical priorities")
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            axis.text(x, y, labels[y, x] or "None", ha="center", va="center", fontsize=8)
    save_figure_atomic(figure, path, dpi=180)
    plt.close(figure)


def save_starting_profile_outcomes(rows: list[dict], path: Path) -> None:
    """Show whether each fixed starting profile creates useful trajectories."""

    profiles = list(dict.fromkeys(str(row["starting_profile"]) for row in rows))
    episodes = list(dict.fromkeys(str(row["episode_id"]) for row in rows))
    initial_failure_rates = []
    final_acceptance_rates = []
    median_actions = []
    for profile in profiles:
        profile_episodes = [
            episode
            for episode in episodes
            if any(
                str(row["episode_id"]) == episode
                and str(row["starting_profile"]) == profile
                for row in rows
            )
        ]
        initial_rows = [
            next(
                row
                for row in rows
                if str(row["episode_id"]) == episode and int(row["step"]) == 0
            )
            for episode in profile_episodes
        ]
        final_rows = [
            max(
                (row for row in rows if str(row["episode_id"]) == episode),
                key=lambda value: int(value["step"]),
            )
            for episode in profile_episodes
        ]
        corrected_rows = [
            final_row
            for initial_row, final_row in zip(initial_rows, final_rows, strict=True)
            if not bool(initial_row["acceptable"]) and bool(final_row["acceptable"])
        ]
        initial_failure_rates.append(
            100.0 * np.mean([not bool(row["acceptable"]) for row in initial_rows])
        )
        final_acceptance_rates.append(
            100.0 * np.mean([bool(row["acceptable"]) for row in final_rows])
        )
        median_actions.append(
            0.0
            if not corrected_rows
            else float(np.median([int(row["step"]) for row in corrected_rows]))
        )
    all_initial_rows = [row for row in rows if int(row["step"]) == 0]
    all_final_rows = [
        max(
            (row for row in rows if str(row["episode_id"]) == episode),
            key=lambda value: int(value["step"]),
        )
        for episode in episodes
    ]
    all_corrected_rows = [
        final_row
        for initial_row, final_row in zip(
            all_initial_rows, all_final_rows, strict=True
        )
        if not bool(initial_row["acceptable"]) and bool(final_row["acceptable"])
    ]
    stress_episodes = [
        episode
        for episode in episodes
        if not any(
            str(row["episode_id"]) == episode
            and str(row["starting_profile"]) == "balanced_reference"
            for row in rows
        )
    ]
    stress_initial_rows = [
        row
        for row in all_initial_rows
        if str(row["starting_profile"]) != "balanced_reference"
    ]
    stress_final_rows = [
        max(
            (row for row in rows if str(row["episode_id"]) == episode),
            key=lambda value: int(value["step"]),
        )
        for episode in stress_episodes
    ]
    stress_initial_by_episode = {
        str(row["episode_id"]): row for row in stress_initial_rows
    }
    stress_corrected_rows = [
        row
        for row in stress_final_rows
        if not bool(stress_initial_by_episode[str(row["episode_id"])]["acceptable"])
        and bool(row["acceptable"])
    ]
    initial_failure_rates.insert(
        0, 100.0 * np.mean([not bool(row["acceptable"]) for row in all_initial_rows])
    )
    final_acceptance_rates.insert(
        0, 100.0 * np.mean([bool(row["acceptable"]) for row in all_final_rows])
    )
    median_actions.insert(
        0,
        0.0
        if not all_corrected_rows
        else float(np.median([int(row["step"]) for row in all_corrected_rows])),
    )
    initial_failure_rates.insert(
        1,
        100.0 * np.mean(
            [not bool(row["acceptable"]) for row in stress_initial_rows]
        ),
    )
    final_acceptance_rates.insert(
        1, 100.0 * np.mean([bool(row["acceptable"]) for row in stress_final_rows])
    )
    median_actions.insert(
        1,
        0.0
        if not stress_corrected_rows
        else float(np.median([int(row["step"]) for row in stress_corrected_rows])),
    )
    labels = [
        "ALL EPISODES",
        "STRESS PROFILES",
        *(value.replace("_", " ") for value in profiles),
    ]
    x = np.arange(len(labels))
    colors = ["#244a68", "#326d96", *(["#4c78a8"] * len(profiles))]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)
    axes[0].bar(x, initial_failure_rates, color=colors)
    axes[0].axhspan(40.0, 80.0, color="#54a24b", alpha=0.15)
    axes[0].set_ylabel("Initial plans outside acceptance (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Stress-profile information gate: 40-80%")
    axes[1].bar(
        x,
        final_acceptance_rates,
        color=["#2f6b2f", "#3b843b", *(["#54a24b"] * len(profiles))],
    )
    axes[1].axhline(90.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Final accepted plans (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Planning-success gate: at least 90%")
    axes[2].bar(
        x,
        median_actions,
        color=["#a85200", "#cb6805", *(["#f58518"] * len(profiles))],
    )
    axes[2].axhspan(1.0, 4.0, color="#54a24b", alpha=0.15)
    axes[2].set_ylabel("Median manual actions in corrected plans")
    axes[2].set_title("Corrected-plan trajectory gate: 1-4 actions")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)
    severity_pass = all(
        float(row["ptv_d1cc_gy"]) <= 70.0
        and float(row["covering_isodose_ratio_57gy"]) <= 1.25
        and float(row["worst_oar_goal_ratio"]) <= 1.50
        and float(row["ptv_v57gy_percent"]) >= 90.0
        for row in rows
        if int(row["step"]) == 0
    )
    figure.suptitle(
        "Fixed starting-profile calibration | "
        f"initial-severity limits: {'PASS' if severity_pass else 'FAIL'}"
    )
    save_figure_atomic(figure, path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_major_variation_tradeoffs(records: list[dict], path: Path) -> None:
    """Show the two states that require a physician choice in each episode."""

    if not records:
        return
    figure, axis = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
    for index, record in enumerate(records):
        x_values = (
            float(record["target_preserved_worst_oar_goal_ratio"]),
            float(record["oar_preserved_worst_oar_goal_ratio"]),
        )
        y_values = (
            float(record["target_preserved_ptv_v57gy_percent"]),
            float(record["oar_preserved_ptv_v57gy_percent"]),
        )
        axis.plot(x_values, y_values, color="#9aa0a6", linewidth=1, alpha=0.65)
        axis.scatter(x_values[0], y_values[0], color="#4c78a8", marker="o", s=38)
        axis.scatter(x_values[1], y_values[1], color="#f58518", marker="s", s=38)
        if index < 12:
            axis.annotate(
                str(record["episode_id"]).replace("tcia-Prostate-", ""),
                (x_values[1], y_values[1]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
    axis.axvline(1.0, color="black", linestyle="--", linewidth=1)
    axis.axhline(95.0, color="black", linestyle=":", linewidth=1)
    axis.axhline(99.0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Worst OAR value / institutional limit")
    axis.set_ylabel("PTV V57 Gy (%)")
    axis.set_title("Major variations: target-preserved and OAR-preserved states")
    axis.grid(alpha=0.2)
    axis.scatter([], [], color="#4c78a8", marker="o", label="Target-preserved state")
    axis.scatter([], [], color="#f58518", marker="s", label="OAR-preserved state")
    axis.legend(frameon=False)
    save_figure_atomic(figure, path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_cohort_anatomy(cases: list, records: list[dict], path: Path) -> None:
    """Show the selected contact anatomy before any dose optimization."""

    maximum_panels = min(len(cases), 12)
    columns = min(maximum_panels, 4)
    rows = int(np.ceil(maximum_panels / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.7 * rows), squeeze=False)
    color_map = ListedColormap(
        ("#202124", "#dadce0", "#e45756", "#b2182b", "#4c78a8", "#f2cf5b")
    )
    for axis, case, record in zip(axes.flat, cases[:maximum_panels], records[:maximum_panels], strict=False):
        structure = str(record["overlap_structure"])
        structure_index = case.structure_names.index(structure)
        oar = case.oars[structure_index]
        overlap = case.target & oar
        overlap_by_slice = overlap.sum(axis=(0, 1))
        slice_index = int(np.argmax(overlap_by_slice))
        if int(overlap_by_slice[slice_index]) == 0:
            slice_index = int(np.rint(np.argwhere(case.target).mean(axis=0))[2])
        category = np.zeros(case.body[:, :, slice_index].shape, dtype=np.uint8)
        category[case.body[:, :, slice_index]] = 1
        category[case.target[:, :, slice_index]] = 2
        if case.clinical_target is not None:
            category[case.clinical_target[:, :, slice_index]] = 3
        category[oar[:, :, slice_index]] = 4
        category[overlap[:, :, slice_index]] = 5
        axis.imshow(category.T, origin="lower", cmap=color_map, vmin=0, vmax=5, interpolation="nearest")
        add_hfs_orientation_labels(axis, "axial")
        case_label = case.case_id.replace("tcia-Prostate-AEC-", "AEC-")
        axis.set_title(
            f"{case_label} | {structure}\n"
            f"OAR overlap {100.0 * float(record['oar_overlap_fraction']):.1f}%; "
            f"PTV overlap {100.0 * float(record['ptv_overlap_fraction']):.1f}%",
            fontsize=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[maximum_panels:]:
        axis.axis("off")
    figure.suptitle(
        "Selected prostate anatomy before planning | head-first supine\n"
        "dark red: prostate/CTV; red: PTV margin; blue: selected OAR; yellow: overlap",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure_atomic(figure, path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plane(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return array[index, :, :].T
    if axis == 1:
        return array[:, index, :].T
    return array[:, :, index].T


def save_representative_review(
    case,
    trajectory,
    config,
    path: Path,
    beam_angles_degrees: tuple[float, ...] | None = None,
) -> None:
    initial = trajectory.steps[0].plan
    final = trajectory.final.plan
    dose_gy = final.dose.detach().float().cpu().numpy() * PRESCRIPTION_GY
    target_center = np.rint(np.argwhere(case.target).mean(axis=0)).astype(int)
    figure, axes = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)
    view_names = ("Sagittal", "Coronal", "Axial")
    for axis_index, (axis, view_name, slice_index) in enumerate(
        zip(axes[0], view_names, target_center, strict=True)
    ):
        image = axis.imshow(plane(dose_gy, axis_index, int(slice_index)), origin="lower", cmap="turbo", vmin=0, vmax=66)
        axis.contour(plane(case.target, axis_index, int(slice_index)), levels=[0.5], colors="white", linewidths=1.6)
        if case.clinical_target is not None:
            axis.contour(
                plane(case.clinical_target, axis_index, int(slice_index)),
                levels=[0.5],
                colors="#ff66cc",
                linewidths=1.0,
            )
        for oar, color in zip(case.oars[:2], ("#3399ff", "#66ff66"), strict=True):
            axis.contour(plane(oar, axis_index, int(slice_index)), levels=[0.5], colors=color, linewidths=1.0)
        if (
            final.optimization_target is not None
            and np.any(final.optimization_target.relaxed_overlap_mask)
        ):
            axis.contour(
                plane(
                    final.optimization_target.relaxed_overlap_mask,
                    axis_index,
                    int(slice_index),
                ),
                levels=[0.5],
                colors="#ff9d00",
                linewidths=2.0,
            )
        for level, color in ((57.0, "cyan"), (60.0, "yellow"), (63.0, "red")):
            axis.contour(plane(dose_gy, axis_index, int(slice_index)), levels=[level], colors=color, linewidths=1.2)
        axis.set_title(f"{view_name} dose")
        add_hfs_orientation_labels(axis, view_name)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(image, ax=axes[0, :], label="Dose (Gy)", shrink=0.8)

    bins = np.linspace(0.0, 72.0, 217)
    colors = {
        "Prostate": "#ff66cc",
        "PTV": "#d62728",
        "PTV-minus-OAR": "#111111",
        "PTV-OAR overlap": "#ff9d00",
        "bladder": "#1f77b4",
        "rectum": "#2ca02c",
        "femur head L": "#9467bd",
        "femur head R": "#c49cde",
    }
    evaluation_masks = prostate_evaluation_masks(case)
    masks = {
        "Prostate": case.clinical_target,
        "PTV": case.target,
        "bladder": evaluation_masks["bladder"],
        "rectum": evaluation_masks["rectum"],
        "femur head L": evaluation_masks["femur_head_l"],
        "femur head R": evaluation_masks["femur_head_r"],
    }
    if (
        final.optimization_target is not None
        and final.optimization_target.cropped_oar_indices
    ):
        masks["PTV-minus-OAR"] = final.optimization_target.coverage_mask
        masks["PTV-OAR overlap"] = final.optimization_target.relaxed_overlap_mask
    for label, plan, linestyle in (("Initial", initial, "--"), ("Final", final, "-")):
        values = plan.dose.detach().float().cpu().numpy() * PRESCRIPTION_GY
        for structure, mask in masks.items():
            axes[1, 0].plot(
                bins,
                cumulative_dvh(values[mask], bins),
                color=colors[structure],
                linestyle=linestyle,
                linewidth=1.7,
                label=f"{structure}: {label}",
            )
    axes[1, 0].axvline(57.0, color="cyan", linewidth=1)
    axes[1, 0].axvline(60.0, color="gold", linewidth=1)
    axes[1, 0].axvline(63.0, color="red", linewidth=1)
    axes[1, 0].set_xlim(0, 72)
    axes[1, 0].set_ylim(0, 102)
    axes[1, 0].set_xlabel("Dose (Gy)")
    axes[1, 0].set_ylabel("Volume receiving at least dose (%)")
    axes[1, 0].set_title("Cumulative DVH")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(frameon=False, fontsize=7, ncol=2)

    metrics = final.metrics
    clinical = clinical_constraint_record(case, final.dose)
    pass_text = "PASS" if is_acceptable_3d(metrics, case, config) else "FAIL"
    metric_text = "Clinical objective review\n\n"
    metric_text += (
        f"Prostate V60: {float(clinical['prostate_v60gy_percent']):5.1f}%  >=99%  "
        f"{'PASS' if clinical['prostate_v60gy_pass'] else 'FAIL'}\n"
        f"PTV D99:      {float(clinical['ptv_d99_gy']):5.2f} Gy >=57 Gy "
        f"{'PASS' if clinical['ptv_d99_pass'] else 'FAIL'}\n"
        f"PTV V57:      {float(clinical['ptv_v57gy_percent']):5.1f}%  >=95%  "
        f"{'PASS' if clinical['ptv_v57gy_acceptable_variation_pass'] else 'FAIL'}\n"
        f"PTV D1cc:     {float(clinical['ptv_d1cc_gy']):5.2f} Gy <=63 Gy "
        f"{'PASS' if clinical['ptv_d1cc_pass'] else 'FAIL'}\n"
        f"Rectum V37:   {float(clinical['rectum_v37gy_percent']):5.1f}%  <=50%  "
        f"{'PASS' if clinical['rectum_v37gy_pass'] else 'FAIL'}\n"
        f"Rectum V46:   {float(clinical['rectum_v46gy_percent']):5.1f}%  <=30%  "
        f"{'PASS' if clinical['rectum_v46gy_pass'] else 'FAIL'}\n"
        f"Bladder V37:  {float(clinical['bladder_v37gy_percent']):5.1f}%  <=50%  "
        f"{'PASS' if clinical['bladder_v37gy_pass'] else 'FAIL'}\n"
        f"Bladder V46:  {float(clinical['bladder_v46gy_percent']):5.1f}%  <=30%  "
        f"{'PASS' if clinical['bladder_v46gy_pass'] else 'FAIL'}\n"
        f"Femur L V43:  {float(clinical['femur_head_l_v43gy_percent']):5.1f}%  <=5%   "
        f"{'PASS' if clinical['femur_head_l_v43gy_pass'] else 'FAIL'}\n"
        f"Femur R V43:  {float(clinical['femur_head_r_v43gy_percent']):5.1f}%  <=5%   "
        f"{'PASS' if clinical['femur_head_r_v43gy_pass'] else 'FAIL'}\n"
        f"Conformity:   {metrics.covering_isodose_ratio_95:5.3f}   <=1.10  "
        f"{'PASS' if metrics.covering_isodose_ratio_95 <= 1.10 else 'FAIL'}\n\n"
        f"Overall clinical review: {pass_text}\n"
        f"Class: {clinical['protocol_acceptance_class']}"
    )
    if metrics.relaxed_overlap_d98 is not None:
        metric_text += (
            f"\nOverlap D98: {metrics.relaxed_overlap_d98 * PRESCRIPTION_GY:.2f} Gy; "
            "no local overlap dose floor"
        )
    axes[1, 1].axis("off")
    axes[1, 1].text(0.02, 0.98, metric_text, va="top", family="monospace", fontsize=8.8)

    actions = [step.action.description for step in trajectory.steps if step.action]
    review = short_review_record(case, final, config)
    review_decision = review["review_decision"]
    if (
        clinical["protocol_acceptance_class"] == "major_variation"
        and trajectory.stopping_reason.startswith("requires_physician_review")
    ):
        tradeoff = tradeoff_state_record(case, trajectory, config)
        review_decision = (
            "Major variation. Do not accept automatically. "
            f"Target-preserved state: step {tradeoff['target_preserved_step']}, "
            f"PTV V57 Gy {tradeoff['target_preserved_ptv_v57gy_percent']:.1f}%, "
            f"worst OAR ratio {tradeoff['target_preserved_worst_oar_goal_ratio']:.2f}. "
            f"OAR-preserved state: step {tradeoff['oar_preserved_step']}, "
            f"PTV V57 Gy {tradeoff['oar_preserved_ptv_v57gy_percent']:.1f}%, "
            f"worst OAR ratio {tradeoff['oar_preserved_worst_oar_goal_ratio']:.2f}. "
            "Record a physician choice and its reason."
        )
    action_text = "Manual review sequence\n\n" + (
        "\n".join(
            textwrap.fill(f"{index}. {value}", width=48, subsequent_indent="   ")
            for index, value in enumerate(actions, start=1)
        )
        if actions
        else (
            "No manual change was required."
            if trajectory.stopping_reason == "acceptable"
            else "No manual change was applied."
        )
    )
    action_text += "\n\nShort review\n"
    action_text += textwrap.fill(review_decision, width=48)
    action_text += f"\n\n{len(final.active_beams)} fixed coplanar fields"
    if beam_angles_degrees is not None:
        angle_text = ", ".join(f"{value:.1f}" for value in beam_angles_degrees)
        action_text += "\nAngles: " + textwrap.fill(angle_text, width=42, subsequent_indent="        ")
    axes[1, 2].axis("off")
    axes[1, 2].text(0.02, 0.98, action_text, va="top", fontsize=8.8, wrap=True)
    figure.suptitle(
        f"Review plan: {case.case_id} | HFS | magenta CTV; white PTV; "
        "orange relaxed overlap; cyan 57 Gy; yellow 60 Gy; red 63 Gy",
        fontsize=14,
    )
    save_figure_atomic(
        figure,
        path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(figure)


CLINICAL_CONFIG = HighLevelSearchConfig3D(
    max_steps=8,
    beam_width=1,
    add_candidates=0,
    remove_candidates=0,
    shift_candidates=0,
    optimizer_iterations=1000,
    optimizer_learning_rate=0.02,
    priority_factor=3.0,
    priority_ceiling=7.59375,
    priority_floor=1.0,
    d95_min=0.0,
    d98_min=0.0,
    d99_min=0.0,
    d50_min=0.0,
    d50_max=float("inf"),
    d02_max=float("inf"),
    d1cc_max=float("inf"),
    clinical_target_v100_min=0.0,
    target_hotspot_threshold=1.00,
    target_hotspot_weight=50.0,
    initial_field_count=7,
    minimum_field_count=7,
    normal_tissue_weight=50.0,
    normal_tissue_threshold=0.5,
    integral_dose_weight=2.0,
    high_dose_normal_tissue_weight=150.0,
    high_dose_normal_tissue_threshold=0.95,
    clinical_dvh_weight=2.0,
    target_normalization_d50=None,
    clinical_target_normalization_d99=1.0,
    target_normalization_interval=100,
    prostate_protocol_tier="variation_acceptable",
    paddick_ci_95_min=0.0,
    covering_isodose_ratio_95_max=1.10,
    r50_max=float("inf"),
    manual_ptv_oar_crop=True,
    ptv_oar_overlap_minimum=0.0,
    overlap_floor_is_acceptance=False,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clinical prostate manual planning pilot")
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/prostate300_local/merged"))
    parser.add_argument(
        "--anatomy-source",
        choices=("parametric", "tcia"),
        default="parametric",
    )
    parser.add_argument("--tcia-root", type=Path, default=Path("data/tcia"))
    parser.add_argument("--tcia-subjects", nargs="+")
    parser.add_argument(
        "--tcia-subject-file",
        type=Path,
        help="Text file with one TCIA patient directory name per line",
    )
    parser.add_argument(
        "--tcia-episode-manifest",
        type=Path,
        help=(
            "CSV with one patient_id and starting_profile per row. The manifest "
            "sets the TCIA cohort, profile assignment, order, and case count."
        ),
    )
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--selection-mode",
        choices=("validation_hard", "oar_stress"),
        default="validation_hard",
    )
    parser.add_argument("--seed-start", type=int, default=200000)
    parser.add_argument("--maximum-anatomy-attempts", type=int, default=50000)
    parser.add_argument("--coarse-grid-size", type=int, default=48)
    parser.add_argument(
        "--stress-structure",
        choices=("balanced", "bladder", "rectum"),
        default="balanced",
    )
    parser.add_argument(
        "--minimum-bladder-overlap-fraction",
        "--minimum-oar-overlap-fraction",
        dest="minimum_bladder_overlap_fraction",
        type=float,
        default=0.135,
    )
    parser.add_argument("--minimum-rectum-overlap-fraction", type=float, default=0.060)
    parser.add_argument(
        "--minimum-bladder-ptv-overlap-fraction",
        "--minimum-ptv-overlap-fraction",
        dest="minimum_bladder_ptv_overlap_fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument("--minimum-rectum-ptv-overlap-fraction", type=float, default=0.025)
    parser.add_argument("--maximum-ptv-overlap-fraction", type=float, default=0.20)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--fluence-size", type=int, default=64)
    parser.add_argument(
        "--delivery-mode",
        choices=("static_7", "static_9", "static_12", "arc_like_360"),
        default="static_7",
    )
    parser.add_argument("--iterations", type=int, default=CLINICAL_CONFIG.optimizer_iterations)
    parser.add_argument("--learning-rate", type=float, default=CLINICAL_CONFIG.optimizer_learning_rate)
    parser.add_argument(
        "--hotspot-objective-gy",
        type=float,
        default=CLINICAL_CONFIG.target_hotspot_threshold * PRESCRIPTION_GY,
        help="Automated PTV D1cc planning objective; the evaluator remains 63 Gy",
    )
    parser.add_argument(
        "--hotspot-objective-weight",
        type=float,
        default=CLINICAL_CONFIG.target_hotspot_weight,
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--initial-target-priority", type=float, default=1.0)
    parser.add_argument("--initial-hotspot-priority", type=float, default=1.0)
    parser.add_argument("--initial-oar-priority", type=float, default=1.0)
    parser.add_argument("--initial-normal-tissue-priority", type=float, default=1.0)
    parser.add_argument(
        "--starting-profiles",
        nargs="+",
        choices=tuple(STARTING_PROFILE_VALUES),
        help=(
            "Run each anatomy from the selected fixed profiles. These values override "
            "the four custom initial-priority arguments."
        ),
    )
    parser.add_argument(
        "--ptv-oar-overlap-minimum",
        type=float,
        default=CLINICAL_CONFIG.ptv_oar_overlap_minimum,
        help=(
            "Relative-dose floor for a manually separated PTV-OAR overlap; "
            "the clinical default is zero and PTV V57 Gy >=95%% controls total undercoverage"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    initial_priority_values = (
        args.initial_target_priority,
        args.initial_hotspot_priority,
        args.initial_oar_priority,
        args.initial_normal_tissue_priority,
    )
    if any(value <= 0.0 for value in initial_priority_values):
        raise ValueError("Initial priorities must be positive")
    if args.cases <= 0 or args.maximum_anatomy_attempts <= 0:
        raise ValueError("Case and anatomy-attempt counts must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("Learning rate must be positive")
    if args.hotspot_objective_gy <= 0.0 or args.hotspot_objective_weight <= 0.0:
        raise ValueError("Hotspot objective dose and weight must be positive")
    if args.coarse_grid_size < 16 or args.grid_size < 16:
        raise ValueError("Grid sizes must be at least 16")
    if not 0.0 <= args.ptv_oar_overlap_minimum <= 1.0:
        raise ValueError("PTV-OAR overlap minimum must be in [0, 1]")
    if not (
        0.0 <= args.minimum_bladder_overlap_fraction <= 1.0
        and 0.0 <= args.minimum_rectum_overlap_fraction <= 1.0
        and 0.0
        <= args.minimum_bladder_ptv_overlap_fraction
        <= args.maximum_ptv_overlap_fraction
        <= 1.0
        and 0.0
        <= args.minimum_rectum_ptv_overlap_fraction
        <= args.maximum_ptv_overlap_fraction
    ):
        raise ValueError("Anatomy overlap fractions must be ordered values from 0 to 1")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output_dir = args.output_dir or Path(
        "outputs"
    ) / f"prostate_ptv_manual_pilot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "status.html").write_text(STATUS_PAGE, encoding="utf-8")
    episode_assignments = None
    if args.tcia_episode_manifest:
        if args.anatomy_source != "tcia":
            raise ValueError("--tcia-episode-manifest requires --anatomy-source tcia")
        if args.tcia_subjects or args.tcia_subject_file:
            raise ValueError(
                "Use --tcia-episode-manifest without --tcia-subjects or --tcia-subject-file"
            )
        if args.starting_profiles:
            raise ValueError(
                "The episode manifest supplies starting profiles; do not use --starting-profiles"
            )
        episode_assignments = load_tcia_episode_manifest(args.tcia_episode_manifest)
        profile_names = tuple(
            dict.fromkeys(row["starting_profile"] for row in episode_assignments)
        )
    else:
        profile_names = tuple(args.starting_profiles or ("custom",))
    anatomy_attempts = 0
    selected_cases = None
    if args.anatomy_source == "tcia":
        if args.seeds:
            raise ValueError("--seeds cannot be used with --anatomy-source tcia")
        if args.tcia_subjects and args.tcia_subject_file:
            raise ValueError("Use either --tcia-subjects or --tcia-subject-file, not both")
        if episode_assignments is not None:
            subject_names = [row["patient_id"] for row in episode_assignments]
            subject_dirs = [args.tcia_root / value for value in subject_names]
        elif args.tcia_subject_file:
            subject_names = [
                value.strip()
                for value in args.tcia_subject_file.read_text(encoding="utf-8").splitlines()
                if value.strip()
            ]
            subject_dirs = [args.tcia_root / value for value in subject_names]
        elif args.tcia_subjects:
            subject_dirs = [args.tcia_root / value for value in args.tcia_subjects]
        else:
            subject_dirs = sorted(
                path
                for path in args.tcia_root.glob("Prostate-AEC-*")
                if path.is_dir()
            )[: args.cases]
        requested_cases = len(subject_dirs) if episode_assignments is not None else args.cases
        if len(subject_dirs) < requested_cases:
            raise ValueError(
                f"Requested {requested_cases} TCIA cases, but found {len(subject_dirs)} subject directories"
            )
        subject_dirs = subject_dirs[:requested_cases]
        missing_dirs = [str(path) for path in subject_dirs if not path.is_dir()]
        if missing_dirs:
            raise ValueError(f"TCIA subject directories are missing: {missing_dirs}")
        anatomy_started = time.perf_counter()
        write_progress(
            output_dir,
            0,
            requested_cases,
            anatomy_started,
            unit="TCIA anatomies loaded",
        )
        selected_cases = []
        for index, path in enumerate(subject_dirs, start=1):
            selected_cases.append(load_tcia_prostate_case(path, args.grid_size))
            write_progress(
                output_dir,
                index,
                requested_cases,
                anatomy_started,
                last_case=path.name,
                unit="TCIA anatomies loaded",
            )
        records = []
        for index, (case, path) in enumerate(
            zip(selected_cases, subject_dirs, strict=True)
        ):
            record = tcia_anatomy_record(case, path)
            if episode_assignments is not None:
                assignment = episode_assignments[index]
                record.update(
                    {
                        "anatomy_stratum": assignment.get("anatomy_stratum"),
                        "assigned_starting_profile": assignment["starting_profile"],
                        "locked_assignment_order": assignment.get("assignment_order"),
                    }
                )
            records.append(record)
        anatomy_attempts = len(subject_dirs)
    elif args.seeds:
        records = []
        for seed in args.seeds:
            case = generate_prostate_case_3d(seed, args.grid_size, difficulty="hard")
            records.append(
                {
                    "seed": seed,
                    "difficulty": "hard",
                    "selection_mode": "explicit_seed",
                    **anatomy_overlap_record(case),
                }
            )
    elif args.selection_mode == "oar_stress":
        records, anatomy_attempts = select_oar_stress_records(
            args.cases,
            args.seed_start,
            args.maximum_anatomy_attempts,
            args.grid_size,
            args.coarse_grid_size,
            args.stress_structure,
            args.minimum_bladder_overlap_fraction,
            args.minimum_rectum_overlap_fraction,
            args.minimum_bladder_ptv_overlap_fraction,
            args.minimum_rectum_ptv_overlap_fraction,
            args.maximum_ptv_overlap_fraction,
        )
    else:
        records = [
            row
            for row in load_records(args.dataset_dir / "trajectory_view.jsonl")
            if row["split"] == "validation" and row["difficulty"] == "hard"
        ][: args.cases]
        if len(records) < args.cases:
            raise ValueError(f"Requested {args.cases} hard validation cases, but found {len(records)}")
        records = [
            {
                **record,
                "selection_mode": "validation_hard",
                **anatomy_overlap_record(
                    generate_prostate_case_3d(int(record["seed"]), args.grid_size, difficulty="hard")
                ),
            }
            for record in records
        ]
    with (output_dir / "cohort_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    if selected_cases is None:
        selected_cases = [
            generate_prostate_case_3d(int(record["seed"]), args.grid_size, difficulty="hard")
            for record in records
        ]
    if episode_assignments is not None:
        episodes = [
            (record, case, assignment["starting_profile"])
            for record, case, assignment in zip(
                records,
                selected_cases,
                episode_assignments,
                strict=True,
            )
        ]
    else:
        episodes = [
            (record, case, profile_name)
            for record, case in zip(records, selected_cases, strict=True)
            for profile_name in profile_names
        ]
    with (output_dir / "episode_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record, case, profile_name in episodes:
            handle.write(
                json.dumps(
                    {
                        "episode_id": f"{case.case_id}__{profile_name}",
                        "case_id": case.case_id,
                        "starting_profile": profile_name,
                        "patient_id": record.get("patient_id"),
                        "anatomy_stratum": record.get("anatomy_stratum"),
                        "locked_assignment_order": record.get("locked_assignment_order"),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    save_cohort_anatomy(selected_cases, records, output_dir / "00_selected_anatomy.png")
    config = replace(
        CLINICAL_CONFIG,
        optimizer_iterations=args.iterations,
        optimizer_learning_rate=args.learning_rate,
        target_hotspot_threshold=args.hotspot_objective_gy / PRESCRIPTION_GY,
        target_hotspot_weight=args.hotspot_objective_weight,
        max_steps=args.max_steps,
        ptv_oar_overlap_minimum=args.ptv_oar_overlap_minimum,
    )
    initial_priority_record = {
        "initial_target_priority": args.initial_target_priority,
        "initial_hotspot_priority": args.initial_hotspot_priority,
        "initial_oar_priority": args.initial_oar_priority,
        "initial_normal_tissue_priority": args.initial_normal_tissue_priority,
    }
    (output_dir / "criteria.json").write_text(
        json.dumps(
            {
                **vars(config),
                **initial_priority_record,
                "starting_profiles": {
                    name: STARTING_PROFILE_VALUES[name]
                    for name in profile_names
                    if name != "custom"
                },
                "material_response_thresholds": {
                    "target_v57_percentage_points": 1.0,
                    "oar_volume_percentage_points": "max(1.0, 10% of current excess)",
                    "ptv_d1cc_gy": 0.3,
                    "covering_isodose_ratio": 0.01,
                    "repeated_nonresponse_limit": 2,
                },
                "initial_state_development_severity_bounds": {
                    "ptv_d1cc_gy_max": 70.0,
                    "covering_isodose_ratio_57gy_max": 1.25,
                    "worst_oar_goal_ratio_max": 1.50,
                    "ptv_v57gy_percent_min": 90.0,
                },
                "clinical_objective_set": clinical_objective_set_record(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    mode = delivery_mode_3d(args.delivery_mode)
    started = time.perf_counter()
    write_progress(output_dir, 0, len(episodes), started, unit="planning episodes")
    rows: list[dict] = []
    outcomes: list[dict[str, float | int | str | bool | None]] = []
    major_variation_reviews: list[dict] = []
    representative_case = None
    representative_trajectory = None
    review_dir = output_dir / "review_plans"
    review_dir.mkdir(exist_ok=True)
    previous_case = None
    engine = None
    custom_priority_values = (
        args.initial_target_priority,
        args.initial_hotspot_priority,
        args.initial_oar_priority,
        args.initial_normal_tissue_priority,
    )
    for index, (record, case, profile_name) in enumerate(episodes, start=1):
        if case is not previous_case:
            engine = TorchImplicitDoseEngine3D(
                case,
                mode.angles_degrees,
                args.fluence_size,
                device=device,
                dtype=torch.float32,
            )
            previous_case = case
        initial_priorities = starting_priorities(
            case,
            profile_name,
            custom_priority_values,
        )
        episode_id = f"{case.case_id}__{profile_name}"
        def on_step(partial_trajectory: PlanningTrajectory3D) -> None:
            preview_rows = rows + [
                step_row(
                    case,
                    partial_trajectory,
                    step,
                    config,
                    episode_id,
                    profile_name,
                )
                for step in partial_trajectory.steps
            ]
            progress_step = (index - 1) * (config.max_steps + 1) + partial_trajectory.final.step + 1
            progress_steps_total = len(episodes) * (config.max_steps + 1)
            if len(episodes) <= 12 or should_update_figures(
                progress_step,
                progress_steps_total,
                maximum_updates=50,
            ):
                save_trajectory_plot(preview_rows, output_dir / "01_ptv_clinical_trajectory.png")
                save_action_map(preview_rows, output_dir / "02_manual_action_map.png")
                save_representative_review(
                    case,
                    partial_trajectory,
                    config,
                    output_dir / "03_representative_ptv_review.png",
                    mode.angles_degrees,
                )
            fraction = min((partial_trajectory.final.step + 1) / (config.max_steps + 1), 0.95)
            write_progress(
                output_dir,
                (index - 1) + fraction,
                len(episodes),
                started,
                last_case=(
                    f"{case.case_id}, {profile_name.replace('_', ' ')}, "
                    f"manual step {partial_trajectory.final.step}"
                ),
                last_target_priority=partial_trajectory.final.plan.priorities.target,
                unit="planning-episode equivalents",
            )

        trajectory = run_manual_sequence(
            case,
            engine,
            mode.active_beams,
            config,
            initial_priorities=initial_priorities,
            on_step=on_step,
        )
        rows.extend(
            step_row(
                case,
                trajectory,
                step,
                config,
                episode_id,
                profile_name,
            )
            for step in trajectory.steps
        )
        write_trajectory_rows(rows, output_dir / "trajectory_steps.csv")
        save_representative_review(
            case,
            trajectory,
            config,
            review_dir / f"{episode_id}.png",
            mode.angles_degrees,
        )
        final_metrics = trajectory.final.plan.metrics
        final_clinical = clinical_constraint_record(case, trajectory.final.plan.dose)
        unproductive_steps = repeated_unproductive_steps(case, trajectory)
        disposition = terminal_disposition(trajectory, case, config)
        outcomes.append(
            {
                "case_id": case.case_id,
                "episode_id": episode_id,
                "starting_profile": profile_name,
                "anatomy_stratum": record.get("anatomy_stratum"),
                "acceptable": trajectory.stopping_reason == "acceptable",
                "initial_acceptable": is_acceptable_3d(
                    trajectory.steps[0].plan.metrics,
                    case,
                    config,
                ),
                "terminal_disposition": disposition,
                "stopping_reason": trajectory.stopping_reason,
                "manual_changes": len(trajectory.steps) - 1,
                "repeated_unproductive_actions": len(unproductive_steps),
                "expert_demonstration_eligible": (
                    trajectory.stopping_reason == "acceptable"
                    and not unproductive_steps
                ),
                **final_clinical,
                "ptv_d98_gy": final_metrics.target_d98 * PRESCRIPTION_GY,
                "clinical_target_d98_gy": clinical_target_d98_gy(
                    case, trajectory.final.plan.dose
                ),
                "ptv_d50_gy": final_metrics.target_d50 * PRESCRIPTION_GY,
                "ptv_d02_gy": final_metrics.target_d02 * PRESCRIPTION_GY,
                "covering_isodose_ratio_57gy": final_metrics.covering_isodose_ratio_95,
                "paddick_ci_57gy": final_metrics.paddick_ci_95,
                "worst_oar_goal_ratio": worst_oar_ratio(final_metrics, config)[1],
            }
        )
        if disposition == "requires_physician_review":
            tradeoff = {
                "case_id": case.case_id,
                "episode_id": episode_id,
                "starting_profile": profile_name,
                "stopping_reason": trajectory.stopping_reason,
                **tradeoff_state_record(case, trajectory, config),
            }
            major_variation_reviews.append(tradeoff)
            write_trajectory_rows(
                major_variation_reviews,
                output_dir / "major_variation_reviews.csv",
            )
        if (
            representative_trajectory is None
            or len(trajectory.steps) > len(representative_trajectory.steps)
        ):
            representative_case = case
            representative_trajectory = trajectory
        if len(episodes) <= 12 or index == len(episodes):
            save_trajectory_plot(rows, output_dir / "01_ptv_clinical_trajectory.png")
            save_action_map(rows, output_dir / "02_manual_action_map.png")
            save_representative_review(
                representative_case,
                representative_trajectory,
                config,
                output_dir / "03_representative_ptv_review.png",
                mode.angles_degrees,
            )
        save_starting_profile_outcomes(
            rows,
            output_dir / "04_starting_profile_outcomes.png",
        )
        save_major_variation_tradeoffs(
            major_variation_reviews,
            output_dir / "05_major_variation_tradeoffs.png",
        )
        write_progress(
            output_dir,
            index,
            len(episodes),
            started,
            last_case=f"{case.case_id}, {profile_name.replace('_', ' ')}",
            last_target_priority=trajectory.final.plan.priorities.target,
            unit="planning episodes",
        )
        print(
            f"[{index}/{len(episodes)}] {episode_id}: {trajectory.stopping_reason}; "
            f"{len(trajectory.steps) - 1} manual changes",
            flush=True,
        )

    write_trajectory_rows(rows, output_dir / "trajectory_steps.csv")
    initial_rows = [row for row in rows if int(row["step"]) == 0]
    final_rows = [row for row in rows if row["stopping_reason"]]
    initial_failure_rate = float(
        np.mean([not bool(row["acceptable"]) for row in initial_rows])
    )
    stress_initial_rows = [
        row
        for row in initial_rows
        if str(row["starting_profile"]) != "balanced_reference"
    ]
    balanced_initial_rows = [
        row
        for row in initial_rows
        if str(row["starting_profile"]) == "balanced_reference"
    ]
    stress_initial_failure_rate = float(
        np.mean([not bool(row["acceptable"]) for row in stress_initial_rows])
    )
    balanced_control_initial_acceptance_rate = float(
        np.mean([bool(row["acceptable"]) for row in balanced_initial_rows])
    )
    final_acceptance_rate = float(
        np.mean([bool(value["acceptable"]) for value in outcomes])
    )
    corrected_outcomes = [
        value
        for value in outcomes
        if not bool(value["initial_acceptable"]) and bool(value["acceptable"])
    ]
    median_manual_changes = (
        0.0
        if not corrected_outcomes
        else float(
            np.median(
                [int(value["manual_changes"]) for value in corrected_outcomes]
            )
        )
    )
    represented_action_classes = sorted(
        {
            str(row["action_type"])
            for row in rows
            if int(row["step"]) > 0
        }
    )
    initial_states_within_severity_bounds = all(
        float(row["ptv_d1cc_gy"]) <= 70.0
        and float(row["covering_isodose_ratio_57gy"]) <= 1.25
        and float(row["worst_oar_goal_ratio"]) <= 1.50
        and float(row["ptv_v57gy_percent"]) >= 90.0
        for row in initial_rows
    )
    profile_summaries = {}
    for profile_name in profile_names:
        profile_outcomes = [
            value for value in outcomes if value["starting_profile"] == profile_name
        ]
        profile_initial = [
            row for row in initial_rows if row["starting_profile"] == profile_name
        ]
        profile_corrected = [
            value
            for value in profile_outcomes
            if not bool(value["initial_acceptable"]) and bool(value["acceptable"])
        ]
        profile_summaries[profile_name] = {
            "episodes": len(profile_outcomes),
            "initial_failure_rate": float(
                np.mean([not bool(row["acceptable"]) for row in profile_initial])
            ),
            "final_acceptance_rate": float(
                np.mean([bool(value["acceptable"]) for value in profile_outcomes])
            ),
            "median_manual_changes": float(
                0.0
                if not profile_corrected
                else np.median(
                    [int(value["manual_changes"]) for value in profile_corrected]
                )
            ),
            "corrected_episodes": len(profile_corrected),
        }
    calibration_gates = {
        "stress_profile_initial_failure_rate_40_to_80_percent": (
            0.40 <= stress_initial_failure_rate <= 0.80
        ),
        "balanced_control_initial_acceptance_at_least_90_percent": (
            balanced_control_initial_acceptance_rate >= 0.90
        ),
        "final_acceptance_rate_at_least_90_percent": final_acceptance_rate >= 0.90,
        "median_manual_actions_1_to_4": 1.0 <= median_manual_changes <= 4.0,
        "more_than_one_action_class": len(represented_action_classes) > 1,
        "initial_states_within_development_severity_bounds": (
            initial_states_within_severity_bounds
        ),
    }
    summary = {
        "status": "complete",
        "experiment": "Clinical prostate manual planning pilot",
        "clinical_objective_set": clinical_objective_set_record(),
        "anatomy_source": args.anatomy_source,
        "cases": len(outcomes),
        "unique_anatomies": len(selected_cases),
        "planning_episodes": len(outcomes),
        "starting_profiles": profile_names,
        "acceptable_cases": sum(bool(value["acceptable"]) for value in outcomes),
        "initial_failure_rate": initial_failure_rate,
        "stress_profile_initial_failure_rate": stress_initial_failure_rate,
        "balanced_control_initial_acceptance_rate": (
            balanced_control_initial_acceptance_rate
        ),
        "final_acceptance_rate": final_acceptance_rate,
        "median_manual_changes": median_manual_changes,
        "represented_action_classes": represented_action_classes,
        "initial_state_severity": {
            "maximum_ptv_d1cc_gy": float(
                max(float(row["ptv_d1cc_gy"]) for row in initial_rows)
            ),
            "maximum_covering_isodose_ratio_57gy": float(
                max(float(row["covering_isodose_ratio_57gy"]) for row in initial_rows)
            ),
            "maximum_worst_oar_goal_ratio": float(
                max(float(row["worst_oar_goal_ratio"]) for row in initial_rows)
            ),
            "minimum_ptv_v57gy_percent": float(
                min(float(row["ptv_v57gy_percent"]) for row in initial_rows)
            ),
        },
        "calibration_gates": calibration_gates,
        "profile_summaries": profile_summaries,
        "terminal_dispositions": {
            label: sum(value["terminal_disposition"] == label for value in outcomes)
            for label in sorted(
                {str(value["terminal_disposition"]) for value in outcomes}
            )
        },
        "major_variations_requiring_physician_review": len(major_variation_reviews),
        "expert_demonstration_eligible_episodes": sum(
            bool(value["expert_demonstration_eligible"]) for value in outcomes
        ),
        "acceptance_classes": {
            label: sum(
                bool(value["acceptable"])
                and value["protocol_acceptance_class"] == label
                for value in outcomes
            )
            for label in (
                "per_protocol",
                "acceptable_target_coverage_variation",
                "acceptable_oar_variation",
            )
        },
        "manual_changes_per_case": [int(value["manual_changes"]) for value in outcomes],
        "stopping_reasons": [str(value["stopping_reason"]) for value in outcomes],
        "anatomy_strata": summarize_anatomy_strata(outcomes),
        "final_median_ptv_d98_gy": float(np.median([float(value["ptv_d98_gy"]) for value in outcomes])),
        "final_median_prostate_v60gy_percent": float(
            np.median([float(value["prostate_v60gy_percent"]) for value in outcomes])
        ),
        "final_median_ptv_d99_gy": float(
            np.median([float(value["ptv_d99_gy"]) for value in outcomes])
        ),
        "final_median_ptv_v57gy_percent": float(
            np.median([float(value["ptv_v57gy_percent"]) for value in outcomes])
        ),
        "final_median_ptv_d1cc_gy": float(
            np.median([float(value["ptv_d1cc_gy"]) for value in outcomes])
        ),
        "final_median_clinical_target_d98_gy": float(
            np.nanmedian([float(value["clinical_target_d98_gy"]) for value in outcomes])
        ),
        "final_median_ptv_d50_gy": float(np.median([float(value["ptv_d50_gy"]) for value in outcomes])),
        "final_median_ptv_d02_gy": float(np.median([float(value["ptv_d02_gy"]) for value in outcomes])),
        "final_median_covering_isodose_ratio_57gy": float(np.median([float(value["covering_isodose_ratio_57gy"]) for value in outcomes])),
        "final_median_paddick_ci_57gy": float(np.median([float(value["paddick_ci_57gy"]) for value in outcomes])),
        "final_median_worst_oar_goal_ratio": float(
            np.median([float(value["worst_oar_goal_ratio"]) for value in outcomes])
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(device),
        "beam_angles_degrees": mode.angles_degrees,
        "delivery_mode": mode.name,
        "configuration": {
            **vars(config),
            **initial_priority_record,
            "starting_profile_values": {
                name: STARTING_PROFILE_VALUES[name]
                for name in profile_names
                if name != "custom"
            },
        },
        "selection": {
            "mode": (
                "tcia_locked_episode_manifest"
                if episode_assignments is not None
                else "tcia_clinical_anatomy"
                if args.anatomy_source == "tcia"
                else "explicit_seed" if args.seeds else args.selection_mode
            ),
            "episode_manifest": (
                str(args.tcia_episode_manifest.resolve())
                if args.tcia_episode_manifest
                else None
            ),
            "anatomy_attempts": anatomy_attempts,
            "seed_start": args.seed_start,
            "stress_structure": args.stress_structure,
            "minimum_bladder_overlap_fraction": args.minimum_bladder_overlap_fraction,
            "minimum_rectum_overlap_fraction": args.minimum_rectum_overlap_fraction,
            "minimum_bladder_ptv_overlap_fraction": args.minimum_bladder_ptv_overlap_fraction,
            "minimum_rectum_ptv_overlap_fraction": args.minimum_rectum_ptv_overlap_fraction,
            "maximum_ptv_overlap_fraction": args.maximum_ptv_overlap_fraction,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_progress(
        output_dir,
        len(episodes),
        len(episodes),
        started,
        status="complete",
        last_case=str(outcomes[-1]["episode_id"]),
        unit="planning episodes",
    )
    print(json.dumps(summary, indent=2), flush=True)


def mark_status_failed_from_arguments(error: BaseException) -> None:
    """Change an existing progress file to failed after an unexpected error."""

    if "--output-dir" not in sys.argv:
        return
    index = sys.argv.index("--output-dir") + 1
    if index >= len(sys.argv):
        return
    output_dir = Path(sys.argv[index])
    progress_path = output_dir / "progress.json"
    if not progress_path.exists():
        return
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    payload["status"] = "failed"
    payload["error_message"] = f"{type(error).__name__}: {error}"
    temporary = output_dir / "progress.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(progress_path)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        mark_status_failed_from_arguments(error)
        raise
