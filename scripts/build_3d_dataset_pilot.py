import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dosim_sim.dataset3d import (
    action_index_3d,
    action_name_3d,
    final_settings_target_3d,
    retention_eligible_3d,
    state_features_3d,
)
from dosim_sim.planning3d import (
    HighLevelSearchConfig3D,
    is_acceptable_3d,
    run_high_level_search_3d,
    run_reference_optimizer_3d,
)
from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D
from dosim_sim.volume3d import generate_case_3d, generate_prostate_case_3d


ANGLES = tuple(float(value) for value in range(0, 360, 30))


def candidate_specs(
    max_attempts: int,
    seed_start: int,
    split_manifest: Path | None = None,
    split: str | None = None,
    start_ordinal: int = 0,
) -> list[dict]:
    """Return deterministic case specifications for one independently writable shard."""
    if split_manifest is None:
        return [
            {
                "seed": seed_start + offset,
                "split": None,
                "split_ordinal": None,
                "difficulty": ("easy", "moderate", "hard")[offset % 3],
            }
            for offset in range(max_attempts)
        ]
    if split is None:
        raise ValueError("--split is required when --split-manifest is provided")
    with split_manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    rows.sort(key=lambda row: int(row["split_ordinal"]))
    selected = [row for row in rows if int(row["split_ordinal"]) >= start_ordinal][:max_attempts]
    if len(selected) < max_attempts:
        raise ValueError(
            f"split {split!r} has only {len(selected)} cases at or after ordinal {start_ordinal}; "
            f"{max_attempts} requested"
        )
    return [
        {
            "seed": int(row["seed"]),
            "split": row["split"],
            "split_ordinal": int(row["split_ordinal"]),
            "difficulty": ("easy", "moderate", "hard")[int(row["split_ordinal"]) % 3],
        }
        for row in selected
    ]


def save_audit(attempts: list[dict], retained: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    difficulties = ("easy", "moderate", "hard")
    attempted = [sum(row["difficulty"] == value for row in attempts) for value in difficulties]
    included = [sum(row["difficulty"] == value for row in retained) for value in difficulties]
    x = np.arange(3)
    axes[0].bar(x - 0.18, attempted, 0.36, label="attempted")
    axes[0].bar(x + 0.18, included, 0.36, label="retained")
    axes[0].set_xticks(x, difficulties)
    axes[0].set_ylabel("Cases")
    axes[0].set_title("Case retention")
    axes[0].legend()
    axes[1].hist([row["trajectory_length"] for row in retained], bins=np.arange(0.5, 13.5), color="#4E79A7")
    axes[1].set_xlabel("High-level actions")
    axes[1].set_ylabel("Retained cases")
    axes[1].set_title("Trajectory length")
    counts = Counter(action for row in retained for action in row["action_names"])
    axes[2].barh(list(counts), list(counts.values()), color="#59A14F")
    axes[2].set_xlabel("Recorded actions")
    axes[2].set_title("Action composition")
    fig.suptitle("Matched three-dimensional endpoint/trajectory pilot dataset", fontweight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_progress(
    output_dir: Path,
    target: int,
    attempts: int,
    retained: int,
    started: float,
    last_case: str | None = None,
    status: str = "running",
) -> None:
    elapsed = time.perf_counter() - started
    rate = retained / elapsed if elapsed > 0 else 0.0
    remaining = max(target - retained, 0)
    payload = {
        "status": status,
        "retained": retained,
        "target": target,
        "attempted": attempts,
        "percent_complete": 100.0 * retained / max(target, 1),
        "elapsed_seconds": elapsed,
        "estimated_seconds_remaining": remaining / rate if rate > 0 else None,
        "last_case": last_case,
    }
    temporary = output_dir / "progress.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "progress.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build matched 3D endpoint and trajectory pilot views")
    parser.add_argument("--retained-cases", type=int, default=40)
    parser.add_argument("--max-attempts", type=int, default=120)
    parser.add_argument("--seed-start", type=int, default=15000)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "iid_test", "ood_test"))
    parser.add_argument("--start-ordinal", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--anatomy", choices=("generic", "prostate"), default="generic")
    parser.add_argument("--fluence-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--deep-iterations", type=int, default=40)
    parser.add_argument("--reference-iterations", type=int, default=240)
    parser.add_argument("--initial-field-count", type=int, default=4)
    parser.add_argument("--normal-tissue-weight", type=float, default=0.0)
    parser.add_argument("--normal-tissue-threshold", type=float, default=0.5)
    parser.add_argument("--integral-dose-weight", type=float, default=0.0)
    parser.add_argument("--clinical-dvh-weight", type=float, default=0.0)
    parser.add_argument("--target-hotspot-threshold", type=float, default=1.10)
    parser.add_argument("--target-hotspot-weight", type=float, default=5.0)
    parser.add_argument("--high-dose-normal-tissue-weight", type=float, default=0.0)
    parser.add_argument("--high-dose-normal-tissue-threshold", type=float, default=0.95)
    parser.add_argument("--target-normalization-d98", type=float)
    parser.add_argument("--target-normalization-d50", type=float)
    parser.add_argument("--target-normalization-interval", type=int, default=0)
    parser.add_argument(
        "--prostate-protocol-tier",
        choices=(
            "off",
            "per_protocol",
            "variation_acceptable",
            "oar_per_protocol",
            "oar_variation_acceptable",
        ),
        default="off",
    )
    parser.add_argument("--d95-min", type=float, default=0.85)
    parser.add_argument("--d98-min", type=float, default=0.0)
    parser.add_argument("--d50-min", type=float, default=0.0)
    parser.add_argument("--d50-max", type=float, default=float("inf"))
    parser.add_argument("--d02-max", type=float, default=1.25)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--deep-max-steps", type=int, default=10)
    parser.add_argument("--deep-beam-width", type=int, default=4)
    parser.add_argument("--paddick-ci-95-min", type=float, default=0.0)
    parser.add_argument("--covering-isodose-ratio-95-max", type=float, default=float("inf"))
    parser.add_argument("--r50-max", type=float, default=float("inf"))
    parser.add_argument("--minimum-field-count", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/3d_dataset_pilot"))
    parser.add_argument(
        "--automatic-train-count",
        type=int,
        help="Assign the first retained cases to train and all remaining retained cases to validation",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    shallow_cfg = HighLevelSearchConfig3D(
        max_steps=args.max_steps,
        beam_width=args.beam_width,
        add_candidates=2,
        remove_candidates=1,
        optimizer_iterations=args.iterations,
        initial_field_count=args.initial_field_count,
        normal_tissue_weight=args.normal_tissue_weight,
        normal_tissue_threshold=args.normal_tissue_threshold,
        integral_dose_weight=args.integral_dose_weight,
        clinical_dvh_weight=args.clinical_dvh_weight,
        target_hotspot_threshold=args.target_hotspot_threshold,
        target_hotspot_weight=args.target_hotspot_weight,
        high_dose_normal_tissue_weight=args.high_dose_normal_tissue_weight,
        high_dose_normal_tissue_threshold=args.high_dose_normal_tissue_threshold,
        target_normalization_d98=args.target_normalization_d98,
        target_normalization_d50=args.target_normalization_d50,
        target_normalization_interval=args.target_normalization_interval,
        prostate_protocol_tier=args.prostate_protocol_tier,
        d95_min=args.d95_min,
        d98_min=args.d98_min,
        d50_min=args.d50_min,
        d50_max=args.d50_max,
        d02_max=args.d02_max,
        paddick_ci_95_min=args.paddick_ci_95_min,
        covering_isodose_ratio_95_max=args.covering_isodose_ratio_95_max,
        r50_max=args.r50_max,
        minimum_field_count=args.minimum_field_count,
    )
    deep_cfg = HighLevelSearchConfig3D(
        max_steps=args.deep_max_steps,
        beam_width=args.deep_beam_width,
        add_candidates=3,
        remove_candidates=2,
        optimizer_iterations=args.deep_iterations,
        priority_ceiling=25.0,
        initial_field_count=args.initial_field_count,
        normal_tissue_weight=args.normal_tissue_weight,
        normal_tissue_threshold=args.normal_tissue_threshold,
        integral_dose_weight=args.integral_dose_weight,
        clinical_dvh_weight=args.clinical_dvh_weight,
        target_hotspot_threshold=args.target_hotspot_threshold,
        target_hotspot_weight=args.target_hotspot_weight,
        high_dose_normal_tissue_weight=args.high_dose_normal_tissue_weight,
        high_dose_normal_tissue_threshold=args.high_dose_normal_tissue_threshold,
        target_normalization_d98=args.target_normalization_d98,
        target_normalization_d50=args.target_normalization_d50,
        target_normalization_interval=args.target_normalization_interval,
        prostate_protocol_tier=args.prostate_protocol_tier,
        d95_min=args.d95_min,
        d98_min=args.d98_min,
        d50_min=args.d50_min,
        d50_max=args.d50_max,
        d02_max=args.d02_max,
        paddick_ci_95_min=args.paddick_ci_95_min,
        covering_isodose_ratio_95_max=args.covering_isodose_ratio_95_max,
        r50_max=args.r50_max,
        minimum_field_count=args.minimum_field_count,
    )
    attempts: list[dict] = []
    retained: list[dict] = []
    started = time.perf_counter()
    write_progress(args.output_dir, args.retained_cases, 0, 0, started)
    specs = candidate_specs(
        args.max_attempts,
        args.seed_start,
        args.split_manifest,
        args.split,
        args.start_ordinal,
    )
    for spec in specs:
        if len(retained) >= args.retained_cases:
            break
        seed = spec["seed"]
        difficulty = spec["difficulty"]
        generator = generate_prostate_case_3d if args.anatomy == "prostate" else generate_case_3d
        case = generator(seed, args.grid_size, difficulty=difficulty)
        engine = TorchImplicitDoseEngine3D(case, ANGLES, args.fluence_size, device=device, dtype=torch.float16)
        trajectory = run_high_level_search_3d(case, engine, shallow_cfg)
        reference = run_reference_optimizer_3d(case, engine, args.reference_iterations, config=deep_cfg)
        search_ok = is_acceptable_3d(trajectory.final.plan.metrics, case, shallow_cfg)
        reference_ok = is_acceptable_3d(reference.metrics, case, shallow_cfg)
        search_tier = "shallow"
        if not search_ok and reference_ok:
            trajectory = run_high_level_search_3d(case, engine, deep_cfg)
            search_ok = is_acceptable_3d(trajectory.final.plan.metrics, case, deep_cfg)
            search_tier = "deep"
        attempt = {
            "seed": seed,
            "case_id": case.case_id,
            "difficulty": difficulty,
            "anatomy": args.anatomy,
            "split": spec["split"],
            "split_ordinal": spec["split_ordinal"],
            "search_acceptable": search_ok,
            "reference_acceptable": reference_ok,
            "search_tier": search_tier,
            "search_violation": trajectory.final.violation_score,
            "trajectory_length": len(trajectory.steps) - 1,
            "stopping_reason": trajectory.stopping_reason,
            "initial_field_count": args.initial_field_count,
            "normal_tissue_weight": args.normal_tissue_weight,
            "normal_tissue_threshold": args.normal_tissue_threshold,
            "integral_dose_weight": args.integral_dose_weight,
            "clinical_dvh_weight": args.clinical_dvh_weight,
            "target_hotspot_threshold": args.target_hotspot_threshold,
            "target_hotspot_weight": args.target_hotspot_weight,
            "high_dose_normal_tissue_weight": args.high_dose_normal_tissue_weight,
            "high_dose_normal_tissue_threshold": args.high_dose_normal_tissue_threshold,
            "target_normalization_d98": args.target_normalization_d98,
            "target_normalization_d50": args.target_normalization_d50,
            "target_normalization_interval": args.target_normalization_interval,
            "prostate_protocol_tier": args.prostate_protocol_tier,
            "d95_min": args.d95_min,
            "d98_min": args.d98_min,
            "d50_min": args.d50_min,
            "d50_max": args.d50_max,
            "d02_max": args.d02_max,
            "paddick_ci_95_min": args.paddick_ci_95_min,
            "covering_isodose_ratio_95_max": args.covering_isodose_ratio_95_max,
            "r50_max": args.r50_max,
            "minimum_field_count": args.minimum_field_count,
        }
        attempts.append(attempt)
        if retention_eligible_3d(search_ok, reference_ok):
            action_names = [action_name_3d(step.action) for step in trajectory.steps[1:]]
            transitions = [
                {
                    "state": state_features_3d(case, before, deep_cfg.max_steps).tolist(),
                    "action_index": action_index_3d(after.action),
                    "action_name": action_name_3d(after.action),
                    "next_state": state_features_3d(case, after, deep_cfg.max_steps).tolist(),
                }
                for before, after in zip(trajectory.steps, trajectory.steps[1:])
            ]
            # Explicit stop label at the accepted terminal state.
            transitions.append(
                {
                    "state": state_features_3d(case, trajectory.final, deep_cfg.max_steps).tolist(),
                    "action_index": action_index_3d(None),
                    "action_name": "stop",
                    "next_state": state_features_3d(case, trajectory.final, deep_cfg.max_steps).tolist(),
                }
            )
            retained.append(
                {
                    **attempt,
                    "initial_features": state_features_3d(case, trajectory.steps[0], deep_cfg.max_steps).tolist(),
                    "final_features": state_features_3d(case, trajectory.final, deep_cfg.max_steps).tolist(),
                    "final_settings": final_settings_target_3d(trajectory.final).tolist(),
                    "trajectory": transitions,
                    "action_names": action_names,
                }
            )
        write_progress(
            args.output_dir,
            args.retained_cases,
            len(attempts),
            len(retained),
            started,
            last_case=case.case_id,
        )
        print(f"attempt={len(attempts)} retained={len(retained)} seed={seed} difficulty={difficulty} search={search_ok} reference={reference_ok}", flush=True)
    elapsed = time.perf_counter() - started
    if args.automatic_train_count is not None:
        if not 0 <= args.automatic_train_count <= len(retained):
            raise ValueError("--automatic-train-count must be between zero and the retained case count")
        train_ordinal = 0
        validation_ordinal = 0
        for index, row in enumerate(retained):
            if index < args.automatic_train_count:
                row["split"] = "train"
                row["split_ordinal"] = train_ordinal
                train_ordinal += 1
            else:
                row["split"] = "validation"
                row["split_ordinal"] = validation_ordinal
                validation_ordinal += 1
    endpoint_path = args.output_dir / "endpoint_view.jsonl"
    trajectory_path = args.output_dir / "trajectory_view.jsonl"
    with endpoint_path.open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps({key: row[key] for key in ("case_id", "seed", "difficulty", "anatomy", "split", "split_ordinal", "initial_features", "final_features", "final_settings")}, separators=(",", ":")) + "\n")
    with trajectory_path.open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps({key: row[key] for key in ("case_id", "seed", "difficulty", "anatomy", "split", "split_ordinal", "initial_features", "final_features", "final_settings", "trajectory")}, separators=(",", ":")) + "\n")
    with (args.output_dir / "attempt_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in attempts:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    save_audit(attempts, retained, args.output_dir / "01_dataset_audit.png")
    case_ids_endpoint = [json.loads(line)["case_id"] for line in endpoint_path.read_text().splitlines()]
    case_ids_trajectory = [json.loads(line)["case_id"] for line in trajectory_path.read_text().splitlines()]
    summary = {
        "status": "matched development dataset; not a learner result",
        "anatomy": args.anatomy,
        "attempted": len(attempts),
        "retained": len(retained),
        "elapsed_seconds": elapsed,
        "case_ids_identical": case_ids_endpoint == case_ids_trajectory,
        "endpoint_contains_trajectory_key": any("trajectory" in json.loads(line) for line in endpoint_path.read_text().splitlines()),
        "feature_count": len(retained[0]["initial_features"]) if retained else None,
        "final_setting_count": len(retained[0]["final_settings"]) if retained else None,
        "transition_count": sum(len(row["trajectory"]) for row in retained),
        "retained_by_difficulty": dict(Counter(row["difficulty"] for row in retained)),
        "split": args.split,
        "start_ordinal": args.start_ordinal if args.split_manifest else None,
        "reference_acceptable_among_retained": float(
            np.mean([bool(row["reference_acceptable"]) for row in retained])
        ) if retained else None,
        "initial_field_count": args.initial_field_count,
        "normal_tissue_weight": args.normal_tissue_weight,
        "normal_tissue_threshold": args.normal_tissue_threshold,
        "integral_dose_weight": args.integral_dose_weight,
        "clinical_dvh_weight": args.clinical_dvh_weight,
        "target_hotspot_threshold": args.target_hotspot_threshold,
        "target_hotspot_weight": args.target_hotspot_weight,
        "high_dose_normal_tissue_weight": args.high_dose_normal_tissue_weight,
        "high_dose_normal_tissue_threshold": args.high_dose_normal_tissue_threshold,
        "target_normalization_d98": args.target_normalization_d98,
        "target_normalization_d50": args.target_normalization_d50,
        "target_normalization_interval": args.target_normalization_interval,
        "prostate_protocol_tier": args.prostate_protocol_tier,
        "d95_min": args.d95_min,
        "d98_min": args.d98_min,
        "d50_min": args.d50_min,
        "d50_max": args.d50_max,
        "d02_max": args.d02_max,
        "paddick_ci_95_min": args.paddick_ci_95_min,
        "covering_isodose_ratio_95_max": args.covering_isodose_ratio_95_max,
        "r50_max": args.r50_max,
        "minimum_field_count": args.minimum_field_count,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_progress(
        args.output_dir,
        args.retained_cases,
        len(attempts),
        len(retained),
        started,
        status="complete" if len(retained) >= args.retained_cases else "attempt_limit",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
