"""Start the bounded prostate DVH calibration and its local status server."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def detached_flags() -> int:
    return (
        getattr(subprocess, "DETACHED_PROCESS", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    )


def timestamped_output_dir(output_dir: Path, current_time: datetime | None = None) -> tuple[Path, str]:
    now = current_time or datetime.now().astimezone()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    resolved = output_dir.resolve()
    return resolved.with_name(f"{resolved.name}_{timestamp}"), timestamp


def main() -> None:
    parser = argparse.ArgumentParser(description="Start local DVH calibration with a token-free status page")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--anatomy-source",
        choices=("parametric", "tcia"),
        default="parametric",
    )
    parser.add_argument("--tcia-root", type=Path, default=Path("data/tcia"))
    parser.add_argument("--tcia-subjects", nargs="+")
    parser.add_argument("--tcia-subject-file", type=Path)
    parser.add_argument("--tcia-episode-manifest", type=Path)
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
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--fluence-size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--hotspot-objective-gy", type=float, default=60.0)
    parser.add_argument("--hotspot-objective-weight", type=float, default=50.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--delivery-mode",
        choices=("static_7", "static_9", "static_12", "arc_like_360"),
        default="static_7",
    )
    parser.add_argument("--initial-target-priority", type=float, default=1.0)
    parser.add_argument("--initial-hotspot-priority", type=float, default=1.0)
    parser.add_argument("--initial-oar-priority", type=float, default=1.0)
    parser.add_argument("--initial-normal-tissue-priority", type=float, default=1.0)
    parser.add_argument(
        "--starting-profiles",
        nargs="+",
        choices=(
            "balanced_reference",
            "oar_omitted",
            "hotspot_low",
            "conformity_low",
            "oar_low",
            "oar_guarded",
            "hotspot_stress",
            "conformity_stress",
        ),
    )
    parser.add_argument("--manual-shifts", type=int, default=2)
    parser.add_argument("--shift-degrees", type=float, default=10.0)
    parser.add_argument(
        "--action-set",
        choices=("target_only", "target_hotspot"),
        default="target_only",
    )
    parser.add_argument("--serve-only", action="store_true")
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Use the output directory exactly as given instead of adding a run timestamp",
    )
    parser.add_argument(
        "--pilot",
        choices=("clinical_dvh", "target_priority", "manual_trajectory", "expert_angle", "ptv_manual"),
        default="clinical_dvh",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/prostate_clinical_dvh_24x24_12case"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    run_timestamp = None
    if not args.serve_only and not args.no_timestamp:
        output_dir, run_timestamp = timestamped_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flags = detached_flags()

    server_stdout = (output_dir / "server.log").open("ab")
    server_stderr = (output_dir / "server.err.log").open("ab")
    run_stdout = (output_dir / "run.log").open("ab")
    run_stderr = (output_dir / "run.err.log").open("ab")
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            str(args.port),
            "--bind",
            "127.0.0.1",
            "--directory",
            str(output_dir),
        ],
        stdin=subprocess.DEVNULL,
        stdout=server_stdout,
        stderr=server_stderr,
        creationflags=flags,
        close_fds=True,
    )
    run = None
    if not args.serve_only:
        if args.pilot == "ptv_manual":
            run_arguments = [
                sys.executable,
                "scripts/run_prostate_ptv_manual_pilot.py",
                "--cases",
                str(args.cases),
                "--anatomy-source",
                args.anatomy_source,
                "--tcia-root",
                str(args.tcia_root.resolve()),
                "--selection-mode",
                args.selection_mode,
                "--seed-start",
                str(args.seed_start),
                "--maximum-anatomy-attempts",
                str(args.maximum_anatomy_attempts),
                "--coarse-grid-size",
                str(args.coarse_grid_size),
                "--stress-structure",
                args.stress_structure,
                "--minimum-bladder-overlap-fraction",
                str(args.minimum_bladder_overlap_fraction),
                "--minimum-rectum-overlap-fraction",
                str(args.minimum_rectum_overlap_fraction),
                "--minimum-bladder-ptv-overlap-fraction",
                str(args.minimum_bladder_ptv_overlap_fraction),
                "--minimum-rectum-ptv-overlap-fraction",
                str(args.minimum_rectum_ptv_overlap_fraction),
                "--maximum-ptv-overlap-fraction",
                str(args.maximum_ptv_overlap_fraction),
                "--max-steps",
                str(args.max_steps),
                "--grid-size",
                str(args.grid_size),
                "--fluence-size",
                str(args.fluence_size),
                "--iterations",
                str(args.iterations),
                "--learning-rate",
                str(args.learning_rate),
                "--hotspot-objective-gy",
                str(args.hotspot_objective_gy),
                "--hotspot-objective-weight",
                str(args.hotspot_objective_weight),
                "--device",
                args.device,
                "--delivery-mode",
                args.delivery_mode,
                "--initial-target-priority",
                str(args.initial_target_priority),
                "--initial-hotspot-priority",
                str(args.initial_hotspot_priority),
                "--initial-oar-priority",
                str(args.initial_oar_priority),
                "--initial-normal-tissue-priority",
                str(args.initial_normal_tissue_priority),
                "--output-dir",
                str(output_dir),
            ]
            if args.seeds:
                run_arguments.extend(["--seeds", *(str(seed) for seed in args.seeds)])
            if args.tcia_subjects:
                run_arguments.extend(["--tcia-subjects", *args.tcia_subjects])
            if args.tcia_subject_file:
                run_arguments.extend(
                    ["--tcia-subject-file", str(args.tcia_subject_file.resolve())]
                )
            if args.tcia_episode_manifest:
                run_arguments.extend(
                    [
                        "--tcia-episode-manifest",
                        str(args.tcia_episode_manifest.resolve()),
                    ]
                )
            if args.starting_profiles:
                run_arguments.extend(["--starting-profiles", *args.starting_profiles])
        elif args.pilot == "target_priority":
            run_arguments = [
                sys.executable,
                "scripts/run_prostate_target_priority_pilot.py",
                "--cases",
                str(args.cases),
                "--output-dir",
                str(output_dir),
            ]
        elif args.pilot == "manual_trajectory":
            run_arguments = [
                sys.executable,
                "scripts/run_prostate_manual_target_trajectory.py",
                "--cases",
                str(args.cases),
                "--action-set",
                args.action_set,
                "--output-dir",
                str(output_dir),
            ]
        elif args.pilot == "expert_angle":
            run_arguments = [
                sys.executable,
                "scripts/run_prostate_expert_angle_clinical_pilot.py",
                "--cases",
                str(args.cases),
                "--manual-shifts",
                str(args.manual_shifts),
                "--shift-degrees",
                str(args.shift_degrees),
                "--output-dir",
                str(output_dir),
            ]
        else:
            run_arguments = [
                sys.executable,
                "scripts/run_prostate_clinical_dvh_pilot.py",
                "--cases-per-stratum",
                "4",
                "--grid-size",
                "64",
                "--fluence-size",
                "24",
                "--iterations",
                "300",
                "--weights",
                "5",
                "--device",
                "cuda:0",
                "--output-dir",
                str(output_dir),
            ]
        run = subprocess.Popen(
            run_arguments,
            stdin=subprocess.DEVNULL,
            stdout=run_stdout,
            stderr=run_stderr,
            creationflags=flags,
            close_fds=True,
        )
    payload = {
        "server_pid": server.pid,
        "run_pid": run.pid if run else None,
        "status_url": f"http://127.0.0.1:{args.port}/status.html",
        "output_dir": str(output_dir),
        "run_timestamp": run_timestamp,
    }
    (output_dir / "processes.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
