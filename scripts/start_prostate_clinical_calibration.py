"""Start the bounded prostate DVH calibration and its local status server."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def detached_flags() -> int:
    return (
        getattr(subprocess, "DETACHED_PROCESS", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Start local DVH calibration with a token-free status page")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--manual-shifts", type=int, default=2)
    parser.add_argument("--shift-degrees", type=float, default=10.0)
    parser.add_argument(
        "--action-set",
        choices=("target_only", "target_hotspot"),
        default="target_only",
    )
    parser.add_argument("--serve-only", action="store_true")
    parser.add_argument(
        "--pilot",
        choices=("clinical_dvh", "target_priority", "manual_trajectory", "expert_angle"),
        default="clinical_dvh",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/prostate_clinical_dvh_24x24_12case"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
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
        if args.pilot == "target_priority":
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
    }
    (output_dir / "processes.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
