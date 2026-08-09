# DOSIM Planning-Trajectory Simulation

This repository tests whether intermediate expert planning actions teach a model more than the same expert's final plan alone.

The latest verified measurements and their interpretation are summarized in [`CURRENT_RESULTS.md`](CURRENT_RESULTS.md).
Instructions for validating and benchmarking the optional PyTorch backend on the four-A100 server are in [`GPU_SERVER.md`](GPU_SERVER.md).
The prespecified case assignments are in [`outputs/splits/case_split_manifest.csv`](outputs/splits/case_split_manifest.csv); they contain no trajectory or outcome fields.
Measured RTX 4060 results through 256-cubed are summarized in [`LOCAL_GPU_RESULTS.md`](LOCAL_GPU_RESULTS.md).
The protocol-stage Medical Physics manuscript is in `paper/planning_trajectory_manuscript_draft.docx`; its Results section is intentionally reserved until the experimental specification and analysis are frozen.

## What exists now

The first executable milestone implements:

1. deterministic synthetic anatomy with one target and two avoidance structures;
2. 12 beam directions with 8 adjustable beamlets each;
3. a transparent plan-quality objective balancing target coverage, hot spots, avoidance dose, and plan complexity;
4. an automated inner optimizer that adjusts beamlet intensities for fixed planning settings;
5. a separate manual planner whose recorded actions are limited to beam-angle and target/OAR-priority changes;
6. images explaining the optimizer/manual boundary and the response to each manual change;
7. tests for reproducibility, dose linearity, optimizer masking, and manual-action integrity;
8. a 64 x 64 x 64 reference implementation of an implicit 3D dose operator and its exact adjoint.

This is an educational dose surrogate, not a clinical dose calculation.
The v0.6-manual acceptability thresholds are provisional feasibility settings; they
will be calibrated and frozen before the endpoint-versus-trajectory comparison.

## Run the high-level manual-planning demo

```powershell
uv run --extra dev python scripts/run_manual_demo.py --seed 10000
```

Outputs appear in `outputs/manual_demo/`:

- `01_nested_workflow.png` separates the recorded manual loop from automated beamlet optimization;
- `02_manual_trajectory.png` shows the reoptimized dose after every beam/priority change;
- `03_manual_metrics.png` aligns the changed settings with the resulting target and OAR metrics;
- `manual_trajectory.csv` contains one row per manual planning decision.

## Run the legacy beamlet-level prototype

```powershell
uv run --extra dev python scripts/run_demo.py --seed 20260809
```

Outputs appear in `outputs/demo/`:

- `01_case_construction.png` explains the anatomy and dose-influence model;
- `02_expert_trajectory.png` shows initial, middle, and final dose states, objective improvement, target coverage, and final beamlet intensities.
- `03_action_sequence.png` shows why the score changes and the exact beam/beamlet adjustment at each step;
- `trajectory.csv` contains one readable row per expert action.

This older demo is retained for provenance, but its individual beamlet adjustments are no longer the manual trajectory used by the experiment.

## Run the 3D planning prototype

```powershell
.\.venv\Scripts\python.exe scripts\run_3d_demo.py --grid-size 64 --iterations 60
```

Outputs appear in `outputs/3d_demo/`:

- `01_3d_anatomy.png` shows the target and two OARs in axial, coronal, and sagittal planes;
- `02_3d_planning_steps.png` shows dose and constraints after four manual edits;
- `trajectory.csv` records only OAR/target priority and beam-angle actions;
- `summary.json` records runtime and memory measurements.

The latest full CPU run completed one five-state 64-cubed trajectory in 17.7 seconds; repeated runs have taken approximately 15-18 seconds. It caches 14.3 MiB of geometric interpolation data and avoids a dense per-case influence matrix. The next server milestone is to port the same forward/adjoint interface to batched PyTorch and distribute cases across the four A100 GPUs. The 2D environment remains the fast unit-test and ablation environment; 64-cubed is the 3D development resolution, 96-cubed is the proposed main resolution, and 128-cubed is a sensitivity analysis.

To reproduce the resolution benchmark:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_3d_operator.py --grids 64 96 128
```

The current CPU forward-plus-adjoint timings are approximately 0.043, 0.297, and 0.707 seconds per optimization iteration at 64-, 96-, and 128-cubed. These are kernel timings, not complete trajectory times, and the A100 implementation should be benchmarked directly before scheduling the main run.

To reproduce the initial 96-cubed environment calibration:

```powershell
.\.venv\Scripts\python.exe scripts\run_3d_calibration.py --cases 12 --grid-size 96 --fluence-size 16 --iterations 60 --device cuda:0
```

This calibration applies only high-level beam-angle and named-priority changes. Fluence optimization remains an automated inner calculation and is not recorded as a planner action.

## Run validation tests

```powershell
uv run --extra dev pytest
```

## Run the legacy beamlet-level environment pilot

```powershell
uv run --extra dev python scripts/run_environment_pilot.py --cases 100 --workers 4
```

This preserves the original optimizer/action-abstraction diagnostic. It is not the current manual-trajectory experiment.

For the current high-level manual planner:

```powershell
uv run --extra dev python scripts/run_manual_pilot.py --cases 20 --workers 4
```

This creates paired before/after metrics, an action-frequency chart, and a per-case manual trajectory audit table.

## Compare the manual policy with the high-level oracle

```powershell
uv run --extra dev python scripts/run_oracle_pilot.py --cases 12 --workers 4
```

The oracle uses the same high-level action library but evaluates several plausible setting changes at each step. Its priority-independent violation score is used to identify cases where the goals are reachable but the rule-based manual policy chooses a weaker sequence. Oracle failure means "not reached by this search," not proof of physical infeasibility.

The latest 12-case diagnostic is preserved in `outputs/oracle_pilot_v2/`, including a cohort comparison and a same-case manual-versus-oracle decision sequence.

## Build a matched endpoint/trajectory dataset pilot

```powershell
uv run --extra dev python scripts/build_oracle_dataset.py --reachable-cases 4 --workers 4
```

This writes matched `endpoints.jsonl` and `trajectories.jsonl` records, compressed anatomy/dose arrays, and a manifest containing every attempted or excluded case. Both views contain identical cases and final plans; only the trajectory view includes intermediate high-level actions.

## How methods and results will be shown

The full experiment will use a fixed visual sequence:

1. **Case card:** anatomy, planning goals, beam candidates, and difficulty.
2. **Trajectory filmstrip:** reoptimized plan states and the exact beam-angle or priority change between them.
3. **Objective story:** total score and each penalty term over planning steps.
4. **Endpoint-versus-trajectory comparison:** paired dots for the same test cases, so the improvement or failure is visible case by case.
5. **Learning curves:** performance versus number of training cases, showing sample efficiency.
6. **Failure atlas:** representative cases where each learner succeeds or fails, grouped by geometry and failure type.

The model-comparison stage will be added only after the environment and expert pass the validation gates in `simulation_plan.md`.
