# DOSIM Planning-Trajectory Simulation

This repository tests whether intermediate expert planning actions teach a model more than the same expert's final plan alone.

The latest verified measurements and their interpretation are summarized in [`CURRENT_RESULTS.md`](CURRENT_RESULTS.md).
Instructions for validating and benchmarking the optional PyTorch backend on the four-A100 server are in [`GPU_SERVER.md`](GPU_SERVER.md).
The prespecified case assignments are in [`outputs/splits/case_split_manifest.csv`](outputs/splits/case_split_manifest.csv); they contain no trajectory or outcome fields.
Measured RTX 4060 results through 256-cubed are summarized in [`LOCAL_GPU_RESULTS.md`](LOCAL_GPU_RESULTS.md).
The protocol-stage Medical Physics manuscript is in `paper/planning_trajectory_manuscript_draft.docx`; its Results section is intentionally reserved until the experimental specification and analysis are frozen.
The completed 240-case training and 60-case validation dataset is in `outputs/pilot300_local_v2/merged/`, with all attempted cases retained in its manifest and a visual audit in `01_dataset_summary.png`.

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

## Validate the revised 3D environment

```powershell
uv run python scripts/validate_3d_environment.py --cases 100 --response-cases 12 --grid-size 64 --iterations 200 --device cuda:0 --output-dir outputs/3d_environment_validation_v4
```

The controlled test compares target- and named-OAR-weight perturbations with neutral continuations of equal length from the same fluence state. The current run produced the expected response in all 12 stratified cases for both controls.

## Audit bounded high-level search

```powershell
uv run python scripts/run_3d_search_pilot.py --cases-per-stratum 10 --grid-size 32 --fluence-size 8 --max-steps 6 --beam-width 3 --output-dir outputs/3d_search_pilot_revised_30_shallow
uv run python scripts/audit_3d_search_failures.py outputs/3d_search_pilot_revised_30_shallow/case_metrics.csv --grid-size 32 --iterations 40 --max-steps 10 --beam-width 4 --output-dir outputs/3d_search_failure_audit_revised
```

Both stages change only beam membership or named target, hot-spot, and OAR priorities. The deeper stage is restricted to routine-search failures that the independent reference solver can reach.

## Run the closed-loop learner development check

```powershell
uv run python scripts/evaluate_3d_closed_loop_pilot.py --dataset-dir outputs/3d_dataset_pilot_revised --seeds 5 --action-weight 0.02 --output-dir outputs/3d_closed_loop_pilot_stop_masked
```

The learned trajectory policy selects one legal high-level action, the inner optimizer recalculates fluence, and the process repeats. Stop is legal only when the current plan satisfies all visible acceptance rules. The endpoint arm in this script is a direct terminal-settings comparator; it is not the prespecified primary iterative endpoint-only policy.

## Run the matched iterative-policy development check

```powershell
uv run python scripts/train_3d_iterative_policy_pilot.py --dataset-dir outputs/3d_dataset_pilot_revised --pretrain-updates 400 --updates 30 --seeds 3 --dtype float32 --deterministic --output-dir outputs/3d_iterative_policy_pilot_deterministic
```

Both arms use the same iterative network, terminal simulator reward, legal-action mask, rollout limit, and optimizer-update count. The endpoint-only arm receives no intermediate demonstration action. The trajectory arm receives the same terminal supervision plus categorical supervision on the recorded high-level actions.

The completed 300-case variance pilot uses `--action-weight 0.20`. The earlier value of 0.02 did not reliably learn the 35-class action labels. The value 0.20 is frozen before any test-partition run.

## Compare angular delivery complexity

```powershell
uv run python scripts/run_3d_delivery_complexity_pilot.py --cases-per-stratum 4 --grid-size 64 --fluence-size 8 --iterations 200 --output-dir outputs/3d_delivery_complexity_pilot
```

This paired engineering pilot compares four static fields, twelve static fields, 19 control points over 180 degrees, and 36 control points over 360 degrees. The arc-like modes use independent fluence maps at sampled angles. They are useful angular-complexity surrogates but are not delivery-realistic VMAT.

## Build a matched endpoint/trajectory dataset pilot

```powershell
uv run --extra dev python scripts/build_oracle_dataset.py --reachable-cases 4 --workers 4
```

This writes matched `endpoints.jsonl` and `trajectories.jsonl` records, compressed anatomy/dose arrays, and a manifest containing every attempted or excluded case. Both views contain identical cases and final plans; only the trajectory view includes intermediate high-level actions.

For a prespecified, shardable 3D pilot, select cases from the frozen split manifest rather than allocating sequential development seeds. For example, the first training shard is generated with:

```powershell
uv run python scripts/build_3d_dataset_pilot.py --split-manifest outputs/splits/case_split_manifest.csv --split train --start-ordinal 0 --max-attempts 100 --retained-cases 60 --fluence-size 8 --device cuda:0 --output-dir outputs/3d_300_train_shard0
```

`--start-ordinal` defines a nonoverlapping manifest range for each GPU process. Training and validation cases must be generated into separate shards and merged only after case-identity and ordinal checks.

`scripts/merge_3d_dataset_shards.py` performs the required duplicate, partition, retained-count, and endpoint/trajectory identity checks before writing a canonical merged dataset. The iterative trainer accepts explicit `--train-cases` and `--heldout-cases` values and uses the stored manifest partition rather than randomly repartitioning merged pilot data.

## Run the prostate anatomy path

Generate matched parametric prostate demonstrations with a 64-cubed dose grid and 12 x 12 fluence maps:

The command below records the stopped four-to-seven-field calibration. Do not use it to create the next learner dataset. Expert angle refinement, manual normal-tissue priority, and coverage-tier labels must be frozen first.

```powershell
uv run python scripts/build_3d_dataset_pilot.py --retained-cases 300 --max-attempts 450 --automatic-train-count 240 --seed-start 106000 --grid-size 64 --anatomy prostate --fluence-size 12 --iterations 20 --deep-iterations 30 --reference-iterations 240 --initial-field-count 4 --minimum-field-count 7 --normal-tissue-weight 50 --normal-tissue-threshold 0.5 --integral-dose-weight 2 --d95-min 0.94 --d02-max 1.22 --paddick-ci-95-min 0.40 --r50-max 15 --max-steps 8 --beam-width 1 --deep-max-steps 10 --deep-beam-width 1 --device cuda:0 --output-dir outputs/prostate_manual_final7_300_v2
```

The generated `progress.json` file reports retained cases, attempts, percent complete, elapsed time, and estimated remaining time. The first 240 retained records receive the `train` split. The last 60 receive the `validation` split.

```powershell
uv run python scripts/build_3d_dataset_pilot.py --anatomy prostate --split-manifest outputs/splits/case_split_manifest.csv --split train --start-ordinal 0 --max-attempts 320 --retained-cases 240 --grid-size 64 --fluence-size 12 --iterations 20 --deep-iterations 40 --reference-iterations 240 --output-dir outputs/prostate_300_train
uv run python scripts/build_3d_dataset_pilot.py --anatomy prostate --split-manifest outputs/splits/case_split_manifest.csv --split validation --start-ordinal 0 --max-attempts 100 --retained-cases 60 --grid-size 64 --fluence-size 12 --iterations 20 --deep-iterations 40 --reference-iterations 240 --output-dir outputs/prostate_300_validation
uv run python scripts/merge_3d_dataset_shards.py outputs/prostate_300_train outputs/prostate_300_validation --expected-train 240 --expected-validation 60 --output-dir outputs/prostate_300_merged
```

Download one public TCIA subject and render the imported contours:

```powershell
uv run python scripts/download_tcia_prostate_subject.py Prostate-AEC-051
uv run --extra clinical python scripts/render_tcia_prostate_case.py data/tcia/Prostate-AEC-051 --grid-size 64 --output outputs/tcia_prostate_preview.png
```

The TCIA contour-only stress test uses at least 24 x 24 fluence maps. Raw DICOM data under `data/tcia` is excluded from Git.

The completed five-seed development run used 24 training cases and 60 validation cases. Run the combined analysis and the action-sequence audit with:

```powershell
uv run python scripts/analyze_3d_iterative_policy_pilot.py outputs/prostate_volume_policy_pilot_seed0 outputs/prostate_volume_policy_seeds1_4 --output-dir outputs/prostate_volume_policy_seeds0_4_analysis
uv run python scripts/analyze_prostate_action_failures.py outputs/prostate_volume_policy_seeds1_4/case_metrics.csv --output-dir outputs/prostate_volume_policy_seeds0_4_analysis
```

The trajectory condition improved the hard stratum but reduced moderate-case acceptability in the 24-case training pilot. A later one-seed run used all 240 training cases. It increased validation acceptability from 76.7% to 83.3% and removed the moderate-case failure loop in that seed. Four additional full-case seeds are required before progression. The 10,000-case experiment is not authorized by one seed.

During training, `progress.json` and `progress.log` report the current phase, completed work, elapsed time, and estimated remaining time. The `review_plans/` directory contains paired easy, moderate, and hard plan images and JSON audit records. Each record shows dose metrics, active beam angles, priorities, and every manual-level action.

## How methods and results will be shown

The full experiment will use a fixed visual sequence:

1. **Case card:** anatomy, planning goals, beam candidates, and difficulty.
2. **Trajectory filmstrip:** reoptimized plan states and the exact beam-angle or priority change between them.
3. **Objective story:** total score and each penalty term over planning steps.
4. **Endpoint-versus-trajectory comparison:** paired dots for the same test cases, so the improvement or failure is visible case by case.
5. **Learning curves:** performance versus number of training cases, showing sample efficiency.
6. **Failure atlas:** representative cases where each learner succeeds or fails, grouped by geometry and failure type.

The model-comparison stage will be added only after the environment and expert pass the validation gates in `simulation_plan.md`.
