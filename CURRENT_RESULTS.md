# Current Results

Verified against generated outputs on 2026-08-09.

## What is implemented

- A reproducible 2D 64 x 64 planning environment with a nested automated beamlet optimizer and high-level manual actions.
- A rule-based manual planner and bounded high-level search oracle using the same action vocabulary.
- Matched endpoint and trajectory dataset records with attempted and excluded cases retained in a manifest.
- A pre-trajectory 10,000-case split manifest containing 7,000 training, 1,000 validation, 1,000 IID-test, and 1,000 reserved OOD-test seeds. Its SHA-256 digest is `c9af78cd282846ee1410f9f688bbb3cb1b82294009560374e6b331310e269a2b`.
- A reproducible 3D 64 x 64 x 64 anatomy generator, implicit dose operator, exact adjoint, automated fluence optimizer, and high-level manual trajectory demonstration.
- An optional batched PyTorch 3D backend, differentiable inner optimizer, four-GPU benchmark script, and A100 server runbook.
- A validated CUDA installation on the local RTX 4060, including successful complete trajectories from 96- through 256-cubed.
- Twenty passing tests with CUDA-enabled Torch, including NumPy/PyTorch forward and adjoint parity, batched-state agreement, inactive-beam masking, difficulty-stratified geometry generation, split integrity, and high-level action integrity.

## 2D manual-planning pilot

The 20-case rule-based manual pilot reached the provisional synthetic goals in 10 of 20 cases (50%). The median trajectory contained five manual actions. The recorded actions were 39 OAR-priority increases, 22 target-priority increases, 3 hotspot-priority increases, 11 beam additions, and 10 beam removals.

This is an interpretable baseline, not the primary demonstration generator.

## 2D manual-versus-oracle pilot

The latest comparison contains 12 valid cases:

| Outcome | Cases |
|---|---:|
| Both reach the provisional goals | 3 |
| Oracle only | 3 |
| Manual policy only | 0 |
| Neither reaches within the search budget | 6 |

For seed 10000, the rule-based planner remained slightly outside the constraints after eight actions. The oracle reached them in three high-level actions: add the 210-degree beam, remove the 90-degree beam, and remove the 0-degree beam.

These results demonstrate that the selected sequence and beam configuration can affect whether a bounded planner reaches the goals. They do not yet establish that trajectory supervision improves a learned model.

## Matched dataset pilot

Five cases were attempted and four oracle-reachable cases were retained. The endpoint and trajectory views contain exactly the same four case IDs and final states. Only the trajectory view contains intermediate states and actions. The excluded case remains in the manifest with its stopping reason.

## 3D planning demonstration

The current 64-cubed example uses 12 candidate coplanar beams and an 8 x 8 fluence map per beam. Its recorded sequence contains four manual actions:

1. increase OAR 1 priority;
2. add the 30-degree beam;
3. increase OAR 2 priority;
4. increase target priority.

The automated inner optimizer recalculates the fluence maps after each action. Its beamlet/fluence updates are not stored as manual labels.

| State | Target D95 | Target D02 | OAR 1 / limit | OAR 2 / limit |
|---|---:|---:|---:|---:|
| Initial four beams | 0.820 | 1.130 | 1.212 | 1.174 |
| Increase OAR 1 priority | 0.804 | 1.192 | 1.118 | 1.145 |
| Add 30-degree beam | 0.835 | 1.157 | 0.985 | 1.045 |
| Increase OAR 2 priority | 0.832 | 1.122 | 0.952 | 0.953 |
| Increase target priority | 0.876 | 1.115 | 0.981 | 0.978 |

The final state passes the current provisional synthetic rules: D95 at least 0.85, D02 at most 1.25, and both OAR mean-dose ratios at most 1.0.

## Initial 3D environment calibration

A 12-case engineering calibration was completed locally at 96 cubed with 12 candidate beams, 16 x 16 fluence maps, and 60 inner-optimizer iterations per state. Each case contained exactly four recorded manual-level changes: two named-OAR priority increases, one geometry-informed beam addition, and one target-priority increase. Fluence updates remained inside the automated inner optimizer.

The initial optimized plans satisfied the provisional synthetic rules in 7 of 12 cases (58.3%). After the four high-level changes, 12 of 12 cases satisfied the rules. Median PTV D95 changed from 0.899 to 0.919, and the median maximum OAR mean-dose-to-limit ratio changed from 0.907 to 0.761. The run required 58.1 seconds (4.84 seconds per case) and 239 MiB peak allocated GPU memory on the local RTX 4060.

This is a small environment-response check, not a learner comparison and not evidence for the primary hypothesis. The uniform final success rate also indicates that the present two-OAR, non-overlap generator may be too easy for a definitive experiment. The next calibration must expand geometry difficulty and evaluate a bounded high-level planner before environment version 1.0 is frozen.

## Measured 3D computational scaling

The CPU reference benchmark uses 12 beams with 8 x 8 fluence maps:

| Grid | Voxels | Implicit cache | Dense float32 matrix avoided | Forward + adjoint |
|---|---:|---:|---:|---:|
| 64 cubed | 262,144 | 14.3 MiB | 0.75 GiB | 0.043 s |
| 96 cubed | 884,736 | 48.9 MiB | 2.53 GiB | 0.297 s |
| 128 cubed | 2,097,152 | 116.8 MiB | 6.00 GiB | 0.707 s |

One complete five-state, 60-iteration-per-state 64-cubed demonstration took 17.7 seconds in the latest full run. Timing varies slightly between runs; the earlier run took approximately 15 seconds.

For the proposed 96-cubed main environment with 16 x 16 fluence maps, a dense float32 influence matrix would be about 10.1 GiB per case. The implicit forward/adjoint design is therefore required. The batched PyTorch implementation and concurrent four-GPU benchmark entry point are ready, but the four A100 GPUs have not yet been benchmarked because server access has not been configured here.

The local RTX 4060 completes the full five-state 96-cubed trajectory in 4.26 seconds and the 128-cubed trajectory in 8.62 seconds. It also completes 192-cubed in 28.15 seconds and 256-cubed in 140.95 seconds; all four final states pass the provisional constraints. Detailed memory, quality, and batching measurements are in `LOCAL_GPU_RESULTS.md`.

## Present interpretation

The experiment mechanics are now demonstrated in both 2D and 3D. The current evidence shows that high-level beam/priority trajectories can be represented, reproduced, optimized, and audited. It does not answer the primary scientific question. The only learner comparison to date is a small pipeline check, not the prespecified iterative-policy experiment.

## Revised 3D validation and matched learner pilot

The revised target objective includes an additional lower-tail coverage term acting on the 10% most underdosed target voxels. Controlled priority tests compare each perturbed continuation with a neutral continuation of equal length from the same initial fluence. At 64 cubed with 200 continuation iterations, all 12 stratified target-priority perturbations increased PTV D95 and all 12 named-OAR perturbations decreased the selected OAR mean dose. All 100 generated geometries were exactly reproducible. Median target D95 change was +0.0106 and median named-OAR mean-dose change was -0.0133.

In an independent 30-case 32-cubed reachability audit, the routine bounded search reached 17 of 22 reference-reachable cases. A deeper high-level audit recovered the five residual cases, yielding 22 of 22 coverage among reference-reachable cases. Eight cases, including eight of ten hard cases, were not reached by the reference optimizer and were not treated as demonstration failures.

The regenerated matched dataset retained 32 acceptable demonstrations from 42 attempts in 311.9 seconds on the local RTX 4060. It contains 14 easy, 13 moderate, and 5 hard cases and 118 labeled transitions. Endpoint and trajectory views contain identical case identifiers and final targets; the endpoint records contain no trajectory field. Every retained case was acceptable to the independent reference check. One reference-reachable hard case was not recovered within the dataset search budget and remains in the failure manifest.

The 32-case network check used identical 31,156-parameter models and identical optimizer-update counts. Both conditions memorized the development set. Across five seeds on a fixed 24-case training and 8-case held-out split, endpoint-only mean final-setting MAE was 0.0870 and trajectory-supervised mean MAE was 0.0884. Trajectory supervision was better in two of five seeds. This result is inconclusive and is not the primary iterative closed-loop comparison.

A subsequent closed-loop development check exposed an action-legality defect: when stop was allowed in an unacceptable state, the trajectory policy stopped early in 25 of 40 case-seed evaluations and reached acceptability in 37.5%, compared with 72.5% for a direct endpoint regressor. Defining stop consistently with the demonstration protocol, so that it is legal only after all acceptance criteria are satisfied, increased trajectory-policy acceptability to 82.5% and reduced mean violation score to 0.00894, compared with 0.0536 for the direct endpoint regressor. Across 40 paired evaluations, trajectory violation was lower in 11, equal in 28, and higher in 1. Mean trajectory length was 4.5 high-level actions.

This closed-loop result is a development finding on eight held-out cases repeated across five training seeds. The endpoint arm is a direct terminal-settings regressor rather than the prespecified iterative endpoint-only policy, so this is not the primary comparison and no inferential claim is made.

## Matched iterative-policy development pilot

The primary comparator mechanics are now implemented. Both conditions used the same 31,156-parameter iterative policy, terminal plan-setting loss, policy-gradient loss computed from terminal simulator outcomes, legal-action mask, ten-action rollout limit, and optimizer-update count. The endpoint-only condition did not receive demonstration states or actions. The trajectory-supervised condition received the same terminal supervision plus categorical action loss at demonstration states. Each selected action was a high-level beam-membership or named-priority change, followed by actual fluence reoptimization.

The deterministic pilot used 24 training cases, eight held-out cases, three initialization seeds, 400 matched pretraining updates, and 30 matched terminal-rollout updates. Float32 geometry and deterministic PyTorch algorithms were used. Across 24 held-out case-seed evaluations per condition, endpoint-only acceptability was 54.2% and trajectory-supervised acceptability was 66.7%. Mean violation scores were 0.1129 and 0.0504, respectively, and mean trajectory lengths were 6.88 and 6.04 actions. Trajectory supervision was favored on aggregate but achieved higher acceptability in only two of three seeds.

Two independent deterministic smoke runs produced byte-identical case-metric files. The frozen full-pilot case-metric SHA-256 digest is `d4d0d0bfbd8c5063b52b72f502599c30b345a3f04f1805ad2dc56a530a665bb0`; the training-history digest is `3658d87617f6bafeafde579e5761858e119c986e68891e142b9d86cd64d9c457`.

This result demonstrates a matched, reproducible comparison path and supports proceeding to the variance pilot. It is not a primary result: the cohort is small, the effect is not consistent across initialization seeds, and the development cases were not drawn from the frozen train/validation manifest.

## Current decision

Environment, demonstration, shared rollout, action-mask parity, and the primary iterative comparator have passed their current engineering checks. The comparison remains a development result until the prespecified 300-case variance pilot is complete. The multi-day 10,000-case study should not begin before that pilot. The next compute milestone is the direct four-A100 correctness and throughput run described in `GPU_SERVER.md`, followed by a 300-case training/validation pilot drawn only from the frozen split manifest.

## Angular delivery complexity pilot

The current high-level planner starts from four cardinal fields but is not limited to four fields; it can add fields from 12 coplanar candidates. A paired 12-case, 64-cubed pilot compared four fixed angular parameterizations using 200 optimizer iterations and 8 x 8 fluence maps:

| Delivery representation | Angular samples | Acceptable | Median D95 | Median maximum OAR ratio | Median optimization time |
|---|---:|---:|---:|---:|---:|
| Four static fields | 4 | 25.0% | 0.976 | 1.174 | 0.72 s |
| Twelve static fields | 12 | 58.3% | 0.987 | 0.901 | 0.71 s |
| 180-degree arc-like sampling at 10 degrees | 19 | 58.3% | 0.988 | 0.901 | 0.91 s |
| 360-degree arc-like sampling at 10 degrees | 36 | 58.3% | 0.989 | 0.864 | 1.68 s |

The main change occurred between four and twelve directions. The full arc-like representation modestly reduced median OAR burden but did not increase the acceptable-plan count. This pilot used independent fluence maps at each sampled angle. It is not a delivery-realistic VMAT model because it omits MLC-aperture continuity, cumulative monitor units, dose-rate modulation, and gantry-speed constraints.

The recommended design is to retain the 12-angle static environment for the primary supervision experiment and add 180-degree and 360-degree arc-like sampling as prespecified delivery-complexity conditions. Manual-level arc actions should change the arc span, rotate its start and stop angles, add an avoidance sector, or add a second arc. Individual control-point fluence remains an inner-optimizer variable and must not become a manual behavior label.
