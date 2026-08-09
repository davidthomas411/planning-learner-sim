# Simulation Plan: Do Planning Trajectories Add Learnable Signal Beyond Endpoints?

## Current implementation status (2026-08-09)

The 2D nested planner, high-level search oracle, matched dataset pilot, implicit 3D NumPy/PyTorch backends, and pre-trajectory 10,000-case split manifest are executable. All 16 tests pass with CUDA-enabled Torch on the local RTX 4060. Complete 96-, 128-, 192-, and 256-cubed demonstration trajectories reach the provisional constraints; the 96-cubed trajectory takes 4.26 seconds and the 256-cubed trajectory takes 140.95 seconds locally. The endpoint-versus-trajectory learner comparison and direct four-A100 benchmark have not started. See `CURRENT_RESULTS.md` and `LOCAL_GPU_RESULTS.md` for measured values and the boundary between demonstrated mechanics and untested scientific claims.

## 1. Decision the simulation must support

The simulation should answer one narrow question:

> When final expert plans are held constant, does supervision from the expert's intermediate planning trajectory improve learned planning performance relative to supervision from final plans alone?

This is a proof-of-principle for collecting process-level data. It is not intended to demonstrate clinical benefit, reproduce Eclipse, or establish deliverability of real radiotherapy plans.

The primary claim is supported only if trajectory supervision improves held-out performance, convergence, or sample efficiency under a controlled comparison. A null result is also useful because it would identify which trajectory elements are redundant with the endpoint.

## 2. Design principles

1. **Matched information:** Both learners receive the same case description, initial plan, final expert plan, training cases, model architecture, parameter count, training budget, and inference-step budget.
2. **Single manipulated variable:** The endpoint learner receives loss only on the final solution. The trajectory learner receives the same final-solution loss plus intermediate action/state supervision.
3. **No test leakage:** Split by generated patient geometry before creating trajectories. All trajectories and perturbations from a case remain in one split.
4. **Independent evaluation:** Score predicted plans with a fixed evaluator and an oracle optimizer, not with the heuristic expert that generated the demonstrations.
5. **Reproducibility:** Store all configuration values, random seeds, code version, dataset hashes, checkpoints, and per-case outputs.
6. **Modest interpretation:** Report results as evidence about the value of process supervision in the synthetic environment, not as evidence of clinical superiority.

## 3. Synthetic planning environment

### 3.1 Case geometry

Use a two-dimensional 64 x 64 voxel grid. Each case contains:

- one elliptical planning target volume (PTV);
- one to three elliptical organs at risk (OARs);
- optional body boundary and avoidance region;
- controlled overlap or proximity between the PTV and OARs;
- 12 equally spaced candidate beam angles, each with 8 beamlets.

Generate shapes from parameterized ellipses so difficulty can be controlled and cases can be reproduced exactly. Reject anatomically trivial or invalid cases using prespecified rules.

Difficulty strata:

- **easy:** no PTV-OAR overlap and generous separation;
- **moderate:** close proximity or small overlap;
- **hard:** overlap, multiple competing OARs, or restricted useful beam angles.

### 3.2 Dose model

For each case, construct a deterministic nonnegative dose-influence matrix `A`, mapping 96 beamlet intensities `x` to voxel dose `d = A x`. Use a simplified ray-based kernel with attenuation and lateral Gaussian spread. Normalize kernels so prescriptions and penalties have comparable scales across geometries.

This model should preserve the essential planning structure—spatial dose deposition and competing objectives—without claiming clinical dose accuracy.

### 3.2a Staged 3D extension

Keep the 2D 64 x 64 environment as a fast correctness and ablation harness, but run the main spatial experiment in three dimensions:

- 64 x 64 x 64 for development and continuous integration;
- 96 x 96 x 96 for the primary 3D dataset;
- 128 x 128 x 128 for a prespecified resolution sensitivity analysis.

Use 12 coplanar candidate angles initially and 16 x 16 fluence pixels per beam for the 96-cubed and 128-cubed runs. Do not save a dense voxel-by-beamlet influence matrix per case: at 96-cubed this would require about 10.1 GiB in float32 for 12 beams with 16 x 16 fluence maps. Instead, save reproducible geometry parameters and evaluate an implicit forward dose operator plus its exact adjoint. Use float16 or bfloat16 for batched GPU kernels, float32 accumulation for plan metrics, and split independent cases across the four A100 GPUs. Add noncoplanar/couch-angle actions only as a later OOD condition after the coplanar environment passes validation.

Treat denser angular sampling as a separate delivery-complexity factor. The initial comparison will include a 180-degree arc-like set sampled every 10 degrees (19 control points) and a 360-degree set sampled every 10 degrees (36 control points). These conditions assign independent fluence maps to sampled gantry angles and must not be described as delivery-realistic VMAT until MLC-aperture continuity, cumulative monitor units, dose-rate modulation, and gantry-speed constraints are represented. High-level arc actions may change an arc's start or stop angle, expand or contract its span, rotate the span, add an avoidance sector, or add a second arc. Individual control-point fluence changes remain inside the automated optimizer and are excluded from manual labels.

### 3.3 Plan state and action

The recorded manual-planning state at step `t` contains:

- case geometry masks and prescription;
- current active beam angles and target/OAR priority settings;
- the beamlet intensities produced by the most recent automated optimization;
- current dose map `d_t`;
- summary metrics: PTV coverage, PTV hot spot, and OAR mean/max dose;
- voxelwise underdose and overdose violation maps;
- step index and remaining action budget.

A recorded action is a high-level manual planning change:

- add or remove one candidate beam angle;
- increase or decrease the target priority;
- increase or decrease the hot-spot priority;
- increase or decrease the priority of a named OAR;
- stop and accept the current plan.

After each manual action, an inner optimizer adjusts all active beamlet intensities with the angles and priorities held fixed. Those optimizer iterations are not treated as manual actions and are not used as behavior labels.

### 3.4 Plan-quality objective

Use a fixed, transparent objective:

```text
J(x) = w_cov * mean[(Rx - d_PTV)_+^2]
     + w_hot * mean[(d_PTV - 1.07 Rx)_+^2]
     + sum_k w_OAR,k * mean[(d_OAR,k - L_k)_+^2]
     + w_complexity * ||x||_1
     + w_smooth * beamlet_smoothness(x)
```

Report component terms as well as the total. Draw clinically inspired but explicitly synthetic OAR limits and weights from bounded distributions. Save the weights with each case so the task is fully specified.

Define acceptability before running experiments, for example:

- PTV D95 >= 95% of prescription;
- PTV D2 <= 107% of prescription;
- every OAR synthetic constraint is met;
- total normalized objective below a threshold calibrated on the training-free oracle.

The exact thresholds should be frozen after the environment-validation pilot and before model comparison.

## 4. Expert and oracle planners

### 4.1 Synthetic expert trajectory generator

Implement a deterministic nested planner:

1. begin from the same initial beam-angle template and neutral priorities;
2. run the automated beamlet optimizer;
3. review the resulting dose and unmet goals;
4. make one high-level beam-angle or target/OAR-priority change;
5. rerun the automated optimizer;
6. record the reviewed state, manual action, reoptimized state, objective components, and stopping reason;
7. stop when acceptable or after the manual-step limit.

The primary synthetic expert should use a bounded high-level search over beam-angle and priority actions; it may evaluate multiple high-level sequences but must never expose beamlet iterations as behavior labels. A separate rule-based planner should use explicit clinical-style heuristics based on target and OAR violations and serve as an interpretable baseline. Use deterministic tie breaking. The main dataset should contain one successful high-level expert trajectory per oracle-reachable case, with all failed or excluded cases retained in a manifest. A secondary experiment may include three perturbed high-level trajectories ending near the same solution to study behavior diversity.

### 4.2 Independent oracle

Use a continuous constrained optimizer over beamlet intensities to calculate the best attainable objective for each case. The oracle supplies:

- a lower-bound/reference objective;
- normalized regret for learned plans;
- an environment correctness check;
- a way to detect systematic weakness in the heuristic expert.

Do not train either learner on oracle trajectories. If the expert final plan is materially worse than the oracle, discard or regenerate the case using a prespecified tolerance.

## 5. Dataset

### 5.1 Records

For every case, store:

- `case_id`, generator version, random seed, and difficulty stratum;
- geometry masks, prescription, limits, weights, and dose-influence matrix or its reproducible parameters;
- common initial state;
- full expert sequence `(s_0, a_1, s_1, ..., a_T, s_T)`;
- final expert plan and dose;
- oracle plan, score, and solver status;
- trajectory length, stopping reason, and quality metrics at each step.

The endpoint view exposes the case, common initial state, and `s_T`. The trajectory view exposes exactly the same items plus intermediate states and actions.

### 5.2 Size and splits

Generate 10,000 valid cases after environment validation:

- 7,000 training;
- 1,000 validation;
- 1,000 in-distribution test;
- 1,000 out-of-distribution test.

Stratify the first three splits by difficulty. Use case-level splitting before trajectory generation.

OOD test conditions should include geometries outside the training ranges but within solver validity:

- greater PTV-OAR overlap;
- three OARs when training contains primarily one or two;
- shifted objective weights;
- missing or restricted beam-angle subsets.

For the sample-efficiency experiment, use nested training subsets of 100, 250, 500, 1,000, 2,500, 5,000, and 7,000 cases. Keep validation and test sets fixed.

## 6. Learners and fair comparison

Use one shared iterative policy architecture `pi_theta(case, state_t) -> action distribution`. It may encode masks and dose maps with a small CNN, encode scalar metrics with an MLP, and combine them to score legal actions.

### 6.1 Endpoint-only learner

Unroll the policy for the same maximum number of steps used by the trajectory model. Train using final-state losses only:

- distance from the expert final intensity/dose;
- final plan-quality objective;
- constraint violation penalty.

Intermediate expert states and actions must not be available to this learner.

### 6.2 Trajectory-supervised learner

Use the identical policy and final-state losses, plus:

- action imitation loss at each expert state;
- next-state or dose-change prediction loss, if included;
- optional monotonic-improvement regularizer.

Tune the relative intermediate-loss weight on validation data and freeze it before test evaluation.

### 6.3 Required baselines

- common initial plan with no learned edits;
- direct endpoint regressor from case to final intensities;
- heuristic expert;
- continuous oracle;
- endpoint learner with matched training compute;
- trajectory learner with matched training compute.

The primary comparison is trajectory-supervised versus endpoint-only iterative policy. Other baselines provide context and diagnose architecture effects.

## 7. Experimental matrix

### 7.1 Primary experiment

Train endpoint-only and trajectory-supervised policies on all 7,000 cases with 10 independent initialization seeds. Evaluate paired predictions on the same fixed IID and OOD cases.

Primary endpoint:

- normalized objective regret: `(J_model - J_oracle) / max(|J_oracle|, epsilon)`.

Secondary endpoints:

- acceptable-plan rate;
- PTV D95 and D2 surrogate metrics;
- each OAR constraint violation;
- steps to acceptability;
- objective improvement per action;
- inference time and number of dose evaluations.

### 7.2 Sample-efficiency experiment

Repeat training on every nested subset size. Plot test regret and acceptable-plan rate against number of training cases. Summarize area under the learning curve and estimate how many endpoint-only cases are required to match the trajectory model at 1,000 cases.

### 7.3 Robustness and ablations

Run these in order after the primary comparison is frozen:

1. **Actions only:** final loss plus action labels, without intermediate-state prediction.
2. **States only:** final loss plus intermediate-state targets, without action imitation.
3. **Trajectory order shuffled:** preserves the quantity of supervision but destroys temporal order.
4. **Sparse trajectory sampling:** retain every 2nd, 5th, or 10th action.
5. **Noisy expert:** add suboptimal actions at 5%, 10%, and 20% of steps.
6. **Multiple valid experts:** train on diverse near-optimal trajectories for the same case.
7. **Equal optimizer updates:** verify that any gain is not explained by extra gradient steps.
8. **Equal labeled targets:** downsample trajectory labels to test whether gains reflect label quantity alone.
9. **Hidden rationale test:** remove violation maps and retain only observable dose/metric state to approximate what screen-derived trajectories may capture.

## 8. Statistical analysis

Treat the test case as the unit of analysis. Because models are evaluated on identical cases, use paired differences.

- Report mean, median, interquartile range, and 95% bootstrap confidence intervals across test cases.
- Aggregate across the 10 training seeds with a hierarchical bootstrap over seed and case.
- For the primary endpoint, report the paired mean difference in normalized regret and its confidence interval.
- For acceptable-plan rate, report paired risk difference and confidence interval.
- Control the false-discovery rate for secondary and ablation comparisons; do not adjust the single preregistered primary comparison.
- Report effect sizes, not only p-values.

Suggested success criteria, to freeze before the main run:

1. the trajectory model has a lower mean IID normalized regret and its 95% confidence interval excludes zero;
2. the improvement is reproduced in at least 8 of 10 training seeds;
3. the acceptable-plan rate does not decrease;
4. either OOD regret or sample efficiency improves without a material increase in inference cost.

These criteria establish a synthetic proof of principle; they do not validate a clinical effect size.

## 9. Validation gates

### Gate A: environment correctness

- dose is linear and nonnegative;
- action application exactly matches logged state transitions;
- objective components reproduce from saved records;
- stronger target weights increase coverage in controlled test cases;
- stronger OAR weights reduce OAR dose in controlled test cases;
- identical seeds reproduce identical cases and expert trajectories.

### Gate B: expert quality

- at least 95% of retained IID cases reach synthetic acceptability;
- expert objective is within a frozen tolerance of the oracle;
- objective usually decreases over time; all increases are explained by an explicitly enabled exploration condition;
- stopping reasons and trajectory lengths are plausible across difficulty strata.

### Gate C: comparison integrity

- identical train/validation/test case IDs for both learners;
- identical initialization-seed schedule, parameter count, action mask, rollout limit, and training compute;
- automated check that endpoint-only data loaders never expose intermediate expert fields;
- test set remains unopened until model and threshold choices are frozen.

### Gate D: reporting integrity

- every figure can be regenerated from a single results table;
- failed runs and solver failures remain logged;
- all exclusions follow prespecified rules;
- conclusions distinguish IID, OOD, and ablation evidence.

## 10. Execution sequence

### Phase 0: specification

- freeze the state/action ontology, objective, acceptability definition, data schema, split rules, and primary analysis;
- write unit tests for geometry, dose, action transitions, objective terms, and serialization.

### Phase 1: environment pilot (100 cases)

- visually inspect representative geometry and dose maps;
- verify all Gate A checks;
- tune only environment scales, expert step sizes, and provisional acceptability thresholds;
- freeze environment version 1.0.

### Phase 2: expert pilot (1,000 cases)

- generate trajectories and oracle solutions;
- assess failure rate, trajectory-length distribution, expert-oracle gap, and difficulty balance;
- revise expert mechanics if Gate B fails, then regenerate all pilot data;
- freeze generator and expert version 1.0.

### Phase 3: model pilot

- use training/validation data only;
- confirm that both models can overfit 32 cases;
- choose architecture, optimizer, rollout length, and trajectory-loss weight;
- run a compute audit and freeze the training configuration.

### Phase 4: main simulation

- generate the 10,000-case versioned dataset;
- train both primary models at 10 seeds;
- evaluate once on IID and OOD test sets;
- produce the primary paired analysis and learning curves.

### Phase 5: ablations

- execute the prespecified ablation order;
- stop low-priority ablations if the primary result is null, except shuffled-order and equal-compute controls, which remain necessary to interpret the null.

### Phase 6: report

- publish the configuration, seeds, dataset manifest, code commit, result table, confidence intervals, and failure log;
- create a concise figure showing endpoint-only versus trajectory-supervised learning curves and final regret;
- explicitly state limitations and the mapping needed from synthetic actions to screen-derived clinical events.

## 11. Expected outputs

1. versioned synthetic dataset with endpoint and trajectory views;
2. environment and dataset data cards;
3. validation report for Gates A-C;
4. model checkpoints and training logs;
5. one tidy per-case results table;
6. primary comparison figure;
7. sample-efficiency figure;
8. robustness/ablation figure;
9. short methods and limitations text suitable for the proposal.

## 12. Risks and mitigations

- **The heuristic trajectory simply reveals the optimizer.** Use an independent oracle and test noisy/multiple experts; frame the finding as learnability of expert process, not discovery of unique clinical reasoning.
- **The endpoint has multiple valid solutions.** Prefer dose/objective losses over raw intensity distance and include diverse near-optimal endpoints in a secondary condition.
- **Trajectory learning gets more labels or compute.** Include equal-update and equal-label controls.
- **The synthetic state contains information unavailable from screen capture.** Run the hidden-rationale ablation and define a mapping from every retained state variable to a proposed clinical data source.
- **The task is too easy.** Use difficulty strata, oracle-normalized regret, and frozen OOD conditions.
- **The task is too hard or unstable.** Validate on 100 cases, check expert-oracle gaps, and freeze a solvable environment before model training.
- **Results do not translate clinically.** Treat the result only as justification to collect and study the missing data modality; clinical validation requires real trajectories and dosimetric review.

## 13. Mapping to future clinical trajectory data

Before claiming that the simulation supports session capture, map the synthetic fields to observable clinical counterparts:

| Synthetic element | Proposed clinical counterpart |
|---|---|
| case geometry masks | CT and structure set |
| prescription and synthetic limits | prescription and planning goals |
| beamlet/beam state | RTPLAN plus TPS interface state |
| dose and violation summaries | dose calculation, DVH, and goal status |
| beam-angle action | beam addition, removal, or angle change |
| target/OAR priority action | optimization-objective or constraint-weight edit |
| action order and timing | session-derived event timeline |
| stop/accept decision | plan review, save, approval, or export event |
| final plan | approved RTPLAN and dose |

Any synthetic feature without a credible clinical counterpart should be removed from the primary learner or labeled as privileged information and tested only in a separate upper-bound analysis.

## 14. Minimal grant-ready version

If time or compute is limited, run a reduced but still interpretable study:

- 2D 64 x 64 environment, one PTV and up to two OARs;
- 2,000 cases split 1,400/200/200/200;
- 100, 500, and 1,400-case learning curves;
- five training seeds;
- identical iterative policy with endpoint-only versus trajectory-supervised loss;
- IID and one OOD overlap condition;
- normalized oracle regret, acceptable-plan rate, and steps to acceptability;
- required equal-compute and shuffled-order controls.

This reduced design should be presented as a feasibility experiment. The 10,000-case design remains the preferred confirmatory simulation.
