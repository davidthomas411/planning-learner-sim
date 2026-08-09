# Methods Walkthrough

This guide explains what the experiment does, what every object means, and how each result will be made understandable.

## The question in plain language

Suppose two students see the same final treatment plan:

- Student A sees only the starting problem and the final answer.
- Student B sees the same problem and final answer, plus every adjustment the expert made along the way.

The experiment asks whether Student B learns to plan better on new synthetic cases. The final answer is identical for both students; only the intermediate teaching signal differs.

## What one synthetic case contains

The first image, `outputs/demo/01_case_construction.png`, shows three ingredients:

1. **Anatomy:** a green target that should receive dose and two colored avoidance structures that should receive less dose.
2. **One optimizer variable's physical effect:** increasing one beamlet produces a spatial band of dose.
3. **Available beam geometry:** 12 candidate directions contain 8 beamlets each.

The simulator uses a 64 x 64 grid. It is deliberately simpler than clinical radiation transport, but it preserves the decision structure: a planner chooses beam geometry and priorities, then an optimizer distributes intensity across the available beamlets.

### The new 3D version

The executable 3D prototype uses 64 x 64 x 64 voxels. `outputs/3d_demo/01_3d_anatomy.png` shows the same synthetic patient in axial, coronal, and sagittal planes. A beam carries an 8 x 8 fluence map. For each beam angle, the simulator rotates every body voxel into beam's-eye coordinates, bilinearly samples that fluence map, applies depth attenuation, and sums dose from all active beams.

It does not construct a dense voxel-by-beamlet matrix. The forward operator calculates dose on demand, and its mathematically matched adjoint sends a voxelwise objective gradient back to fluence pixels. A unit test verifies their inner products agree. This is the key interface that will be replaced by batched GPU kernels on the four-A100 server.

The 3D demonstration contains four recorded manual actions: increase OAR 1 priority, add the 30-degree beam, increase OAR 2 priority, and increase target priority. After each edit, the automated fluence optimizer runs to completion. The final example reaches target D95 0.876, target D02 1.115, and OAR mean-dose ratios 0.981 and 0.978, satisfying the current provisional rules.

## How a plan is scored

Every candidate plan receives one number called the objective score. Lower is better. It is the sum of five visible components:

1. target underdose;
2. target hot spots;
3. avoidance-structure overdose;
4. total beamlet intensity;
5. neighboring-beamlet roughness.

The component values are never hidden. `outputs/manual_demo/03_manual_metrics.png` shows the target and OAR response after every manual setting change. This prevents a low total score from concealing a bad trade-off.

The v0.6-manual feasibility rule currently calls a plan acceptable when:

- target D95 is at least 0.85 relative dose;
- target D02 is at most 1.25 relative dose;
- mean dose to each avoidance structure is below that case's stored limit.

These are synthetic, provisional thresholds. They are not clinical constraints. They will be calibrated across the environment pilot and frozen before any learner comparison.

## Two nested loops

The experiment now explicitly separates optimization from manual planning.

### Automated inner loop

For a fixed set of beam angles and fixed target/OAR priorities, the optimizer adjusts beamlet intensities until it cannot improve its objective. These beamlet iterations are implementation details. They are not labeled as human actions and are not the trajectory-supervision target.

### Recorded manual loop

After each optimizer run, the synthetic planner reviews target D95, target D02, OAR mean-dose ratios, and the dose map. Its allowed actions are:

- increase target priority;
- increase target hot-spot priority;
- increase the priority of a specific OAR;
- add a beam angle;
- remove a beam angle.

After one such action, the automated optimizer runs again from the revised settings. The transition from the reviewed plan through the manual action to the new optimized plan is one stored trajectory step.

### Rule-based planner versus search oracle

Two high-level planners now use the same manual action vocabulary:

- the **rule-based planner** chooses an action from the currently largest visible target or OAR violation;
- the **high-level search oracle** evaluates a limited beam of plausible beam/priority sequences and keeps the sequences with the lowest priority-independent clinical violation.

Neither exposes beamlet adjustments as manual actions. The search oracle is used to test whether the high-level action space can reach the goals and to generate successful synthetic-expert demonstrations. Failure of this search is labeled `not_reached_by_oracle`; it is not proof that a physical plan is impossible.

The current demonstration uses seed `10000`. Its eight recorded actions include OAR-priority increases, target-priority increases, removal of the 0-degree beam, and addition of a 240-degree beam. The case deliberately remains slightly outside the provisional constraints after eight actions, making the failure visible rather than hiding it.

For the same seed, the search oracle reaches all provisional goals in three high-level actions: add the 210-degree beam, remove the 90-degree beam, and remove the 0-degree beam. This paired case is the clearest example of why trajectories matter: the final outcome depends on the order and combination of high-level decisions, not merely on repeatedly increasing penalties.

## What exactly is stored as a trajectory

For manual action `t`, the dataset stores:

```text
case anatomy, prescription, and planning goals
+ current active beam angles
+ current target, hot-spot, and OAR priorities
+ current optimized dose and plan-quality metrics
+ one manual beam-angle or priority change
+ reoptimized dose and plan-quality metrics
```

The inner optimizer's beamlet iterations may be retained for debugging, but they are not exposed to the behavior learner as manual labels. The readable trajectory is `outputs/manual_demo/manual_trajectory.csv`. Each row is one high-level decision.

## What the two learners will receive

Both learners will use the same iterative policy architecture and the same cases, initial states, final plans, number of parameters, training budget, and inference-step limit.

The endpoint-only learner receives:

```text
case + initial planning settings -> expert final settings and plan
```

The trajectory-supervised learner receives:

```text
case + reviewed optimized plan at step t
-> expert beam/priority change at step t
-> next reoptimized plan
```

It also receives the same final-plan loss used by the endpoint learner. Therefore, intermediate supervision is the intended difference between the primary conditions.

## How model results will be shown

No conclusion will depend on a single aggregate number. Results will be shown in six linked views:

1. **Paired-case plot:** every test case has one endpoint-only result and one trajectory result joined by a line. This shows how often the benefit occurs and exposes cases that get worse.
2. **Learning curve:** objective regret and acceptable-plan rate versus 100, 250, 500, 1,000, 2,500, 5,000, and 7,000 training cases.
3. **Planning movie/filmstrip:** the same held-out case planned by both models, with identical color scale and step numbers.
4. **Constraint plot:** target and avoidance metrics over steps, including the frozen acceptability thresholds.
5. **Failure atlas:** representative successes and failures grouped by overlap, number of avoidance structures, and out-of-distribution condition.
6. **Ablation plot:** full trajectory supervision compared with shuffled, sparse, noisy, action-only, and state-only trajectories.

Each figure will link back to a tidy per-case results table so the displayed result can be audited.

## Current boundary

The nested 2D environment, matched endpoint/trajectory dataset builder, 3D implicit dose operator, automated inner optimizer, and high-level manual trajectory are now executable. Ten validation tests pass. The learner comparison and four-A100 benchmark have not started.

The earlier beamlet-action prototypes remain useful as validation history:

- environment v0.1 reached provisional acceptability in only 7 of 100 cases;
- removing direct target-OAR overlap and loosening the synthetic OAR limits in v0.2 increased this to 29 of 100 cases;
- adding objective terms aligned to D95, D02, and mean OAR limits did not solve the problem: the v0.5 diagnostic reached acceptability in 9 of 30 cases;
- despite the low acceptability rate, v0.5 removed a median of roughly 99.5% of the numerical objective.

That contrast motivated the nested redesign. The optimizer now has a narrower job: find beamlet intensities for the planner's chosen angles and priorities. The manual trajectory contains the higher-level decisions used to change that optimization problem.

The first 12-case oracle comparison found:

- 3 cases where both the rule-based planner and search oracle reached the goals;
- 3 cases reached only by the search oracle;
- 0 cases reached only by the rule-based planner;
- 6 cases not reached by either method within the search budget.

The matched dataset pilot now retains four oracle-reachable cases from five attempts, with identical endpoint/trajectory case IDs and every exclusion preserved in the manifest. Scaling this pilot is not yet justified: a continuous feasibility optimizer and environment calibration are still required before calling remaining cases physically infeasible or freezing the training dataset. Current measurements are consolidated in `CURRENT_RESULTS.md`.
