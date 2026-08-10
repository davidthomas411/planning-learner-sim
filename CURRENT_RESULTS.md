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

## Three-dimensional image encoder and prostate phantom

The matched image-plus-scalar policy uses 11 current-state volume channels: body, target, three OAR masks, dose, target underdose, target hot spot, and three OAR excess-dose maps. Both conditions use the same 93,476-parameter network. The endpoint condition does not receive action labels. A one-seed development run used 24 training cases and all 60 held-out validation cases. The endpoint acceptable-plan rate was 46.7%, and the trajectory-supervised rate was 58.3%. Mean violation decreased from 0.177 to 0.106. This is a validation-stage engineering result. It is not a primary result and has no multi-seed confidence interval.

A prostate-specific parametric generator now produces a pelvic body, PTV-like prostate target, bladder, rectal wall, and two femoral-head shapes. The femoral heads share one priority group. Across 300 anatomy-only cases, all structures were valid. Median total target-OAR overlap fractions were 0.000, 0.056, and 0.243 for easy, moderate, and hard cases. At 64 cubed with 12 x 12 fluence maps, four of four sampled hard cases were reachable by the independent reference optimizer. An end-to-end dataset smoke test retained one easy, one moderate, and one hard case in 8.0 seconds. Both the bounded manual-level search and the reference optimizer met all dose rules. The selected prostate policy input is 32 cubed.

The full prostate development dataset is now complete. It contains 240 training cases and 60 validation cases retained from 322 attempts. Retention requires both an acceptable bounded-search demonstration and an acceptable independent reference plan. The retained cohort contains 108 easy, 107 moderate, and 85 hard cases. Easy and moderate retention were 100%; hard retention was 79.4%. The mean demonstration contained 2.29 manual actions before stop. No reference-reachable case was missed by bounded search. The endpoint SHA-256 digest is `9a729d95a22e329c4fca3461350ebca6a2e0954bea309341b25958d30d1f686e`; the trajectory digest is `0f35e0753a4d66c1cb0d02f2e12f0587a227329fd27b9c917a514d9dfa7acda0`; and the attempt-manifest digest is `2f23a1c61d45ba3c3163e16d10bb5c2294c79ff7d2978c25ac843396a5c0e721`.

A five-seed prostate image-policy variance pilot used 24 training cases and all 60 validation cases. It produced 300 paired case-seed evaluations. Endpoint and trajectory acceptability were 75.7% and 70.0%, respectively. The paired difference was -5.7 percentage points. The hierarchical-bootstrap 95% interval was -17.0 to +4.7 percentage points. Mean violation scores were 0.0574 and 0.0422. The paired violation difference was -0.0152, with a 95% interval from -0.0512 to +0.0155. One seed favored trajectory supervision for acceptability, one seed tied, and three seeds favored endpoint supervision. Three seeds favored trajectory supervision for mean violation.

Both conditions accepted all easy cases. On moderate cases, endpoint and trajectory acceptability were 90.9% and 66.4%. On hard cases, the rates were 16.0% and 29.3%. Mean hard-case violation fell from 0.2192 to 0.1068 with trajectory supervision. Thus, trajectory supervision helped the hard stratum but harmed the moderate stratum under this small-data schedule. The trajectory action accuracy was 43.7%.

Action sequences were available for seeds 1 through 4. All 25 failed moderate-case trajectory rollouts reached the 10-action limit. Twenty used the same dominant loop: six increases in femoral-head priority followed by four increases in hot-spot priority. Failed moderate rollouts contained a mean of 5.64 femoral-head priority increases and 4.12 hot-spot priority increases. Accepted moderate rollouts contained means of 2.14 and 1.35. The next model run must use all 240 training cases and must test recovery from repeated-action states before the main experiment.

A subsequent one-seed calibration used all 240 training cases, the same 60 validation cases, 100 pretraining updates, and 15 closed-loop updates per condition. It completed on the local RTX 4060 in 1429.8 seconds and used 7119 MB of peak CUDA memory. Endpoint and trajectory acceptability were 76.7% and 83.3%. The paired difference was +6.7 percentage points. Mean violation fell from 0.0625 to 0.0243. There were 46 cases accepted by both models, four accepted only by the trajectory model, no cases accepted only by the endpoint model, and ten accepted by neither model.

Both models accepted all 23 easy cases. Endpoint and trajectory models accepted 21 and 22 of 22 moderate cases. They accepted 2 and 5 of 15 hard cases. Hard-case mean violation fell from 0.2492 to 0.0972. All moderate trajectory plans were acceptable, so the repeated moderate-case action loop was absent in this seed. The trajectory action accuracy was 37.1%. This is a one-seed validation result. Its case-bootstrap interval does not measure training-seed uncertainty and must not be used as confirmatory evidence.

The trainer now writes a live progress record with phase, completed work units, elapsed time, and estimated remaining time. It also saves paired review plans for easy, moderate, and hard cases. Each review plan contains initial and final dose images, anatomy contours, target and OAR metrics, beam angles, priorities, and the complete high-level action sequence. The raw per-case table contains the same audit fields for all 120 validation plans.

One subject from the TCIA Prostate Anatomical Edge Cases collection was downloaded and imported. The RTSTRUCT contained prostate, bladder, rectum, and separate left and right femoral-head contours. The RTSTRUCT referenced the downloaded 178-slice CT series. A 5-mm isotropic prostate margin produced the PTV-like target. The contour-only importer removed disconnected couch pixels, padded the pelvic field of view in physical coordinates, and resampled the masks to 64 cubed. The reference optimizer did not meet the synthetic rules with 12 x 12 or 16 x 16 fluence maps. Violation decreased to 0.020 at 20 x 20. The case met all rules at 24 x 24 and 32 x 32. The external-anatomy test will therefore use at least 24 x 24 fluence maps. CT density is not yet used in dose calculation.

## Prostate field-count and spatial-objective revision

The original prostate environment started with four cardinal fields and accepted plans using only PTV D95, PTV D02, and OAR mean-dose limits. Human review showed a four-field cross-shaped dose distribution and poor dose spill. In the one-seed 240-case learner run, 43 of 60 endpoint plans still used four fields. The resulting synthetic acceptability label did not indicate clinical acceptability.

A paired 12-case field-count pilot used identical anatomy, fluence resolution, and optimizer settings. With the original objective, increasing from four to seven static fields reduced median R50 from 23.82 to 12.17 and reduced the median maximum OAR mean-dose ratio from 1.15 to 0.81. Additional fields produced smaller OAR gains and did not improve Paddick CI95 because the inner objective had no normal-tissue term. The 36-point arc-like condition was an angular-sampling test, not delivery-realistic VMAT.

The inner objective now includes a normal-tissue excess-dose term above 0.50 relative dose and a normal-tissue integral-dose term. A 12-case calibration selected base weights 50 and 2, respectively. At seven fields, this setting produced median Paddick CI95 0.572, median R50 10.11, median PTV D95 0.984, and median maximum OAR mean-dose ratio 0.848. A planner-controlled normal-tissue priority now multiplies both base terms. The manual action set can increase or decrease this priority after review of coverage, OAR dose, and dose spill. These values remain properties of a dose surrogate and are not clinical planning constraints.

The revised manual task starts from four cardinal fields and requires a final plan with at least seven fields, PTV D95 at least 0.94, PTV D02 at most 1.22, Paddick CI95 at least 0.40, R50 at most 15, and all OAR mean doses within their stored limits. The outer process can only add or remove one beam or change one named target, hot-spot, or OAR priority. The inner optimizer controls beamlet fluence and is not recorded as manual behavior.

A 12-case calibration retained 12 of 13 attempts in 72.2 seconds. The cohort contained 61 manual actions plus 12 stop labels. Trajectories contained 3 to 8 actions, with a mean of 5.08. Recorded actions included 39 beam additions, 2 beam removals, 14 hot-spot-priority increases, 3 OAR-priority increases, and 1 target-priority increase. The easy, moderate, and hard review cases ended with 7, 7, and 8 fields. Their R50 values changed from 34.9 to 12.4, 11.8 to 10.3, and 16.3 to 10.0, respectively.

Three attempted 300-case generations were stopped during validation. The first exposed loss of the hard stratum. The second lacked stored train and validation labels. The third was stopped after review showed that 30-degree add-or-remove actions did not represent expert beam-angle refinement. Their output directories remain as failed calibration records and are not part of a final dataset. No revised 300-case generation is active.

A deterministic rule-based angle-selection pilot started from seven equally separated fields and allowed two 10-degree refinements. The rule used beam-eye-view PTV-OAR overlap, weighted by the current OAR dose-to-limit ratios. Across 12 paired cases, median maximum OAR dose-to-limit ratio decreased by 0.0262 and median Paddick CI95 increased by 0.0063. Median R50 increased by 0.272. Thus, the geometry rule contained anatomy-dependent OAR information but did not control dose spill. Ten-degree refinement will be retained for further development. The older 30-degree shift action is disabled. A 5-degree sensitivity test is planned after the 10-degree representation is integrated.

Target coverage will use explicit goal tiers. The primary standard tier requires D95 at least 0.94. A separate compromise tier may use D95 at least 0.90 only after a standard-goal reference attempt fails and all noncoverage rules pass. In a deterministic 12-hard-case audit, 4 cases met the standard tier, 2 met the compromise tier, and 6 remained unreached. Standard and compromise cases will not be pooled in the primary comparison.

## Current decision

Environment, demonstration, shared rollout, action-mask parity, and the primary iterative comparator have passed their current engineering checks. The 300-case generic dataset, 300-case prostate dataset, scalar-model variance pilot, matched three-dimensional image encoder, one-subject TCIA contour import, five-seed small-data pilot, and one-seed 240-case calibration are complete. The 240-case result removes the immediate moderate-case failure, but one seed cannot measure training variance. The 10,000-case experiment should not begin until at least four additional 240-case seeds reproduce the direction of effect and the saved plans pass human review.

## Protocol-inspired prostate DVH objectives

The prostate path now uses an explicit 60 Gy in 20 fraction evaluation. It reports PTV D98, D99, and D02 in Gy and NRG Table 10 rectum, bladder, and femoral-head Vx values in percent volume. Differentiable dose-volume penalties can be activated in the inner optimizer with a separate base weight. The planning configuration can require either the represented per-protocol tier or the variation-acceptable tier. This tier is now part of stop legality and the violation score. OAR priority changes use the worst active DVH-goal ratio when a protocol tier is active.

A three-case, seven-field weight scan compared clinical DVH weights 2, 5, 10, and 20. Median PTV D98 increased from 58.31 Gy at weight 2 to 58.70 Gy at weight 20. Median CI95 decreased from 0.57 to 0.53, and median R50 increased from 11.24 to 11.49. The hard case's worst OAR constraint ratio increased from 1.30 to 1.46. Thus, a larger combined DVH weight improved target coverage but worsened the hard-case OAR and conformality tradeoff. Weight 5 is retained as a provisional balanced value; it is not frozen.

A strict end-to-end smoke test required seven fields, D95 at least 0.94, D02 at most 1.10, CI95 at least 0.40, R50 at most 15, and the represented variation-acceptable DVH tier. No case was retained from three low-resolution attempts, and the reference optimizer also failed all three. A subsequent 64-cubed, 24 x 24 fluence check produced median CI95 0.80 and median R50 4.99, but D02 remained 70.8-72.9 Gy. Increasing hot-spot priority from 1 to 25 did not reduce D02 below 66 Gy. This showed that D02 was an invalid replacement for the protocol D0.03-cc maximum because it represents a much larger target volume. D02 remains an engineering diagnostic and has been removed from the protocol pass label. After this correction, two of the three fixed seven-field 24 x 24 plans met the represented variation-acceptable D98, D99, and OAR goals. The hard case remained outside the goals. The next calibration must report D02 separately and preserve an independently declared engineering hot-spot bound.

The expanded fixed-plan calibration used 12 cases, a 64-cubed grid, seven fields, 24 x 24 fluence maps, clinical DVH weight 5, and 300 optimizer iterations. Eight of 12 plans met every represented variation-acceptable D98, D99, and OAR goal. All four easy and all four moderate cases passed. None of the four hard cases passed. Median D98 was 59.46 Gy, median CI95 was 0.851, and median R50 was 4.30. The hard cases had D98 values from 57.79 to 58.46 Gy; three also exceeded the per-protocol bladder V60 limit, although that limit was not the principal variation-tier failure. This establishes the next manual task: increase target priority on hard cases, then adjust bladder priority or beam angle only if the resulting DVH requires it.

A paired target-priority calibration used the same four hard cases and fixed all other settings. The tested target-priority multipliers were 1.00, 1.75, 3.0625, and 5.359375. The represented variation-acceptable pass rates were 0%, 75%, 100%, and 100%, respectively. All OAR limits passed at every tested priority. Median D98 increased from 58.32 Gy at priority 1.00 to 59.30 Gy at priority 3.0625. Median CI95 decreased from 0.866 to 0.836, and median R50 increased from 4.01 to 4.09. Priority 5.359375 produced no additional accepted plans and caused a further conformity and dose-spill cost. The provisional manual rule is therefore a target-priority increase from 1.00 to 1.75, followed by review, with a second increase to 3.0625 only if the target still fails.

The closed-loop check used the same four hard cases, seven fixed equally spaced fields, 24 x 24 fluence maps, 300 inner-optimizer iterations per state, and the represented variation-acceptable review tier. All four cases passed. Three cases required one target-priority change from 1.00 to 1.75. One case required a second change to 3.0625. No OAR priority or beam-angle change was required. The worst OAR variation-limit ratio remained between 0.41 and 0.61 across all recorded states. Individual dose and DVH review images were written for every case. An initial implementation used seven fields selected from a 30-degree candidate grid; it was rejected because its angles did not match the fixed-plan calibration. The reported closed-loop result uses 0, 51.4, 102.9, 154.3, 205.7, 257.1, and 308.6 degrees.

The rule was then evaluated on all 15 hard validation cases. A target-only rule identified five cases in which target coverage passed but the separate engineering D02 limit of 75 Gy remained exceeded. Continuing to increase target priority worsened the hot spot and was rejected as an invalid manual response. The corrected rule increased target priority only when coverage failed and increased hot-spot priority when coverage passed but D02 remained high. This corrected sequence accepted all 15 cases within five manual steps. The trajectories contained 21 target-priority actions and seven hot-spot-priority actions. The median action count was one, with a range of one to four. Final D02 ranged from 71.85 to 74.30 Gy. Median D98 was 59.08 Gy, median maximum OAR variation-limit ratio was 0.450, median CI95 was 0.846, and median R50 was 4.40. All reported OAR variation limits passed. The 75-Gy D02 limit remains an engineering constraint and is not a surrogate for a clinical D0.03-cc constraint.

An anatomy-only beam-angle rule was evaluated after priority calibration. The rule selected 5- or 10-degree field shifts from beam-eye-view target-OAR overlap and did not optimize candidate plans before the manual choice. Two 10-degree shifts reduced acceptance from 15/15 to 10/15. One 10-degree shift reduced acceptance to 12/15. One 5-degree shift reduced acceptance to 14/15. The 5-degree condition produced a small median reduction in the worst per-protocol OAR ratio (-0.015) and a near-zero median D98 change (+0.003 Gy), but median D02 increased by 0.79 Gy and median R50 increased by 0.19. Among the 11 cases with a baseline per-protocol OAR ratio above 1, only six improved that ratio. Thus, the tested beam-eye-view rule does not reliably predict dosimetric benefit and is excluded from demonstration generation. The negative result is retained as a calibration finding.

The local status page now includes a live figure gallery. Pilot scripts rewrite available summary figures after each completed case, and the browser refreshes both progress and images every two seconds without model calls.

The calibration runner now writes an atomic `progress.json` update after each completed plan and a local `status.html` page that polls it every two seconds. The polling loop is independent of the model and consumes no tokens. A detached local launcher starts the GPU run and HTTP status server without opening visible console windows and records both process identifiers.

## Local 300-case train/validation dataset

The development-resolution variance-pilot dataset was generated from the frozen split manifest with 8 x 8 fluence maps. It contains 240 training cases and 60 validation cases retained from 405 attempted cases. The endpoint and trajectory views contain exactly the same case identifiers, settings, and terminal states; only the trajectory view contains intermediate high-level actions. No training case appears in validation.

| Difficulty | Attempted | Retained | Retention rate |
|---|---:|---:|---:|
| Easy | 136 | 136 | 100.0% |
| Moderate | 135 | 125 | 92.6% |
| Hard | 134 | 39 | 29.1% |
| All | 405 | 300 | 74.1% |

The retained training partition contains 110 easy, 101 moderate, and 29 hard cases. Validation contains 26 easy, 24 moderate, and 10 hard cases. The median demonstration contained two high-level actions before stop, the mean contained 2.75, and the range was one to ten. Eight reference-reachable attempts were not recovered by bounded high-level search; demonstration coverage among reference-reachable attempts was therefore 97.4%. All retained demonstrations were acceptable and reference-acceptable.

The canonical merged endpoint SHA-256 digest is `298e025e7c574d6d251647033b2cdc7de0dc03dc2c78abcd114c216c398f82dc`; the trajectory digest is `2f5172f6dfc79e88157119c762428a65a7ef817bcc60021ab2c7f97dd3f25c64`; and the complete attempt-manifest digest is `2ef0608fb190658c2cadf45d49e589114d384b1eff944157dd98079423d7d127`.

An initial 4 x 4 fluence attempt retained only 21 of 120 validation cases, all easy, and was rejected before learner training. The corrected 8 x 8 representation restored moderate and hard cases. The remaining hard-case underrepresentation is retained as an observed property of the frozen generator and feasibility rules and must be reported in difficulty-stratified learner results.

## Ten-seed iterative-policy variance pilot

Both conditions used the same 240 training cases, 60 validation cases, 31,156-parameter network, terminal simulator reward, optimizer-update count, action mask, and ten-action limit. The endpoint condition did not receive intermediate action labels. The trajectory condition received the same terminal supervision plus categorical action supervision. All 600 paired case-seed evaluations were retained.

The initial trajectory coefficient of 0.02 did not reliably teach the action labels. Final action cross-entropy was frequently near the 35-class chance value of 3.56. With this underweighted condition, endpoint acceptability was 61.5% and trajectory acceptability was 60.5%. The paired difference was -1.0 percentage point, with a hierarchical-bootstrap 95% interval from -7.7 to +5.8 percentage points.

The action coefficient was increased to 0.20 as a validation-stage calibration. This change produced 50.6% mean final training action accuracy across the ten trajectory models. Endpoint acceptability remained 61.5%. Trajectory acceptability was 68.8%, for a paired difference of +7.3 percentage points with a hierarchical-bootstrap 95% interval from +1.3 to +14.0 percentage points. Nine of ten seeds favored trajectory supervision. There were 73 trajectory-only acceptable outcomes and 29 endpoint-only acceptable outcomes.

Mean violation score was 0.1115 for endpoint-only training and 0.0525 for trajectory supervision. The paired difference was -0.0590, with a 95% interval from -0.0905 to -0.0318. All ten seeds favored trajectory supervision on mean violation.

| Validation difficulty | Cases | Endpoint acceptable | Trajectory acceptable |
|---|---:|---:|---:|
| Easy | 26 | 90.8% | 93.5% |
| Moderate | 24 | 53.3% | 68.3% |
| Hard | 10 | 5.0% | 6.0% |

This is a development and hyperparameter-selection result on the validation partition. It is not the prespecified primary test-set result. The action coefficient is now frozen at 0.20 for the next comparison. The low hard-case success rate and the use of a compact scalar/centroid encoder remain progression failures for the full 10,000-case computation.

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
