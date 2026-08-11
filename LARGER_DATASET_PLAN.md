# Dataset scale-up plan

## Purpose

The study tests whether intermediate planning decisions provide useful supervision beyond the final plan. The next data stages must increase anatomical and planning complexity without changing this comparison.

The endpoint-only and trajectory-supervised conditions must use the same patients, initial plans, final plans, model architecture, optimization budget, and evaluation budget. Only the intermediate state and action labels may differ.

## Stage 1: TCIA clinical-anatomy calibration and locked evaluation

### Source and case selection

The source is the TCIA Prostate Anatomical Edge Cases collection. It contains 131 patients with CT and RTSTRUCT data.

The anatomy quality-control step will import every available patient. A case will be excluded only if a required contour is absent or empty, contour geometry cannot be mapped to CT, the PTV is empty, or a required structure is outside the usable image field. Every exclusion will remain in the audit record.

The importer produced 116 unique valid patients. The 15 patients used for current method calibration form a development set. The remaining 101 patients form a locked clinical-anatomy evaluation set. It will not contain duplicate patients or synthetic copies.

The cohort will contain two prespecified anatomy strata:

1. `margin_only`: the prostate/CTV does not overlap the bladder or rectum on the 64-cubed planning grid. The 5 mm PTV margin can overlap these structures.
2. `interface_overlap`: the source prostate/CTV contour has rasterized overlap with the bladder or rectum. These anatomical edge cases will remain unchanged and will be reported separately.

All valid `margin_only` cases will be selected first. The remaining places will use `interface_overlap` cases sampled across the observed PTV-OAR overlap range. This rule will avoid selection by planning outcome.

### Planning method

Each patient will start with seven fixed coplanar fields. Each field will have a 64 by 64 fluence map. The inner optimizer will use 1000 iterations after each high-level change. The outer process can create a PTV-minus-bladder or PTV-minus-rectum optimization target and can change only target, hot-spot, named-OAR, or normal-tissue priority. The limit is eight high-level changes.

The prescription is 60 Gy in 20 fractions. The institutional objectives remain the per-protocol goals. A trial-informed acceptable-variation policy is fixed before the locked evaluation. PTV V57 Gy can decrease from at least 99% to at least 95% only when all institutional OAR limits pass. Rectum or bladder volume can exceed an institutional volume limit by no more than 5 percentage points only when standard target coverage passes. Prostate V60 Gy at least 99%, PTV D1cc at most 63 Gy, femoral-head limits, and conformity remain fixed. Simultaneous target and OAR variation is not accepted automatically and requires physician review. Per-protocol and acceptable-variation plans are reported separately.

### Outputs

The pilot will save the following data:

- one anatomy and overlap manifest;
- one exclusion log;
- every intermediate planning state and action;
- final dose metrics and cumulative DVHs;
- one review image for each patient;
- cohort plots for acceptance, manual action count, PTV coverage, hot spots, conformity, and OAR limits;
- results for each anatomy stratum;
- a live local status page with current figures.

The 15-patient set is for method development. The remaining 101 patients are not a training cohort and will be evaluated once after the planning policy is frozen.

### Short plan review

The numerical acceptance test will review every plan. A structured secondary review will inspect every failed plan and a fixed sample of passing plans from both anatomy strata. The short review will record six items:

1. anatomy and contour import are plausible;
2. PTV coverage meets the declared limits;
3. the PTV hot spot is acceptable and is in a plausible location;
4. the 57 Gy isodose has acceptable conformity to the PTV;
5. bladder, rectum, and femoral-head DVHs meet the declared limits; and
6. each recorded high-level action is a reasonable response to the preceding plan.

The terminal disposition will be `per protocol`, `acceptable target variation`, `acceptable OAR variation`, `requires physician review`, or `technical failure`. Planning failures will also be classified as correctable, major anatomical conflict, action nonresponse, or action-budget exhaustion. The automated review is an engineering quality-control method. It is not a clinical plan approval or a substitute for a licensed dosimetrist or medical physicist. A publishable study should include blinded review of a prespecified sample by a qualified human reviewer.

### Progression gate

The current balanced template accepted 14 of 15 plans under the exact institutional limits, but all 15 plans stopped without a manual action. This is adequate for plan-quality calibration but is not an informative trajectory dataset.

Before scale-up, four fixed starting templates will be tested on the 15 development patients: balanced reference, guarded OAR stress, hotspot stress, and conformity stress. These are starting conditions, not optimizer-selected candidates. The balanced profile provides required stop examples. Each stress profile starts with target priority 3.0. The OAR, hotspot, and normal-tissue priorities are 0.10, 0.04, and 0.02. If a related clinical metric fails, the reviewer increases the active priority by a factor of 3.0. A target-underweighted profile was tested and rejected because prostate D99 normalization preserved target coverage while creating a hot spot and a conformity failure; it did not isolate the intended target-priority decision. The four retained templates produce 60 development episodes but only 15 independent patients.

The response to each action is measured. The fixed material-response thresholds are 1 percentage point for PTV V57 Gy, 0.3 Gy for PTV D1cc, 1 percentage point or 10% of the current excess for an OAR volume metric, and 0.01 for the 57 Gy covering-isodose ratio. After two consecutive nonresponsive actions of the same class, the reviewer must change strategy or require physician review. Repeated nonresponsive actions remain negative calibration records and are not expert demonstration labels.

Progression requires initial acceptance in at least 90% of balanced-control plans, an initial failure rate of 40% to 80% among stress-profile plans, final per-protocol or acceptable-variation status in at least 90% of all development episodes, a median of one to four manual actions among corrected plans, and more than one represented action class. Initial development states must also avoid gross errors: PTV D1cc must not exceed 70 Gy, the 57 Gy covering-isodose ratio must not exceed 1.25, the worst OAR objective ratio must not exceed 1.50, and PTV V57 Gy must remain at least 90%. These are development-severity bounds, not treatment-plan acceptance criteria. Failed episodes remain in the audit record.

The first complete 60-episode calibration failed this progression gate. Its initial failure rate was 30.0%, and its maximum initial OAR objective ratio was 1.560. Final acceptance was 93.3%, corrected-plan median action count was one, and six action classes were represented. The next initial-state screen therefore replaces OAR omission with priority 0.05 and tests stronger but nonzero hotspot and normal-tissue stress priorities of 0.04 and 0.02. All 15 development patients are used for this screen; no conclusion is based on the earlier three-patient subset.

The final guarded-profile calibration passed every progression gate. Balanced-control initial acceptance was 93.3%, stress-profile initial failure was 46.7%, and final automatic acceptance was 91.7%. Median action count among corrected plans was one. All initial severity limits passed. Fifty-five of 60 episodes are eligible expert demonstrations; five physician-review episodes remain in the audit data without expert action labels.

The 101-patient TCIA assignment is now frozen in `data/manifests/tcia_locked101_profiles.csv`. It assigns one profile per patient before dose calculation: 10 balanced control, 21 guarded OAR, 35 hotspot stress, and 35 conformity stress. The allocation is proportional within 31 margin-only and 70 interface-overlap cases. Within each stratum, assignments are spread across ranked maximum PTV-OAR overlap. The manifest SHA-256 digest is `7dc86d32fb94680aeac8e5e8438fb22f4a74355f2b61c11e7c722535647843ec`. This distribution supplies stop examples without allowing the balanced control to dominate the planning-decision set. Patient identity remains the independent unit. No validation dose has been calculated at the time of manifest freeze.

## Stage 2: PortPy prostate external planning study

PortPy is the preferred next clinical-complexity source. The official dataset contains 129 prostate patients. Each patient has CT images, contours, 72 candidate beam directions, Eclipse dose-influence data, expert-selected beams, and ECHO IMRT and VMAT reference plans. This allows a direct test against a commercial-system-derived dose model and reference plans.

The first PortPy study will use 20 patients selected before dose review. It will download the expert-selected beam subset and shared patient data. This is the lowest-storage test of data import, objective mapping, dose reconstruction, DVH agreement, and plan review. These 20 patients form an integration set. They will not enter the external performance estimate.

The second PortPy study will lock the remaining 109 prostate patients as the external evaluation set. This study will compare high-level priority changes with ECHO IMRT and VMAT plans. No PortPy patient in this set will be used to select objectives, tolerances, action rules, or model settings. A later study can use PortPy for training, but it must then use a different clinical dataset for external testing.

For one inspected prostate patient, the nine expert-selected beams plus shared anatomy and reference data require about 1.26 GB. A 20-patient integration cohort should therefore require about 25 GB before outputs. All 129 patients with expert beams can require about 163 GB. The full 72-beam data should be used only for the beam-angle and VMAT study. One full patient is about 7.6 GB. The complete 129-patient, 72-beam collection is therefore close to 1 TB before derived outputs. The local computer does not have enough free storage for either complete form with safe working space. The data should remain on the GPU server or network storage. Downloads should use only the required beams for each experiment.

The PortPy study requires a separate implementation gate:

1. Reconstruct one supplied ECHO dose from its influence data.
2. Match the supplied DVH within declared numerical tolerances.
3. Map prostate PTV and OAR names without manual case-specific code.
4. Reproduce one expert-selected IMRT plan with fixed objectives.
5. Review dose planes, DVHs, conformity, and hot spots for five patients.
6. Freeze the PortPy objective mapping before the 20-patient run.

PortPy uses a non-commercial academic license, and its hosted dataset is marked CC BY-NC 4.0. The final study must record the exact dataset revision and license.

## Stage 3: Large trajectory dataset

The large training dataset will start only after the TCIA and PortPy progression gates pass.

The recommended first scale is 2,000 unique parametric anatomies. Each anatomy receives one starting template assigned before optimization and one matched endpoint and trajectory record. The four starting templates are balanced across anatomy difficulty and overlap strata. This scale tests the data pipeline, action balance, training stability, and learning curves without committing to the final computation.

The recommended confirmatory scale is 10,000 unique parametric anatomies. The existing frozen manifest assigns 7000 training, 1000 validation, 1000 IID-test, and 1000 reserved OOD-test cases. This stage will use fixed code, objectives, acceptable-variation policy, starting-template assignment, action vocabulary, patient splits, and primary endpoints.

Clinical patients are limited in number. The 101 locked TCIA patients remain 101 independent statistical units. Controlled variants can be used in later training experiments, but they must not increase the reported patient count. All variants from one patient must remain in one dataset split.

For each training patient, 10 to 20 small, audited contour variants can be created. The deformation limits must be fixed before generation. The prostate, bladder, rectum, femoral heads, and body must remain anatomically valid. The 5 mm PTV must be recalculated from the changed prostate/CTV. A variant will be rejected if it creates a disconnected structure, a structure outside the body, or an undeclared CTV-OAR overlap.

The large dataset should contain a balanced range of initial planning states. The variables may include target priority, hot-spot priority, named-OAR priority, normal-tissue priority, and a small set of clinically plausible beam configurations. The manual labels will remain high-level changes. Fluence edits will remain inner-optimizer operations.

## Role of MAISI

MAISI is not required for the present water-equivalent dose study. The current dose surrogate does not use CT Hounsfield units. A synthetic CT image would therefore add image texture but would not change the calculated dose.

The current NVIDIA CT model can generate a body, bladder, prostate, and femurs. Its published label dictionary does not contain a separate rectum label. It also does not generate a radiotherapy PTV. MAISI must therefore not replace the audited prostate planning masks in the main experiment.

MAISI can be used later for a visual-realism or density-correction study. In that study, a CT can be generated from an audited mask, and the rectum and PTV must remain from the planning mask. The generated CT must pass anatomical and density quality control before use.

## Data splits and statistics

All splits will use independent source patients:

- development: code and objective calibration;
- training: model fitting;
- validation: fixed model and coefficient selection;
- test: one final locked comparison;
- external test: PortPy patients not used in development.

No variant, trajectory, plan, or dose from one source patient may cross a split boundary.

The primary result will use paired patient-seed outcomes. The report will include the paired mean difference in normalized regret with a 95% confidence interval. It will also include acceptable-plan risk difference, results by anatomy stratum, learning curves, manual action count, and compute time. Patient-level bootstrap resampling will keep all variants and training seeds for one patient together.

The 101-patient TCIA evaluation can estimate a 90% acceptance rate with an approximate 95% confidence interval half-width of 5.9 percentage points. Under an assumed 30% paired-discordance rate, 101 paired patients provide approximately 80% power for a 15-percentage-point paired difference. They do not provide adequate power for a 10-percentage-point difference, which would require approximately 233 paired patients under the same assumption. Therefore, TCIA is a clinical-anatomy validation study, not the primary small-effect comparison.

## Compute and storage plan

The final 60-episode development run completed in 2168.8 seconds on the local RTX 4060. This is 36.1 seconds per episode, including the required manual reoptimizations. At the same observed rate, the locked 101-patient run should take about 61 minutes after the anatomy cache is complete. A practical reservation is 1 to 1.5 hours. The full anatomy import can take longer on its first pass because it must decode DICOM files.

The four A100 GPUs should be used for the 2,000-case and 10,000-case stages. Patient shards can run independently on the four GPUs. Each shard must use a distinct output folder and a fixed subject list. A merge step will reject duplicate patient and trajectory identifiers.

Expected planning time, based on the current local rate and ideal four-GPU sharding, is approximately:

- 101 TCIA plans with one starting template: approximately 61 minutes on the local RTX 4060 at the observed complete-episode rate;
- 2,000 parametric trajectories: about 9.6 hours on four GPUs before overhead;
- 5,000 parametric trajectories: about 24 hours on four GPUs before overhead;
- 10,000 parametric trajectories: about 48 hours on four GPUs before overhead.

These are planning estimates. Model training, data transfer, PortPy influence-matrix loading, and repeated seed runs are additional costs. The first server benchmark must measure the actual A100 rate before the final compute reservation.

## Stop rules

The study will stop before a larger stage if any of the following conditions occurs:

- fewer than 100 valid unique TCIA patients are available for the requested pilot;
- the anatomy audit shows systematic contour-mapping errors;
- the demonstration acceptance rate remains below 95% in the intended training domain;
- endpoint and trajectory views do not contain identical patients and final plans;
- the PortPy dose reconstruction does not match the supplied reference within the frozen tolerance;
- the full run would exceed available disk space;
- a patient-level split audit finds data leakage.

These conditions are method results. They are not reasons to remove difficult patients after results are known.

## Primary sources

- TCIA Prostate Anatomical Edge Cases: https://www.cancerimagingarchive.net/collection/prostate-anatomical-edge-cases/
- PortPy: https://github.com/PortPy-Project/PortPy
- PortPy dataset: https://huggingface.co/datasets/PortPy-Project/PortPy_Dataset
- NVIDIA NV-Generate-CTMR: https://github.com/NVIDIA-Medtech/NV-Generate-CTMR
