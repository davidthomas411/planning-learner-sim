# Process-Level Treatment Planning Trajectories: Project Summary and Evidence Notes

## Working concept

Current radiotherapy auto-planning and knowledge-based planning systems learn primarily from endpoint data: CT images, structure sets, prescriptions, approved plans, and delivered dose distributions. The proposed project asks whether treatment-planning process data, including the sequence of dosimetrist actions and refinements that transform an initial automated plan into an approved plan, contains additional supervisory signal that is not recoverable from final approved plans alone.

The central asset to justify is not screen recording itself. The asset is structured planning trajectory data.

## Core hypothesis

Expert treatment-planning trajectories contain information that is lost when only final approved plans are retained. Capturing these trajectories may allow future auto-planning systems to learn the small, plan-specific refinements that expert dosimetrists currently apply after automated plan generation.

A concise grant framing:

> Published autoplanning systems can often produce clinically acceptable plans, but a measurable fraction still require minor human edits. These minor edits represent the knowledge gap this project aims to capture. We hypothesize that planning-session trajectories encode expert behaviors that could train future models to reduce or eliminate these manual refinements.

## Clinical gap this project targets

The relevant gap is not that auto-planning fails. The relevant gap is that auto-planning often gets close enough to be clinically useful, but still requires minor expert review, adjustment, or replanning.

This is the gap between:

- **acceptable as-is**, and
- **acceptable after minor edits**.

That difference is the opportunity space for process-level learning.

## Key external evidence

### MD Anderson multi-site KBP/autoplanning study

MD Anderson developed knowledge-based planning/autoplanning models for 10 cancer sites: anorectal, bladder, breast/chest wall, cervix, esophagus, head and neck, liver, lung/mediastinum, prostate, and prostate with nodes. Physician review rated **88%** of KBP-generated plans as **acceptable as-is** and **98%** as **acceptable after minor edits**. This defines a **10 percentage-point gap** between immediate clinical usability and usability after minor human refinement. Citation from chat: `turn5search129`.

The same paper explicitly defines the review categories:

- rating 5: use as-is, clinically acceptable without change;
- rating 4: minor edits not necessary, stylistic but not clinically important;
- rating 3: minor edits necessary, faster than starting from scratch or expected to have minimal outcome effect;
- rating 2: major edits, preferable to start from scratch;
- rating 1: unusable.

This supports the claim that the “minor edit” gap is an operationally defined clinical category, not a vague impression. Citation from chat: `turn5search129`.

The same study states that knowledge-based planning typically generates initial optimization objectives and constraints, but additional modification is necessary to create an optimal plan for each individual patient. It also states that the treatment planner still reviews DVH estimates and refines optimization objectives until an optimal plan is achieved. Citation from chat: `turn5search129`.

The study also documents that dosimetrists used clinical experience to test and introduce fields and planning structures, including normal tissue rings and OAR avoidance structures, into the automated models through an iterative process. Citation from chat: `turn5search129`.

A concrete example from the study: the lung/mediastinum model initially did not meet V5 and V10 lung constraints. After the team contoured a lung avoidance structure in 30 optimal cases and trained that structure into the model, plans met those constraints without compromising coverage. This is evidence that expert planning insight can be converted into model-improving planning structures and objectives. Citation from chat: `turn5search129`.

### Radiation Planning Assistant clinical acceptability study

The Radiation Planning Assistant study reviewed automated contouring and planning outputs for 75 cases. At least three physicians reviewed each case, with 31 radiation oncologists from 16 institutions in six countries on five continents using a 5-point Likert scale. Citation from chat: `turn5search124`.

Reported use-as-is versus with-minor-edits gaps were site dependent:

- postmastectomy breast plans: **44% usable as-is**, **91% usable with minor edits**, a **47 percentage-point gap**;
- whole-brain plans: **67% usable as-is**, **99% usable with minor edits**, a **32 percentage-point gap**;
- head and neck VMAT plans: **87% acceptable as-is**, **96% acceptable with minor edits**, a **9 percentage-point gap**;
- cervix VMAT plans: **99% acceptable as-is**, **100% acceptable with minor edits**, a **1 percentage-point gap**.

This provides independent evidence that the minor-edit gap exists and varies by treatment site and planning complexity. Citation from chat: `turn5search124`.

The study used a 5-point scale similar to the MD Anderson acceptability rubric: unusable, major edits, minor edits required, minor edits not required, and use as-is. For VMAT plans, a score of at least 4 was considered clinically acceptable; for simple plans, a score of at least 3 was considered clinically acceptable. Citation from chat: `turn5search124`.

### RapidPlan model validation evidence

A 2025 JACMP paper on automated RapidPlan model validation states that RapidPlan provides efficiency and quality improvements but requires extensive validation before clinical use. It describes this validation as an iterative process requiring numerous refinements to fine-tune and optimize a RapidPlan model. Citation from chat: `turn5search122`.

In a liver SBRT validation example, the authors used 76 model-set patients and 17 validation patients. The validation loop involved 17 different model iterations and 405 automatically generated plans with 118.7 hours of active planning time. Citation from chat: `turn5search122`.

The same paper cites prior RapidPlan tuning efforts requiring large numbers of replans: 300 replans for a breast model study, 226 replans for head and neck, and 333 replans for rectal cancer. Citation from chat: `turn5search122`.

This supports a complementary point: even when endpoint-based models are used, expert-guided iterative refinement is a major part of making the model clinically viable.

### Scan-to-plan challenge

A fully automated prostate/prostate-bed scan-to-plan challenge received 13 entries. The top submission met 81.8% of minimum objectives across all cases using consensus contours and met all objectives in 1 of 10 cases. Using its own autocontours, the same system met 89.5% of objectives and met all objectives in 4 of 10 cases. Citation from chat: `turn5search163`.

Despite a “hard” rule that participants should not check or edit contours or plans, 69% of participants reported looking at their results before submission. This supports the idea that automation exists, but full automation without review/editing remains limited by performance and trust. Citation from chat: `turn5search163`.

## Internal project evidence from this chat

### Email thread with Jenia

In the email thread titled **RE: dosimetry planning database idea**, the proposed clinical concept was to turn on Citrix session screen recording for Eclipse, with IS&T support, to create recordings of planning sessions and build a database of each step dosimetry takes for each plan. The analogy used was an “RT Black-box.” Jenia replied that it was a “really cool idea” and described it as “R01-level” with feasibility considerations. Citation from chat: `turn1search50`.

The email included an example output from a recorded Eclipse planning session analyzed into six phases:

1. loading and initialization,
2. context switch or non-planning navigation,
3. course and plan setup,
4. first field setup,
5. MLC aperture fitting,
6. opposing field and finalization.

The detected planning events included prescription setup, target selection, isocenter/beam configuration, gantry angles, MLC fitting, and plan review. Citation from chat: `turn1search50`.

### Existing project abstract

The internal abstract **Foundation Dataset for Radiation Treatment Planning_project-abstract_2026-08-04_JV.docx** frames the project as a process-level foundation dataset for radiation treatment planning. It states that current commercial and academic knowledge-based planning systems learn from static endpoints such as anatomy and approved plans, while the proposed dataset would model planning trajectories, including explored beam arrangements, aperture refinements, changes to optimization parameters, and review criteria. Citation from chat: `turn4search111`.

### Existing screen-recording feasibility work

The internal **expert_patterns_dashboard.html** reported a proof-of-concept Eclipse screen-recording analysis with:

- 132.7 second Eclipse TPS video duration,
- 1095 dense frames extracted,
- 15 scene changes detected,
- six clinical phases identified,
- event types including UI navigation, gantry changes, energy selection, MLC fitting, aperture editing, collimator changes, beam creation, and plan review.

This is preliminary feasibility evidence that planning-session recordings can be transformed into structured planning-event timelines. Citation from chat: `turn4search115`.

## Important conceptual distinction

The proposal should not be framed as:

> Auto-planning does not work.

The evidence does not support that. The stronger and more accurate framing is:

> Auto-planning often works well enough to be clinically useful, but published studies show a measurable difference between plans usable as-is and plans usable after minor human edits. Those edits represent expert process knowledge that current endpoint-trained models do not explicitly learn.

## Why current endpoint datasets are incomplete

Endpoint datasets contain:

- CT,
- structures,
- prescription,
- approved plan,
- final dose distribution,
- possibly RTPLAN parameters.

They generally do not contain:

- rejected beam arrangements,
- alternative objective weights,
- intermediate DVHs,
- failed optimization attempts,
- ring/helper structure creation logic,
- manual aperture refinements,
- planner rationale for tradeoffs,
- review-driven replanning decisions,
- sequence of optimization parameter changes.

This missing process information is what the proposed session-capture dataset would collect.

## Simplified proof-of-principle experiment

For a two-page Varian-style proposal, the cleanest experiment is fully synthetic. It does not require real plans, real Eclipse data, or manual planning.

### Aim

Test whether learning from planning trajectories improves performance beyond learning from final endpoint solutions alone.

### Synthetic environment

Create a toy planning world with:

- a target,
- several avoidance structures,
- candidate beams/actions,
- an objective function balancing target coverage and OAR sparing,
- automatically generated cases.

The domain can be abstract. It should preserve the structure of treatment planning: sequential decision-making under competing objectives.

### Synthetic expert

Implement a heuristic expert planner that:

1. starts from an initial plan,
2. evaluates a plan-quality score,
3. modifies the plan,
4. evaluates again,
5. repeats until convergence.

Record the full trajectory:

```text
state_0 -> action_1 -> state_1 -> action_2 -> ... -> final_solution
```

### Dataset variants

Generate two matched datasets:

1. **Endpoint dataset**

```text
problem -> final_solution
```

2. **Trajectory dataset**

```text
problem -> full expert action sequence -> final_solution
```

Both datasets include the final solution. Only the trajectory dataset contains expert behavior.

### Model comparison

Compare:

- **Endpoint learner:** trained only on problem-to-final-solution examples.
- **Behavior learner:** trained on the same final solutions plus expert trajectories.

Evaluate both on held-out synthetic cases.

### Metrics

Use simple quantitative metrics:

- final plan-quality score,
- target coverage term,
- OAR sparing term,
- number of steps to reach acceptable solution,
- sample efficiency as a function of training-set size,
- performance on harder or out-of-distribution synthetic cases.

### Expected result

If the behavior learner reaches better quality, better convergence, or similar quality with fewer training cases, this supports the premise that trajectory data contains useful supervisory signal beyond final endpoints.

## How this supports data collection

The synthetic experiment is not meant to prove clinical superiority. It is meant to justify collecting the missing data modality.

A grant-ready statement:

> The published literature demonstrates a measurable use-as-is versus minor-edit gap in automated planning systems. Our synthetic proof-of-principle experiment will test whether access to planning trajectories can reduce such gaps in a controlled setting where the endpoint solution is held constant. If trajectory-supervised agents outperform endpoint-only agents, this provides direct evidence that treatment-planning session recordings are a valuable data source rather than an archival convenience.

## Key grant message

The most important message for Varian:

> RapidPlan and related KBP systems show that endpoint learning is clinically useful. However, published data show that a measurable fraction of automated plans still require expert minor edits. Those edits are currently performed by dosimetrists but are not captured in approved-plan archives. We propose to capture the planning behavior that closes this final clinical acceptability gap.

## What remains unknown

No public dataset was found in this chat that directly logs:

```text
initial RapidPlan output -> dosimetrist edit sequence -> final approved plan
```

That missing dataset is the novelty. The published literature documents the gap and the resource burden, but not the actual expert trajectories that close the gap.

## Suggested two-page proposal structure

1. **Problem:** Endpoint-only KBP/autoplanning is clinically useful but leaves a measurable minor-edit gap.
2. **Evidence:** MD Anderson 88% as-is vs 98% after minor edits; RPA site-specific gaps; RapidPlan iterative tuning burden.
3. **Hypothesis:** Planning trajectories encode the expert actions that close this gap.
4. **Proof-of-principle:** Fully synthetic planning environment comparing endpoint-only learning to trajectory-supervised learning.
5. **Future clinical translation:** Use Citrix/Eclipse screen recording plus VLM/event extraction to build clinical planning trajectory datasets.
6. **Impact:** Reduce minor manual replanning, improve automation, preserve expert dosimetrist knowledge, and create a new process-level data modality for radiotherapy planning AI.

## Short version for proposal text

Published autoplanning systems demonstrate high but incomplete clinical acceptability. In a 10-site MD Anderson KBP study, 88% of automated plans were acceptable as-is, increasing to 98% after minor edits. The Radiation Planning Assistant study showed similar use-as-is versus minor-edit gaps that varied by site, including a 47 percentage-point gap for postmastectomy breast plans and a 9 percentage-point gap for head and neck VMAT. These data define a practical opportunity: current endpoint-trained systems are close to clinical utility, but expert dosimetrists still supply plan-specific modifications that are not captured in approved-plan archives. We propose to test whether planning trajectories provide additional learning signal beyond final plans and, if confirmed, to create a process-level treatment-planning dataset from recorded Eclipse sessions.
