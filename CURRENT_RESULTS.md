# Current Results

Verified against generated outputs on 2026-08-09.

## What is implemented

- A reproducible 2D 64 x 64 planning environment with a nested automated beamlet optimizer and high-level manual actions.
- A rule-based manual planner and bounded high-level search oracle using the same action vocabulary.
- Matched endpoint and trajectory dataset records with attempted and excluded cases retained in a manifest.
- A reproducible 3D 64 x 64 x 64 anatomy generator, implicit dose operator, exact adjoint, automated fluence optimizer, and high-level manual trajectory demonstration.
- Ten passing tests covering reproducibility, dose linearity, optimizer masking, trajectory action integrity, dataset separation, and 3D forward/adjoint consistency.

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

## Measured 3D computational scaling

The CPU reference benchmark uses 12 beams with 8 x 8 fluence maps:

| Grid | Voxels | Implicit cache | Dense float32 matrix avoided | Forward + adjoint |
|---|---:|---:|---:|---:|
| 64 cubed | 262,144 | 14.3 MiB | 0.75 GiB | 0.043 s |
| 96 cubed | 884,736 | 48.9 MiB | 2.53 GiB | 0.297 s |
| 128 cubed | 2,097,152 | 116.8 MiB | 6.00 GiB | 0.707 s |

One complete five-state, 60-iteration-per-state 64-cubed demonstration took 17.7 seconds in the latest full run. Timing varies slightly between runs; the earlier run took approximately 15 seconds.

For the proposed 96-cubed main environment with 16 x 16 fluence maps, a dense float32 influence matrix would be about 10.1 GiB per case. The implicit forward/adjoint design is therefore required. The four A100 GPUs have not yet been benchmarked because PyTorch is not installed in the local environment and server access has not been configured here.

## Present interpretation

The experiment mechanics are now demonstrated in both 2D and 3D. The current evidence shows that high-level beam/priority trajectories can be represented, reproduced, optimized, and audited. It does not yet answer the primary scientific question, because endpoint-only and trajectory-supervised learners have not been trained and compared on a frozen dataset.

The next compute milestone is the batched PyTorch 3D backend and direct four-A100 benchmark. The next scientific milestone is environment calibration followed by a frozen expert dataset and matched learner pilot.
