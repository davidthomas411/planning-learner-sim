# Performance plan before larger simulation runs

## Skill review

The curated Codex skill catalog and the available experimental path were checked on 2026-08-10. No CUDA, PyTorch, scientific-computing, or numerical-performance skill was available. The Jupyter skill can support interactive analysis, but it does not improve simulation throughput. No additional skill was installed.

## Local profile

A deterministic 64 × 64 × 64 prostate plan with seven fields, 24 × 24 fluence maps, the clinical DVH objective, and 100 optimizer iterations required 2.13 seconds on the local GPU. The fluence and dose SHA-256 values were recorded for exact comparisons.

The profile showed many small PyTorch operations and low GPU occupancy. Two changes were tested and rejected:

1. Caching interpolation indices and weights produced exact output hashes but did not reduce measured time and increased cached memory.
2. Reducing finite-loss check frequency produced exact output hashes but did not reduce measured time.

No numerical optimizer change was retained.

## Retained efficiency changes

1. New launcher-created output folders include a local timestamp in `YYYYMMDD_HHMMSS` format.
2. Numerical progress updates after every completed case.
3. Live figures update at most 50 times during a large run. This prevents plotting time from increasing with every case while preserving regular visual review.
4. Dose calculation, objective terms, optimizer steps, floating-point precision, and deterministic settings are unchanged.

## Four-A100 execution plan

The first server benchmark will use one independent process on each A100. Each process will receive a nonoverlapping manifest shard. All four shards will use one timestamped parent run directory.

A second benchmark may use two processes per A100 because the current workload does not saturate a large GPU. This configuration will be accepted only if all of the following conditions pass:

- identical case identifiers and settings;
- identical retained and rejected case decisions;
- exact serialized metric hashes for deterministic float32 runs;
- no increase in failed processes or nonfinite values;
- a material increase in completed cases per GPU-hour.

Mixed precision, reduced optimizer iterations, `torch.compile`, and changed DVH approximations remain disabled for the primary experiment. Each requires a separate paired validation because it can change optimization trajectories or boundary-case acceptance.

## Gate before the next large run

Run a 32-case throughput benchmark with the frozen clinical configuration. Compare one and two processes per GPU. Record wall time, GPU memory, case throughput, numerical hashes, and plan acceptance. Select the faster configuration only when the numerical equivalence checks pass.
