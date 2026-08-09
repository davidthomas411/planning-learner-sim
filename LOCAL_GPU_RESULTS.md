# Local GPU Results

Verified on 2026-08-09 using the local display GPU.

## Hardware and software

- GPU: NVIDIA GeForce RTX 4060
- Dedicated VRAM reported by NVIDIA: 8,188 MiB
- Driver: 576.80
- PyTorch: 2.11.0+cu128
- CUDA runtime used by PyTorch: 12.8
- Geometry/interpolation cache: float16
- Master fluence and Adam optimizer state: float32

The default PyPI resolution installed a CPU-only Torch package on Windows. The working CUDA build was installed with:

```powershell
uv pip install --python .\.venv\Scripts\python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

This local environment override is intentionally not encoded as a mandatory project dependency because the correct CUDA wheel depends on the target machine and driver.

## Correctness

All 20 tests pass with CUDA-enabled Torch installed. The additional tests compare PyTorch forward dose and adjoint values with the NumPy reference, verify batched and individual evaluations agree, confirm inactive beams remain exactly zero, verify difficulty-stratified geometry generation, validate split integrity, and restrict recorded 3D planning to the declared high-level action vocabulary.

The first pure-float16 full optimization produced NaNs because Adam's optimizer state was also float16. The implementation now uses the normal mixed-precision pattern: float16 cached geometry with float32 master fluence and optimizer moments. A finite-loss guard stops any future non-finite trajectory immediately.

## 300-case dataset generation

The local RTX 4060 generated a frozen-manifest development dataset containing 240 retained training cases and 60 retained validation cases. The corrected 8 x 8 fluence jobs attempted 405 cases. Training required 2396 seconds for 329 attempts; validation required 584 seconds for 76 attempts and ran concurrently for part of the training job. The combined process-time rate was 7.36 seconds per attempted case, and corrected dataset wall time was approximately 40 minutes.

The retained dataset contains 136 easy, 125 moderate, and 39 hard cases. A preceding 4 x 4 fluence calibration was rejected because it retained only easy cases. This failure establishes that fluence resolution is an experimental feasibility parameter and must remain frozen at 8 x 8 for the development-resolution variance pilot.

## Complete trajectory measurements

Each measurement is the same five-state, four-manual-action demonstration with 60 automated fluence-optimization iterations per state, 12 candidate beam angles, and 16 x 16 fluence pixels per beam.

| Grid | Voxels | Full trajectory | Peak Torch memory | Final D95 | Final D02 | OAR ratios |
|---|---:|---:|---:|---:|---:|---:|
| 96 cubed | 884,736 | 4.26 s | 0.23 GiB | 0.906 | 1.242 | 0.661, 0.767 |
| 128 cubed | 2,097,152 | 8.62 s | 0.56 GiB | 0.912 | 1.247 | 0.651, 0.766 |
| 192 cubed | 7,077,888 | 28.15 s | 1.90 GiB | 0.912 | 1.248 | 0.664, 0.763 |
| 256 cubed | 16,777,216 | 140.95 s | 4.51 GiB | 0.911 | 1.246 | 0.667, 0.761 |

All four resolutions pass the current provisional synthetic rules. The stable metrics from 128 through 256 cubed are reassuring: increasing resolution is changing runtime and memory much more than the example's planning conclusion.

## Batched kernel observations

The differentiable forward-plus-backward benchmark can batch multiple fluence states for one anatomy:

- 96 cubed reaches approximately 228 batched states per second at batch 128, using 2.67 GiB peak allocated memory.
- 128 cubed reaches approximately 68 batched states per second at batch 32, using 1.89 GiB.
- A single 256-cubed state fits, using approximately 4.20 GiB in the kernel benchmark.

These are optimizer-state evaluations, not complete cases. Independent anatomies currently have separate geometry caches and are not yet combined into one batch.

## What this means for scheduling

The local RTX 4060 is sufficient for:

- all development and correctness work;
- the 96-cubed main spatial resolution;
- 128-cubed sensitivity experiments;
- targeted 192-cubed and even 256-cubed demonstrations.

At the measured sequential 96-cubed rate, 10,000 copies of this fixed five-state trajectory would take approximately 11.8 GPU-hours. The high-level search oracle evaluates many alternative action sequences, so full expert-dataset generation will take materially longer unless candidate states and independent cases are batched. The four-A100 server remains valuable for oracle search, repeated learner training seeds, and ablations, but it is no longer required to continue local model and environment development.

The revised two-stage 32-cubed dataset pilot retained 32 demonstrations from 42 attempts in 311.9 seconds, or 7.43 seconds per attempted case, including reference solves and deeper search for selected failures. Linear extrapolation at this development resolution is approximately 37 minutes for 300 attempted cases and 20.6 hours for 10,000 attempted cases on one RTX 4060. These are scheduling bounds rather than a 96-cubed projection; the four-A100 benchmark and a complete 96-cubed two-stage case timing are required before estimating the primary run.
