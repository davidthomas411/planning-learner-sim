# Four-A100 Server Runbook

The GPU backend preserves the same experiment boundary as the CPU reference:

- a human-level trajectory action changes a beam angle or target/OAR priority;
- the PyTorch inner optimizer changes fluence pixels for those fixed settings;
- fluence updates are never exported as manual action labels.

## 1. Clone and create the environment

```bash
git clone https://github.com/davidthomas411/planning-learner-sim.git
cd planning-learner-sim
uv sync --extra dev --extra gpu
```

If the default Torch package does not match the server's CUDA driver, install the CUDA-enabled Torch build recommended for that server, then rerun the checks below.

## 2. Verify all four GPUs

```bash
nvidia-smi
uv run python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

Expected device count: `4`. Stop if CUDA is unavailable or the devices are not the expected A100s.

## 3. Run CPU and Torch correctness tests

```bash
uv run pytest -q
```

With Torch installed, the suite also checks:

- PyTorch forward dose against the NumPy reference;
- PyTorch adjoint against the NumPy reference;
- batched results against individual evaluations;
- exact zero fluence for inactive beams.

## 4. Benchmark all four A100s concurrently

Start with 64- and 96-cubed volumes:

```bash
uv run python scripts/benchmark_torch_3d.py \
  --devices 0 1 2 3 \
  --grids 64 96 \
  --fluence-size 16 \
  --batch-size 2 \
  --dtype float16
```

The output is written to `outputs/gpu_benchmark/torch_operator.csv`. It records each GPU separately, including iteration time, batched fluence states per second, peak allocated memory, and geometry-cache memory. A batched state is not a complete case or trajectory.

Only after 96-cubed succeeds should the 128-cubed sensitivity benchmark be attempted:

```bash
uv run python scripts/benchmark_torch_3d.py \
  --devices 0 1 2 3 \
  --grids 128 \
  --fluence-size 16 \
  --batch-size 1 \
  --dtype float16
```

## 5. Interpretation

The benchmark times one differentiable forward-plus-backward iteration. It does not directly equal cases per second for dataset generation because a trajectory contains multiple optimizer iterations and several human-level states. Use the measured iteration time to choose batch size and then time a complete trajectory before projecting the full dataset schedule.

## 6. Required preflight before the 300-case pilot

Run the 100-case environment validation and the 30-case two-stage search audit on one A100 before distributing jobs. Then time at least 30 complete 96-cubed attempted cases, including reference solves and deep-search failures. Record median, 90th percentile, and total seconds per attempted and retained case. Use those measurements, rather than operator-kernel throughput, to project the 300-case and 10,000-case schedules.

The local 32-cubed two-stage pilot required 311.9 seconds for 42 attempted cases (7.43 seconds per attempt). This projects to approximately 37 minutes for 300 attempts or 20.6 hours for 10,000 attempts on one RTX 4060 at development resolution. It is not a valid A100 or 96-cubed estimate. With four A100s, cases should be sharded by seed so that each process owns one GPU and writes a separate manifest shard; shards should be merged only after duplicate-case and completeness checks.
