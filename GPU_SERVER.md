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

## 7. Frozen-manifest sharding for the 300-case pilot

Generate training and validation data from the existing split manifest. Each process owns a nonoverlapping ordinal interval; no process samples replacement seeds. A representative four-process launch is:

```bash
uv run python scripts/build_3d_dataset_pilot.py --split-manifest outputs/splits/case_split_manifest.csv --split train --start-ordinal 0   --max-attempts 100 --retained-cases 60 --device cuda:0 --output-dir outputs/pilot300/train_shard0 &
uv run python scripts/build_3d_dataset_pilot.py --split-manifest outputs/splits/case_split_manifest.csv --split train --start-ordinal 100 --max-attempts 100 --retained-cases 60 --device cuda:1 --output-dir outputs/pilot300/train_shard1 &
uv run python scripts/build_3d_dataset_pilot.py --split-manifest outputs/splits/case_split_manifest.csv --split train --start-ordinal 200 --max-attempts 100 --retained-cases 60 --device cuda:2 --output-dir outputs/pilot300/train_shard2 &
uv run python scripts/build_3d_dataset_pilot.py --split-manifest outputs/splits/case_split_manifest.csv --split validation --start-ordinal 0 --max-attempts 120 --retained-cases 60 --device cuda:3 --output-dir outputs/pilot300/validation_shard0 &
wait
```

These commands target 240 retained training cases and 60 retained validation cases. The attempt ranges are deliberately larger than the retained targets because unreachable cases are preserved but excluded from learned-policy training. If a shard does not reach its retained target, continue from the first unused split ordinal; do not change the acceptance rule or reuse an ordinal. Before training, verify that retained case identifiers are unique, all endpoint/trajectory pairs match, no training identifier appears in validation, and the combined retained counts are exactly 240 and 60.

Perform those checks and create the canonical merged dataset with:

```bash
uv run python scripts/merge_3d_dataset_shards.py \
  outputs/pilot300/train_shard0 outputs/pilot300/train_shard1 \
  outputs/pilot300/train_shard2 outputs/pilot300/validation_shard0 \
  --expected-train 240 --expected-validation 60 \
  --output-dir outputs/pilot300/merged
```

The merge fails on duplicate split ordinals, duplicate case identifiers, endpoint/trajectory disagreement, test-partition contamination, or incorrect retained counts. It writes canonical file hashes to `summary.json`.

The first server execution should use the development grid and optimizer settings to establish correctness. The 96-cubed configuration should be started only after the four shards reproduce the expected retention fields and action-mask tests. Dataset generation is case-parallel; model initialization seeds should subsequently be assigned one at a time to GPUs, with complete per-seed checkpoints and metrics rather than data-parallel mixing of seed states.

After the development-resolution merge, distribute the ten initialization seeds as nonoverlapping jobs with explicit manifest partitions:

```bash
for spec in "0 3 0" "3 3 1" "6 2 2" "8 2 3"; do
  set -- $spec
  uv run python scripts/train_3d_iterative_policy_pilot.py \
    --dataset-dir outputs/pilot300/merged --train-cases 240 --heldout-cases 60 \
    --seed-start "$1" --seeds "$2" --pretrain-updates 400 --updates 60 \
    --dtype float32 --deterministic --device "cuda:$3" \
    --output-dir "outputs/pilot300/iterative_policy_seeds_$1" &
done
wait
```

The initial execution should retain deterministic float32 calculations. Reduced precision may be evaluated later as a separate throughput condition only after its per-case metrics agree within a frozen numerical tolerance.
