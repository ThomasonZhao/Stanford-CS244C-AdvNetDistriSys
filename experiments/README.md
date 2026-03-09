# Experiments

`harness.py` runs ZeRO experiment matrices and writes per-case logs, profiles, and summary JSON.
It launches distributed jobs through the current Python interpreter (`python -m torch.distributed.run`) with explicit master address/port assignment, so it does not depend on a separate `torchrun` binary being on `PATH`.

This file is the current experiment playbook and status tracker. It replaces the old short-form project status notes for anything related to experiment execution.

## Current Status

As of March 9, 2026, the repo is in a good state for single-host CUDA experiments and for preparing real multi-host validation.

What is already validated:

- single-host 2-GPU CUDA runs work for ZeRO stages 0, 1, 2, and 3 when using `--collective-impl torch`
- the harness and plotting pipeline produce machine-readable summaries and the main deliverable figures
- measured ZeRO state-memory instrumentation is wired through the engines, training loop, harness, and plotting code
- stage 2 now correctly keeps only sharded persistent gradient state
- multi-node launch semantics were validated by emulating 2 nodes on one host with `nnodes=2` and `nproc_per_node=1`

What is not yet proven on real hardware:

- real multi-host NCCL runs across separate machines
- real `tc` shaping on the target setup
- final bandwidth crossover results using real network throttling instead of simulated bandwidth

Important operational constraints:

- for CUDA runs, use `--collective-impl torch`
- do not use the custom `ring` backend for multi-GPU CUDA runs in this repo
- the custom send/recv ring path hangs with CUDA tensors under NCCL and now fails fast with an explicit error
- overlap metrics exist, but there is still no overlap enable/disable switch, so overlap is not yet a valid experiment axis

## Completed Runs

The current repo-local experiment suite is under `experiments/results/repro2gpu_suite/`.

The most important artifacts are:

- `experiments/results/repro2gpu_suite/suite_summary.json`
- `experiments/results/repro2gpu_suite/correctness_loss.png`
- `experiments/results/repro2gpu_suite/throughput_grouped.png`
- `experiments/results/repro2gpu_suite/communication_sensitivity.png`
- `experiments/results/repro2gpu_suite/measured_state_memory_bs64.png`
- `experiments/results/repro2gpu_suite/memory_capacity.png`

Key findings from the completed 2-GPU suite:

- tiny-model correctness matched across stages within `5e-4` final loss
- small-model measured state memory followed the expected ordering: `stage 0 > stage 1 > stage 2 > stage 3`
- at small batch size `128` per GPU, only stage 3 succeeded on 2x RTX 3060 12 GB
- simulated low bandwidth hurts stage 3 the most
- stage 2 memory now reflects the intended sharded-gradient design after the fix

Representative measured small-model state memory at batch size `64` per GPU:

- stage 0: `2736.4 MB`
- stage 1: `2052.3 MB`
- stage 2: `1710.3 MB`
- stage 3: `1368.2 MB`

Representative small-model full-bandwidth throughput:

- stage 0: `4123` tokens/s
- stage 1: `3411` tokens/s
- stage 2: `3863` tokens/s
- stage 3: `2896` tokens/s

## What To Do Next

Do the remaining work in this order.

### 1. Validate Real Multi-Host Launch

Goal:

- prove that the same code path works across real hosts, not just inside a single-host emulation

Pass criteria:

- distributed sanity passes across hosts
- tiny stage 0 passes across hosts
- tiny stage 3 passes across hosts
- no rendezvous, interface, firewall, or NCCL timeout issues

Recommended first topology:

- 2 hosts
- 2 GPUs per host
- `nnodes=2`
- `nproc_per_node=2`

Start with the sanity script on host 0:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=2 \
  --master_addr=<host0-ip> \
  --master_port=29500 \
  scripts/distributed_sanity.py
```

Run the matching command on host 1 with `--node_rank=1`.

Then validate tiny stage 0 on both hosts:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=2 \
  --master_addr=<host0-ip> \
  --master_port=29501 \
  train_zero.py \
  --zero-stage 0 \
  --collective-impl torch \
  --data-mode synthetic \
  --model-size tiny \
  --seq-len 128 \
  --batch-size 4 \
  --max-steps 3 \
  --dtype bfloat16 \
  --profile-memory-interval 1
```

Repeat the same with `--zero-stage 3`, and run the same command on host 1 with `--node_rank=1`.

Do not start expensive multi-host sweeps until all three checks pass.

### 2. Validate Real Bandwidth Throttling

Goal:

- prove that `tc` changes actual throughput on the real target setup

Use two real hosts or VMs connected through the interface you intend to shape.

On the server host:

```bash
python3 scripts/validate_bandwidth.py server --bind 0.0.0.0 --port 5201
```

On the client host:

```bash
python3 scripts/validate_bandwidth.py validate \
  --target-host <server-ip> \
  --port 5201 \
  --device eth0 \
  --rate 10gbit \
  --duration-s 5 \
  --json-output /tmp/tc_validation.json
```

Decision rule:

- if shaped throughput clearly drops, `tc` is usable for the real sweep
- if it does not, use `--bandwidth-mode simulated` and state that clearly in the report

Environment warning:

- local container runs are not enough to validate `tc`
- in this container, traffic to the container's own non-loopback IP routes through `lo`
- this container also lacks `CAP_NET_ADMIN`, so `tc` apply fails even as root

### 3. Run the Real Multi-Host Experiment Set

Only do this after steps 1 and 2 pass.

Priority order:

1. memory by stage at full bandwidth
2. throughput by stage at full bandwidth
3. throughput versus bandwidth sweep
4. torch collectives as the CUDA baseline
5. custom collectives only on CPU or after the CUDA ring path is actually fixed

Use the harness once cross-host bring-up is proven. For example, on host 0:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python experiments/harness.py \
  --name real_multihost_small_bw_sweep \
  --results-dir experiments/results \
  --stages 0 1 2 3 \
  --model-sizes small \
  --bandwidth-gbps 0 2 10 \
  --nproc-per-node 2 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr <host0-ip> \
  --steps 20 \
  --seq-len 128 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --collective-impl torch \
  --data-mode synthetic \
  --dtype bfloat16 \
  --profile-memory-interval 1 \
  --bandwidth-mode simulated \
  --case-timeout-s 1800
```

Run the same command on host 1 with `--node-rank 1`.

Switch `--bandwidth-mode simulated` to `tc` only after `scripts/validate_bandwidth.py` proves the real shaping works.

## What Not To Do

Do not do these yet:

- do not run expensive multi-host sweeps before real tiny multi-host smoke tests pass
- do not trust `tc` results until `iperf3` confirms a real throughput drop
- do not use the CUDA `ring` backend for the final experiments in the current repo state
- do not present overlap as an evaluated experiment axis until a real overlap toggle exists

## What Done Means

The experiment work is done when all of this is true:

- stages 0 to 3 are validated on real GPU hardware
- real multi-host launch works across separate machines
- bandwidth manipulation is validated, either with real `tc` or a justified simulated fallback
- final figures exist for memory, throughput, and communication sensitivity
- we can clearly answer:
  - how memory changes from stage 0 to stage 3
  - how communication changes from stage 0 to stage 3
  - where each stage becomes impractical as bandwidth drops

## Quick Start

```bash
python3 experiments/harness.py --config experiments/configs/week3_smoke_matrix.json
```

Outputs are written under `experiments/results/<name>/`:

- `summary.json`: run-level metadata + all case results
- `cases/<case_id>.json`: idempotent per-case result payload
- `logs/<case_id>.log`: stdout/stderr from the launched distributed job
- `profiles/<case_id>.json`: per-step profiler output from rank 0 (default)

Each case JSON also includes a theoretical state-memory breakdown (`params_mb`, `grads_mb`, `optimizer_mb`, `total_mb`) computed from model size and ZeRO stage.
If the training command records memory snapshots (`--profile-memory-interval > 0`), the harness also extracts peak host/CUDA memory fields and the measured state-memory breakdown into each case result in `summary.json`.

## CLI Matrix Example

```bash
python3 experiments/harness.py \
  --name week3_medium_bandwidth \
  --stages 0 1 2 3 \
  --model-sizes medium \
  --bandwidth-gbps 0 1 2.5 5 10 25 50 \
  --bandwidth-mode simulated \
  --nproc-per-node 2 \
  --master-addr 127.0.0.1 \
  --master-port-base 29500 \
  --case-timeout-s 1800 \
  --profile-memory-interval 1 \
  --steps 100
```

## Config-File Format

Harness config files are JSON with two top-level objects:

- `defaults`: scalar CLI overrides
- `matrix`: sweep axes (`stages`, `model_sizes`, `bandwidth_gbps`)

Example files:

- `experiments/configs/week3_smoke_matrix.json`
- `experiments/configs/week3_medium_bandwidth.json`

## Bandwidth Modes

- `simulated` (default): injects delay inside collective ops via
  - `ZERO_SIM_BW_GBPS`
  - `ZERO_SIM_LATENCY_MS`
- `tc`: calls `./infra/throttle.sh apply/delete` around each case
- `none`: no bandwidth manipulation

Before trusting `--bandwidth-mode tc`, validate the real link with `iperf3`:

```bash
python3 scripts/validate_bandwidth.py server --bind 0.0.0.0 --port 5201
```

On the peer host:

```bash
python3 scripts/validate_bandwidth.py validate \
  --target-host <server-ip> \
  --port 5201 \
  --device eth0 \
  --rate 10gbit \
  --duration-s 5 \
  --json-output /tmp/tc_validation.json
```

If this check does not show a real throughput drop, do not trust `tc` results for the sweep. Fall back to `--bandwidth-mode simulated` and state that clearly in the evaluation. In containerized environments, note that root may still be unable to apply `tc` unless the container has `CAP_NET_ADMIN`.

## Idempotency

Use `--skip-existing` to reuse completed per-case JSON records and rerun only missing/failed cases.

## Launch Notes

- Single-node runs default to `--master-addr 127.0.0.1`, which avoids brittle hostname-based rendezvous.
- `--master-port-base` allocates one port per case in the sweep (`base + case_index`).
- `--case-timeout-s` marks hung cases as failed instead of blocking the full matrix indefinitely.

## 2-GPU Reproduction Suite

The repo is set up to run a 2-GPU-first reproduction suite that targets the paper's main takeaways:

- correctness parity across ZeRO stages 0-3
- memory reduction across stages
- throughput tradeoffs across stages
- communication sensitivity under different bandwidth settings

The recommended execution order is:

1. Correctness suite on `tiny`
2. Memory suite on `small` and `medium` or larger per-GPU microbatches until failure
3. Throughput suite on stable `tiny` and `small` workloads
4. Communication sensitivity sweep on `small` with at least two simulated bandwidth points
5. Overlap experiment only if the codebase exposes an overlap on/off switch

The current completed local suite already covers this flow for a single host with 2 GPUs, using `torch` collectives and simulated bandwidth where appropriate.

### Suggested Fixed Workload Knobs

Keep these fixed inside each comparison:

- world size: 2 GPUs
- optimizer: AdamW defaults from `train_zero.py`
- dtype: `bfloat16`
- data mode: `synthetic`
- collective backend for CUDA validation: `torch`
- sequence length: `128`

### Correctness Check

Run the same `tiny` workload under stages `0 1 2 3` and compare loss curves and final losses. Numerical parameter parity is already covered by `zero/tests/test_zero_stages.py`; this suite adds the CUDA/NCCL runtime check.

### Memory Experiment

Use either:

- a model-size ladder (`tiny`, `small`, `medium`) with a fixed microbatch, or
- a per-GPU microbatch ladder at fixed model size

The goal is to find the largest trainable configuration per stage on 2 GPUs and record whether each case succeeds, OOMs, or times out.

### Throughput Experiment

Pick several stable workloads that all stages can run, then compare steady-state step time, tokens/sec, forward/backward time, optimizer time, and communication time.

### Communication Sensitivity Experiment

Rerun the same stage sweep at multiple simulated bandwidth settings, for example:

- unlimited / no simulation (`0`)
- constrained bandwidth (`1` or `2` Gbps)
- moderate bandwidth (`5` or `10` Gbps)

This exposes the extra communication cost of ZeRO-2 and especially ZeRO-3.

### Machine-Readable Records

Every run should produce machine-readable JSON records that include:

- run ID / case ID
- stage, world size, global batch size, per-GPU microbatch size, grad accumulation steps
- model dimensions and model size label
- sequence length, dtype, optimizer settings
- success / failure / OOM status
- final loss and loss-curve summary
- peak GPU memory, measured state memory breakdown, and any per-rank memory available
- average warm-step throughput and timing breakdowns
- communication time and bandwidth setting
- overlap-derived metrics if available

### Current Limitation

The repo currently has overlap metrics but does not expose a user-facing overlap enable/disable switch for the ZeRO engines. That means the overlap experiment should be skipped unless such a toggle is added.
