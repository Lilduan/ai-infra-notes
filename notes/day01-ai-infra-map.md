# Day 1 - AI Infra Map

Date: 2026-03-15
Backfilled on: 2026-03-16
Topic: AI infra map, performance dimensions, local environment, inference critical path
Time spent: 60-90 min target

## Core Philosophy

AI infra is not "more ML code." Its core job is to optimize the **end-to-end critical path** of training or inference under hard constraints: `latency`, `throughput`, `memory`, `bandwidth`, `utilization`, `stability`, and `cost`.

This philosophy implies two rules:

1. Measure before optimizing.
2. Think in system boundaries and data movement, not only in model math.

## 1. What Training Infra / Inference Infra / Platform Infra Mean

### Training Infra

Training infra exists to turn data and compute into updated model weights reliably and efficiently.

From the core philosophy perspective, training infra is about keeping the training loop moving with high hardware utilization while controlling communication overhead, failures, and reproducibility. Its main concerns are:

- data input pipeline
- distributed execution
- checkpointing
- experiment management
- fault tolerance
- cluster scheduling

The critical path is usually:

`data load -> host preprocess -> host-to-device transfer -> forward -> backward -> optimizer step -> checkpoint / log`

### Inference Infra

Inference infra exists to turn a user request into model output with predictable latency and acceptable cost.

It expresses the same philosophy in a stricter real-time setting: every unnecessary copy, synchronization, queue wait, or decode step delay directly hurts user-visible latency or service throughput. Its main concerns are:

- request parsing and tokenization
- batching / scheduling
- model execution
- KV cache management
- sampling / decoding
- response streaming
- tail latency control

The critical path is usually:

`request arrival -> tokenize -> preprocess -> model forward -> decode / sample -> postprocess -> response`

### Platform Infra

Platform infra exists to make training infra and inference infra usable at team scale.

Its role is to standardize environment, deployment, observability, resource allocation, access control, and operational workflows. It serves the same philosophy indirectly: if engineers cannot reproduce runs, observe failures, or deploy safely, system performance does not matter because the platform itself becomes the bottleneck.

Typical responsibilities:

- environment and dependency management
- job submission and resource isolation
- deployment and rollout
- metrics, logs, tracing, alerting
- artifact and model version management
- permissions and governance

## 2. Performance Dimensions and Their System Meaning

- `latency`: time from request start to usable result; this is the most direct user-visible metric on the inference path.
- `throughput`: useful work completed per unit time, such as requests/sec or tokens/sec; this reflects how well the system amortizes fixed overhead.
- `memory`: the capacity pressure of parameters, activations, KV cache, and buffers; memory limits often decide feasible batch size and model size first.
- `bandwidth`: how fast data moves between memory layers or devices; many inference workloads are bottlenecked by movement, not arithmetic.
- `utilization`: how much of the hardware stays busy doing useful work; low utilization usually means bubbles, stalls, or serialization in the pipeline.
- `stability`: whether the system keeps predictable behavior under load, failures, or noisy inputs; unstable systems destroy benchmark credibility.
- `cost`: the resource price paid for the achieved latency/throughput target; infra is not "fastest possible," it is "fast enough within budget."

### Tensions Between Metrics

- Higher batch size can improve `throughput` but hurt `latency`.
- Larger models can improve quality but increase `memory` pressure and `cost`.
- Aggressive optimization can raise average performance while hurting `stability`.
- Better `utilization` often requires deeper queues or batching, which can worsen tail latency.

## 3. Local Environment Status on This Mac

### What I Verified

- Homebrew-installed Python: `3.12.13`
- Project virtual environment: `.venv`
- PyTorch: `2.10.0`
- `torch.profiler` imports and reports CPU operator timings correctly
- MPS availability on this machine: `False`

### Interpretation

The local environment is now **ready** for CPU-side PyTorch experiments and basic profiling. The only notable limitation is that this machine currently reports `torch.backends.mps.is_available() == False`, so this setup should be treated as a CPU learning environment, not a Mac GPU environment.

### Environment Target

- Python: `3.12`
- Package isolation: `venv`
- Base packages: `torch`, `numpy`
- Profiling entry point: `torch.profiler`

### Setup Commands Used

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch numpy
```

### Minimal Tensor Demo Verified

```python
import torch

x = torch.randn(1024, 1024)
y = torch.randn(1024, 1024)
z = x @ y

print(torch.__version__)
print(z.shape)
```

### Minimal Profiler Demo Verified

```python
import torch
from torch.profiler import profile, ProfilerActivity

x = torch.randn(1024, 1024)
y = torch.randn(1024, 1024)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    z = x @ y

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))
```

## 4. One Inference Request Critical Path

```text
input text
  -> tokenize
  -> host preprocess
  -> numerical tensors
  -> model forward
  -> logits
  -> sampling / decoding
  -> output tokens
  -> detokenize / postprocess
  -> final response
```

### System Interpretation of Each Stage

- `tokenize`: converts user text into model-consumable token ids; it is CPU-side work and can become visible at small batch sizes.
- `host preprocess`: prepares tensors, masks, and request metadata; poor implementation here adds Python overhead and copies.
- `model forward`: the core numerical execution on backend kernels; this is where compute and memory bandwidth usually dominate.
- `sampling / decoding`: converts logits into next-token decisions; repeated decode steps make per-token overhead visible.
- `postprocess`: turns model output into user-facing text or structured response; often ignored, but still part of user-visible latency.

## Output For Day 1

- A working definition of training / inference / platform infra
- A performance-dimension checklist for later benchmarks
- A working local PyTorch environment based on Python `3.12`
- A first-pass inference critical path diagram

## Remaining Gap

The main remaining limitation is not installation but hardware path: MPS is currently unavailable on this machine, so later profiling and benchmarks should be interpreted as CPU results unless that changes.
