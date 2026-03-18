# AI Infra Learning TODO

Start date: 2026-03-18
Scope: first 2 weeks only
Background: already familiar with Transformer and basic deep learning



## Week 1

### Day 1 - 2026-03-18
Focus: build the AI infra map and set up the environment
Status: **completed**
Execution note: the work was backfilled on 2026-03-16, and completion was confirmed on 2026-03-18. The planned deliverables exist in `notes/day01-ai-infra-map.md` and `env/local-checklist.md`.

- Write a one-page note answering: what is training infra, what is inference infra, what is platform infra.
- List the main performance dimensions: latency, throughput, memory, bandwidth, utilization, stability, cost.
- Prepare a local environment with Python, PyTorch, and a profiler you can use on your machine.
- Draw the full path of one inference request from input text to output tokens.

Deliverable:
- `notes/day01-ai-infra-map.md`
- A simple environment checklist you can reuse later

### Day 2 - 2026-03-19
Focus: refresh PyTorch execution from a systems angle
Execution note: started early on 2026-03-18 after Day 1 completion was confirmed.

- Review tensor layout, autograd, operator dispatch, and eager execution.
- Trace a small model forward pass and identify where compute, memory movement, and synchronization may happen.
- Write down which parts are Python overhead and which parts are backend execution.
- Record 5 questions you still cannot answer about PyTorch internals.

Deliverable:
- `notes/day02-pytorch-execution.md`

### Day 3 - 2026-03-20
Focus: understand the inference critical path

- Run a small Transformer or LLM inference example.
- Break total time into tokenizer, host preprocessing, model forward, sampling, and postprocessing.
- Measure latency for batch size 1 and batch size 4.
- Note which part of the path would become the bottleneck first.

Deliverable:
- `notes/day03-inference-critical-path.md`
- A table with latency numbers for at least 2 batch sizes

### Day 4 - 2026-03-21
Focus: profiling basics

- Learn the basic outputs of PyTorch profiler or an equivalent profiler available in your environment.
- Profile one inference run and identify the top 5 hottest operators.
- For each hot operator, state whether it is likely compute-bound or memory-bound.
- Summarize which measurements you still do not have but want next.

Deliverable:
- `notes/day04-profiling-basics.md`
- One saved profiling capture or screenshot set

### Day 5 - 2026-03-22
Focus: memory and data movement

- Study host memory vs device memory, pinned memory, copies, and synchronization points.
- Inspect your inference script and mark every likely host-to-device or device-to-host transfer.
- Explain how unnecessary synchronization can silently hurt throughput.
- Write a short note on why memory bandwidth is often more important than FLOPS for inference.

Deliverable:
- `notes/day05-memory-data-movement.md`

### Day 6 - 2026-03-23
Focus: batching and throughput

- Compare single-request inference with batched inference.
- Measure latency and throughput under at least 3 batch sizes.
- Explain why throughput can improve while tail latency worsens.
- Write a short note about when dynamic batching is useful.

Deliverable:
- `notes/day06-batching.md`
- A small benchmark table

### Day 7 - 2026-03-24
Focus: weekly consolidation

- Re-read all week 1 notes.
- Write a 1-2 page weekly summary: what AI infra is, what the inference hot path is, and where optimization usually matters.
- List the top 10 terms you want to master next week.
- Clean up your benchmark scripts and notes so they are reusable.

Deliverable:
- `notes/week1-summary.md`

## Week 2

### Day 8 - 2026-03-25
Focus: LLM inference specifics

- Study prefill vs decode.
- Explain why decode is usually memory-bandwidth sensitive.
- Learn the role of KV cache and write down how it changes the performance model.
- Add prefill and decode as separate stages in your inference notes.

Deliverable:
- `notes/day08-prefill-decode-kv-cache.md`

### Day 9 - 2026-03-26
Focus: serving system concepts

- Learn the purpose of request queueing, scheduling, continuous batching, and admission control.
- Read the architecture overview of one open-source serving system such as vLLM or TensorRT-LLM.
- Summarize the design in your own words instead of copying implementation details.
- Note what problems a model server solves that a plain script does not.

Deliverable:
- `notes/day09-serving-architecture.md`

### Day 10 - 2026-03-27
Focus: quantization and model format tradeoffs

- Study FP16, BF16, INT8, and 4-bit at a systems level.
- Write down the tradeoff among speed, memory footprint, accuracy risk, and engineering complexity.
- If possible, run one quantized model and compare memory use or latency with a higher-precision baseline.
- Record where quantization helps and where it does not.

Deliverable:
- `notes/day10-quantization.md`

### Day 11 - 2026-03-28
Focus: PyTorch distributed basics

- Learn the difference between data parallel, tensor parallel, and pipeline parallel.
- Understand process group, rank, world size, collective communication, and all-reduce.
- Draw one diagram showing how gradients or activations move in each parallel strategy.
- Note which strategies are used more often in training and which in inference.

Deliverable:
- `notes/day11-distributed-basics.md`

### Day 12 - 2026-03-29
Focus: communication costs

- Study why distributed jobs are often limited by communication rather than raw compute.
- Learn the basic purpose of NCCL and common collectives: all-reduce, all-gather, reduce-scatter, broadcast.
- Write down how communication overhead scales as model size or GPU count grows.
- Explain why topology awareness matters.

Deliverable:
- `notes/day12-communication-costs.md`

### Day 13 - 2026-03-30
Focus: observability and failure thinking

- Define the minimum metrics you would watch for an inference service: latency, throughput, error rate, queue length, GPU memory, utilization.
- Define the minimum metrics you would watch for a training job: step time, data time, communication time, loss health, memory, checkpoint success.
- List 5 likely failure modes in AI infra and how you would detect each one.
- Write one page on why observability is part of infra, not an afterthought.

Deliverable:
- `notes/day13-observability.md`

### Day 14 - 2026-03-31
Focus: second-week consolidation and next-step setup

- Write a two-week summary covering inference path, profiling, KV cache, batching, quantization, and distributed basics.
- Identify your strongest area and weakest area from the past 14 days.
- Decide which of these you will study next: CUDA internals, PyTorch internals, inference serving systems, or distributed training.
- Draft one small project for week 3-4, for example an inference benchmark harness or a minimal model serving demo.

Deliverable:
- `notes/week2-summary.md`
- `notes/week3-project-draft.md`

## Rules For Execution

- Spend at least 60-90 minutes each day.
- Every day must produce a written note, not just reading.
- Every benchmark must include environment, batch size, model, and measured output.
- If a topic feels abstract, force it into a diagram or a timing breakdown.
- At the end of each week, compress everything into one summary you could use for interview prep.
