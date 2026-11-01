# End-to-end benchmark: RedHatAI speculator suite

**Historical result, before the incremental draft cache and transfer fixes.** See
[the benchmark refresh](runtime-refresh.md) for current measurements on the saved 128-prompt subset.

Measured November 1, 2026 (UTC). In this initial implementation,
MineDraft improved throughput at concurrency 8; both speculative policies lost at
concurrency 32. Our untrained gap policy was slower at both loads.

## Throughput and request latency

One complete pass over the dataset per method and concurrency level. Speedup is relative to one non-speculative target GPU at the same total concurrency; values below 1× are slowdowns. Latency includes prefill and decoding.

| Total concurrency | Method | GPUs | Output tokens/s | Throughput speedup | Mean latency | p95 latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | Non-speculative | 1 | 545 | 1.00× | 2.66 s | 3.87 s |
| 8 | Non-speculative, 2 replicas | 2 | 550 | 1.01× | 2.64 s | 3.82 s |
| 8 | MineDraft policy | 2 | 685 | 1.26× | 2.11 s | 4.87 s |
| 8 | Ours, untrained DFlash1 | 2 | 466 | 0.85× | 3.12 s | 4.65 s |
| 32 | Non-speculative | 1 | 2,087 | 1.00× | 2.71 s | 3.96 s |
| 32 | Non-speculative, 2 replicas | 2 | 2,057 | 0.99× | 2.72 s | 3.93 s |
| 32 | MineDraft policy | 2 | 1,359 | 0.65× | 4.19 s | 8.50 s |
| 32 | Ours, untrained DFlash1 | 2 | 1,220 | 0.58× | 4.69 s | 6.82 s |

The two-replica baseline measures both GPUs together; it is not an extrapolation. With the same two-GPU budget, MineDraft's throughput ratios are **1.24× / 0.66×** at concurrency 8 / 32. Ours reaches **0.85× / 0.59×**.

Concurrency is a system total: two replicas receive 4 or 16 requests each. These measurements describe performance at those loads; they do not establish the maximum throughput of either GPU allocation.

## What the current implementation achieves

MineDraft's smaller-load throughput gain comes with worse p95 latency. At higher concurrency, its larger output advance does not translate into throughput.

Observed output advance per emitting decode step is **3.50** tokens for MineDraft and **1.05** for ours at concurrency 8 (3.50 and 1.06 at 32). Non-speculative decoding advances one token. This counts committed output tokens, excluding the first prefill token and steps without output. Waiting between steps still contributes to request latency.

Both policies used the existing DFlash1 checkpoint, without gap training. This
version copied target-feature history and recomputed context K/V each round,
without an incremental draft cache. The benchmark included those costs but did
not isolate their contribution to the slowdown. The updated runtime fixes these
issues.

## Task breakdown

Mean request-latency speedup relative to one target GPU, within the mixed workload. Each row contains all prompts from that source file.

| Source | Requests | MineDraft, C=8 | Ours, C=8 | MineDraft, C=32 | Ours, C=32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HumanEval | 164 | 1.63× | 0.85× | 0.84× | 0.58× |
| math_reasoning | 80 | 2.20× | 0.85× | 1.10× | 0.58× |
| qa | 80 | 0.94× | 0.85× | 0.47× | 0.58× |
| question | 80 | 1.15× | 0.86× | 0.60× | 0.58× |
| rag | 80 | 1.10× | 0.85× | 0.57× | 0.58× |
| summarization | 80 | 0.93× | 0.85× | 0.47× | 0.57× |
| tool_call | 200 | 1.20× | 0.86× | 0.64× | 0.58× |
| translation | 80 | 0.92× | 0.90× | 0.50× | 0.60× |

## Setup and timing

- **Data:** all 844 prompts from [RedHatAI/speculator_benchmarks](https://huggingface.co/datasets/RedHatAI/speculator_benchmarks/tree/2ae86affa2cb97a972b7fc681dd51c04fbff083e). `writing.jsonl` is byte-identical to `question.jsonl`, so it is counted once. All eight distinct files, including `tool_call.jsonl`, are included.
- **Prompts:** HumanEval uses its original completion prefix; other tasks use Qwen3's chat template with thinking disabled. No input truncation; 21–2,798 input tokens.
- **Generation:** greedy, EOS respected, maximum 256 new tokens. Every trial generates 153,856 tokens; 392/844 requests (46.4%) hit the length cap, including all HumanEval requests. This is a generation-speed benchmark, not a task-accuracy evaluation.
- **Models:** `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c`; `z-lab/Qwen3-4B-DFlash-b16` revision `b74e3a329c4d963783143b1e970d95b002be72bd`. BF16, 15 draft tokens per window. MineDraft denotes our implementation of its two-cohort schedule.
- **Runtime:** vLLM `564ac0de1b228568cafcbfbe8ccaccc82d6023eb`; PyTorch `2.14.0+cu132`. V2 runner, TP=PP=DP=1, eager execution, batch-invariant kernels, CPU async scheduling and prefix caching disabled for every mode. 32 engine slots, 2,048 batched-token budget, 4,096 context limit, 8 GiB target KV cache per GPU.
- **Hardware:** NVIDIA B300 SXM6 GPUs 5 and 6, driver 610.43.02. Comparisons run sequentially; only the two AR replicas run simultaneously.
- **Load:** closed-loop continuous batching, replacing completed requests immediately up to the concurrency limit. Full corpus shuffled with seed 20261101, using the same input order for each method. Two target replicas receive disjoint round-robin shards of that order.
- **Boundary:** engine submission to final output, including request setup, prefill, drafting, verification, GPU copies, scheduling, and detokenization. Throughput includes draining outstanding draft work at the end. Model loading, 32-request warmups, prompt tokenization, dataset I/O, and HTTP transport are excluded.
- **Controls:** runtime source frozen throughout; profiling and per-step trace logging disabled. All generated token sequences match the paired non-speculative reference in every measured run. The common eager/batch-invariant settings define this prototype comparison; it is not a comparison against a separately optimized serving configuration.

These are single-pass measurements; run-to-run variation was not measured. Raw timings, tokens, pinned data manifest, source hashes, and the benchmark harness remain local under `.local/speculator_benchmark/`. The tracked artifact is this report.
