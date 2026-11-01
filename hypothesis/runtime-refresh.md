# Benchmark refresh: process-based drafting

November 1, 2026 (UTC). Qwen3-4B + DFlash1 on two B300 GPUs.

Saved **128-prompt subset**, 16 prompts per task, with a **128-token output cap**. All methods below use two GPUs. C is total concurrent requests; every run produces the same 13,740 output tokens.

| Method | C=8 tokens/s | C=32 tokens/s |
| --- | ---: | ---: |
| AR, two GPUs | 541 | 1,830 |
| **MineDraft** | **1,001** | **2,704** |
| **Ours, existing checkpoint** | **662** | **1,714** |
| Synchronous, same backend | 1,551 | 3,201 |
| Native DFlash, two GPUs | 1,916 | 5,331 |

MineDraft, ours, synchronous decoding and selected replay points were rerun once. AR and native DFlash use two replicas; their unchanged controls are reused from earlier measurements on the same day. The [844-prompt results](runtime-and-training-targets.md) remain historical; they were not rerun here.

**MineDraft improves by 42% / 32% at C=8 / 32 after the runtime fixes.** Ours improves by 18% / 20%, but still loses to AR at C=32 with the existing checkpoint. Native DFlash remains faster at these fixed loads.

## Throughput and latency

Against the same **two-GPU AR budget**, MineDraft reaches **1.85× / 1.48×** throughput, and ours reaches **1.22× / 0.94×**. The synchronous control improves by only 4–5%; the much larger MineDraft gain is consistent with the [source audit's process-isolation diagnostic](minedraft-reproduction.md).

| Method | Mean latency, C=8 / 32 | p95 latency, C=8 / 32 |
| --- | ---: | ---: |
| Non-speculative, two replicas | 1.55 / 1.59 s | 1.92 / 1.94 s |
| Native DFlash, two replicas | 0.43 / 0.52 s | 0.75 / 0.90 s |
| Synchronous, same backend | 0.53 / 0.94 s | 0.93 / 1.64 s |
| MineDraft policy | 0.83 / 1.04 s | 1.44 / 1.80 s |
| Ours, existing checkpoint | 1.26 / 1.78 s | 1.61 / 2.28 s |

Proposal quality is essentially unchanged: MineDraft and synchronous decoding commit **3.31 tokens per emitting decode step**, versus **1.07** for ours. This measure excludes the first prefill token and steps without output; elapsed time includes waiting.

## Matching the verifier batch capacity

At C=32, MineDraft splits requests into two cohorts of roughly 16; synchronous decoding can verify all 32 together. One additional synchronous C=16 run matches MineDraft's nominal verifier capacity.

| Method | Total concurrency | Nominal verifier batch | Tokens/s | Mean latency |
| --- | ---: | ---: | ---: | ---: |
| Synchronous | 16 | 16 | 2,359 | 0.67 s |
| MineDraft | 32 | 16 | 2,704 | 1.04 s |

MineDraft delivers **1.15× throughput** in this comparison, while supporting twice the total concurrency. The larger concurrency also has higher request latency. Both use the same two GPUs and backend. This is a finite workload with startup, refill and drain; actual batch membership varies. It measures neither identical scheduled work nor maximum serving capacity. The fixed-C table and this comparison answer different questions.

## Replay: updated training targets

**These are simulated proposal qualities, not a trained model.** Every run keeps the production draft process, real DFlash forward passes, feature transfers and target verification. A local harness substitutes only positive-gap token proposals after receiving the real predictions. Gap-zero predictions remain unchanged; no future hidden states are supplied.

q is the independent probability that each of the 15 future tokens matches the corrected greedy trajectory. It is a marginal token-match probability, not prefix-conditioned acceptance. All runs use the same saved prompts and deterministic error draws.

| Future-token match probability | C=8 tokens/s | C=32 tokens/s |
| --- | ---: | ---: |
| Existing checkpoint | 662 | 1,714 |
| 25% | 812 | 2,062 |
| 50% | 1,075 | 2,664 |
| 65% | 1,336 | 3,183 |
| 75% | 1,605 | 3,801 |
| 90% | 2,283 | 5,148 |
| 95% | 2,674 | 5,930 |
| 100% | 3,152 | 6,980 |

The first tested uniform quality meeting each target is:

| Total concurrency | ≥1.20× two AR replicas | ≥1.10× MineDraft | ≥1.10× strongest measured SD control |
| --- | ---: | ---: | ---: |
| 8 | Already met | 65% | 90% |
| 32 | 50% | 65% | 95% |

These are coarse, single-pass targets; they do not locate precise thresholds or establish that training can achieve them. The strongest SD control is the best observed native DFlash, synchronous, or MineDraft result at that concurrency, within the two-GPU budget.

At C=32, the 95% run is **1.112×** the strongest control, only about 1.1% above the chosen 1.10× goal. A single pass cannot resolve that boundary reliably.

Perfect future proposals reach **1.64× / 1.31×** the strongest SD control. They still retain the checkpoint's initial window, including its errors. With only 107 output tokens per request on average, that startup cost matters: overall advance is 6.15 even with perfect future proposals. Different startup handling or longer outputs could change these targets.

## Measurement details

- All **21 new trials** (six policy/load cases, one capacity control, fourteen replay cases) match every saved AR token exactly. Production source stayed unchanged throughout; the runtime includes both process isolation and admission-time cohort assignment fixes.
- The reused single-GPU controls reach 535 / 1,821 tokens/s for AR and 1,928 / 3,911 for native DFlash at C=8 / 32. They are included when selecting the strongest measured SD control for replay targets.
- The [original setup](speculator-benchmark.md#setup-and-timing) supplies model, dataset and vLLM revisions. This subset has 21–1,462 input tokens; 87/128 outputs hit the 128-token cap. BF16, greedy generation, eager execution, batch-invariant kernels, 15 candidates, 32 engine slots and an 8 GiB target KV cache remain unchanged.
- Closed-loop continuous batching replaces completed requests immediately. Timing includes prefill, drafting, verification, transfers, scheduling, detokenization and final worker drain. Model loading, warmup and prompt tokenization are excluded. Profiling and per-step tracing are off.
- Replay observation/substitution time is included (0.05%–0.26% of elapsed time); summary counters are computed afterward. No draft cost is subtracted.
- Each new configuration ran once. Reused controls are identified above; variation between runs was not measured. These fixed-load results do not establish saturated capacity or complete performance fidelity to the original MineDraft runtime.

Raw outputs, source hashes and benchmark/replay scaffolding remain local under `.local/benchmark_refresh_20261101/`. The earlier [threaded-runtime report](runtime-and-training-targets.md) is retained as history.
