# Runtime fixes and drafter training targets

Measured November 1, 2026 (UTC). Qwen3-4B + DFlash1 on B300 GPUs. One pass per configuration; no repeated benchmark trials.

**Historical runtime:** these results use the earlier threaded worker. The later
[MineDraft source audit](minedraft-reproduction.md) found and fixed host execution
contention with a separate draft process. The full benchmark and replay sweep
below have not been rerun; their speedups and training thresholds do not describe
the updated runtime. See the [targeted benchmark refresh](runtime-refresh.md)
for new measurements of the corrected runtime and selected replay qualities.

**The runtime fixes recover MineDraft speedup over AR, but native vLLM DFlash remains faster.** Replay quantifies separate training targets for beating AR and for beating native DFlash.

## Measured runtime performance

All 844 prompts, a 256-token output cap, and the same settings as the [original benchmark](speculator-benchmark.md). These are measurements of our implementation. Both speculative policies use two GPUs and the existing, untrained-for-gaps checkpoint.

| Method | GPUs | C=8 tokens/s | C=8 speedup | C=32 tokens/s | C=32 speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Non-speculative | 1 | 545 | 1.00× | 2,086 | 1.00× |
| Non-speculative, 2 replicas | 2 | 548 | 1.01× | 2,098 | 1.01× |
| MineDraft policy | 2 | 774 | 1.42× | 2,615 | 1.25× |
| Ours, existing checkpoint | 2 | 577 | 1.06× | 1,559 | 0.75× |

Speedups above use one target GPU as the reference. Against **two target replicas**, MineDraft reaches **1.41× / 1.25×**, and ours reaches **1.05× / 0.74×** at C=8 / 32. Concurrency is the system total; this does not measure maximum serving capacity.

| Method | Mean latency, C=8 / 32 | p95 latency, C=8 / 32 |
| --- | ---: | ---: |
| Non-speculative | 2.66 / 2.71 s | 3.87 / 3.96 s |
| Non-speculative, 2 replicas | 2.65 / 2.69 s | 3.84 / 3.90 s |
| MineDraft policy | 1.87 / 2.16 s | 3.67 / 4.25 s |
| Ours, existing checkpoint | 2.51 / 3.66 s | 3.68 / 5.41 s |

## What changed

- Transfer only newly committed target features, packed into one GPU-to-GPU copy.
- Keep persistent draft K/V by request generation; retire it safely after cancellation or preemption.
- Use ragged attention and fused RMS normalization, without padding every request to the longest context.
- Upload verifier candidates once per batch through reusable pinned buffers.

MineDraft improved from **685 → 774 tokens/s** at C=8 and **1,359 → 2,615** at C=32. Its committed advance stayed at **3.50 tokens per emitting decode step**. The improvement came with essentially unchanged proposal quality.

The runtime remains eager. Draft K/V uses bounded slots sized to the configured context limit (2.5 GiB for 32 slots × 4,096 tokens); token results still pass through the CPU coordinator.

## Scheduling controls

A separate, balanced subset uses 128 prompts (16 per task) and a 128-token cap. The synchronous control uses our cached backend; it serializes drafting and decode verification. Native DFlash uses vLLM's own drafter on the target GPU. All configurations are eager.

| Method | GPUs | C=8 tokens/s | C=32 tokens/s | Advance at C=32 |
| --- | ---: | ---: | ---: | ---: |
| Non-speculative | 1 | 535 | 1,821 | 1.00 |
| Non-speculative, 2 replicas | 2 | 541 | 1,830 | 1.00 |
| Native vLLM DFlash | 1 | 1,928 | 3,911 | 3.34 |
| Native vLLM DFlash, 2 replicas | 2 | 1,916 | 5,331 | 3.34 |
| Synchronous, same backend | 2 | 1,494 | 3,038 | 3.31 |
| MineDraft policy | 2 | 703 | 2,045 | 3.31 |

Synchronous drafting beats MineDraft by **2.13× / 1.49×**, despite using the same
backend and committing the same 3.31 tokens per emitting decode step. MineDraft
usually verifies roughly half the active requests per pass at this fixed total
concurrency. This comparison confounds verifier batch size with execution
overhead. The later source audit identifies a separate implementation defect:
threaded overlap is slower even when both variants use identical cohorts.
These measurements do not establish an intrinsic MineDraft disadvantage. We
have not demonstrated a benefit over native DFlash on this model/hardware pair.

A nine-step diagnostic found 105.36 ms of target kernels, 11.38 ms of draft kernels,
and 3.11 ms of simultaneous kernels within a 188.53 ms window. This confirms some
GPU overlap and relatively little draft work to hide. It does not isolate the
individual contributions of batch size, CPU scheduling, or kernel-launch overhead
to the remaining slowdown. Profiling was off for every throughput measurement.

## Replay: how good must future proposals become?

**The following results use simulated candidate quality, not a trained drafter.** They use the same 128-prompt subset and its measured baselines. Every trial runs the real draft network, feature transfers, scheduler, and target verifier. We replace only proposals across positive prefix gaps with oracle tokens plus controlled errors. Gap-zero proposals remain the checkpoint's predictions. Future hidden states are never supplied.

Here **qᵢ** is the probability that future candidate i matches the verifier's corrected greedy trajectory. In the uniform sweep, all 15 positions have qᵢ=q, with independent position errors from one fixed noise realization keyed by request and output position. This is a marginal match rate, not the usual acceptance rate conditioned on all preceding proposals being correct. Speedups below are relative to **two non-speculative replicas**.

| Future-token match probability | C=8 speedup | C=32 speedup |
| --- | ---: | ---: |
| Existing checkpoint | 1.04× | 0.78× |
| 10% at every position | 1.06× | 0.84× |
| 25% at every position | 1.24× | 0.92× |
| 40% at every position | 1.40× | 1.06× |
| 50% at every position | 1.63× | 1.23× |
| 65% at every position | 1.92× | 1.46× |
| 75% at every position | 2.27× | 1.72× |
| 85% at every position | 2.73× | 2.09× |
| 90% at every position | 3.16× | 2.37× |
| 95% at every position | 3.71× | 2.80× |
| 100% at every position | 4.30× | 3.25× |

Define a useful target as **20% more throughput than two AR replicas**, and separately **10% more than MineDraft**. We also compare against the strongest measured SD control (native DFlash with one or two replicas, synchronous cached DFlash, or MineDraft). The first tested uniform probability meeting each target is:

| Concurrency | ≥1.20× two AR replicas | ≥1.10× MineDraft | ≥1.10× strongest SD control |
| --- | ---: | ---: | ---: |
| 8 | 25% | 50% | 100% |
| 32 | 50% | 65% | 100% |

These are tested grid points, not precise thresholds or evidence that training can achieve them. At C=32, the 50% run is 1.097× MineDraft—within about 0.3% of the 1.10× goal. A single pass cannot resolve that boundary.

At **95%** uniform future accuracy, ours reaches **1.04× / 0.96×** the strongest SD control. Even **perfect** future proposals reach only **1.21× / 1.12×**. With this runtime, improving the drafter enough to beat AR is a substantially easier goal than improving it enough to beat native DFlash.

**“Perfect future” still pays for a poor initial window.** The first 15 candidates
come from the existing checkpoint and remain unchanged by replay. Our policy keeps
their suffix after corrections. In the perfect-future C=32 run, those initial
windows contain 1,344 incorrect candidates across 128 requests: **10.5 corrections
per request**. These account for **60.7% of the 2,214 per-request emitting decode
steps**. This is a count of request-level steps, not a fraction of elapsed time.

The average completion here is only 107 tokens. Startup therefore matters greatly:
even perfect future proposals produce an overall advance of 6.15 tokens per step.
The thresholds above apply to this policy and workload; they are not a fundamental
accuracy requirement for the idea. Improving initial suffix recovery, refreshing
poor retained candidates, or amortizing startup over longer outputs could change
them. Those changes were not measured in this sweep.

| Error pattern (75% mean probability) | C=8 speedup | C=32 speedup |
| --- | ---: | ---: |
| Independent, flat 75% | 2.27× | 1.72× |
| 95% at position 1 → 55% at 15 | 2.31× | 1.82× |
| 55% at position 1 → 95% at 15 | 2.34× | 1.72× |
| Whole window correct or wrong | 2.27× | 1.72× |

The existing checkpoint's positive-gap proposals on this subset (C=32) all use a 16-position gap. Its future-token match rates are only a few percent:

| Position in future window | Matches corrected trajectory | Matches after an earlier error |
| --- | ---: | ---: |
| 1 | 2.3% | — |
| 4 | 2.7% | 2.7% |
| 8 | 1.9% | 1.9% |
| 12 | 3.5% | 3.5% |
| 15 | 3.9% | 3.9% |

The tested 75%-mean error patterns have similar throughput, with the front-loaded
profile about 6% faster at C=32 in this pass. This is not evidence of a universal
advantage for that pattern. Training evaluation should report **match probability
by gap and position**, including **after an earlier candidate is corrected**.
Evaluate on held-out trajectories, then compare the trained checkpoint against
both AR and native DFlash.

## Validation and scope

- All complete benchmark, control, and replay outputs match the saved non-speculative token sequences exactly.
- Validation passes: 19 CPU tests, an opt-in GPU comparison of cached outputs and stored K/V against upstream full-context DFlash, and both native preemption/abort stress runs (408 output tokens each). The GPU test covers mixed lengths, reordered batches, gaps, zero-length deltas and request-ID reuse.
- Model revisions, vLLM revision, greedy decoding, batch-invariant kernels, and engine timing boundaries follow the original report. Graphs, prefix caching, and CPU async scheduling remain disabled.
- Replay counters are computed outside the timed interval. Recording observations and substituting oracle tokens remain inside it; their CPU wall time is 0.03–0.28% of elapsed time. Draft compute and transfer costs are fully included.
- The full corpus generates 153,856 tokens per trial; 392/844 requests hit the 256-token cap. The replay subset and shorter cap are a separate workload.

Runtime source was frozen during measurement. Raw results, source hashes, the original implementation, and all benchmark/replay scaffolding remain local under `.local/runtime_optimization/`. No oracle behavior is present in the production plugin.
