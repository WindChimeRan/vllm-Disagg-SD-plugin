# MineDraft source audit

Audited November 1, 2026 against [`electron-shaders/MineDraft@272c855`](https://github.com/electron-shaders/MineDraft/tree/272c8556797495fe9c9f6a3008337bcd5cc2d031).

**Two reproduction defects were fixed:** running target and draft launch threads in one process, and assigning cohorts before native admission. The current runtime uses a separate draft process and assigns cohorts only when requests enter execution.

| Reference behavior | Current adaptation |
| --- | --- |
| Verify A while drafting B, then swap; save B's proposals for its turn. | Matches tested startup, steady-state and single-cohort schedules. Rejected suffixes are discarded. |
| Separate target and draft ranks. | Separate processes with CUDA IPC feature transfer. |
| Admit up to 2m requests for verifier batches of m. | `max_num_seqs` is total admission capacity; use 2m with enough concurrent requests. |
| Balance admitted cohorts in mixed-prefill mode. | Waiting requests have no membership. Preemption releases it until readmission. |
| Start target scoring before drafting the other cohort. | Our adapter submits drafting before native target preparation. The performance impact of this ordering remains unisolated. |

Source anchors: [pipeline and proposal handoff](https://github.com/electron-shaders/MineDraft/blob/272c8556797495fe9c9f6a3008337bcd5cc2d031/minedraft/plugin/spec_decode/spec_decode_worker.py#L1225), [separate ranks](https://github.com/electron-shaders/MineDraft/blob/272c8556797495fe9c9f6a3008337bcd5cc2d031/minedraft/plugin/distributed/parallel_state.py#L334), [2m admission](https://github.com/electron-shaders/MineDraft/blob/272c8556797495fe9c9f6a3008337bcd5cc2d031/minedraft/plugin/engine/llm_engine.py#L27), [cohort assignment](https://github.com/electron-shaders/MineDraft/blob/272c8556797495fe9c9f6a3008337bcd5cc2d031/minedraft/plugin/core/scheduler.py#L127).

## Evidence

- Selected actual upstream methods, executed with CPU engine/model doubles, match 80 rounds of proposal handoff and cohort transitions. Upstream's mixed-prefill admission/recycling methods also match 20,000 randomized lifecycle events, including 3,067 admissions.
- A native vLLM stress run matches all 11 admissions/readmissions and maintains the same cohort membership as the running request set. Both policies complete seven requests and 408 tokens exactly matching AR, including aborts and four forced preemptions.
- The numerical GPU test compares draft outputs and cached K/V against upstream full-context DFlash, and IPC candidates against the local backend. Cases include unequal lengths, gaps, reordering, empty deltas and request-ID reuse.

With two fixed cohorts of 16, identical candidate tokens and committed outputs, 96 measured decode steps take **13.74 ms/step serially, 19.13 ms with threads, and 12.08 ms with processes**. This uses Qwen3-4B + DFlash1 on two B300s, with 15 candidates per request and one pass per configuration; prefill, warmup, profiling and shutdown are excluded. A separate process-isolation trace shows 77% of draft kernel time overlapping target kernels, versus 24% with threads. This supports host contention as a cause, without isolating the GIL specifically.

[Current end-to-end benchmark](runtime-refresh.md) measures the corrected runtime under continuous batching.

## Scope

This is a MineDraft-style DFlash adaptation. It uses current vLLM's unified prefill/decode scheduler and matches upstream's mixed-prefill cohort rule, rather than its older prefill-only mode. Models, hardware, sampling and transport differ; the complete historical engine and paper performance numbers were not reproduced. Eager execution itself is consistent with the upstream benchmark scripts.

Pinned source files, detailed checks and traces remain local under `.local/minedraft_audit/`. The longer audit narrative is preserved locally with the older reports.
