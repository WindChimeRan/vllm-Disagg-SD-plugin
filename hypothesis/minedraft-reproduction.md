# MineDraft reproduction audit

November 1, 2026. Audited upstream commit
[`272c855`](https://github.com/electron-shaders/MineDraft/tree/272c8556797495fe9c9f6a3008337bcd5cc2d031).

**The audit found two implementation problems, now fixed:** shared-process launch
contention and assigning cohorts before native admission. The alternating batch
algorithm was present, but threads made concurrent execution slower than serial
execution. Queued requests could also distort the balance of the active cohorts.

## What matches the source?

| Upstream behavior | Our implementation |
| --- | --- |
| Verify A while drafting B; then swap. Save B's proposals for its next verification. | Matches in the tested startup and steady-state schedules. MineDraft discards the rejected suffix. |
| Target and drafter use separate distributed ranks. | Previously two threads in one process. **Fixed:** a spawned draft process, behind the existing `Worker` interface. |
| Configure a verifier batch of m and admit up to 2m requests. | Our `max_num_seqs` remains the **total** admission limit. Set it to 2m and supply enough requests for two full cohorts. |
| Assign cohorts when requests enter execution; mixed-prefill mode balances admitted requests. | **Fixed:** registration and admission are separate. Queued requests have no cohort; preemption releases membership until readmission. |
| Start asynchronous target scoring before drafting the other cohort. | We submit drafting before native target preparation. Both now progress independently, but launch order differs. |
| Benchmark scripts enable eager execution and disable asynchronous output processing. | Our eager execution is consistent with those scripts; it was not, by itself, a reproduction defect. |

Source anchors: [pipeline and proposal handoff](https://github.com/electron-shaders/MineDraft/blob/272c8556797495fe9c9f6a3008337bcd5cc2d031/minedraft/plugin/spec_decode/spec_decode_worker.py#L1225),
[separate target ranks](https://github.com/electron-shaders/MineDraft/blob/272c8556797495fe9c9f6a3008337bcd5cc2d031/minedraft/plugin/distributed/parallel_state.py#L334),
[2m admission](https://github.com/electron-shaders/MineDraft/blob/272c8556797495fe9c9f6a3008337bcd5cc2d031/minedraft/plugin/engine/llm_engine.py#L27),
[cohort assignment](https://github.com/electron-shaders/MineDraft/blob/272c8556797495fe9c9f6a3008337bcd5cc2d031/minedraft/plugin/core/scheduler.py#L127),
[benchmark settings](https://github.com/electron-shaders/MineDraft/blob/272c8556797495fe9c9f6a3008337bcd5cc2d031/scripts/experiment_4_bs_parallel.sh#L43).

## Admission follow-up

Previously, arrivals A, B, C, D received cohorts before any GPU slots were
allocated. If only A and B were admitted, then A finished and queued C was
cancelled, admitting D left **B and D in the same cohort**. The corrected code
assigns D to the vacated cohort, preserving overlap.

The adapter now reports admission after native scheduling assigns work, before
target execution. Repeated prefill chunks keep membership; cancellation and
preemption release it; an empty running set resets the alternation phase.

The independent reference executes upstream's actual `_get_next_batch_flag`
and `_recycle_batch_flag` methods. All **3,067 admissions across 20,000 randomized
lifecycle events** match its mixed-prefill rule, including out-of-order
admission, queued cancellation, preemption, and late draft completion. A real
vLLM integration run also matches all 11 admissions/readmissions and maintains
exact equality between native running requests and cohort membership.

This comparison uses upstream's **mixed-prefill branch**. The current adapter
uses vLLM's [unified prefill/decode scheduler](https://github.com/vllm-project/vllm/blob/564ac0de1b228568cafcbfbe8ccaccc82d6023eb/vllm/v1/core/sched/scheduler.py#L698).
Upstream's older prefill-only mode instead uses phase-dependent cohort assignment;
we are not claiming to reproduce that scheduling mode or its benchmark settings.

## Does overlapping identical work help?

Two fixed cohorts of 16 requests, Qwen3-4B + DFlash1, two B300s. All variants use
the same MineDraft policy, 15 candidates per request, identical feature spans,
candidate tokens, verifier batches, and committed outputs. The serial control
also uses these two cohorts; it does **not** verify all 32 requests together.

| Execution | Milliseconds per engine step | Relative to matched serial |
| --- | ---: | ---: |
| Serial draft + verify | 13.74 | 1.00× |
| Previous threaded overlap | 19.13 | 0.72× |
| New process overlap | **12.08** | **1.14×** |

Each measurement covers 96 decode steps and 4,984 new tokens. We exclude prefill,
16 warmup decode steps, profiling, and shutdown. The 32 distinct prompts come
from the saved RedHatAI benchmark manifest, restricted to at most 64 prompt
tokens so admission finishes in one prefill batch. EOS is ignored to maintain
fixed cohort sizes. This is a focused diagnostic, not a new full benchmark or a
measurement of saturated serving throughput. One pass per configuration.

A separate process-isolation probe measured 12.01 ms/step before integrating the
worker. Its 12-step CUDA trace showed **77% of draft kernel time overlapping
target kernels**, versus **24%** with threads. Target and draft kernel-active
times were nearly unchanged: approximately 125 ms and 14.4 ms respectively.
This supports host execution contention as the cause; it does not isolate the
Python GIL from every other shared-process overhead. Throughput timing excludes
all profiled steps.

The new worker passes immutable packed features through CUDA IPC, retains their
storage until completion, and drains before target storage is released. The
verification rules remain unchanged. The admission correction above was applied
after this timing diagnostic; it does not change its fixed, fully admitted cohorts.

## Evidence and limits

- Executed selected **actual upstream methods** with CPU model/engine doubles:
  80 rounds match verification membership, draft membership, candidate positions,
  and proposal handoff across startup, varying rejection positions, and a single
  remaining cohort. Separately checked upstream preemption invalidation. This
  does not execute the original vLLM engine or distributed transport.
- The new process worker reproduces every candidate and committed token from
  the serial control. CPU checks cover isolation, backpressure, initialization
  failure, crashes, timeout, and draining. The GPU check compares IPC results
  with the local backend, whose outputs and K/V are checked against upstream
  full-context DFlash; it includes gaps, empty deltas, reordering and ID reuse.
- Both policies pass the integration stress case after the change: arrivals,
  queued admission, abort, chunked prefill and four forced preemptions; each
  completes seven requests with all 408 tokens matching the saved AR reference.
  Validation totals after the admission correction: 27 CPU tests pass. The GPU
  numerical/IPC test passed after the process change; those components did not
  change in the admission follow-up.
- This remains a **MineDraft-style DFlash adaptation**, not a reproduction of
  the paper's performance numbers. Target/draft models, speculative width,
  hardware, sampling, communication, and prefill scheduling mode differ. We have
  not run their complete historical runtime here.

The earlier fixed-C comparison also used different verifier batch sizes:
MineDraft at total C=32 usually verifies 16 requests, while synchronous SD
verifies up to 32. It cannot establish an intrinsic MineDraft disadvantage.
The identified admission mismatch is now resolved and checked against upstream
for mixed-prefill scheduling. Before comparing serving capacity, use matched
verifier batch sizes and a saturated load. Complete historical-runtime and
paper-performance reproduction remain outside this DFlash adaptation's scope.

Raw traces, pinned upstream files, source hashes, and diagnostic scripts stay
local in `.local/minedraft_audit/`. The earlier full benchmark and replay results
describe the threaded runtime; they were not rerun for this audit.
A subsequent [128-prompt benchmark refresh](runtime-refresh.md) measures the
corrected runtime, including a comparison at matched nominal verifier capacity.
