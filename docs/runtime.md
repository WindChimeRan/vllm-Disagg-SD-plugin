# Runtime prototype

One runtime, two research policies and a synchronous control, two GPUs. The target uses native vLLM
verification and KV management; DFlash1 runs batched inference on the second GPU.

| Component | Owns |
| --- | --- |
| `policy.py` | Which requests verify and where drafting starts |
| `runtime.py` | Candidate positions, outstanding jobs, request generations |
| `dflash.py`, `cache.py` | Incremental draft execution and context slots |
| `transport.py`, `dflash_worker.py` | One batch in flight on a separate process; CUDA IPC feature ownership |
| `vllm/` | Native scheduling, target features, candidate injection and cleanup |

The core has no torch or vLLM dependency. A policy never changes committed token
counts or target KV. The worker receives immutable feature deltas and returns results;
only the coordinator changes request state.
The draft process has independent Python and CUDA launch scheduling. Packing
and CUDA IPC live in the DFlash adapter; the process transport has no torch or
vLLM dependency. New scheduling ideas implement `Policy`; a future remote transport implements
`Worker`. Neither requires changing candidate ownership or verification rules.

## The two policies

**`minedraft`:** alternate two request cohorts. Verify A while drafting B, then
verify B while drafting A from its updated prefix. Requests join a cohort when
native scheduling admits them, using MineDraft's mixed-prefill balance rule.
Waiting requests do not affect cohort balance. An empty cohort cannot strand
the remaining requests. This implements MineDraft's
overlap pattern with DFlash1, not its full published runtime or performance tuning.

**`overlap`:** while verifying one window, draft beyond it using the last available
anchor and the actual future positions. Leave the target's bonus position open.
After a correction, keep uncommitted suffix tokens and verify them again. A gap
is filled by the target before later candidates become eligible.

Example: verify positions 10–24 while drafting 26–40. If position 12 is corrected,
reverify 13–24 under that correction. Position 25 remains the target's responsibility.
The existing checkpoint has **not** been trained for these gaps.

**`sync`:** draft a decode batch, then verify it, using the same backend. This
control measures the effect of overlapping decode work. Prefill admission remains
under native scheduling.

## Correctness and lifecycle

- Candidate tokens are untrusted proposals. Target verification is contiguous
  from the committed prefix; rejected target KV is handled by native vLLM.
- Preserve target features only through the accepted prefix. Immutable deltas
  own their storage and carry CUDA events for cross-device copies. Each delta is
  transferred once, then released from the target-side store.
- A correction keeps the request generation. Preemption or ID reuse creates a
  new generation, so old draft results cannot attach to the new request state.
- Registration does not grant cohort membership. Native admission enrolls a
  generation once, including before a partial prefill; preemption releases it
  until native readmission. Draining all admitted requests resets cohort phase.
- One draft batch and one target batch may be in flight. A draft failure or
  timeout propagates; it does not silently switch to another decoding method.
- The parent retains source features, the packed IPC tensor and its readiness
  event until the child completes. Shutdown drains and joins the child before
  releasing target storage. A failed or timed-out child is stopped first.

The draft worker owns a bounded context cache keyed by request generation.
Retirement travels with the next worker batch, so cancellation or preemption
cannot race an in-flight draft. Only newly committed features are projected to
K/V. Query K/V uses scratch space after the committed prefix and is overwritten
on the next call; future rotary positions do not create fake committed KV.

vLLM currently has no general request-eligibility hook. The scheduler adapter
temporarily presents eligible running requests to native scheduling, reserves
capacity for held requests, then restores order. Held requests retain their KV.
Native preemption therefore chooses among the visible requests. This is the main
version-sensitive seam; the upstream scheduler is neither copied nor patched.

## Supported scope

The adapter targets vLLM main commit
[`564ac0de1b228568cafcbfbe8ccaccc82d6023eb`](https://github.com/vllm-project/vllm/commit/564ac0de1b228568cafcbfbe8ccaccc82d6023eb).
The custom speculator interface is experimental; compatibility with later commits
requires rerunning the integration checks.

The initial backend supports unquantized Qwen3 + DFlash1, greedy text generation,
TP=PP=DP=1, the `uni` executor, V2 model runner, and eager execution. Prefix
caching, LoRA, structured output, and cross-node transport are outside this version.
`async_scheduling=False` disables vLLM's CPU scheduling pipeline; requests still
arrive and finish continuously while the two GPUs can execute concurrently.

The backend reuses the upstream DFlash1 weights, projections and MLPs, with fused
RMS normalization and ragged paged attention from the pinned vLLM build. It packs
feature deltas into one GPU-to-GPU transfer and uploads verifier candidates in
one batch. Draft K/V reserves one slot per active request, sized to the configured
context limit; the 32-slot, 4,096-token benchmark uses 2.5 GiB for draft K/V.
This remains an eager prototype. The CPU coordinator receives one batched token
result, and CUDA graphs are not supported.

## Run

Use a CUDA environment with the pinned vLLM checkout and its dependencies installed.
Install this package and the pinned DFlash implementation:

```bash
uv pip install -e '.[runtime,test]'
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_PLUGINS=disagg_sd
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_BATCH_INVARIANT=1  # compare exact tokens across different batch shapes

python examples/continuous_batch.py \
  --target Qwen/Qwen3-4B --draft z-lab/Qwen3-4B-DFlash-b16 \
  --policy minedraft --output .local/minedraft
```

Use `--policy overlap` for our policy, `--policy target` for the token reference,
and `--reset-at 9` to exercise native preemption/replay. The example submits
eight requests in three waves and aborts one.
Add `--max-num-seqs 4 --long-prefill` to exercise queued admission and chunked
prefill. Use the same prompts and batch-invariant setting for the target reference.

```bash
python -m pytest -q
python examples/check_results.py .local/target .local/minedraft .local/overlap
```

The checker requires exact token parity and, when a joint `profile.json` is present,
positive overlap between actual GEMM/attention kernels on the two GPUs. Host
timestamps alone are not evidence of concurrent GPU execution. Generated outputs
and traces stay under ignored `.local/`.

`--profile` saves only the parent process as `target-profile.json`; it cannot
establish overlap with the spawned drafter. Use Nsight Systems with child-process
capture, or aligned traces from both processes, for that check. Put engine
creation under `if __name__ == "__main__":` when embedding this spawned runtime
in a script, as the example does.

## Validation on November 1, 2026 (UTC)

Qwen3-4B + DFlash-b16, two B300 GPUs, BF16, with the pinned vLLM revision above:

| Check | MineDraft policy | Overlap policy |
| --- | --- | --- |
| Eight arrivals, one abort | 7 completed | 7 completed |
| Exact target-only token parity, batch-invariant mode | 408 / 408 | 408 / 408 |
| Queued arrivals + chunked prefill + forced preemption | 408 / 408 | 408 / 408 |
| Process worker: 128-prompt refresh at C=8 and C=32 | 13,740 / 13,740 per run | 13,740 / 13,740 per run |
| Earlier threaded worker: full 844-prompt benchmark | 153,856 / 153,856 | 153,856 / 153,856 |

The numerical GPU check compares the cached, ragged forward and stored context
K/V against upstream full-context DFlash. It covers mixed lengths, reordered
batches, positive gaps, empty deltas, and request-ID reuse. The spawned worker's
candidate tokens must also exactly match the local backend for every batch. Run it with
`DISAGG_TEST_CUDA=1 python -m pytest tests/test_dflash_cuda.py -q` on two GPUs;
it uses a small synthetic checkpoint without downloading weights.

The [MineDraft source audit](../hypothesis/minedraft-reproduction.md) found that
threaded overlap was slower than serial execution for identical cohorts. The
spawned worker reduces that diagnostic from 19.13 to 12.08 ms/step, versus 13.74
ms for serial execution, with identical candidates and outputs. The process
isolation probe shows 77% of draft kernel time overlapping target kernels;
profiling is excluded from throughput timing. The
[targeted benchmark refresh](../hypothesis/runtime-refresh.md) measures both
corrected policies, the synchronous control and selected replay qualities on
128 saved prompts. The [full 844-prompt benchmark](../hypothesis/runtime-and-training-targets.md)
still describes the earlier threaded worker.

Without batch-invariant mode, both speculative runs diverged from the target-only
run on one of seven completions. Enabling vLLM's batch-invariant kernels removed
the mismatch without changing either policy. Exact-token validation therefore
uses that setting on all compared runs; ordinary BF16 kernels are not claimed
to produce identical tokens across batch shapes.

Both stress runs pass again with the process worker: four active slots, a
557-token prompt, and four native request preemptions. The 27 CPU checks cover
policy transitions, suffix retention,
late results after reset/ID reuse, cache retirement, admission restoration,
configuration rejection, and worker failures. Raw validation and benchmark
artifacts remain local under `.local/`.

The admission audit additionally checks 20,000 randomized lifecycle events
against upstream's mixed-prefill assignment/recycling methods and checks cohort
membership against the real native running set during the MineDraft stress run.
