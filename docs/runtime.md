# Run the plugin

The target uses native vLLM verification and KV management. DFlash1 runs in a separate process on the second GPU, receiving only newly committed target features and keeping its draft KV cache between calls.

| Policy | Behavior |
| --- | --- |
| `minedraft` | Verify one request cohort while drafting the other. |
| `overlap` | Draft beyond an unresolved window and retain candidates after corrections. The checkpoint still needs gap training. |
| `sync` | Draft, then verify; a control using the same backend. |

## Setup

Use a CUDA environment with vLLM commit [`564ac0de`](https://github.com/vllm-project/vllm/commit/564ac0de1b228568cafcbfbe8ccaccc82d6023eb) and its dependencies installed. The custom speculator interface is experimental; this is the tested revision.

```bash
uv pip install -e '.[runtime,test]'
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_PLUGINS=disagg_sd
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_BATCH_INVARIANT=1

python examples/continuous_batch.py \
  --target Qwen/Qwen3-4B --draft z-lab/Qwen3-4B-DFlash-b16 \
  --policy minedraft --output .local/minedraft
```

Use `--policy overlap` for ours or `--policy target` for the token reference. The example submits eight requests in waves and aborts one. Add `--max-num-seqs 4 --long-prefill --reset-at 9` to exercise queued admission, chunked prefill and preemption.

**Supported:** unquantized Qwen3 + DFlash1, greedy text generation, TP=PP=DP=1, V2 runner, `uni` executor, eager execution, `async_scheduling=False`, prefix caching off. CPU async scheduling is disabled; requests still arrive and finish continuously. Batch-invariant kernels are used for exact output comparisons across batch shapes.

## Check outputs

```bash
python -m pytest -q
python examples/check_results.py .local/target .local/minedraft .local/overlap
DISAGG_TEST_CUDA=1 python -m pytest tests/test_dflash_cuda.py -q
```

Use identical prompts/settings for the three output directories. The GPU test needs two GPUs and uses a synthetic checkpoint. The validated source passes 27 CPU tests, the numerical/IPC GPU check, and both abort/preemption stress runs. [Measured throughput, latency and replay targets](../hypothesis/runtime-refresh.md).

## Implementation

| Component | Responsibility |
| --- | --- |
| `policy.py`, `runtime.py` | Scheduling policy, candidate positions, request generations |
| `dflash.py`, `cache.py` | Incremental batched drafting and bounded context slots |
| `transport.py`, `dflash_worker.py` | Separate draft process, packed CUDA IPC features and storage lifetime |
| `vllm/` | Native admission, target features, candidate injection and cleanup |

The coordinator owns request state; policies never modify committed tokens or target KV. Preemption creates a new generation and invalidates old draft results. The draft worker retains immutable input storage until completion and drains before shutdown. Native admission, rather than arrival, assigns MineDraft cohorts.

The main version-sensitive adapter temporarily filters eligible running requests before native scheduling and restores their order afterward. [MineDraft source comparison](../hypothesis/minedraft-reproduction.md).

When embedding the engine in a script, create it under `if __name__ == "__main__":` for process spawning. `--profile` captures only the parent as `target-profile.json`. To measure GPU overlap, capture both processes with Nsight Systems or aligned traces; the output checker accepts a joint `profile.json` for that check.
