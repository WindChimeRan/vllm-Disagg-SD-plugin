# Runtime benchmark

Qwen3-4B + DFlash1 on two B300 GPUs. **128 prompts, 128-token output cap.** C is total concurrent requests. Measured November 1, 2026.

| Method | C=8 tokens/s | C=32 tokens/s |
| --- | ---: | ---: |
| AR, two GPUs | 541 | 1,830 |
| **MineDraft** | **1,001** | **2,704** |
| **Ours, existing checkpoint** | **662** | **1,714** |
| Synchronous, same backend | 1,551 | 3,201 |
| Native DFlash, two GPUs | 1,916 | 5,331 |

All rows use two GPUs. AR and native DFlash use two replicas; their unchanged measurements were reused. The other rows were rerun once. All outputs match the saved AR reference.

**MineDraft reaches 1.85× / 1.48× AR throughput; ours reaches 1.22× / 0.94×.** Native DFlash remains faster at these loads. Our checkpoint has not been trained for prefix gaps.

## Batch size matters

At C=32, MineDraft verifies roughly 16 requests per cohort. Against synchronous C=16, it reaches **2,704 versus 2,359 tokens/s: 14.6% higher throughput**, with mean latency of **1.04 versus 0.67 seconds**. This matches nominal verifier capacity at different total concurrency; it is not a maximum-capacity measurement.

## Training targets from replay

Replay executes the real drafter, transfers and verifier, then replaces only future proposals across gaps with controlled errors. The initial window stays unchanged. Here q is each future token's probability of matching the corrected target trajectory, not prefix-conditioned acceptance.

The first tested q meeting each throughput target was:

| Concurrency | ≥20% over two-GPU AR | ≥10% over MineDraft | ≥10% over strongest SD control |
| --- | ---: | ---: | ---: |
| 8 | Already met | 65% | 90% |
| 32 | 50% | 65% | 95% |

Perfect future proposals reach **3,152 / 6,980 tokens/s**. These are simulated training targets from single passes. The 95% result at C=32 only narrowly clears its goal; it is not a precise threshold or evidence that training can achieve it.

<details>
<summary>Latency, replay measurements and reproducibility</summary>

| Method | Mean latency, C=8 / 32 | p95 latency, C=8 / 32 |
| --- | ---: | ---: |
| AR, two replicas | 1.55 / 1.59 s | 1.92 / 1.94 s |
| MineDraft | 0.83 / 1.04 s | 1.44 / 1.80 s |
| Ours, existing checkpoint | 1.26 / 1.78 s | 1.61 / 2.28 s |
| Synchronous, same backend | 0.53 / 0.94 s | 0.93 / 1.64 s |
| Native DFlash, two replicas | 0.43 / 0.52 s | 0.75 / 0.90 s |

| Future-token match probability | C=8 tokens/s | C=32 tokens/s |
| --- | ---: | ---: |
| 25% | 812 | 2,062 |
| 50% | 1,075 | 2,664 |
| 65% | 1,336 | 3,183 |
| 75% | 1,605 | 3,801 |
| 90% | 2,283 | 5,148 |
| 95% | 2,674 | 5,930 |
| 100% | 3,152 | 6,980 |

- **Data:** 16 prompts from each of eight tasks in [RedHatAI/speculator_benchmarks](https://huggingface.co/datasets/RedHatAI/speculator_benchmarks/tree/2ae86affa2cb97a972b7fc681dd51c04fbff083e), shuffled with seed 20261101. HumanEval uses completion prefixes; other tasks use Qwen3's chat template with thinking disabled. Input lengths are 21–1,462 tokens. Each trial produces 13,740 tokens; 87/128 outputs hit the cap.
- **Models:** `Qwen/Qwen3-4B` at `1cfa9a7208912126459214e8b04321603b3df60c`; `z-lab/Qwen3-4B-DFlash-b16` at `b74e3a329c4d963783143b1e970d95b002be72bd`.
- **Runtime:** vLLM `564ac0de1b228568cafcbfbe8ccaccc82d6023eb`, PyTorch `2.14.0+cu132`, V2 runner, BF16, greedy, eager, batch-invariant kernels, TP=PP=DP=1. CPU async scheduling and prefix caching are off. There are 15 candidates, 32 engine slots, a 2,048-token batch budget, 4,096-token context limit and 8 GiB target KV cache.
- **Timing:** closed-loop continuous batching, including prefill, decoding, transfers, scheduling, detokenization and final worker drain. Loading, warmup and tokenization are excluded; profiling is off. Each new configuration ran once; run-to-run variation was not measured.
- **Proposal quality:** MineDraft and synchronous decoding advance 3.31 tokens per emitting decode step; ours advances 1.07. This excludes the first prefill token and steps without output.
- **Replay:** independent position errors use one deterministic draw per request/output position, seeded with 20261102. No future hidden states are supplied. Real draft costs and oracle substitution remain timed; observation/substitution takes 0.05–0.26% of elapsed time. The strongest SD control includes saved single-GPU native DFlash (1,928 / 3,911 tokens/s), two replicas, synchronous and MineDraft results. Perfect future proposals still retain initial-window errors; overall advance is 6.15 on these short outputs, averaging 107 tokens per request.
- **Validation:** all 21 new trials match every AR token. The runtime includes process isolation and admission-time cohort assignment fixes; production source stayed unchanged. [MineDraft source audit](minedraft-reproduction.md).
- **Local artifacts:** raw results, source hashes and scripts are under `.local/benchmark_refresh_20261101/`; older reports are retained locally. The full 844-prompt workload was not rerun for this refresh.

</details>
