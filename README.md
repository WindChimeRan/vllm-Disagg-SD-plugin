# vllm-Disagg-SD-plugin

A vLLM plugin that runs speculative drafting and verification on separate GPUs.

## Goal

- **Single node:** one GPU for drafting, others for verification. Start with 1:1.
- **Cross node:** local DGX Spark / vLLM-metal for drafting, cloud B300 for verification. Start with Spark + one B300.
- Stay compatible with the latest vLLM.
- **DFlash1 only.** Other drafters may serve as benchmarks; integrations are not planned.

## Approach

**Baseline — overlap across requests:** draft one batch while verifying another, as in [MineDraft](https://arxiv.org/html/2603.18016v3).

**Our idea — overlap within one request:** exploit DFlash1's parallel drafting to keep useful candidates after a correction. If position 3 is wrong, correct it and reverify the existing tokens from position 4 onward, while the drafter predicts farther ahead.

Retrain DFlash1 with **skip-gram gaps between available anchors and drafting positions**, so drafting can continue without waiting for the full intervening prefix. Reuse candidate tokens; recompute verification under the corrected prefix.

[MineDraft design](design/MineDraft.md) · [Our method](design/OurMethod.md) · [First hypothesis test](hypothesis/suffix-reuse.md)

## Runtime

The prototype provides both policies through one out-of-tree plugin, using native
vLLM verification and a separate batched DFlash1 worker.
[Architecture, supported scope, and running instructions](docs/runtime.md).
[Latest benchmark and drafter training targets](hypothesis/runtime-refresh.md).
[MineDraft reproduction audit](hypothesis/minedraft-reproduction.md).

## Main challenges

- Target-feature transfer may become communication-bound.
- Manage draft KV state and corrections while both GPUs are working.
- Keep enough later candidates useful to justify reverification.

## Related work

- [MineDraft](https://arxiv.org/html/2603.18016v3), [PEARL](https://arxiv.org/abs/2408.11850), [SSD / Saguaro](https://arxiv.org/html/2603.03251) — overlapping drafting and verification.
- [StarSD](https://arxiv.org/html/2601.21622), [SpecActor](https://arxiv.org/html/2511.16193) — separate draft and verifier services.
- [SpecEdge](https://arxiv.org/html/2505.17052) — edge/cloud speculation.
- [ReTrace](https://arxiv.org/abs/2608.29748), [ReAnchor](https://arxiv.org/html/2610.26531), [Rejoin](https://github.com/cadelew/Rejoin-Speculative-Decoding) — reuse after rejection.
- [H-Spec](https://arxiv.org/abs/2609.24197) — drafter KV-cache design.

[Full literature review, code links, and vLLM issues/PRs](docs/related-work.md).
