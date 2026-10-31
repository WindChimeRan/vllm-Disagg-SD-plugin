# Related work

Literature and implementation review checked on **2026-10-31**. Code links identify public repositories or PRs inspected during the review; they do not imply current-vLLM compatibility or independently reproduced performance. Where no code link is given, an implementation was not verified in this review.

## Disaggregated drafting, overlap, and shared draft servers

| Work | Relevant idea | Code / scope |
| --- | --- | --- |
| [MineDraft](https://arxiv.org/html/2603.18016v3) | Alternate drafting and verification across batches. Baseline for overlap across requests. | [Code](https://github.com/electron-shaders/MineDraft); the published setup pins vLLM 0.9.2. |
| [PEARL](https://arxiv.org/abs/2408.11850) | Parallel drafting and verification, pre-/post-verification, and adaptive draft length. | [Original code](https://github.com/smart-lty/ParallelSpeculativeDecoding), [nano-PEARL](https://github.com/smart-lty/nano-PEARL). nano-PEARL separates draft/target GPU groups but is based on nano-vllm. |
| [SSD / Saguaro: Speculative Speculative Decoding](https://arxiv.org/html/2603.03251) | Predraft continuations for possible acceptance lengths and bonus tokens while verification runs. Appendix E discusses missing target activations and longer self-conditioning for EAGLE. | [Code](https://github.com/tanishqkumar/ssd); custom engine with NCCL/NVLink communication. |
| [StarSD: One-for-Many Speculative Decoding](https://arxiv.org/html/2601.21622) | Share one drafter across target instances, with per-session KV state and overlapping communication/inference. Particularly relevant to target-conditioned drafting. | [Code](https://github.com/NICE-HKU/StarSD); EAGLE-based server/client implementation sends accepted target hidden states back to the drafter. |
| [SwiftSpec](https://arxiv.org/html/2506.11309) | Disaggregated speculative decoding, parallel tree generation, independent model parallelism, and optimized kernels. | [Code](https://github.com/ByteDance-Seed/SwiftSpec); reported performance includes quantization and kernel optimizations. |
| [SpecActor: Fast LLM Post-training via Decoupled and Fastest-of-N Speculation](https://arxiv.org/html/2511.16193) | Decoupled draft/verify workers, draft-ahead execution, and flexible worker allocation for inference and RL rollouts. | [SGLang PR #22520](https://github.com/sgl-project/sglang/pull/22520), open and unmerged; includes separate-engine launch code and M:N routing. |
| [NebulaSD: Many-for-Many Speculative Decoding](https://arxiv.org/html/2609.29364) | Independent draft/target pools, stage-specific batching, host KV storage, and overlapping state movement with computation. | Four-GPU testbed plus larger profile-driven simulations; large-scale simulation assumes idealized state movement. |
| [SPECTRE](https://arxiv.org/html/2605.08151) | Share a remote small-model service between ordinary requests and speculative jobs; adapt ordinary/parallel speculation. | [SGLang RFC #22044](https://github.com/sgl-project/sglang/issues/22044), [PR #22272](https://github.com/sgl-project/sglang/pull/22272); PR closed without merging. |
| [PipeInfer: Accelerating LLM Inference using Asynchronous Pipelined Speculation](https://arxiv.org/abs/2407.11798) | Continuous asynchronous speculation and early cancellation in a distributed inference pipeline. | Broader pipeline-parallel target execution; a different topology from a complete target on one verifier device. |
| [Speculative Decoding in Decentralized LLM Inference: Turning Communication Latency into Computation Throughput](https://arxiv.org/abs/2511.11733) | Decentralized verification and communication amortization. | Its semantic-importance-based acceptance thresholds require separate consideration from exact target-distribution preservation. |

## Edge/cloud execution and communication

| Work | Relevant idea | Code / scope |
| --- | --- | --- |
| [SpecEdge](https://arxiv.org/html/2505.17052) | Consumer-device drafting, cloud verification, proactive drafting, and server batching across clients. | [Code](https://github.com/kaist-ina/specedge). Distinguish its cached-prefill option from complete request latency. |
| [PicoSpec](https://arxiv.org/html/2603.19133) | Pipelined edge/cloud speculation, rollback handling, and split rejection sampling. | Reference for network overlap and sampling traffic. |
| [PipeSD](https://arxiv.org/html/2605.13319) | Token-batch communication scheduling and adaptive verification triggering. | [Code](https://github.com/Ghanyunhe/PipeSD); separate cloud verification service. |
| [SLED](https://arxiv.org/html/2506.09397) | Adaptive edge speculation and shared batched verification. | Reference for serving multiple clients with a shared verifier. |
| [WISP](https://arxiv.org/html/2601.11652) | Rejection prediction and verification-time estimation for adaptive speculation and scheduling. | Autoregressive early-stop savings need reevaluation for one-pass DFlash blocks. |
| [A Distributed Speculative Decoding Solution for Edge-Cloud Agile Large Model Serving (DSD / DSD-Sim)](https://arxiv.org/html/2511.21669) | Adaptive speculation windows under network and batching constraints. | Simulator-based evaluation; distinct from the decentralized DSD paper above. |
| [DSSD](https://arxiv.org/html/2507.12000) | Split acceptance and residual sampling between device and server to reduce vocabulary-distribution traffic. | Addresses sampling communication; does not by itself remove DFlash target-feature traffic. |
| [DISCO](https://arxiv.org/abs/2610.35957) | Reconstruct draft probabilities with a server-side draft-model replica. | Trades communication for server computation and memory. |
| [Speculation at a Distance: Where Edge-Cloud Speculative Decoding Actually Pays Off](https://arxiv.org/html/2606.25091) | Analyze when communication, wasted predrafting, and server load permit a benefit. | Analytical position paper; motivates evaluating cloud capacity alongside single-request latency. |

## Parallel drafting, suffix reuse, and cache design

These works are especially relevant to the proposed reuse of positions after a correction. Reusing candidate tokens, reusing hidden states to generate new candidates, and reusing valid target execution state are different operations.

| Work | Relevant idea | Relationship to the proposal |
| --- | --- | --- |
| [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/html/2602.06036) | Predict a block in one forward pass conditioned on target features. | Base drafter. Standard training predicts a contiguous block from an anchor; the proposed training allows an anchor-to-draft gap. |
| [H-Spec: Parallel Speculative Decoding Without a Drafter-Side KV Cache](https://arxiv.org/abs/2609.24197) | Combine direct target-KV reuse with last-token target hidden states in a parallel drafter. | Cache-design reference, requiring an architecture change; target-KV access still needs consideration across devices. |
| [ReTrace: Rejected-Trajectory Conditioning for Speculative Decoding](https://arxiv.org/abs/2608.29748) | Use rejected-suffix representations and target-aware correction signals to condition the next DFlash block. | Uses rejected representations to regenerate candidates; compare with retaining existing candidate tokens for reverification. |
| [Carryover Drafting: Recycling Rejected States for Speculative Decoding](https://arxiv.org/html/2609.14717) | Recycle rejected target hidden states as temporary drafter KV context, with parallel draft-verify-draft training. | Relevant training and state-reuse baseline; rejected target states are auxiliary draft context, not committed target KV. |
| [ReAnchor: Reusing Rejected Verifier States for Diffusion Drafting](https://arxiv.org/html/2610.26531) | Generate shifted parallel draft blocks from discarded verifier states and attach continuations through token matching. | Close DFlash-specific prior work on anchors, block extension, and post-correction positional alignment; training-free. |
| [Rejoin Speculative Decoding](https://github.com/cadelew/Rejoin-Speculative-Decoding) | Retain suffix candidates after a correction and reverify them under the repaired prefix. | Research repository, not a paper. Its autoregressive-drafter study reports unfavorable costs and alignment; a useful comparison for testing whether parallel drafting and gap training change the outcome. |

## Upstream integration references

Statuses below are a snapshot from **2026-10-31**. Prefill/decode disaggregation and disaggregated drafter training are separate from the online draft/verifier split targeted here.

| vLLM issue / PR | Status | Relevance |
| --- | --- | --- |
| [RFC #42109: Disaggregated Speculative Decoding with Standalone Parallel Draft Model](https://github.com/vllm-project/vllm/issues/42109) | Open | Standalone draft servers, N:M sharing, isolated request state, and predictive continuation caches. [Public proof of concept](https://github.com/laviier/vllm/tree/disagg_draft). |
| [#63333: Remote draft placement foundation](https://github.com/vllm-project/vllm/pull/63333) | Open, unmerged | Placement configuration, compatibility negotiation, session identities, resource bounds, and failure handling. |
| [#63334: Standalone GPU draft execution](https://github.com/vllm-project/vllm/pull/63334) | Open, unmerged | Native execution and KV maintenance for supported EAGLE/EAGLE3/MTP speculators. DFlash is not in its supported-method set. |
| [#63335: CUDA IPC feature transport](https://github.com/vllm-project/vllm/pull/63335) | Open, unmerged | Same-host target-feature transport with explicit storage lifetimes. |
| [#63205: Custom speculators and configuration classes](https://github.com/vllm-project/vllm/pull/63205) | Merged | Model Runner V2 plugin registration and speculator configuration; a prospective integration entry point. |
| [#64306: Request lifecycle hooks](https://github.com/vllm-project/vllm/pull/64306) | Open, unmerged | Request add/remove and shutdown hooks for per-request remote state. |
| [#64844: Speculative round contract](https://github.com/vllm-project/vllm/pull/64844) | Open, unmerged | Proposed plugin control over verification plans, accepted paths, and candidate budgets. |
| [#33736: Hidden-state extraction system](https://github.com/vllm-project/vllm/pull/33736) | Merged | Feature-extraction reference; does not implement an online remote draft/verify loop. |

The remote-placement series is scoped to same-host execution and does not implement predictive fan-out caching or claim draft/verify compute overlap. Its recent correctness checks use separate processes on one physical GPU; those checks do not establish cross-GPU acceleration.

SGLang's [parallel speculative decoding roadmap #27462](https://github.com/sgl-project/sglang/issues/27462) is another design reference for outcome enumeration, GPU-side continuation selection, versioned buffers, and fallback. Its [protocol/configuration foundation #27634](https://github.com/sgl-project/sglang/pull/27634) merged, but that change alone provides no distributed inference runtime. The roadmap explicitly defers target-hidden-state transport and DFlash support; it closed through inactivity, and several linked runtime follow-ups closed unmerged.
