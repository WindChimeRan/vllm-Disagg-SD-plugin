# Our method: overlap within one request

A parallel drafter can learn to produce useful future continuations across
unresolved prefix gaps, allowing drafting and verification to progress
concurrently within one request.

![Proposed two-GPU system and a timeline of drafting ahead while verifying and correcting an earlier block.](figures/within-request-overlap.svg)

**Keep both GPUs working on the same request.** While the verifier checks an
earlier block, DFlash1 drafts farther ahead. A correction need not discard every
later candidate.

![Retain candidates after correcting position 3, and draft future positions using anchors separated by an unresolved gap.](figures/correction-and-gaps.svg)

**Two ingredients:** reverify retained tokens under the corrected prefix, and
train DFlash1 to predict across gaps from available anchors. Its parallel block
predictions do not feed each sampled token into the next prediction.

These are proposed schedules, not measured timings.

## Comparison with MineDraft

| | [MineDraft](https://arxiv.org/html/2603.18016v3#S5) | Our proposed method |
|---|---|---|
| Concurrent work | Verify batch A while drafting batch B | Verify earlier positions while drafting later positions for the same requests |
| Draft conditioning | Each request's resolved prefix from its previous verification | Available anchors, potentially separated from future positions by unresolved gaps |
| After rejection | Resolve that request before its next drafting turn; the other batch continues | Correct the token, reverify retained candidates, and continue drafting ahead |
| Required concurrency | Two populated request groups sustain overlap | Could sustain overlap with one group, potentially one request |
| Drafter requirement | Supports different drafting methods | Exploits DFlash1's parallel predictions and proposed gap training |

## Pilot study

**Similar EAL, better suffix recovery with DFlash.** We tested Qwen3-4B with
DFlash1-b16 and the Qwen3-0.6B AR drafter on all 164 HumanEval prompts. Ordinary
EAL was similar—**6.41 versus 6.49**—while the chance of retaining at least one
useful suffix token after a correction was **35.5% versus 17.0%**. Mean reused
tokens per correction were **0.98 versus 0.65**.

This supports testing suffix reuse during overlap.
[Full experiment report](../hypothesis/suffix-reuse.md).

The runtime now supports continuous batching with both policies. Gap training
remains future work; [runtime measurements and oracle replay](../hypothesis/runtime-refresh.md)
measure the current checkpoint and estimate the proposal quality needed for speedup.
