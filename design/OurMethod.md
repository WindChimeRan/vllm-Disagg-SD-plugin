# Our method: overlap within one request

![Proposed two-GPU system and a timeline of drafting ahead while verifying and correcting an earlier block.](figures/within-request-overlap.svg)

**Keep both GPUs working on the same request.** While the verifier checks an
earlier block, DFlash1 drafts farther ahead. A correction need not discard every
later candidate.

![Retain candidates after correcting position 3, and draft future positions using anchors separated by an unresolved gap.](figures/correction-and-gaps.svg)

**Two ingredients:** reverify retained tokens under the corrected prefix, and
train DFlash1 to predict across gaps from available anchors. Its parallel block
predictions do not feed each sampled token into the next prediction.

These are proposed schedules, not measured timings. Our [first experiment](../hypothesis/suffix-reuse.md)
tests suffix survival; gap training and asynchronous execution are next.
