# Similar EAL, better suffix recovery with DFlash

**DFlash and the AR drafter have similar ordinary acceptance, yet DFlash is about
twice as likely to preserve a useful token after a correction.** This is a useful
signal for our idea: the verifier may be able to continue with existing DFlash
candidates while the drafter works ahead.

We tested Qwen3-4B with two drafters—DFlash1-b16 and Qwen3-0.6B—on all 164
HumanEval prompts. At a rejection, we commit the verifier's correction and try
the remaining old candidates. We also draft a fresh block from that same
corrected prefix as a reference.

| Measurement | DFlash1 | AR drafter |
|---|---:|---:|
| Ordinary decoding EAL | **6.41** | **6.49** |
| Fresh candidates accepted after correction | 4.58 | 4.52 |
| Corrections followed by at least one accepted old token | **35.5%** | **17.0%** |
| Mean consecutive old tokens accepted | **0.98** | **0.65** |

EAL counts accepted candidates plus the verifier's correction/bonus token, using
full blocks and excluding terminal rounds. Fresh and reused candidate counts
exclude that correction/bonus.

The two baseline averages are close, while DFlash's suffix survives more often
and yields about **1.5× as many reused tokens per rejection**. The difference is
also visible in our original position-3 example: after correcting position 3,
DFlash accepts **0.97** old tokens from position 4 onward, versus **0.59** for AR.

This supports the intuition behind parallel drafting: later DFlash candidates
were produced without feeding back the earlier wrong draft token. Training,
target-feature access, and rejection contexts also differ between these models,
so the result leaves open how much each factor contributes.

The gain is concentrated in short recoveries. DFlash accepts four or more old
tokens in **8.5%** of events, versus **6.0%** for AR; AR has more of the rare long
surviving tails.

Locally, fresh drafts arrive quickly. Here is the cost of one isolated
continuation step:

| One DFlash continuation step | Mean time | Mean new tokens committed |
|---|---:|---:|
| Verify the old suffix | 22.40 ms | 1.98 |
| Draft and verify a fresh block | 27.07 ms | 5.56 |

New tokens here include the next correction/bonus. Dividing total measured time
by total new tokens gives **11.32 ms/token for reuse** and **4.86 ms/token for
fresh drafting**. These are isolated steps from the same corrected prefix,
measured in eager FP32 on B300. Fresh drafting itself takes only about 4 ms.

Our proposed system changes when fresh candidates become available. The next
experiment should measure whether using the old suffix during that wait reduces
latency, with reuse enabled and disabled in the same DFlash pipeline. The current
result gives us a reason to test that overlap, and a clear target for gap training:
make the surviving continuations longer.

<details>
<summary>Setup, checks, and data</summary>

2026-10-31. Greedy decoding, thinking disabled, block size 16 (one anchor plus
15 candidates), maximum 1,024 new tokens. Two prompts reached the output cap.
There were 9,454 eligible DFlash rejections and 8,752 AR rejections. Empty suffixes
and events near the output cap were excluded. Each drafter contributed its own
natural rejection events; within each event, old and fresh trials shared the
corrected prefix. Trials were rolled back before ordinary decoding continued.

Both FP32 runs produced identical output tokens on all 164 prompts, with zero
greedy or trial-continuation mismatches. All 18,206 cache-isolation checks passed.
The BF16 repeat had a similar DFlash retention result, but showed 290 differences
between cached and full-sequence greedy choices; FP32 supplies the consistency
checks reported here. No gap training or distributed execution has been run.

Experiment code, traces, figures, and detailed statistics are stored locally in
`experiments/suffix_reuse/` and excluded from version control.

</details>
