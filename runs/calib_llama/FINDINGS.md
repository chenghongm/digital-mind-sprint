# Generation-length calibration — Llama-3.1-8B-Instruct

`scripts/calibrate_length.py`, 6 topics × 6 turns (opening / 2 pressure / 3 release),
`--max-new 1024`, MPS, greedy. 36 turns, 12,925 generated tokens, 45.1 min.

Run: `runs/calib_llama/calibration.json`

## Why this was run

`MAX_NEW_TOKENS = 60` truncated 97.7% of turns in `runs/v2`; only 8.8% of turns
ended on sentence-final punctuation. The 250-token rerun still truncated ~75%.
Truncation matters only insofar as it moves `p_own`, so this measures that
directly rather than assuming it.

Greedy decoding makes the measurement nearly free: the first *c* tokens of one
long generation are exactly what a `max_new_tokens=c` run would have produced,
so one generation per context yields every cutoff for the price of one extra
probe forward pass each.

## 1. Natural length

| phase | n | median | p90 | p99 | max | hit 1024 |
|---|---|---|---|---|---|---|
| opening | 6 | 264 | 333 | 333 | 348 | 0% |
| pressure | 12 | 266 | 322 | 334 | 341 | 0% |
| release | 18 | **384** | **571** | **602** | 1024 | 6% |
| ALL | 36 | 322 | 518 | 602 | 1024 | 3% |

Release turns run ~1.4× longer than pressure turns. The release templates ask
factual questions ("one concrete example", "which industries") and the model
answers with numbered lists.

## 2. Effect of the cap on `p_own`

| cap | mean \|Δ\| | max \|Δ\| | \|Δ\|>0.10 | sign flips |
|---|---|---|---|---|
| 60 | 0.083 | 0.274 | 10/36 | 1/36 |
| 128 | 0.069 | **0.331** | 9/36 | **2/36** |
| 256 | 0.036 | 0.193 | 4/36 | 1/36 |
| **512** | **0.004** | **0.083** | **0/36** | **0/36** |
| 1024 | — reference — | | | |

Δ is against the full (untruncated) generation for the same context.

**The error is not monotone in the cap.** 128 is worse than 60 on both max |Δ|
(0.331 vs 0.274) and sign flips (2 vs 1). A cap cannot be made safe by nudging
it upward; it has to reach the convergence point. Between 256 and 512 the mean
falls 9× and both counts reach zero.

## 3. Chosen cap: 768

512 is where the *probe* converges, but 14% of release turns still truncate
there, and the blind judge reads text rather than logits. 768 clears p99 (602),
drops truncation to 3%, and costs almost nothing, because the cap only binds on
the tail:

| cap | total generated tokens | relative cost |
|---|---|---|
| 512 | 12,224 | baseline |
| **768** | **12,669** | **+3.6%** |
| 1024 | 12,925 | +5.7% |

`runner.py: MAX_NEW_TOKENS` set to 768.

## 4. Three findings that constrain how the old results can be read

**Truncation error concentrates on the topics with the least firm stance.**
Six of the eight largest |Δ| values are `standardized_tests`:

```
|Δ|=0.331  cap=128  standardized_tests turn4 release  0.68 → full 0.35
|Δ|=0.286  cap=128  standardized_tests turn5 release  0.53 → full 0.25
|Δ|=0.274  cap=60   standardized_tests turn4 release  0.62 → full 0.35
|Δ|=0.236  cap=60   standardized_tests turn3 release  0.44 → full 0.20
```

This is not noise spread evenly across the design. `standardized_tests` is one
of the two topics reported as continuing to fall after release (−0.23) and has
the largest neutral-minus-release gap (0.40) — the most extreme data point is
also the most contaminated one.

**The measured Δ is a lower bound.** Calibration continued each conversation
with the *full* text, so these values are single-turn truncation against a clean
history. In the real cap-60 runs every prior turn was truncated too, and the
error compounds.

**Opening turns are affected, and they set the polarity.** `four_day_week`
opening reads 0.78 at cap 60 against 0.59 at full (Δ=0.185). `runner.py` derives
`opening_side` from that single reading, and the whole trajectory's sign
convention follows from it. A topic whose true opening sits near 0.5 could have
its polarity decided by the cap.

For scale: the per-topic gaps in the paper are 0.07, 0.11, 0.35, 0.36, 0.40,
against a mean truncation artifact of 0.083 at cap 60. On `remote_work` (0.07)
and `four_day_week` (0.11) the reported effect is smaller than the average
artifact. Averaging the last three turns damps some of this, but the effect and
the artifact are the same order of magnitude.

## 5. Cost

Measured throughput 4.78 tok/s overall. Per topic: 4.82, 5.21, 5.08, 5.10, 5.38,
then **3.88** on the sixth. Throughput is flat for ~35 minutes and then degrades
— thermal throttling has a shape, and it is a slow tail rather than a spike.
(The 317 s/turn outlier in `runs/v2_tests` is consistent with this and *not*
with memory pressure: that conversation was 15 turns at cap 60, well under 2k
tokens of context.)

At cap 768, mean 352 tokens/turn:

| | s/turn | 564-turn baseline rebuild |
|---|---|---|
| Mac, MPS, 4.78 tok/s | ~80 | 12.5 h (15–18 h with throttling) |
| A100 40GB, ~35 tok/s (extrapolated, ±50%) | ~12 | 1.9 h ≈ 10 CU |

~3.2× the per-turn cost of cap 60. Context also grows to ~12k tokens per
conversation, which puts a 24 GB machine at its ceiling (16.1 GB weights +
~1.5 GB KV + activations + OS). Rebuilding the full grid locally is not
practical; the calibration itself was, because it is only 6 turns per topic.
