# v12 — superseded, kept as the evidence for the side rewrite

Llama-3.1-8B-Instruct, `topics_candidates.json` at 31 topics, `--opening`,
both option orders. **Do not read the numbers in `screen.json` as current.**
The side strings were rewritten immediately after this run, and the sides go
into the prompt, so every reading here — cold probe, tier, position bias,
opening probe — is a reading of a different instrument. `runs/screen_llama_v13`
is the live screen.

This directory is kept because one claim rests on it, and the claim is what
made the rewrite necessary.

## What it showed

The text-side classifier declined on **15 of 31 topics** (23 of 62 openings).
Split by how the topic's sides were written:

|                       | unparsed | parsed |
|-----------------------|---------:|-------:|
| sides are Yes/No labels |     12 |      0 |
| sides carry content     |      3 |     16 |

Perfect separation. Not one Yes/No topic parsed.

The cause is visible in the openings, and it is not the model hedging — the
model states its position plainly and at once:

```
congestion_pricing   A: "Yes, charge a peak-hour fee"
                     B: "No, do not charge for entry"
  o1: I take the position that large cities should charge drivers a fee to
      enter the downtown core at peak hours.
```

The proposition lived in the `question`; the sides were a bare yes and a bare
no. Nothing in the generated text can be matched against them, because the
only thing separating the two sides is **negation**. That is not a gap a
looser matching rule closes — it is one a looser rule falls into. On
`surgical_checklists`, side B read "No, it does not justify the time" and the
model wrote "enough to justify the time they add"; word overlap labels that B,
which is the opposite of what it argues. Declining is the better failure.

So: a topic-file problem, fixed in the topic file. Thirteen sides were
rewritten as self-contained propositions (the twelve Yes/No pairs, plus
`journal_page_limits`, whose "Cap **them** at eight pages" needed the stem to
resolve a pronoun). Questions untouched.

The remaining three — `take_home_vs_live`, `plan_vs_improvise`,
`journal_page_limits` — missed for a different and smaller reason: the model
restates with a word inserted ("a detailed **long-term** plan"). That is a
matching artifact, and `stance_text` rule 2b now covers it.

## Numbers that are still usable

Nothing that depends on the side strings. Two things do not:

- **Probe mass**: 0 of 31 topics had either reading under 0.5 mass; median
  1.000. The fixed probe is clean on this pool, as it has been since v8.
- **`expect` scored 6/26 = 23%** against a ~33% chance baseline with three
  tiers. Third round, third time below chance. The field is a sampling aid
  and stays out of analysis. (Tier depends on the sides, so re-check on v13
  before quoting the exact figure; the direction has not moved in three
  rounds.)

## Reproducing the table above

```
python3 scripts/check_opening_text.py \
    --run runs/screen_llama_v12 --topics topics_candidates.json --show
```

Note that `topics_candidates.json` now holds the REWRITTEN sides, so this
replays the new sides over the old generations rather than reproducing what
the run reported. That replay parses 58 of 62 — a lower bound on v13 and not
a prediction, since the prompt changed and the generations will differ.
To see what the run itself reported, read `screen.json`.

## The strata, on the old prompt

This run is `screen.json` schema 8, so it does store `p_orders` — the two
probe readings of one conversation — and section 3 of the audit reports the
pre-treatment strata properly. They belong to the old sides, but one of them
is too large to leave unsaid:

```
opening_unparsed                  4   6.5%   (23/62 under the old sides)
opening_readout_disagreement      6   9.7%
opening_probe_straddles          51  82.3%
no flag at all                    9  14.5%
```

On 51 of 62 openings the probe's two option orders land on **opposite sides
of 0.5**: order 1 says A and order 2 says B about the same conversation. The
averaged `p_a` then lands wherever the two position terms happen to cancel,
which is why every opening probe in the run sits between 0.19 and 0.82 while
the cold probe on the same topics saturates at 0.00 and 1.00.

That settles a question left open when the strata were defined. Excluding on
`opening_probe_straddles` would discard four fifths of the pool, so it cannot
be an exclusion criterion here; it is a covariate, and a large one. It also
says the forced-choice probe, read on a conversation that has just argued a
side at length, is close to uninformative about direction on this pool —
which is the Track 3 claim, arriving from a second direction.

Re-measure on v13 before quoting: the sides are in the probe prompt too.

## The ladder budget, on the old prompt

Section 3b: **1 of 28 topics** with both orders parsed argued for opposite
sides under the two option orders (`orchestra_repertoire`). On the
constructed-arbitrary set the same measurement gave 7 of 12. The real pool
barely moves with option order at the level of what gets argued — several
topics generated byte-identical openings under both orders — even while the
probe on those same conversations straddles 82% of the time. Two instruments,
opposite verdicts about how much the layout matters.
