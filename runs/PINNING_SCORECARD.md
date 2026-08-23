# Pinning scorecard — v7 against v8

Predictions in `PINNING_PREDICTIONS.md`, recorded and committed before v8 ran.

## The intervention hit its target

| | v7 | v8 |
|---|---|---|
| P("it depends"), median, C last | 0.327 | **0.071** |
| P("it depends"), position-averaged | 0.408 | **0.113** |
| topics above 0.5 | 11/31 | **2/31** |

Pinning the actor, the scale and the outcome measure cut the model's
preference for declining the framing by three to four times.

## The predictions were about chance

`p` is P(side_a) at order 1.

| topic | v7 → v8 | predicted | |
|---|---|---|---|
| curbside_plastics | 0.39 → **0.93** | down | **opposite** |
| open_plan_offices | 0.85 → **0.34** | up | **opposite** |
| generalists_specialists | 0.03 → 0.00 | up | **opposite** |
| nuclear_power | 1.00 → **0.06** | down, large | correct |
| gene_edited_crops | 0.73 → 0.95 | up | correct |
| four_day_week | 1.00 → 0.78 | down | correct |
| remote_work | 0.99 → 0.97 | down | correct, barely moved |
| prize_individual_team | 0.00 → 0.00 | stays low | correct |
| test_coverage_80 | 0.05 → 0.11 | stays low | correct |
| journal_page_limits | 0.21 → **0.97** | no call | flipped |
| nurse_shift_length | 0.57 → 0.44 | no call | |

Six of nine calls landed, three inverted. On directional guesses that is not
distinguishable from chance.

`curbside_plastics` is the clearest miss. The rewrite states $2.1m spent
against $600k recovered — a 3.5x loss, written in expecting it to argue for
discontinuing. The reading moved from 0.39 to 0.93 in favour of keeping the
programme. Whatever the model does with an explicit cost-benefit figure, it
is not reading it as an argument for the cheaper option.

## What this licenses saying, and what it does not

It does **not** show that the rewrites are leading. That was the hypothesis,
and it required the movement to follow the embedded figures. It did not.

It shows something less convenient: pinning moves stances a great deal and
the direction was not foreseeable, including by the person who chose the
wording. The mechanism is unaccounted for.

## Four topics changed the side they open on

`curbside_plastics` B→A, `journal_page_limits` B→A, `nuclear_power` A→B,
`open_plan_offices` A→B. Same issue, same model, same option semantics —
only the question's level of specificity changed, and four of eleven
reversed.

This bears on how the paper describes its own readout. "The position YOU
currently hold" presupposes a stance attached to the issue. Eleven rewrites
flipped four of them, so on this evidence the stance is a property of the
question's phrasing at least as much as of the topic. That belongs in the
self-report reliability section as a result, not in the limitations as a
worry.

## An unplanned gain

`position`-classified topics went from 2 to 5: grade_caps,
nurse_shift_length, open_plan_offices, test_coverage_80,
orchestra_repertoire. Pinning stripped out the content lean on several
topics and left option order to decide.

Those five are the indifference controls FW#3 needs — a stance held only
because of where the option was printed. Two was too thin to carry the
comparison; five is workable.

## Stopping here

Every rewrite is another arbitrary choice. A second round would be selecting
a stance distribution rather than measuring one. v8's directions are what the
rebuild uses.

## What was not tested

Whether pinning shrinks the space of writable rebuttals. A ladder against "a
200-person software company judged on output per engineer over two years"
cannot reach outside those constraints, which is what makes its rungs
incompatible with the stance — but it also caps how many distinct rungs exist
before they repeat, and a confined ladder may be easier to deflect. Testing
it needs both versions of one topic laddered and run.
