# dog_vs_cat under reframing — two axes, not one

Instrument: `scripts/framing_dogcat.py`, run twice (binary, then `--ternary`).
Model: Llama-3.1-8B-Instruct, bf16, MPS. Probe mass 1.00 throughout, so
none of this is the tokenization fault of `PITFALLS.md` #1.

## Why the topic was picked apart

`dog_vs_cat` came out of the v8 screen as the pool's most settled topic:
0.96 for the dog, 3% of mass on the "it depends" exit. It also has no
literature to cite and no numbers in the question, which made it the natural
place to ask whether a stance that firm is a property of the model or of the
wording.

## Binary: five wordings, same two options

| variant | dog 1st | dog 2nd | P(dog) | mass |
|---|---|---|---|---|
| original | 0.93 | 0.99 | **0.96** | 1.00 |
| general | 0.79 | 0.94 | **0.86** | 1.00 |
| no_lonely | 0.01 | 0.16 | 0.09 | 0.99 |
| unpredict | 0.00 | 0.00 | 0.00 | 1.00 |
| welfare | 0.00 | 0.00 | 0.00 | 1.00 |

`original` names psychological needs; `general` asks which animal is better
for company and drops every parameter; the other three name a practical
constraint. The first reading of this table — that detail flipped the stance
and vagueness restored it — is wrong, because `general` is the least
specific of the five and sits with `original`. The variants sort by which
dimension the question names, not by how much detail it carries.

That also retires the design: `no_lonely` was written as the neutral
baseline, but naming a fifty-hour week and 55 m² is itself a practical
framing. There was no neutral cell.

## Ternary: the same five with an exit

Third option `It depends on the circumstances` (the v6 wording that drew the
most mass, so the exit is offered at its most attractive). All six
permutations of the three options, because C printed last confounds the exit
with this model's slot preference.

| variant | P(C) | C min | C max | P(dog\|AB) |
|---|---|---|---|---|
| original | 0.03 | 0.01 | 0.09 | 0.93 |
| general | **0.69** | 0.43 | 0.88 | 0.87 |
| no_lonely | 0.24 | 0.10 | 0.36 | 0.06 |
| unpredict | 0.01 | 0.00 | 0.02 | 0.00 |
| welfare | 0.01 | 0.00 | 0.02 | 0.00 |

## Two axes that move independently

**Specification sets the exit rate.** `general` puts 69% of its mass on
declining. Its binary 0.86 was the answer to "if you must", printed in the
same column as `original`'s 0.96 with nothing to tell them apart.

**The criterion sets the side.** At low exit rates the reading swings the
full range — 0.93 for the dog when the question asks about psychological
needs, 0.00 when it asks what is easier to keep well. Same two options, same
model, no counter-argument anywhere.

`no_lonely` sits between at 0.24 and shows the axes are not the same thing.
It carries hours and square metres but asks only "which is the better pet",
naming no outcome measure. Pinning parameters is not pinning the criterion.

## Relation to the pinning round

This reproduces `PINNING_SCORECARD.md` inside one topic. There, rewriting
eleven questions cut median P(depends) from 0.327 to 0.113 across the pool;
here one question moves 0.69 to 0.03. Same mechanism at two levels, which is
the strongest evidence so far that the v7→v8 gain was specification and not
the particular figures chosen.

## What it settles for the rebuild

**The v8 `dog_vs_cat` wording is clean.** 0.03 exit, mass 1.00, and both
poles firm — `welfare` and `unpredict` refuse the exit as hard as `original`
does. The worry that the practical side was only answering under duress does
not hold.

**Binary readings are not interpretable alone.** Everything the screen
reports as a stance needs its exit rate beside it. `PITFALLS.md` #13.

**The ladders need a criterion audit.** Rungs 2–4 of the drafted
`dog_vs_cat` ladder argue from the apartment, rehoming rates and outdoor
access — the criterion this table shows is worth 0.93 on its own. A ladder
that drifts onto a second criterion will register a flip without ever
applying pressure. `PITFALLS.md` #14, and the reason `scripts/criterion_sweep.py`
runs before any more rungs are written.

## Caveat

One topic, one model. The criterion sweep is what says whether the swing
generalises; until it reports, this is a demonstration that the failure mode
exists, not a measure of how common it is.
