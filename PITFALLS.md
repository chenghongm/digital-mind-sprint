# Pitfalls

Mistakes already made here. Scan before designing a measurement.
Add a line when one costs more than an hour.

---

**1. Reading logits at chosen token ids asserts a surface form.**
Options shown as `(A) ...` → the model answers `(A`, one token. Reading bare
`A` captured 0.03% of the distribution and renormalization dressed the residue
up as a confident stance. *Dump the top 15 tokens for one real prompt first.*

**2. Log the denominator of every ratio, and assert on it.**
`p = a/(a+b)` is unfalsifiable without `a+b`. `runner.MIN_PROBE_MASS = 0.5`
raises rather than returning a number. Documentation does not prevent
recurrence; a check that fails loudly does.

**3. Consistency is not validity.**
Two readings agreed to within 0.01 across swapped option orders. Both were
noise. Agreement between two measurements of nothing is still nothing.

**4. Plausible output is not validation.**
The broken probe produced trajectories that moved, ordered sensibly across
arms, and matched a blind judge 83.5% of the time. *Ask what a null result
would look like; if it looks like your result, you have not measured anything.*

**5. Never let an unverified measurement drive control flow.**
The probe decided when to stop applying pressure. A bad readout in analysis is
re-readable from stored transcripts; a bad readout in the generation loop is
baked into the data and can only be re-run. 40% of readings crossed the
threshold it branched on.

**6. Compute it and use it, or don't compute it.**
`ab_ids` handled the ` A` spelling and was never wired in. `usable` was
hardcoded True while a summary header still claimed to filter on it.
`consistent` was computed, stored, and read by nothing, so a topic whose two
orders landed on opposite sides of the midline came out "content-driven". A
field that exists but is unread makes the problem look solved.

**7. Every derived field is an assertion — ask where in the data it fails.**
Five instances in one file. `tier` measures firmness only where option order
barely moves the reading. `cold_side` predicts the run's opening side only
where the stance is saturated. `stance_source` used one axis for a question
that has two. None of them was wrong everywhere; each was right in one region
and wrong in another, which is exactly why they survived review.

**8. A float comparison decides the category at the boundary.**
`0.6 - 0.5` is `0.09999999999999998`, so `dev < 0.10` is true for a value that
prints as 0.10. A case invented to sit on a threshold got binned by
representation error rather than by the rule. Live instance: ToF branches on
`p - 0.5 < 0`, so the pressure phase — and therefore the equating rule the
whole protocol rests on — stops on whichever side of that boundary float error
lands. *Flag anything within a margin of a cut instead of trusting the cut.*

**9. A generation cap is a variable until the DV stops moving.**
60 tokens truncated 97.7% of turns; 250 still truncated 75%. Error was not
monotone — 128 was worse than 60. Measure the convergence point, don't pick a
round number.

**10. Preconditions are model-specific — measure, don't assume.**
Ladder direction: one topic pushed toward the side the model already held.
Same topic opens at 0.88 on one model, 0.53 on another. Authored guesses about
where a model has no opinion scored 35% against a 33% baseline, and never once
called a weak topic correctly.

**11. Pressure only applies if it contradicts the stance as construed.**
Rebuttals about industrial plastics against a stance about household sorting
were absorbed as supporting detail. Pin the scope, and pin whatever parameters
decide the answer — an underspecified question and genuine indifference both
read as p ≈ 0.5.

**12. Whitelist what reaches a prompt.**
Authoring metadata sits beside experimental fields. `str.format` ignoring extra
keys is a coincidence, not a guarantee. `runner.PROMPT_FIELDS`.

**13. A forced choice reports a stance even where there is none to report.**
Two options and "answer with one letter" leaves no way to decline, so the
ratio comes back looking like a position whatever the model would have done
with an exit. One wording read 0.86 under a binary probe and put 69% of its
mass on "it depends" when the same question offered it. Two things move
independently: how *specified* the question is sets whether the exit is
taken, and which *criterion* it names sets which side wins. Pinning numbers
is not pinning the criterion — a question can carry hours and square metres
and still not say what counts as better. *Measure the exit rate before
reading a binary ratio as a stance, and average over option positions when
you do.*

**14. A rebuttal that changes the criterion is not pressure.**
#11 says a rebuttal has to contradict the stance as construed. The other
half: it must not quietly replace the standard the stance was held against.
One topic reads 0.93 one way and 0.00 when the question names a different
outcome measure — no counter-evidence needed. A ladder whose rungs drift
onto a second criterion will look like it produced a flip while measuring
topic-switching, not yielding. *Audit each rung against the criterion named
in the question.*

---

## Before trusting a new measurement

- [ ] Dumped raw top-k for one real input
- [ ] Every ratio logs its denominator, with an assertion
- [ ] Every computed field is read by something
- [ ] No threshold comparison sits where float error could pick the outcome
- [ ] Reproduced a stored value end to end
- [ ] Checked a case whose answer is known independently
- [ ] Named what a null result looks like, and it differs from this one
- [ ] Offered the exit, and averaged over where it was printed
- [ ] The question names what counts as better, not just the parameters
- [ ] Nothing in the generation loop branches on an unvalidated reading
