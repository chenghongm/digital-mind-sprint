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

**6. A defensive variable nothing reads is worse than none.**
`ab_ids` handled the ` A` spelling, was never wired into the probe, and made
the problem look solved.

**7. A generation cap is a variable until the DV stops moving.**
60 tokens truncated 97.7% of turns; 250 still truncated 75%. Error was not
monotone — 128 was worse than 60. Measure the convergence point, don't pick a
round number.

**8. Preconditions are model-specific — measure, don't assume.**
Ladder direction: one topic pushed toward the side the model already held.
Same topic opens at 0.88 on one model, 0.53 on another.

**9. Pressure only applies if it contradicts the stance as construed.**
Rebuttals about industrial plastics against a stance about household sorting
were absorbed as supporting detail. Pin the scope, and pin whatever parameters
decide the answer — an underspecified question and genuine indifference both
read as p ≈ 0.5.

**10. Whitelist what reaches a prompt.**
Authoring metadata sits beside experimental fields. `str.format` ignoring extra
keys is a coincidence, not a guarantee. `runner.PROMPT_FIELDS`.

---

## Before trusting a new measurement

- [ ] Dumped raw top-k for one real input
- [ ] Every ratio logs its denominator, with an assertion
- [ ] Reproduced a stored value end to end
- [ ] Checked a case whose answer is known independently
- [ ] Named what a null result looks like, and it differs from this one
- [ ] Nothing in the generation loop branches on an unvalidated reading
