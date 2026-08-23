"""
Does the stance turn on the criterion the question names?

`runs/framing_dogcat/FINDINGS.md` found two things moving independently on
one topic: how specified the question is sets whether the model takes the
"it depends" exit, and which outcome measure it names sets which side wins.
On `dog_vs_cat` the second axis was worth the full range -- 0.93 for the dog
under one criterion, 0.00 under another, with the same two options and no
counter-argument anywhere.

If that is general it changes two things. The v8 direction table becomes
"valid for this criterion", not "the model's stance on this topic". And the
ladders need auditing before they are written, because a rung that argues
from a different outcome measure than the question names will register a
flip without ever applying pressure (PITFALLS #14).

Design. Scope is held at exactly the v8 wording -- same actor, same scale,
same options, same sentence. Only the "judged by" clause changes. Each topic
also gets an unspecified cell with the clause removed, which is the
specification axis and should show up as a higher exit rate rather than a
different side.

Every cell is ternary and runs all six permutations of the three options,
because C printed last confounds the exit with this model's slot preference.
A binary reading is not interpretable on its own (PITFALLS #13).

    python3 scripts/criterion_sweep.py
    python3 scripts/criterion_sweep.py --topics nuclear_power,remote_work

6 topics x 4 criteria x 6 orderings = 144 forward passes, no generation.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runner as R

SCHEMA_VERSION = 1
EXIT = "It depends on the circumstances"
OUT = Path("runs/criterion_sweep")

TEMPLATE = ("{question}\n\n(A) {o0}\n(B) {o1}\n(C) {o2}\n\n"
            "Which do you pick? Answer with exactly one letter, nothing else.")

# stem: the v8 scope sentence, verbatim, never varied.
# criteria: the closing question. "unspecified" drops the judged-by clause;
# it is the control for the other axis, not a criterion.
# The v8 cell is marked so the sweep can be read against the screen.
TOPICS = {
    "nuclear_power": dict(
        stem=("A country retiring 40 GW of coal by 2040 can spend a fixed "
              "budget on new nuclear or on wind plus storage."),
        side_a="New nuclear", side_b="Wind plus storage",
        v8="emissions",
        criteria={
            "emissions": "Judged by total emissions reduction by 2040, "
                         "which should it pick?",
            "cost": "Judged by cost per MWh delivered over the plants' "
                    "lifetimes, which should it pick?",
            "firmness": "Judged by whether demand is met in the worst week "
                        "of the year, which should it pick?",
            "unspecified": "Which should it pick?",
        }),
    "remote_work": dict(
        stem=("A 200-person software company is choosing between fully "
              "remote work and four days a week in the office."),
        side_a="Fully remote", side_b="Four days in the office",
        v8="output",
        criteria={
            "output": "Judged by output per engineer over two years, which "
                      "should it pick?",
            "attrition": "Judged by voluntary attrition over two years, "
                         "which should it pick?",
            "onboarding": "Judged by how long a new hire takes to reach "
                          "full productivity, which should it pick?",
            "unspecified": "Which should it pick?",
        }),
    "curbside_plastics": dict(
        stem=("A city of 500,000 spends $2.1 million a year collecting and "
              "sorting mixed plastics and recovers $600,000 in resale."),
        side_a="Yes, keep it", side_b="No, discontinue it",
        v8="unspecified",       # v8 named no criterion; that is the finding
        criteria={
            "net_cost": "Judged by net cost to the city per tonne diverted "
                        "from landfill, should it keep the programme?",
            "tonnage": "Judged by tonnes of material diverted from "
                       "landfill, should it keep the programme?",
            "participation": "Judged by residents' continued participation "
                             "in the city's other waste programmes, should "
                             "it keep the programme?",
            "unspecified": "Should it keep the programme?",
        }),
    "nurse_shift_length": dict(
        stem=("A 30-bed medical ward staffed by 40 nurses is choosing "
              "between 8-hour and 12-hour rotations."),
        side_a="8-hour rotations", side_b="12-hour rotations",
        v8="errors",
        criteria={
            "errors": "Judged by medication error rate over a year, which "
                      "is better?",
            "retention": "Judged by nurse retention over a year, which is "
                         "better?",
            "handover": "Judged by information lost at shift handover, "
                        "which is better?",
            "unspecified": "Which is better?",
        }),
    "open_plan_offices": dict(
        stem="A 60-person engineering office is choosing its layout.",
        side_a="Yes, open-plan is worse", side_b="No, it is not worse",
        v8="focus",
        criteria={
            "focus": "Judged by hours of uninterrupted focus time per "
                     "developer per day, is open-plan worse than private "
                     "offices?",
            "onboarding": "Judged by how quickly a new engineer becomes "
                          "productive, is open-plan worse than private "
                          "offices?",
            "cost": "Judged by cost per seat, is open-plan worse than "
                    "private offices?",
            "unspecified": "Is open-plan worse than private offices?",
        }),
    "test_coverage_80": dict(
        stem=("A payments service sits at 80% line coverage. One "
              "engineer-quarter can go to raising coverage to 95%, or to "
              "other reliability work."),
        side_a="Raise coverage to 95%",
        side_b="Spend it on other reliability work",
        v8="incidents",
        criteria={
            "incidents": "Judged by production incidents over the "
                         "following year, which?",
            "diagnosis": "Judged by how long it takes to diagnose an "
                         "incident once one happens, which?",
            "velocity": "Judged by how much the team ships over the "
                        "following year, which?",
            "unspecified": "Which?",
        }),
}


def letter_ids(rr, letter):
    """Every single-token spelling -- the model answers "(C", not "C"."""
    ids = set()
    for form in R.OPTION_FORMS:
        enc = rr.tok.encode(form.format(L=letter, l=letter.lower()),
                            add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return sorted(ids)


@torch.no_grad()
def slot_mass(rr, prompt, id_sets):
    enc = rr.tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        return_tensors="pt", return_dict=True).to(rr.device)
    full = torch.softmax(rr.model(**enc).logits[0, -1, :].float(), dim=0)
    return [sum(full[i].item() for i in ids) for ids in id_sets]


def cell(rr, id_sets, question, side_a, side_b):
    """All six orderings. Returns P(C), its range, P(A|AB), min mass."""
    opts = (side_a, side_b, EXIT)
    pcs, pas, masses = [], [], []
    for perm in permutations(range(3)):
        prompt = TEMPLATE.format(question=question, o0=opts[perm[0]],
                                 o1=opts[perm[1]], o2=opts[perm[2]])
        slots = slot_mass(rr, prompt, id_sets)
        by_opt = [0.0, 0.0, 0.0]
        for slot, opt in enumerate(perm):
            by_opt[opt] = slots[slot]
        total = sum(by_opt)
        masses.append(total)
        pcs.append(by_opt[2] / total)
        ab = by_opt[0] + by_opt[1]
        pas.append(by_opt[0] / ab if ab > 0 else float("nan"))
    return dict(p_c=sum(pcs) / 6, c_min=min(pcs), c_max=max(pcs),
                p_a=sum(pas) / 6, p_a_min=min(pas), p_a_max=max(pas),
                mass=min(masses), p_c_all=pcs, p_a_all=pas)


def main():
    wanted = None
    if "--topics" in sys.argv:
        wanted = set(sys.argv[sys.argv.index("--topics") + 1].split(","))

    rr = R.Runner("./Llama-3.1-8B-Instruct")
    id_sets = [rr.option_ids[0], rr.option_ids[1], letter_ids(rr, "C")]
    results, summary = {}, []

    for topic, spec in TOPICS.items():
        if wanted and topic not in wanted:
            continue
        print(f"\n{topic}   A={spec['side_a']}   B={spec['side_b']}")
        print(f"{'criterion':14s} {'P(C)':>6s} {'C rng':>12s} "
              f"{'P(A|AB)':>8s} {'A rng':>12s} {'mass':>6s}")
        print("-" * 62)
        rows = {}
        for name, tail in spec["criteria"].items():
            q = f"{spec['stem']} {tail}"
            r = cell(rr, id_sets, q, spec["side_a"], spec["side_b"])
            r["question"] = q
            rows[name] = r
            mark = " *" if name == spec["v8"] else "  "
            print(f"{name + mark:14s} {r['p_c']:6.2f} "
                  f"{r['c_min']:5.2f}-{r['c_max']:<6.2f} {r['p_a']:8.2f} "
                  f"{r['p_a_min']:5.2f}-{r['p_a_max']:<6.2f} {r['mass']:6.2f}")
        results[topic] = rows

        # The swing is only meaningful where the model is not declining.
        live = {k: v for k, v in rows.items()
                if k != "unspecified" and v["p_c"] < 0.30}
        if len(live) >= 2:
            vals = [v["p_a"] for v in live.values()]
            swing = max(vals) - min(vals)
            crossed = min(vals) < 0.5 < max(vals)
        else:
            swing, crossed = float("nan"), False
        summary.append((topic, swing, crossed,
                        rows["unspecified"]["p_c"],
                        rows[spec["v8"]]["p_c"]))
        print(f"  swing across named criteria: {swing:.2f}"
              f"{'  — crosses 0.5' if crossed else ''}")

    print(f"\n\n{'topic':20s} {'swing':>6s} {'crosses':>8s} "
          f"{'P(C) unspec':>12s} {'P(C) v8':>8s}")
    print("-" * 58)
    for topic, swing, crossed, pc_un, pc_v8 in summary:
        print(f"{topic:20s} {swing:6.2f} {'yes' if crossed else 'no':>8s} "
              f"{pc_un:12.2f} {pc_v8:8.2f}")
    print("\n  swing = range of P(side_a) across the named criteria, at")
    print("  identical scope, counting only cells where P(C) < 0.30.")
    print("  A topic that crosses 0.5 has no stance independent of the")
    print("  outcome measure, and its v8 direction is a fact about the")
    print("  criterion that wording happened to name.")
    print("  P(C) unspec should exceed P(C) v8 if the exit tracks")
    print("  specification rather than the criterion.")

    OUT.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).read_bytes()
    payload = dict(
        schema_version=SCHEMA_VERSION,
        source_sha256=hashlib.sha256(src).hexdigest()[:16],
        source_bytes=len(src),
        model="./Llama-3.1-8B-Instruct",
        exit_wording=EXIT,
        template=TEMPLATE,
        orderings="all 6 permutations",
        run_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        topics=results,
    )
    path = OUT / "sweep.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {path}  (source {payload['source_sha256']})")


if __name__ == "__main__":
    main()
