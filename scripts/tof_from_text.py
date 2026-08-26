"""What ToF would have been if the flip test had read the TEXT.

ToF stops the pressure phase, so it decides the data (PITFALLS #5). It is
computed from the order-averaged probe. `runs/repl_b1/FINDINGS.md` s7 found
that probe and text disagree on 50% of turns in the `tof = -1` cells and on
6.7% where ToF <= 5 -- monotone in ToF -- which raises the question this
script answers: on the turns that were actually generated, where would the
pressure phase have stopped if `elicited_side` had been the flip test?

    python3 scripts/tof_from_text.py --run runs/repl_b1

No model, no torch. Reads stored transcripts only.

THE COUNTERFACTUAL IS ONE-SIDED, and the script refuses to pretend otherwise.
The pressure phase stopped where the PROBE said so. Turns past that point
were never generated. So:

  text_tof <= probe_tof   is a real observation: the text had already crossed
                          while the probe had not, on turns that exist.
  text_tof  > probe_tof   is unobservable. If the text has not crossed by the
                          time the probe stopped the phase, nothing says what
                          it would have done on rung 6. Reported as `>N`,
                          never as a number.

Same caveat as scripts/tof_robustness.py: a counterfactual over the rungs
actually applied, not over the ladder.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stance_text import UNPARSED                              # noqa: E402


def text_tof(rec, persistent=False):
    """First pressure turn whose elicitation states the opposite side.

    persistent=True instead wants the first turn after which it never comes
    back -- the text oscillates on some cells (A?AAABBB...ABABBAAAA) and the
    probe rule, being instantaneous, has no analogue of that. Both are
    reported because neither is obviously the right one.
    """
    opening = rec["opening_side"]
    pressure = [t for t in rec["turns"] if t["phase"] == "pressure"]
    crossed = [t["turn_idx"] for t in pressure
               if t["elicited_side"] not in (UNPARSED, opening)]
    if not crossed:
        return None
    if not persistent:
        return crossed[0]
    # last run of consecutive crossed turns must reach the end of the phase
    idxs = {t["turn_idx"] for t in pressure}
    last = max(idxs)
    k = last
    while k - 1 in crossed and k - 1 in idxs:
        k -= 1
    return k if last in crossed else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    rows, seen = [], set()
    for f in sorted(Path(args.run).glob("meta/*.json")):
        rec = json.load(open(f))
        if not rec["condition"].startswith("pressure"):
            continue
        # The three pressure arms share an identical pressure phase (same
        # opening, same ladder, deterministic decode); counting them
        # separately would triple every number. One row per (topic, order).
        key = (rec["topic"], str(rec["option_order"]))
        if key in seen:
            continue
        seen.add(key)
        n_press = sum(1 for t in rec["turns"] if t["phase"] == "pressure")
        rows.append(dict(
            topic=rec["topic"], order=str(rec["option_order"]),
            probe_tof=rec["tof"], n_pressure=n_press,
            first=text_tof(rec), persist=text_tof(rec, True),
            dis=sum(1 for t in rec["turns"] if t["agrees"] is False),
            cmp=sum(1 for t in rec["turns"] if t["agrees"] is not None)))

    hdr = (f"{'topic':<22}{'o':>2}{'probe ToF':>11}{'text ToF':>10}"
           f"{'persistent':>12}{'pressure':>10}{'text!=probe':>13}")
    print(hdr); print("-" * len(hdr))
    earlier = later = same = 0
    for r in sorted(rows, key=lambda r: (r["topic"], r["order"])):
        p = r["probe_tof"]
        ft = str(r["first"]) if r["first"] is not None else f">{r['n_pressure']}"
        pe = str(r["persist"]) if r["persist"] is not None else "-"
        if r["first"] is None:
            pass
        elif p < 0 or r["first"] < p:
            earlier += 1
        elif r["first"] == p:
            same += 1
        else:
            later += 1
        rate = f"{r['dis']}/{r['cmp']}"
        print(f"{r['topic']:<22}{r['order']:>2}{p:>11}{ft:>10}{pe:>12}"
              f"{r['n_pressure']:>10}{rate:>13}")

    print(f"\ntext crossed EARLIER than the probe: {earlier} of {len(rows)}")
    print(f"same turn: {same}   later: {later}   "
          f"never within the turns generated: "
          f"{sum(1 for r in rows if r['first'] is None)}")
    print("\n`>N` means the text had not crossed by the time the probe stopped")
    print("the phase at N pressure turns. Turns past that were never")
    print("generated, so no number can be put there.")

    neg = [r for r in rows if r["probe_tof"] < 0]
    if neg:
        print(f"\nThe {len(neg)} cells the probe recorded as tof = -1:")
        for r in neg:
            ft = r["first"] if r["first"] is not None else "never"
            print(f"  {r['topic']:<22}o{r['order']}  text crossed at {ft}, "
                  f"probe never did, {r['dis']}/{r['cmp']} turns disagree")


if __name__ == "__main__":
    main()
