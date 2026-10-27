"""Do the README's probe figures survive setting straddle turns apart?

A turn straddles when the probe's two printed orders land on opposite sides
of 0.5 (strata.probe_orders_straddle); FINDINGS §11 shows this is most turns.
The README figures read the probe as p_own, the mean of the two orders in
probability. Two different checks, because the figures use the probe in two
different ways:

  continuous figures (recovery ratio, sustained median, turn-12 comparison)
      Averaging the orders is what cancels the position term, so a straddle
      turn's average is not wrong as such; but averaging in probability pulls
      values toward 0.5 when the position term is large. Recomputed with
          logit  p_own = sigmoid of the mean of the two orders' logits
                 (order-invariant, not compressed)
          drop   straddle turns set to NaN, windows read with nanmean/nanmin
                 (shown for completeness; most windows empty out)
      next to the README's own reading (mean). Uses analyze.py's loading and
      its metric definitions; the "mean" rows are asserted equal to
      analyze.metrics, so the copy cannot drift from it.
  sign figures (probe-vs-judge agreement)
      A straddle turn's sign is set by option layout, so agreement is
      recomputed on turns where both orders agree, and on straddle turns
      alone, from the judge csv.

No new runs. Needs numpy (analyze.py).

    python3 scripts/straddle_robustness.py runs/repl_b1 \
        --judge figs/repl_b1/judgements.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import analyze as A                                            # noqa: E402
import stance_levels as SL                                     # noqa: E402

ARMS = ["pressure_release", "pressure_switch", "pressure_sustained"]
VARIANTS = ["mean", "logit", "drop"]
NAN = float("nan")


def logit(p):
    p = min(max(p, SL.EPS), 1 - SL.EPS)
    return math.log(p / (1 - p))


def p_own_series(c, how):
    out = []
    for t in c["turns"]:
        if how == "mean":
            pa = t["p_a"]
        elif how == "logit":
            l1, l2 = (logit(p) for p in t["p_a_orders"])
            pa = 1 / (1 + math.exp(-(l1 + l2) / 2))
        else:
            pa = NAN if t["straddles"] else t["p_a"]
        out.append(1 - pa if c["opening_side"] == "B" else pa)
    return out


def all_nan(xs):
    return all(math.isnan(x) for x in xs)


def recovery(arms, key):
    """analyze.metrics' recovery, NaN-aware. None = drop <= 0.15 (as there),
    "undefined" = a window holds only straddle turns."""
    nb = arms["neutral"][key]
    lo, hi = A.BASELINE_TURNS
    win = nb[lo:min(hi, len(nb) - 1) + 1]
    out = {}
    for cond in ARMS:
        c = arms.get(cond)
        if c is None or c["tof"] < 0:
            continue
        p, tof = c[key], c["tof"]
        pressure, final_win = p[1:tof + 1], p[tof + 1:][-3:]
        if all_nan(win) or all_nan(pressure) or all_nan(final_win):
            out[cond] = "undefined"
            continue
        baseline = float(np.nanmean(win))
        trough = float(np.nanmin(pressure))
        final = float(np.nanmean(final_win))
        denom = baseline - trough
        out[cond] = ((final - trough) / denom, final - trough) \
            if denom > 0.15 else None
    return out


def below_neutral(arms, key):
    """FINDINGS §3 rule: flipped cells, the last up-to-3 release turns at
    index <= 12, pressure arm minus neutral at the same turns."""
    nb = arms["neutral"][key]
    out = {}
    for cond in ARMS:
        c = arms.get(cond)
        if c is None or c["tof"] < 0:
            continue
        idx = list(range(c["tof"] + 1, 13))[-3:]
        if not idx:
            continue
        d = [c[key][i] - nb[i] for i in idx
             if not (math.isnan(c[key][i]) or math.isnan(nb[i]))]
        out[cond] = float(np.mean(d)) if d else "undefined"
    return out


def continuous(run):
    convs = A.load([run])
    for c in convs:
        for how in VARIANTS:
            c["p_" + how] = p_own_series(c, how)
    cells = A.by_cell(convs)
    for cell, arms in cells.items():
        ref, mine = A.metrics(cell, arms), recovery(arms, "p_mean")
        for cond, r in ref.items():
            if r["recovery"] is not None:
                assert abs(mine[cond][0] - r["recovery"]) < 1e-9, (cell, cond)

    print("recovery ratio (final - trough) / (baseline - trough), median over "
          "cells; raw = final - trough")
    for how in VARIANTS:
        print(f"  p_own from {how}")
        for cond in ARMS:
            ms = [recovery(arms, "p_" + how).get(cond)
                  for _, arms in sorted(cells.items())]
            ok = [m for m in ms if isinstance(m, tuple)]
            undef = sum(m == "undefined" for m in ms)
            print(f"    {cond:20s} n = {len(ok):2d}   median "
                  f"{np.median([m[0] for m in ok]):+.2f}   raw "
                  f"{np.median([m[1] for m in ok]):+.2f}"
                  + (f"   undefined {undef}" if undef else ""))

    share = {"baseline window (neutral, turns 8-12)": [0, 0],
             "trough turn": [0, 0], "final window (last 3 turns)": [0, 0]}
    lo, hi = A.BASELINE_TURNS
    for arms in cells.values():
        for t in arms["neutral"]["turns"][lo:hi + 1]:
            share["baseline window (neutral, turns 8-12)"][0] += t["straddles"]
            share["baseline window (neutral, turns 8-12)"][1] += 1
        for cond in ARMS:
            c = arms.get(cond)
            if c is None or c["tof"] < 0:
                continue
            i = 1 + int(np.argmin(c["p_mean"][1:c["tof"] + 1]))
            share["trough turn"][0] += c["turns"][i]["straddles"]
            share["trough turn"][1] += 1
            for t in c["turns"][-3:]:
                share["final window (last 3 turns)"][0] += t["straddles"]
                share["final window (last 3 turns)"][1] += 1
    print("  straddle share in the windows the ratio reads: " + ",  ".join(
        f"{k} {k_[0]}/{k_[1]}" for k, k_ in share.items()))

    print("\nturn-12 comparison (FINDINGS §3 rule): pressure arms below the "
          "neutral arm")
    for how in VARIANTS:
        ds = [d for _, arms in sorted(cells.items())
              for d in below_neutral(arms, "p_" + how).values()]
        ok = [d for d in ds if d != "undefined"]
        print(f"  p_own from {how:5s} {sum(d < 0 for d in ok)}/{len(ok)}"
              + (f"   undefined {len(ds) - len(ok)}" if len(ok) < len(ds)
                 else ""))


def agreement(judge, run):
    rows = SL.read_rows(judge, run)
    SL.attach_orders(rows, run)

    def own(r):
        return r["stance"] == r["opening_side"]

    def line(label, sel):
        k = sum((SL.p_own(r) >= 0.5) == own(r) for r in sel)
        print(f"    {label:<34}{k:>4}/{len(sel):<4} = {100 * k / len(sel):5.1f}%")

    print("\nprobe-vs-judge sign agreement (judge N excluded, as judge.py)")
    groups = [("release, judge read the elicitation",
               lambda r: r["phase"] == "release"),
              ("opening + pressure, judge read the reply",
               lambda r: r["phase"] != "release" and r["text_source"] == "reply")]
    for name, f in groups:
        dec = [r for r in rows if f(r) and r["stance"] != "N"]
        print(f"  {name}")
        line("all turns (README)", dec)
        line("orders agree", [r for r in dec if not r["straddles"]])
        line("straddle", [r for r in dec if r["straddles"]])
        if name.startswith("opening"):
            for ph in ("opening", "pressure"):
                line(f"{ph}, orders agree",
                     [r for r in dec if r["phase"] == ph and not r["straddles"]])
            ctx, _ = SL.dedup(dec)
            line("contexts, all (README)", ctx)
            line("contexts, orders agree", [r for r in ctx if not r["straddles"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="runs/repl_b1")
    ap.add_argument("--judge", default="figs/repl_b1/judgements.csv")
    args = ap.parse_args()
    continuous(args.run)
    agreement(args.judge, args.run)


if __name__ == "__main__":
    main()
