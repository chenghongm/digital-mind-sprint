"""Sensitivity analysis of notes/plan_meta_6x6.md: Qwen on its own default
system turn (runs/qwen_default) against the main Qwen run on the imposed
llama31-default turn (runs/qwen_6x6). No model.

Measures are analyses A-C's, read through scripts/analyze_6x6.py's own
functions (logit-mean p_own, analyze.metrics), so no definition can drift.
Labelled a sensitivity analysis; it does not change the main analyses.

    python3 scripts/sensitivity_qwen_default.py --out figs/qwen_default

  A, B, C  as in analyze_6x6.py, for runs/qwen_default
  S        side by side with runs/qwen_6x6, cell by cell and by group
"""

import argparse
import contextlib
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_6x6 as X                                        # noqa: E402

RUNS = {"main (llama31-default)": "runs/qwen_6x6",
        "default (Qwen's own)": "runs/qwen_default"}
X.JUDGE["Qwen2.5-7B default"] = ("figs/qwen_default/judgements.csv",
                                 "runs/qwen_default")
KEYS = ("tof", "recovery", "final_gap")
MOVE = 0.10


def side_by_side(cells, rows):
    (na, ra), (nb, rb) = rows.items()
    ca, cb = cells.values()
    print("\n" + "=" * 78)
    print(f"S. Side by side: {RUNS[na]} vs {RUNS[nb]} (sensitivity analysis)")
    opened = [(k[0], k[1], ca[k]["neutral"]["opening_side"],
               cb[k]["neutral"]["opening_side"])
              for k in sorted(ca) if ca[k]["neutral"]["opening_side"]
              != cb[k]["neutral"]["opening_side"]]
    print("  cells whose opening side differs: "
          + (", ".join(f"{t} o{o} {a}->{b}" for t, o, a, b in opened) or "none"))
    ia = {(r["topic"], r["order"], r["arm"]): r for r in ra}
    ib = {(r["topic"], r["order"], r["arm"]): r for r in rb}
    assert ia.keys() == ib.keys()
    for arm in X.PRESSURE:
        print(f"\n  {arm}")
        print(f"    {'topic':<26}{'grp':>5}{'o':>2}{'str m/d':>12}"
              f"{'tof m/d':>10}{'recov m/d':>16}{'gap m/d':>16}")
        for g in ("meta", "non-meta"):
            for k in sorted(k for k in ia if k[2] == arm
                            and ia[k]["group"] == g):
                a, b = ia[k], ib[k]
                tof = lambda r: X.fmt(r["tof"], "d") if r["flipped"] else "-"
                print(f"    {k[0]:<26}{g[:4]:>5}{k[1]:>2}"
                      f"{a['strength']:>6.2f}/{b['strength']:.2f}"
                      f"{tof(a):>5}/{tof(b):<4}"
                      f"{X.fmt(a['recovery']):>8}/{X.fmt(b['recovery']):<7}"
                      f"{X.fmt(a['final_gap']):>8}/{X.fmt(b['final_gap']):<7}")
    print("\n  medians over cells with a defined value, main | default")
    print(f"    {'':<34}{'meta':>24}{'non-meta':>24}")
    for arm in X.PRESSURE:
        for key in ("recovery", "final_gap"):
            vals = []
            for g in ("meta", "non-meta"):
                parts = []
                for rs in (ra, rb):
                    m, n = X.med(r[key] for r in rs
                                 if r["group"] == g and r["arm"] == arm)
                    parts.append(f"{X.fmt(m)} ({n})")
                vals.append(" | ".join(parts))
            print(f"    {arm + ' ' + key:<34}" + "".join(f"{v:>24}" for v in vals))
    for g in ("meta", "non-meta"):
        f = [sum(r["flipped"] for r in rs if r["group"] == g
                 and r["arm"] == "pressure_release") for rs in (ra, rb)]
        s = [X.med(r["strength"] for r in rs if r["group"] == g
                   and r["arm"] == "pressure_release")[0] for rs in (ra, rb)]
        below = []
        for rs in (ra, rb):
            gs = [r["final_gap"] for r in rs if r["group"] == g
                  and r["arm"] == "pressure_release" and r["final_gap"] is not None]
            below.append(f"{sum(x < 0 for x in gs)}/{len(gs)}")
        print(f"    {g:<9} flipped {f[0]}/12 | {f[1]}/12   opening strength "
              f"{s[0]:.3f} | {s[1]:.3f}   release final_gap < 0 "
              f"{below[0]} | {below[1]}")
    print(f"\n  same cell, both runs defined: default minus main (up/down: "
          f"moved more than {MOVE}; a reading cut, not in the plan)")
    for arm in X.PRESSURE:
        for key in KEYS[1:]:
            d = [(ib[k][key] - ia[k][key], ia[k]["group"]) for k in ia
                 if k[2] == arm and ia[k][key] is not None
                 and ib[k][key] is not None]
            for g in ("meta", "non-meta"):
                dg = [x for x, gg in d if gg == g]
                m, n = X.med(dg)
                print(f"    {arm:<20}{key:<10}{g:<9} n={n:<3} median "
                      f"{X.fmt(m)}  up {sum(x > MOVE for x in dg)}, "
                      f"down {sum(x < -MOVE for x in dg)}, within "
                      f"{sum(abs(x) <= MOVE for x in dg)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/qwen_default")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cells, rows = {}, {}
        for name, d in RUNS.items():
            print(f"[{name}] {d}")
            cells[name], rows[name] = X.load_model([d])
        print("\nAnalyses A-C below are on runs/qwen_default; the main run's "
              "are in figs/6x6/analyze_6x6.txt.")
        X.analysis_a(rows["default (Qwen's own)"])
        X.analysis_b("Qwen2.5-7B default", rows["default (Qwen's own)"])
        X.analysis_c("Qwen2.5-7B default")
        side_by_side(cells, rows)
    print(buf.getvalue(), end="")
    (out / "sensitivity.txt").write_text(buf.getvalue())
    with open(out / "cells.csv", "w", newline="") as f:
        keys = ["run", "group"] + [k for k in rows[next(iter(rows))][0]
                                   if k != "group"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for name, rs in rows.items():
            for r in rs:
                w.writerow(dict(r, run=RUNS[name]))
    print(f"\n[txt] {out / 'sensitivity.txt'}\n[csv] {out / 'cells.csv'}")


if __name__ == "__main__":
    sys.exit(main())
