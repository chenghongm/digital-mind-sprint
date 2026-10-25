"""Compute-unit watch for the plan in notes/plan_meta_6x6.md. No model.

The CU balance is visible only on the Colab panel, so this cannot read it.
What it can read is the run itself: runner schema 8 stores per-turn `secs`,
and autopush puts the meta files on the remote every few minutes. From those
it estimates the CU spent so far, the CU per conversation, and what the rest
of the plan will cost at that rate, against the last balance read off the
panel.

    python3 scripts/cu_watch.py --balance 117 --branch colab-meta
    python3 scripts/cu_watch.py --balance 117 --local         # runs/ on disk

`--balance` is the panel reading taken when nothing in PLAN had run yet, or
the latest reading together with `--spent-before`, the CU the watched runs
had already used when it was taken. Estimates count generation time only;
`--overhead` (default 1.15) covers model loading and idle runtime, and is a
guess until a panel reading is compared with this script's figure.

Loop it in a terminal tab while a batch runs:

    while true; do python3 scripts/cu_watch.py --balance 117 --branch colab-meta; sleep 900; done
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# run directory -> planned conversations (notes/plan_meta_6x6.md, Design)
PLAN = {
    "runs/llama_6x6": 120,
    "runs/qwen_6x6": 120,
}
CU_PER_HOUR = 5.3          # measured, A100 (notes/compute_estimate_en.md)
FALLBACK_CU_PER_CONV = 0.70  # upper per-conversation figure, used until a
                             # run has its own timing
WARN_MARGIN = 20           # warn when the projected balance falls below this


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                          text=True)


def metas_remote(branch, run):
    r = git("ls-tree", "-r", "--name-only", f"origin/{branch}", "--",
            f"{run}/meta")
    out = []
    for name in r.stdout.split():
        if name.endswith(".json"):
            s = git("show", f"origin/{branch}:{name}")
            if s.returncode == 0:
                out.append(json.loads(s.stdout))
    return out


def metas_local(run):
    return [json.load(open(f)) for f in sorted((ROOT / run).glob("meta/*.json"))]


def secs_of(rec):
    """Generation seconds for one conversation, or None without timing."""
    t = [x.get("secs") for x in rec.get("turns", [])]
    if t and all(isinstance(v, (int, float)) and v > 0 for v in t):
        return sum(t)
    w = rec.get("wall_secs")
    return w if isinstance(w, (int, float)) and w > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balance", type=float, required=True,
                    help="CU on the Colab panel")
    ap.add_argument("--spent-before", type=float, default=0.0,
                    help="CU the watched runs had used when --balance was read")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--branch", help="read runs from origin/<branch>")
    src.add_argument("--local", action="store_true", help="read runs/ on disk")
    ap.add_argument("--overhead", type=float, default=1.15)
    args = ap.parse_args()

    if args.branch:
        f = git("fetch", "-q", "origin", args.branch)
        if f.returncode != 0:
            print(f"[note] origin/{args.branch} not fetched: "
                  f"{f.stderr.strip() or 'no such branch yet'}")

    spent, rate_secs, rate_n = 0.0, 0.0, 0
    remaining_convs = 0
    print(f"{'run':18} {'done':>9} {'timed':>6} {'gen h':>6} {'CU est':>7}")
    for run, planned in PLAN.items():
        recs = (metas_remote(args.branch, run) if args.branch
                else metas_local(run))
        timed = [s for s in (secs_of(r) for r in recs) if s is not None]
        h = sum(timed) / 3600
        cu = h * CU_PER_HOUR * args.overhead
        spent += cu
        rate_secs += sum(timed)
        rate_n += len(timed)
        remaining_convs += max(planned - len(recs), 0)
        print(f"{run:18} {len(recs):>4}/{planned:<4} {len(timed):>6} "
              f"{h:>6.2f} {cu:>7.1f}")

    if rate_n:
        per_conv = rate_secs / rate_n / 3600 * CU_PER_HOUR * args.overhead
        basis = f"measured on {rate_n} conversations"
    else:
        per_conv = FALLBACK_CU_PER_CONV
        basis = "no timed conversation yet; planning figure"
    need = remaining_convs * per_conv
    left = args.balance - (spent - args.spent_before)
    after = left - need
    print(f"\nCU per conversation  {per_conv:.2f}  ({basis})")
    print(f"spent, estimated     {spent:.1f}")
    print(f"balance now, est.    {left:.1f}   (panel {args.balance:g}, "
          f"minus {spent - args.spent_before:.1f} spent since)")
    print(f"remaining plan       {remaining_convs} conversations, "
          f"~{need:.1f} CU")
    print(f"balance after plan   {after:.1f}")
    if after < WARN_MARGIN:
        print(f"\n*** WARNING: projected balance {after:.1f} CU is below "
              f"{WARN_MARGIN}. Buy compute before the next batch "
              f"(plan: decided on the balance alone). ***")
    print("\nThe panel is the ground truth: compare its reading with "
          "'balance now' and adjust --overhead.")


if __name__ == "__main__":
    sys.exit(main())
