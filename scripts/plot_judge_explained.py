"""Judge-vs-probe figure, with its takeaway on the page.

Two panels, one per measurement. When this figure was made each phase had
been judged on only one passage; since 2026-10-10 opening and pressure turns
also have elicitation verdicts, but the panels keep their original split:

    opening + pressure turns   judged on the REPLY        (n = 309 decided)
    release turns              judged on the ELICITATION  (n = 720)

plot_judge.py refuses to pool them (select_source); this keeps them side by
side instead. Neither panel validates the probe on its own (FINDINGS §7,
"the figures below do not supply that validation"): the elicitation is an
answer to a direct question on a discarded branch, like the probe, and the
reply-judged turns are the ones where the probe and the text part company.

    python3 scripts/plot_judge_explained.py figs/repl_b1/judgements.csv \
        --out figs/repl_b1/judge_validity_explained.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_judge import BINS, load                              # noqa: E402

BLUE, GREY, ORANGE = "#2a78d6", "#c9c8c3", "#eb6834"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"

# The definitions of probe, judge and sign agreement live in the README under
# this figure, not on it (2026-10-16). Only the takeaway stays on the image,
# and its percentages are filled from the data.
TAKEAWAY = (
    "Takeaway: on release turns the judge and the probe mostly agree "
    "({elic:.0%}); on opening and pressure replies much less ({reply:.0%}), "
    "and the paper's 83.5% does not replicate.\nNeither panel validates the "
    "probe: the elicitation answers a direct question on a discarded branch "
    "just as the probe does, so agreement there is close to expected "
    "(FINDINGS §7).\nRecords: one per turn of one conversation. Distinct "
    "contexts: the five arms of a topic share the opening and the pressure "
    "arms share the pressure phase word for word, so a shared turn counts "
    "once."
)


def panel(ax, rows, title):
    ns, own, neu, oth, labels = [], [], [], [], []
    for lo, hi in BINS:
        sub = [r for r in rows if lo <= r["p_own"] < hi]
        n = len(sub)
        labels.append(f"{lo:.1f}–{min(hi, 1.0):.1f}")
        ns.append(n)
        own.append(sum(r["j_own"] == "own" for r in sub) / n if n else 0)
        neu.append(sum(r["j_own"] == "N" for r in sub) / n if n else 0)
        oth.append(sum(r["j_own"] == "other" for r in sub) / n if n else 0)
    own, neu, oth = map(np.array, (own, neu, oth))
    x = np.arange(len(labels))
    kw = dict(width=0.68, edgecolor=SURF, linewidth=2)
    ax.bar(x, own, color=BLUE, label="argues its opening side", **kw)
    ax.bar(x, neu, bottom=own, color=GREY, label="takes no side", **kw)
    ax.bar(x, oth, bottom=own + neu, color=ORANGE,
           label="argues the other side", **kw)
    for xi, n in zip(x, ns):
        ax.text(xi, 1.03, f"n={n}", ha="center", fontsize=10, color=INK2)
    dec = [r for r in rows if r["j_own"] != "N"]
    agree = np.mean([(r["p_own"] >= 0.5) == (r["j_own"] == "own")
                     for r in dec])
    # n counts records (one per conversation turn); arms share opening and
    # pressure text, so the distinct-context count is shown beside it
    # (PITFALLS #18). Release passages differ by arm.
    distinct = len({(r["topic"], r["order"], r["turn_idx"],
                     r["condition"] if r["phase"] == "release" else "")
                    for r in dec})
    ax.set_title(f"{title}\nsign agreement {agree:.1%}  (n = {len(dec)} "
                 f"records, {distinct} distinct contexts)",
                 fontsize=13, color=INK, loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=INK2)
    ax.set_xlabel("probe  P(opening side)", fontsize=11, color=INK2)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_facecolor(SURF)
    return agree, len(dec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="figs/repl_b1/judgements.csv")
    ap.add_argument("--out", default="figs/repl_b1/judge_validity_explained.png")
    args = ap.parse_args()

    rows = load(args.csv)
    reply = [r for r in rows if r.get("text_source") == "reply"]
    # release only: since 2026-10-10 the csv also holds elicitation verdicts
    # for opening and pressure turns, which this panel's title excludes
    elic = [r for r in rows if r.get("text_source") == "elicited"
            and r["phase"] == "release"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True,
                             facecolor=SURF)
    a1 = panel(axes[0], reply, "Opening + pressure turns, judge reads the reply")
    a2 = panel(axes[1], elic, "Release turns, judge reads the elicitation")
    axes[0].set_ylabel("share of turns, blind judge", fontsize=11, color=INK2)
    axes[1].legend(frameon=False, fontsize=10.5, loc="upper center",
                   bbox_to_anchor=(-0.05, -0.16), ncol=3)
    fig.text(0.0, -0.13, TAKEAWAY.format(reply=a1[0], elic=a2[0]),
             fontsize=10.5, color=INK, va="top", linespacing=1.55)
    fig.text(0.99, -0.29, f"Source: {args.csv} · scripts/plot_judge_explained.py",
             ha="right", fontsize=9, color=INK2)
    fig.savefig(args.out, dpi=160, facecolor=SURF, bbox_inches="tight")
    print(f"reply {a1[0]:.1%} n={a1[1]} | elicited {a2[0]:.1%} n={a2[1]}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
