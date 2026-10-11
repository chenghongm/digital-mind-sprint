"""Judge verdicts by phase, with the passage each bar was judged on.

Replaces plot_judge.py's judge_phases.png for the replication csv. That
figure has to be drawn from ONE text source (select_source), and in
figs/repl_b1/judgements.csv each phase was judged on only one:

    opening, pressure      the normal reply          (n = 60, 252)
    release-phase turns    the branch elicitation    (n = 144 per arm)

so whichever source it was given, two of its three bars came out n = 0.
Here every turn is used once, under the source it was actually judged on,
and the source is printed over each group. Pooling is still refused where
it would double-count: if any turn appears under more than one source the
script stops.

The release phase is split by arm. "after release" pooled all five arms'
continuation turns, including pressure_sustained (whose continuation is
more pressure) and the two neutral arms (never pressured), so it described
none of them.

Each bar is the share of that group's turns in four exclusive classes:
holds its opening side and concedes nothing / holds and concedes a point /
takes no side / argues the other side. Numbers are in FINDINGS 7, "What the
judge found, in the paper's own terms".

    python3 scripts/plot_judge_phases.py figs/repl_b1/judgements.csv \
        --out figs/repl_b1/judge_phases.png
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_judge import load                                    # noqa: E402

BLUE, LIGHT, GREY, ORANGE = "#2a78d6", "#9cc6f2", "#c9c8c3", "#eb6834"
INK, INK2 = "#0b0b0b", "#52514e"

# (label, phase, condition or None for all arms)
GROUPS = [
    ("opening\n(all arms)", "opening", None),
    ("under pressure\n(3 pressure arms)", "pressure", None),
    ("after release,\nsame topic", "release", "pressure_release"),
    ("after release,\ntopic switched", "release", "pressure_switch"),
    ("pressure\ncontinues", "release", "pressure_sustained"),
    ("no pressure,\nsame topic", "release", "neutral"),
    ("no pressure,\ntopic switched", "release", "neutral_switch"),
]

CAPTION = (
    "Blind judge (claude-sonnet-4-5) reads one passage, with no topic, "
    "condition, turn or probe value, and says which side it argues and "
    "whether it grants a specific point to the other side.\n"
    "\"Own side\" is the side the model opened on. Opening and pressure turns "
    "were judged on the model's normal reply; continuation turns on the "
    "elicitation, a two-sentence stance answer\non a discarded branch. The "
    "two groups are different texts: do not compare bar heights across the "
    "divider, and treat \"concedes\" on the right with caution: a\n"
    "two-sentence position statement leaves little room to grant a point. "
    "Llama-3.1-8B-Instruct, runs/repl_b1, 6 topics x 2 orders."
)


def shares(rows):
    n = len(rows)
    own = [r for r in rows if r["j_own"] == "own"]
    conc = sum(r["conc"] for r in own)
    return n, np.array([
        (len(own) - conc) / n,
        conc / n,
        sum(r["j_own"] == "N" for r in rows) / n,
        sum(r["j_own"] == "other" for r in rows) / n,
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="figs/repl_b1/judgements.csv")
    ap.add_argument("--out", default="figs/repl_b1/judge_phases.png")
    args = ap.parse_args()

    rows = load(args.csv)
    dup = [k for k, c in Counter((r["src"], r["conv_id"], r["turn_idx"])
                                 for r in rows).items() if c > 1]
    if dup:
        raise SystemExit(f"{len(dup)} turns are judged under more than one "
                         f"text source (e.g. {dup[0]}); this figure uses each "
                         f"turn once. Filter the csv to one source per turn.")

    data, srcs = [], []
    for label, phase, cond in GROUPS:
        sub = [r for r in rows if r["phase"] == phase
               and (cond is None or r["condition"] == cond)]
        if not sub:
            raise SystemExit(f"no rows for {label!r}")
        s = {r["text_source"] for r in sub}
        if len(s) != 1:
            raise SystemExit(f"{label!r} mixes sources {sorted(s)}")
        srcs.append(s.pop())
        data.append(shares(sub))
        n, v = data[-1]
        print(f"{label.replace(chr(10), ' '):34s} n={n:4d} {srcs[-1]:9s} "
              + "  ".join(f"{x:.2f}" for x in v))

    x = np.arange(len(GROUPS))
    vals = np.array([v for _, v in data])
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    styles = [
        dict(color=BLUE, label="holds its own side, concedes nothing"),
        dict(color=LIGHT, edgecolor=BLUE, hatch="///",
             label="holds its own side AND concedes a point"),
        dict(color=GREY, label="takes no side"),
        dict(color=ORANGE, label="argues the other side"),
    ]
    bottom = np.zeros(len(GROUPS))
    for k, st in enumerate(styles):
        ax.bar(x, vals[:, k], 0.66, bottom=bottom, linewidth=0.6, **st)
        for xi, b, v in zip(x, bottom, vals[:, k]):
            if v >= 0.06:
                ax.text(xi, b + v / 2, f"{v:.0%}", ha="center", va="center",
                        fontsize=8, color=INK if k != 0 else "white")
        bottom += vals[:, k]
    for xi, (n, _) in zip(x, data):
        ax.text(xi, 1.015, f"n={n}", ha="center", fontsize=8, color=INK2)

    # divider between the two text sources
    cut = next(i for i, s in enumerate(srcs) if s != srcs[0]) - 0.5
    ax.axvline(cut, color=INK2, lw=1, ls="--")
    names = {"reply": "judged on the REPLY", "elicited":
             "judged on the ELICITATION (continuation turns, by arm)"}
    ax.text((cut - 0.5) / 2, 1.09, names[srcs[0]], ha="center", fontsize=9,
            color=INK, weight="bold")
    ax.text((cut + len(GROUPS) - 0.5) / 2, 1.09, names[srcs[-1]],
            ha="center", fontsize=9, color=INK, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in GROUPS], fontsize=8.5)
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("share of turns, blind judge")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("What the model's text argues, by phase", fontsize=11,
                 pad=26)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.13))
    fig.text(0.01, -0.06, CAPTION, fontsize=7.2, color=INK2, va="top")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
