"""Figure for FINDINGS_6x6 E: first-person rates by phase and arm, two models.

Reads the per-turn csvs written by scripts/first_person.py. Two panels,
singular (I, me, my ...) and plural (we, us, our ...) first person, on one
y scale. Seven groups: the shared opening and pressure phases, then the
release phase of each of the five arms (three pressure arms, two neutral
arms). Bars are the turn means quoted in E; whiskers are 95% bootstrap
intervals that resample (topic, order) cells, the independent unit.

    python3 scripts/plot_first_person_6x6.py \
        --qwen figs/qwen_6x6/first_person.csv \
        --llama figs/llama_6x6/first_person.csv \
        --out figs/6x6/first_person_arms.png
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"
GROUPS = [("opening", None, "opening"),
          ("pressure", None, "pressure"),
          ("release", "pressure_sustained", "pressure_\nsustained"),
          ("release", "pressure_release", "pressure_\nrelease"),
          ("release", "pressure_switch", "pressure_\nswitch"),
          ("release", "neutral", "neutral"),
          ("release", "neutral_switch", "neutral_\nswitch")]
CATS = [("fp_sing", "First person singular  (I, me, my ...)"),
        ("fp_plur", "First person plural  (we, us, our ...)")]
N_BOOT = 2000


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, _ in CATS:
            r[k] = float(r[k])
        r["cell"] = (r["topic"], r["order"])
    return rows


def select(rows, phase, arm):
    return [r for r in rows if r["phase"] == phase
            and (arm is None or r["arm"] == arm)]


def mean_ci(rows, k, rng):
    """Turn mean, and a 95% interval from resampling cells."""
    by = defaultdict(list)
    for r in rows:
        by[r["cell"]].append(r[k])
    cells = list(by.values())
    boot = []
    for _ in range(N_BOOT):
        pick = rng.integers(len(cells), size=len(cells))
        vals = np.concatenate([cells[i] for i in pick])
        boot.append(vals.mean())
    m = np.mean([r[k] for r in rows])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return m, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", default="figs/qwen_6x6/first_person.csv")
    ap.add_argument("--llama", default="figs/llama_6x6/first_person.csv")
    ap.add_argument("--out", default="figs/6x6/first_person_arms.png")
    args = ap.parse_args()

    models = [("Qwen2.5-7B", load(args.qwen), BLUE),
              ("Llama-3.1-8B", load(args.llama), ORANGE)]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharey=True,
                             facecolor=SURF)
    x = np.arange(len(GROUPS))
    w = 0.36
    for ax, (k, title) in zip(axes, CATS):
        ax.set_facecolor(SURF)
        for j, (name, rows, color) in enumerate(models):
            stats = [mean_ci(select(rows, p, a), k, rng) for p, a, _ in GROUPS]
            m = np.array([s[0] for s in stats])
            err = np.array([[s[0] - s[1] for s in stats],
                            [s[2] - s[0] for s in stats]])
            xs = x + (j - 0.5) * (w + 0.02)
            ax.bar(xs, m, w, color=color, edgecolor=SURF, lw=1, label=name,
                   zorder=2)
            ax.errorbar(xs, m, yerr=err, fmt="none", ecolor=INK2, lw=1,
                        capsize=2.5, zorder=3)
        ax.axvline(1.5, color=INK2, lw=0.8, ls=(0, (3, 3)))
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=6)
        ax.set_ylabel("per 1000 words", fontsize=9, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([g[2] for g in GROUPS], fontsize=8.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    top = axes[0]
    ymax = top.get_ylim()[1]
    top.text(1.55, ymax * 0.97, "release phase, by arm  →", fontsize=8.5,
             color=INK2, va="top")
    top.legend(frameon=False, fontsize=9, loc="upper right")
    fig.text(0.01, 0.005,
             "Opening and pressure text is shared by all arms of a cell. "
             "Release questions are informational except in pressure_sustained,"
             " which keeps pushing back.\nBars: turn means (all at most 0.4 in"
             " the four informational arms); whiskers: 95% bootstrap over the "
             "24 (topic, order) cells per model.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(args.out, dpi=200, facecolor=SURF)
    print(f"[fig] {args.out}")


if __name__ == "__main__":
    main()
