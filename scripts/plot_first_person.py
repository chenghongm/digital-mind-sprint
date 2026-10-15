"""Figure for FINDINGS 10: first-person language under pressure.

Reads the per-turn csv written by scripts/first_person.py. Four panels:

  top     per (topic, order) cell, the opening's rate and the mean over its
          pressure turns, for singular first person and for first-person
          stance statements (12 cells; arms share this text, so each cell
          appears once)
  bottom  every pressure turn (84): rate against the probe's p_own, marked by
          the blind judge's reading of the same reply. The within-cell
          Spearman printed on each panel is the one FINDINGS 10 quotes.

    python3 scripts/plot_first_person.py figs/repl_b1/first_person.csv \
        --out figs/repl_b1/first_person.png
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from scipy import stats                                        # noqa: E402

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a9a8a3"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
LABEL = {"fp_sing": "first person singular\n(I, me, my, I'd ...)",
         "fp_stance": "first person + stance\n(\"I would argue\", \"in my view\" ...)"}


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ("fp_sing", "fp_stance", "p_own"):
            r[k] = float(r[k])
        r["cell"] = (r["topic"], r["order"])
    return [r for r in rows if r["phase"] in ("opening", "pressure")]


def within_cell_rho(rows, k):
    x = np.array([r[k] for r in rows])
    y = np.array([r["p_own"] for r in rows])
    for c in {r["cell"] for r in rows}:
        m = np.array([r["cell"] == c for r in rows])
        x[m] -= x[m].mean()
        y[m] -= y[m].mean()
    return stats.spearmanr(x, y)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)


def paired(ax, rows, k):
    by = defaultdict(lambda: {"open": None, "press": []})
    for r in rows:
        if r["phase"] == "opening":
            by[r["cell"]]["open"] = r[k]
        else:
            by[r["cell"]]["press"].append(r[k])
    pairs = [(v["open"], np.mean(v["press"])) for v in by.values()]
    up = sum(b > a for a, b in pairs)
    for a, b in pairs:
        ax.plot([0, 1], [a, b], color=GREY, lw=1.2, marker="o", ms=4.5,
                mfc=SURF, mec=GREY, zorder=2)
    ma, mb = np.median([p[0] for p in pairs]), np.median([p[1] for p in pairs])
    ax.plot([0, 1], [ma, mb], color=BLUE, lw=2.4, marker="o", ms=8,
            mec=SURF, mew=1.5, zorder=3)
    ax.text(1.06, mb, f"median\n{ma:.1f} → {mb:.1f}", va="center",
            fontsize=8, color=INK)
    d = np.array([b - a for a, b in pairs])
    p = stats.wilcoxon(d).pvalue
    ax.set_title(f"{LABEL[k]}\nup in {up}/12 cells, Wilcoxon p = {p:.3f}",
                 fontsize=9, color=INK, loc="left")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["opening", "under pressure\n(mean per cell)"])
    ax.set_xlim(-0.25, 1.45)
    ax.set_ylabel("per 1000 words", fontsize=8.5, color=INK2)
    style(ax)


def scatter(ax, rows, k):
    pres = [r for r in rows if r["phase"] == "pressure"]
    groups = [("own", BLUE, "o", "judge: argues its opening side"),
              ("other", ORANGE, "^", "judge: argues the other side"),
              ("N", GREY, "s", "judge: no side")]
    for key, col, mk, lab in groups:
        g = [r for r in pres if r["judge"] == key]
        if g:
            ax.scatter([r["p_own"] for r in g], [r[k] for r in g], s=26,
                       color=col, marker=mk, edgecolor=SURF, linewidth=0.8,
                       label=f"{lab} (n={len(g)})", zorder=3)
    rho, p = within_cell_rho(rows, k)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_title(f"{LABEL[k].splitlines()[0]}, each pressure turn\n"
                 f"within-cell Spearman with p_own {rho:+.2f} (p = {p:.2f})",
                 fontsize=9, color=INK, loc="left")
    ax.set_xlabel("probe p_own (1 = opening side)", fontsize=8.5,
                  color=INK2)
    ax.set_ylabel("per 1000 words", fontsize=8.5, color=INK2)
    ax.set_xlim(-0.02, 1.02)
    style(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="figs/repl_b1/first_person.csv")
    ap.add_argument("--out", default="figs/repl_b1/first_person.png")
    args = ap.parse_args()
    rows = load(args.csv)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), facecolor=SURF)
    for ax in axes.flat:
        ax.set_facecolor(SURF)
    paired(axes[0, 0], rows, "fp_sing")
    paired(axes[0, 1], rows, "fp_stance")
    scatter(axes[1, 0], rows, "fp_sing")
    scatter(axes[1, 1], rows, "fp_stance")
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Under pressure the model keeps saying \"I\", whichever side "
                 "it argues", fontsize=12, color=INK, x=0.06, ha="left")
    fig.text(0.06, -0.05,
             "Llama-3.1-8B-Instruct, runs/repl_b1. 12 (topic, order) cells; "
             "the five arms share the opening and the three pressure arms "
             "share the pressure phase,\nso each cell counts once. Rates are "
             "per 1000 words of the model's reply, quotations removed. "
             "Judge = blind claude-sonnet-4-5 reading the same reply.\n"
             "Word counts are style, not stance (FINDINGS 10). Source: "
             f"{args.csv} · scripts/plot_first_person.py",
             fontsize=7.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = Path(args.out)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURF)
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
