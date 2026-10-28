"""Figures 1 and 2 for the first-person / stance-strength draft (稿2).

Reads the v3 per-turn csvs written by scripts/first_person.py.

Figure 1: opening vs initial pressure, per (topic, order) cell. Three panels
(singular, stance phrases, plural), each with both models. Thin lines join a
cell's opening rate to its mean over pressure turns; dots are the turn means
quoted in the draft's table; the label is up/down/flat over 24 cells and the
Wilcoxon p.

Figure 2: stance-phrase rate vs p_own on opening + pressure turns, one column
per model. Top row pooled, bottom row after removing each cell's mean from
both axes (the within-cell Spearman). Colour is the blind judge's reading.

    python3 scripts/plot_first_person_draft2.py --out figs/6x6
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9893"
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"
CATS = [("fp_sing", "Singular first person\n(I, me, my ...)"),
        ("fp_stance", "First-person stance phrases\n(I would argue, in my view ...)"),
        ("fp_plur", "Plural first person\n(we, us, our ...)")]
JUDGE = [("own", "judge: keeps opening stance", BLUE),
         ("other", "judge: on the other side", ORANGE),
         ("N", "judge: neither", GREY)]


def load(path):
    rows = [r for r in csv.DictReader(open(path))
            if r["phase"] in ("opening", "pressure")]
    for r in rows:
        for k, _ in CATS:
            r[k] = float(r[k])
        r["p_own"] = float(r["p_own"])
        r["cell"] = (r["topic"], r["order"])
    return rows


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def per_cell(rows, k):
    op, pr = {}, defaultdict(list)
    for r in rows:
        if r["phase"] == "opening":
            op[r["cell"]] = r[k]
        else:
            pr[r["cell"]].append(r[k])
    cells = sorted(op)
    return (np.array([op[c] for c in cells]),
            np.array([np.mean(pr[c]) for c in cells]))


def fig1(models, out):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.4), facecolor=SURF)
    for ax, (k, title) in zip(axes, CATS):
        style(ax)
        for j, (name, rows, color) in enumerate(models):
            a, b = per_cell(rows, k)
            x0, x1 = 2 * j, 2 * j + 1
            jit = np.random.default_rng(j).uniform(-0.06, 0.06, len(a))
            for ai, bi, d in zip(a, b, jit):
                ax.plot([x0 + d, x1 + d], [ai, bi], color=color, alpha=0.22,
                        lw=0.8, zorder=1)
            m0 = np.mean([r[k] for r in rows if r["phase"] == "opening"])
            m1 = np.mean([r[k] for r in rows if r["phase"] == "pressure"])
            ax.plot([x0, x1], [m0, m1], color=color, lw=2.2, marker="o",
                    ms=6, zorder=3)
            for xx, mm in ((x0, m0), (x1, m1)):
                ax.annotate(f"{mm:.1f}", (xx, mm), xytext=(0, 7),
                            textcoords="offset points", ha="center",
                            fontsize=8, color=INK, zorder=4)
            d = b - a
            up, dn = int((d > 0).sum()), int((d < 0).sum())
            flat = len(d) - up - dn
            p = stats.wilcoxon(a, b).pvalue if up + dn else float("nan")
            ax.text(x0 + 0.5, -0.13, f"{name}\nup {up} / down {dn} / flat "
                    f"{flat}\np = {p:.3f}".replace("p = 0.", "p = ."),
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=7.8, color=color)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["opening", "pressure"] * 2, fontsize=8)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=6)
    axes[0].set_ylabel("per 1000 words", fontsize=9, color=INK2)
    fig.text(0.01, 0.01,
             "Thin lines: one (topic, order) cell, its opening reply vs the "
             "mean over its initial-pressure replies (24 cells per model). "
             "Thick line and labels: reply means.\nCounts and Wilcoxon "
             "signed-rank p are over cells, uncorrected. Word list v3, quotes "
             "removed.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    path = os.path.join(out, "draft2_fig1_opening_vs_pressure.png")
    fig.savefig(path, dpi=200, facecolor=SURF)
    print(f"[fig] {path}")


def centred(rows, k):
    by = defaultdict(list)
    for r in rows:
        by[r["cell"]].append(r)
    mx = {c: np.mean([r[k] for r in v]) for c, v in by.items()}
    my = {c: np.mean([r["p_own"] for r in v]) for c, v in by.items()}
    return (np.array([r[k] - mx[r["cell"]] for r in rows]),
            np.array([r["p_own"] - my[r["cell"]] for r in rows]))


def fig2(models, out, k="fp_stance"):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.4), facecolor=SURF)
    for j, (name, rows, _) in enumerate(models):
        x = np.array([r[k] for r in rows])
        y = np.array([r["p_own"] for r in rows])
        xc, yc = centred(rows, k)
        jit = np.random.default_rng(j).uniform(-0.25, 0.25, len(rows))
        for i, (ax, xx, yy, lab) in enumerate(
                ((axes[0, j], x + jit, y, "pooled"),
                 (axes[1, j], xc + jit, yc, "within cell"))):
            style(ax)
            for jv, jlab, col in JUDGE:
                m = np.array([r["judge"] == jv for r in rows])
                if m.any():
                    ax.scatter(xx[m], yy[m], s=14, color=col, alpha=0.6,
                               lw=0, label=jlab)
            rho, p = stats.spearmanr(*(
                (x, y) if i == 0 else (xc, yc)))
            ax.text(0.98, 0.97, f"Spearman ρ = {rho:+.2f}\np = {p:.3f}"
                    .replace("0.", "."), transform=ax.transAxes, ha="right",
                    va="top", fontsize=8.5, color=INK, zorder=5,
                    bbox=dict(fc=SURF, ec="none", alpha=0.85, pad=2))
            ax.set_title(f"{name}, {lab}  (n = {len(rows)} replies)",
                         loc="left", fontsize=9.5, color=INK, pad=6)
            if i == 0:
                ax.set_ylabel("p_own", fontsize=9, color=INK2)
                ax.set_ylim(0, 1)
            else:
                ax.set_ylabel("p_own − cell mean", fontsize=9, color=INK2)
                ax.axhline(0, color=INK2, lw=0.6)
                ax.axvline(0, color=INK2, lw=0.6)
            ax.set_xlabel("stance phrases per 1000 words" if i == 0 else
                          "stance phrases per 1000 words − cell mean",
                          fontsize=8.5, color=INK2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.045), markerscale=1.6)
    fig.text(0.01, 0.01,
             "Opening + initial-pressure replies only. Top: raw values. "
             "Bottom: each cell's mean removed from both axes, then Spearman. "
             "x jittered ±0.25 for display only;\np uncorrected and nominal "
             "(replies within a cell are not independent). Colour: blind "
             "judge (claude-sonnet-4-5, T = 0) reading of each reply.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    path = os.path.join(out, "draft2_fig2_stance_vs_pown.png")
    fig.savefig(path, dpi=200, facecolor=SURF)
    print(f"[fig] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", default="figs/qwen_6x6/first_person.csv")
    ap.add_argument("--llama", default="figs/llama_6x6/first_person.csv")
    ap.add_argument("--out", default="figs/6x6")
    args = ap.parse_args()
    models = [("Llama-3.1-8B", load(args.llama), ORANGE),
              ("Qwen2.5-7B", load(args.qwen), BLUE)]
    fig1(models, args.out)
    fig2(models, args.out)


if __name__ == "__main__":
    main()
