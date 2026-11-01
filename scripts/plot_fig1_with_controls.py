"""Figure 1 with the two control word classes added (2026-10-31).

Top row: the three first-person panels exactly as in
plot_first_person_draft2.fig1. Bottom row: hedge (may, might, could,
perhaps, maybe ...) and articles (a, an, the; negative control), counted
with the lists fixed in scripts/hedge_controls.py, same reply set
(opening + initial pressure, de-duplicated), same per-cell lines, same
Wilcoxon over 24 cells. Articles sit near 100 per 1000 words, so each
panel has its own y scale.

    python3 scripts/plot_fig1_with_controls.py --out figs/6x6
"""
import argparse, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_first_person_draft2 as d2                          # noqa: E402
import first_person as fp                                      # noqa: E402
import hedge_controls as hc                                    # noqa: E402

CTRL = [("hedge", "Hedge words — comparison class\n(may, might, could, perhaps ...)"),
        ("article", "Articles — negative control\n(a, an, the)")]


def add_controls(rows, run):
    cells = fp.load_cells(run)
    txt = {}
    for (topic, order), arms in cells.items():
        for t in arms["pressure_release"]["turns"]:
            txt[(topic, str(order), str(t["turn_idx"]))] = t["model_text"]
    for r in rows:
        h = hc.rates(txt[(r["topic"], r["order"], r["turn_idx"])])
        r["hedge"], r["article"] = h["hedge"], h["article"]
    return rows


def panel(ax, models, k, title):
    d2.style(ax)
    for j, (name, rows, color) in enumerate(models):
        a, b = d2.per_cell(rows, k)
        x0, x1 = 2 * j, 2 * j + 1
        jit = np.random.default_rng(j).uniform(-0.06, 0.06, len(a))
        for ai, bi, dd in zip(a, b, jit):
            ax.plot([x0 + dd, x1 + dd], [ai, bi], color=color, alpha=0.22,
                    lw=0.8, zorder=1)
        m0 = np.mean([r[k] for r in rows if r["phase"] == "opening"])
        m1 = np.mean([r[k] for r in rows if r["phase"] == "pressure"])
        ax.plot([x0, x1], [m0, m1], color=color, lw=2.2, marker="o", ms=6,
                zorder=3)
        for xx, mm in ((x0, m0), (x1, m1)):
            ax.annotate(f"{mm:.1f}", (xx, mm), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=d2.INK, zorder=4)
        d = b - a
        up, dn = int((d > 0).sum()), int((d < 0).sum())
        p = stats.wilcoxon(a, b).pvalue if up + dn else float("nan")
        ax.text(x0 + 0.5, -0.13, f"{name}\nup {up} / down {dn} / flat "
                f"{len(d) - up - dn}\n" + ("p < .001" if p < .001 else f"p = {p:.3f}".replace("p = 0.", "p = .")),
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7.8, color=color)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["opening", "pressure"] * 2, fontsize=8)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(bottom=0)
    ax.set_title(title, loc="left", fontsize=9.5, color=d2.INK, pad=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/6x6")
    args = ap.parse_args()
    models = [("Llama-3.1-8B",
               add_controls(d2.load("figs/llama_6x6/first_person.csv"), "runs/llama_6x6"),
               d2.ORANGE),
              ("Qwen2.5-7B",
               add_controls(d2.load("figs/qwen_6x6/first_person.csv"), "runs/qwen_6x6"),
               d2.BLUE)]
    fig = plt.figure(figsize=(10.5, 8.8), facecolor=d2.SURF)
    gs = fig.add_gridspec(2, 6, hspace=0.75, wspace=0.9)
    top = [fig.add_subplot(gs[0, 2 * i:2 * i + 2]) for i in range(3)]
    bot = [fig.add_subplot(gs[1, 1:3]), fig.add_subplot(gs[1, 3:5])]
    for ax, (k, t) in zip(top, d2.CATS):
        panel(ax, models, k, t)
    for ax, (k, t) in zip(bot, CTRL):
        panel(ax, models, k, t)
    for ax in (top[0], bot[0]):
        ax.set_ylabel("per 1000 words", fontsize=9, color=d2.INK2)
    fig.text(0.01, 0.01,
             "Thin lines: one (topic, order) cell, its opening reply vs the "
             "mean over its initial-pressure replies (24 cells per model). "
             "Thick line and labels: reply means.\nCounts and Wilcoxon "
             "signed-rank p are over cells, uncorrected. First-person lists v3; "
             "hedge and article lists fixed before running (2026-10-30). "
             "Quotes removed.\nEach panel has its own y scale.",
             fontsize=7.5, color=d2.INK2)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.17)
    path = os.path.join(args.out, "draft2_fig1_with_controls.png")
    fig.savefig(path, dpi=200, facecolor=d2.SURF)
    print(f"[fig] {path}")


if __name__ == "__main__":
    main()
