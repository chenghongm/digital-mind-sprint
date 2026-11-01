"""Figure 2 (redo, 2026-10-31): correlation summary.

One dot per (word class, target, method, model): Spearman rho between a
word-class rate and a probe target, on opening + initial-pressure replies
(Llama n = 148, Qwen n = 180), with a 95% interval from resampling
(topic, order) cells (2000 draws; a cell drawn twice counts as two cells).

word classes: singular first person, first-person stance phrases (v3),
              hedge words, articles (negative control)
targets:      p_own (held the opening side?), |p_own - 0.5| (how far from
              50/50, either side)
methods:      pooled; within-cell (both axes minus cell mean);
              within-cell + turn (ranks residualised on within-cell turn,
              then Pearson -- same as hedge_controls.py)

Plural first person (fp_plur) is computed and written to the csv but not
drawn: it appears in only 15% (Llama) / 8% (Qwen) of these replies, so most
values are 0, rank correlations are unstable, and some bootstrap draws are
all-zero (rho undefined; those draws are skipped). Added 2026-10-31.

    python3 scripts/plot_fig2_rho_summary.py --out figs/6x6
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_first_person_draft2 as d2                          # noqa: E402
from plot_fig1_with_controls import add_controls               # noqa: E402

CLASSES = [("fp_sing", "Singular first person"),
           ("fp_stance", "First-person stance phrases"),
           ("hedge", "Hedge words"),
           ("article", "Articles (negative control)")]
EXTRA = [("fp_plur", "Plural first person (not drawn)")]
TARGETS = [("p_own", "vs p_own\n(held the opening side?)"),
           ("dist", "vs |p_own − 0.5|\n(how far from 50/50, either side)")]
METHODS = ["pooled", "within cell", "within cell + turn"]
N_BOOT = 2000


def demean(v, cid):
    v = v.astype(float).copy()
    for c in np.unique(cid):
        m = cid == c
        v[m] -= v[m].mean()
    return v


def resid(a, b):
    a = stats.rankdata(a); b = stats.rankdata(b)
    a = a - a.mean(); b = b - b.mean()
    return a - (a @ b) / (b @ b) * b if b @ b else a


def rho(x, y, cid, turn, method):
    if method == "pooled":
        return stats.spearmanr(x, y)[0]
    xc, yc = demean(x, cid), demean(y, cid)
    if method == "within cell":
        return stats.spearmanr(xc, yc)[0]
    tc = demean(turn, cid)
    return stats.pearsonr(resid(xc, tc), resid(yc, tc))[0]


def arrays(rows, k, target):
    x = np.array([r[k] for r in rows])
    p = np.array([r["p_own"] for r in rows])
    y = p if target == "p_own" else np.abs(p - 0.5)
    return x, y


def estimate(rows, k, target, method, rng):
    cells = sorted({r["cell"] for r in rows})
    idx = {c: [i for i, r in enumerate(rows) if r["cell"] == c] for c in cells}
    x, y = arrays(rows, k, target)
    cid = np.array([cells.index(r["cell"]) for r in rows])
    turn = np.array([float(r["turn_idx"]) for r in rows])
    est = rho(x, y, cid, turn, method)
    boot = []
    for _ in range(N_BOOT):
        pick = rng.integers(len(cells), size=len(cells))
        ii = np.concatenate([idx[cells[j]] for j in pick])
        newc = np.concatenate([[n] * len(idx[cells[j]]) for n, j in enumerate(pick)])
        b = rho(x[ii], y[ii], newc, turn[ii], method)
        if np.isfinite(b):
            boot.append(b)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return est, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/6x6")
    args = ap.parse_args()
    models = [("Llama-3.1-8B",
               [r for r in add_controls(d2.load("figs/llama_6x6/first_person.csv"),
                                        "runs/llama_6x6")], d2.ORANGE),
              ("Qwen2.5-7B",
               [r for r in add_controls(d2.load("figs/qwen_6x6/first_person.csv"),
                                        "runs/qwen_6x6")], d2.BLUE)]
    rng = np.random.default_rng(0)
    res = {}
    for name, rows, _ in models:
        for k, _ in CLASSES + EXTRA:
            for t, _ in TARGETS:
                for m in METHODS:
                    res[(name, k, t, m)] = estimate(rows, k, t, m, rng)
    with open(os.path.join(args.out, "fig2_rho_summary.csv"), "w") as f:
        f.write("model,word_class,target,method,rho,ci_lo,ci_hi\n")
        for (n, k, t, m), (e, lo, hi) in res.items():
            f.write(f"{n},{k},{t},{m},{e:.3f},{lo:.3f},{hi:.3f}\n")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 7.2), sharey=True,
                             facecolor=d2.SURF)
    ypos, ylab, y = [], [], 0
    for ci, (k, klab) in enumerate(CLASSES):
        for m in METHODS:
            ypos.append(y); ylab.append(m); y += 1
        y += 0.9
    for ax, (t, tlab) in zip(axes, TARGETS):
        d2.style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=d2.GRID, lw=0.6)
        ax.axvline(0, color=d2.INK2, lw=0.9)
        yi = 0
        for k, _ in CLASSES:
            if k == "article":
                y0 = ypos[yi] - 0.5
                ax.axhspan(y0, y0 + 3, color="#f1efe9", zorder=0)
            for m in METHODS:
                for j, (name, _, col) in enumerate(models):
                    e, lo, hi = res[(name, k, t, m)]
                    yy = ypos[yi] + (j - 0.5) * 0.32
                    ax.plot([lo, hi], [yy, yy], color=col, lw=1.6, zorder=2)
                    ax.plot(e, yy, "o", color=col, ms=5.5, zorder=3,
                            label=name if (yi == 0) else None)
                yi += 1
        ax.set_xlim(-0.6, 0.6)
        ax.set_title(tlab, loc="left", fontsize=10, color=d2.INK, pad=8)
        ax.set_xlabel("rank correlation ρ  (95% interval, cells resampled)",
                      fontsize=8.5, color=d2.INK2)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(ylab, fontsize=8)
    axes[0].invert_yaxis()
    for ci, (k, klab) in enumerate(CLASSES):
        axes[0].text(-0.62, ypos[ci * 3] - 0.75, klab, fontsize=9.5,
                     color=d2.INK, ha="left", va="bottom", fontweight="bold",
                     transform=axes[0].transData, clip_on=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, ncol=2, loc="upper right",
               bbox_to_anchor=(0.98, 0.995))
    fig.text(0.01, 0.01,
             "Opening + initial-pressure replies (Llama n = 148, Qwen n = 180). "
             "pooled: raw Spearman. within cell: each (topic, order) cell's mean "
             "removed from both axes, then Spearman.\nwithin cell + turn: partial "
             "Spearman (ranks residualised on within-cell turn position, then "
             "Pearson on the residuals).\nIntervals: 2000 bootstrap draws of "
             "the 24 cells per model; uncorrected for the 48 estimates shown.\n"
             "Plural first person not drawn (present in 15% of Llama / 8% of "
             "Qwen replies); its estimates are in fig2_rho_summary.csv.",
             fontsize=7.5, color=d2.INK2)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.90, bottom=0.16, wspace=0.08)
    path = os.path.join(args.out, "draft2_fig2_rho_summary.png")
    fig.savefig(path, dpi=200, facecolor=d2.SURF)
    print(f"[fig] {path}")


if __name__ == "__main__":
    main()
