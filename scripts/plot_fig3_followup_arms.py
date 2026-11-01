"""Figure 3 (2026-11-01): first-person rates in the 12 follow-up turns, by
condition. Same numbers as the draft's follow-up table (288 replies per
model per condition = 24 cells x 12 turns): singular / stance phrases /
plural, per 1000 words. Bars are reply means; whiskers are 95% bootstrap
intervals resampling (topic, order) cells (2000 draws). The dashed line in
each panel is that model's initial-pressure mean, for reference.

    python3 scripts/plot_fig3_followup_arms.py --out figs/6x6
"""
import argparse, csv, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_first_person_draft2 as d2                          # noqa: E402

ARMS = [("pressure_sustained", "sustained\npressure"),
        ("pressure_release", "release\nsame\ntopic"),
        ("pressure_switch", "release\nswitch\ntopic"),
        ("neutral", "neutral\nsame\ntopic"),
        ("neutral_switch", "neutral\nswitch\ntopic")]
CATS = [("fp_sing", "Singular first person\n(I, me, my ...)"),
        ("fp_stance", "First-person stance phrases\n(I would argue, in my view ...)"),
        ("fp_plur", "Plural first person\n(we, us, our ...)")]
N_BOOT = 2000


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, _ in CATS:
            r[k] = float(r[k])
        r["cell"] = (r["topic"], r["order"])
    return rows


def mean_ci(rows, k, rng):
    by = defaultdict(list)
    for r in rows:
        by[r["cell"]].append(r[k])
    cells = list(by.values())
    boot = [np.concatenate([cells[i] for i in rng.integers(len(cells), size=len(cells))]).mean()
            for _ in range(N_BOOT)]
    return np.mean([r[k] for r in rows]), *np.percentile(boot, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/6x6")
    args = ap.parse_args()
    models = [("Llama-3.1-8B", load("figs/llama_6x6/first_person.csv"), d2.ORANGE),
              ("Qwen2.5-7B", load("figs/qwen_6x6/first_person.csv"), d2.BLUE)]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.0), facecolor=d2.SURF)
    x = np.arange(len(ARMS)); w = 0.36
    log = []
    for ax, (k, title) in zip(axes, CATS):
        d2.style(ax)
        for j, (name, rows, col) in enumerate(models):
            st = []
            for arm, _ in ARMS:
                sel = [r for r in rows if r["phase"] == "release" and r["arm"] == arm]
                assert len(sel) == 288, (name, arm, len(sel))
                st.append(mean_ci(sel, k, rng))
                log.append((name, k, arm, *st[-1]))
            m = np.array([s[0] for s in st])
            err = np.array([[s[0] - s[1] for s in st], [s[2] - s[0] for s in st]])
            xs = x + (j - 0.5) * (w + 0.02)
            ax.bar(xs, m, w, color=col, edgecolor=d2.SURF, lw=1,
                   label=name, zorder=2)
            ax.errorbar(xs, m, yerr=err, fmt="none", ecolor=d2.INK2, lw=1,
                        capsize=2.5, zorder=3)
            for xx, mm, top in zip(xs, m, [s[2] for s in st]):
                ax.annotate(f"{mm:.1f}", (xx, top), xytext=(0, 3),
                            textcoords="offset points", ha="center",
                            fontsize=7, color=d2.INK)
            pm = np.mean([r[k] for r in rows if r["phase"] == "pressure"])
            ax.axhline(pm, color=col, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([a[1] for a in ARMS], fontsize=7.8)
        ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left", fontsize=9.5, color=d2.INK, pad=6)
    axes[0].set_ylabel("per 1000 words", fontsize=9, color=d2.INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, ncol=2, loc="upper right",
               bbox_to_anchor=(0.99, 1.0))
    fig.text(0.01, 0.01,
             "12 follow-up turns per cell, 288 replies per model per condition "
             "(24 cells x 12). Bars: reply means; whiskers: 95% bootstrap over "
             "cells. Dashed lines: each model's initial-pressure mean.\n"
             "Only 'sustained pressure' keeps asking for a stance; the other four "
             "ask factual questions, so near-zero rates there reflect question "
             "type, not weaker stance.", fontsize=7.5, color=d2.INK2)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.27, wspace=0.22)
    path = os.path.join(args.out, "draft2_fig3_followup_arms.png")
    fig.savefig(path, dpi=200, facecolor=d2.SURF)
    with open(os.path.join(args.out, "fig3_followup_arms.csv"), "w") as f:
        f.write("model,word_class,condition,mean,ci_lo,ci_hi\n")
        for r in log:
            f.write(",".join([r[0], r[1], r[2]] + [f"{v:.2f}" for v in r[3:]]) + "\n")
    print(f"[fig] {path}")


if __name__ == "__main__":
    main()
