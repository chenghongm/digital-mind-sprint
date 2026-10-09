"""
Full-grid compute estimate, binary vs degree. Feeds notes/compute_estimate.md.

No model, no torch. Every input is read from a run directory except the two
constants below, which name where they come from.

Cost model
----------
Per-turn wall time is linear in characters generated. The slope is fitted
on runs/repl_b1/fill and fill_v2 (per-turn `secs` and `model_chars`, same
positions, very different reply lengths -- the two runs CONTEXT_ABLATION_PLAN
s5-2 used). Those turns are generation only. A grid turn also runs the stance
elicitation (also generation, so its characters go through the same slope)
and two probe orders (forward passes only). The probe cost is the per-turn
constant OVERHEAD, set so the model reproduces the one full-turn wall time
ever measured: CONSOLE_SECS below.

    python3 scripts/compute_estimate.py
"""
import glob
import json
import statistics as S

# HANDOFF s6c: "[1/36] neutral__000__o1 ... (284s)", read off the Colab
# console on 2026-08-26, replication batch 1, A100. Never stored in a file.
CONSOLE_CONV = "runs/repl_b1/meta/neutral__000__o1.json"
CONSOLE_SECS = 284.0

# Colab Resources panel, A100, HANDOFF s6c (2026-08-25); analyze.py uses it.
CU_PER_HOUR = 5.3

# Grid as scoped in HANDOFF s6a/s6c: option (B), 31 candidates + 3 controls.
TOPICS, ORDERS = 34, 2
PRESSURE_ARMS = ("pressure_release", "pressure_switch", "pressure_sustained")
NEUTRAL_ARMS = ("neutral", "neutral_switch")
RELEASE_TURNS, MAX_PRESSURE_TURNS, RUNGS = 12, 15, 5   # runner.py


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def fit():
    xs, ys = [], []
    for d in ("fill", "fill_v2"):
        for c in load(f"runs/repl_b1/{d}/*.json"):
            for t in c["turns"]:
                xs.append(t["model_chars"])
                ys.append(t["secs"])
    mx, my = S.mean(xs), S.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / \
        sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b, len(xs)


def chars(t):
    return len(t["model_text"]) + len(t.get("elicited_text") or "")


def main():
    a, b, n = fit()
    print(f"[fit] secs = {a:.3f} + {b*1000:.3f} s/kchar   (n={n} turns, "
          f"fill + fill_v2)")
    print(f"      at 3328 chars ~ 628 tok: {b*3328/628:.4f} s/tok")

    conv = json.load(open(CONSOLE_CONV))
    gen = sum(a + b * chars(t) for t in conv["turns"])
    overhead = (CONSOLE_SECS - gen) / len(conv["turns"])
    print(f"[calib] {CONSOLE_CONV}: {len(conv['turns'])} turns, "
          f"generation predicts {gen:.0f}s of {CONSOLE_SECS:.0f}s -> "
          f"overhead {overhead:.2f} s/turn (probe + elicitation prefill)")

    def sec(ts):
        return S.mean(a + b * chars(t) + overhead for t in ts)

    b1 = load("runs/repl_b1/meta/*.json")
    n27 = load("runs/repl_b1_neu27/meta/*.json")
    press = [c for c in b1 if c["condition"] in PRESSURE_ARMS]
    ph = {p: [t for c in press for t in c["turns"] if t["phase"] == p]
          for p in ("opening", "pressure", "release")}
    s = {p: sec(v) for p, v in ph.items()}
    s["neutral"] = sec([t for c in n27 for t in c["turns"]])
    p_turns = len(ph["pressure"]) / len(press)
    print(f"[s/turn] opening {s['opening']:.1f}  pressure {s['pressure']:.1f}"
          f"  release {s['release']:.1f}   (repl_b1, {len(press)} pressure-arm"
          f" convs)")
    print(f"         neutral {s['neutral']:.1f}   (repl_b1_neu27, {len(n27)} "
          f"convs x 28 turns)")
    print(f"         mean pressure turns per conv {p_turns:.1f} "
          f"(stop-at-flip, flip-rule both)")

    def pressure_conv(k):
        return s["opening"] + k * s["pressure"] + RELEASE_TURNS * s["release"]

    def scenario(name, p_turns, levels):
        # neutral arm runs as long as the longest pressure arm (repl_b1
        # FINDINGS: otherwise final_gap has no turn-matched reference)
        n_len = 1 + (MAX_PRESSURE_TURNS if p_turns > RUNGS else p_turns) \
            + RELEASE_TURNS
        cells = TOPICS * ORDERS
        p_secs = cells * len(PRESSURE_ARMS) * levels * pressure_conv(p_turns)
        n_secs = cells * len(NEUTRAL_ARMS) * n_len * s["neutral"]
        convs = cells * (len(PRESSURE_ARMS) * levels + len(NEUTRAL_ARMS))
        turns = cells * (len(PRESSURE_ARMS) * levels * (1 + p_turns +
                         RELEASE_TURNS) + len(NEUTRAL_ARMS) * n_len)
        h = (p_secs + n_secs) / 3600
        return name, convs, turns, h, h * CU_PER_HOUR

    rows = [
        scenario("binary  (stop-at-flip)", p_turns, 1),
        scenario("degree D1 (1 ladder, 15 rungs fixed)", MAX_PRESSURE_TURNS, 1),
        scenario("degree D2 (3 levels x 5 rungs fixed)", RUNGS, 3),
        scenario("degree D3 (3 levels x 15 rungs fixed)", MAX_PRESSURE_TURNS, 3),
    ]
    base = rows[0][3]
    print(f"\n{'scenario':40s} {'convs':>6s} {'turns':>7s} {'A100 h':>7s} "
          f"{'CU':>6s} {'x':>5s}")
    for name, convs, turns, h, cu in rows:
        print(f"{name:40s} {convs:6d} {turns:7.0f} {h:7.1f} {cu:6.0f} "
              f"{h/base:5.2f}")

    # --- what past work actually spent against what it needed -------------
    # (1) Context ablation: every file carries wall_secs, so spent and needed
    # are both read, not modelled. Needed = the corpus that hit its target
    # (fill_v4), the 2x2 on it (ablation_v4) and the planned robustness
    # method (ablation_splice). Spent on top: three corpora that missed the
    # length target (fill, fill_v2, fill_v3) and cell D on one of them.
    def wall(d):
        return sum(c.get("wall_secs", 0)
                   for c in load(f"runs/repl_b1/{d}/*.json")) / 3600
    need = ["fill_v4", "ablation_v4", "ablation_splice"]
    lost = ["fill", "fill_v2", "fill_v3", "ablation_v3"]
    print("\n[overhead 1] context ablation, recorded wall_secs")
    for d in lost + need:
        print(f"    {d:16s} {wall(d):5.2f} h {wall(d)*CU_PER_HOUR:5.2f} CU"
              f"  {'superseded' if d in lost else 'kept'}")
    hn, hl = sum(map(wall, need)), sum(map(wall, lost))
    f1 = (hn + hl) / hn
    print(f"    spent / needed = {hn+hl:.2f} / {hn:.2f} h = {f1:.2f}x")

    # (2) Replication batch 1: generation, the grid's own protocol. Its
    # neutral arms ran 13 turns and had to be re-run at 28 (repl_b1_neu27)
    # because final_gap had no turn-matched reference once ToF > 12. Not
    # timed in the files (schema 5/6), so costed with the model above.
    neu13 = [c for c in b1 if c["condition"] in NEUTRAL_ARMS]
    h_press = sum(sum(a + b * chars(t) + overhead for t in c["turns"])
                  for c in press) / 3600
    h_neu13 = sum(sum(a + b * chars(t) + overhead for t in c["turns"])
                  for c in neu13) / 3600
    h_neu27 = sum(sum(a + b * chars(t) + overhead for t in c["turns"])
                  for c in n27) / 3600
    f2 = (h_press + h_neu13 + h_neu27) / (h_press + h_neu27)
    print(f"\n[overhead 2] replication batch 1 (cost model)")
    print(f"    pressure arms {len(press)} convs {h_press:.2f} h | neutral "
          f"13-turn {len(neu13)} convs {h_neu13:.2f} h (superseded) | "
          f"neu27 {len(n27)} convs {h_neu27:.2f} h")
    print(f"    spent / needed = {f2:.2f}x   ({h_neu13*CU_PER_HOUR:.1f} CU "
          f"superseded)")

    print(f"\n{'scenario':40s} {'clean CU':>9s} {'x'+format(f2,'.2f'):>8s} "
          f"{'x'+format(f1,'.2f'):>8s}")
    for name, convs, turns, h, cu in rows:
        print(f"{name:40s} {cu:9.0f} {cu*f2:8.0f} {cu*f1:8.0f}")


if __name__ == "__main__":
    main()
