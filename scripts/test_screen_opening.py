"""
screen_topics section 5, without a model.

Section 5 used to derive the reported opening side, the flip, and therefore
the ladder budget from the PROBE. Since schema 8 it derives them from the
generated text, through the same stance_text runner.py uses. That block runs
once per screen, after an hour of generation, which is a bad place to find a
typo -- so the derivation lives in opening_verdicts() / opening_row() and is
driven here against the 26 real v11 openings.

Usage:  python3 scripts/test_screen_opening.py [--run runs/screen_llama_v11]
"""

import argparse
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# screen_topics imports runner, which imports torch. The section under test
# touches neither.
for name, mod in (("torch", types.ModuleType("torch")),
                  ("transformers", types.ModuleType("transformers"))):
    sys.modules.setdefault(name, mod)
sys.modules["torch"].no_grad = lambda: (lambda f: f)
sys.modules["torch"].cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules["torch"].backends = types.SimpleNamespace(
    mps=types.SimpleNamespace(is_available=lambda: False))
sys.modules["transformers"].AutoModelForCausalLM = object
sys.modules["transformers"].AutoTokenizer = object

sys.path.insert(0, str(ROOT / "scripts"))
import screen_topics as S                                   # noqa: E402
import stance_text as ST                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/screen_llama_v11")
    ap.add_argument("--topics", default="topics_arbitrary.json")
    args = ap.parse_args()

    # --- unit: the swap is undone exactly once -----------------------------
    item = {"side_a": "Cats win", "side_b": "Dogs win"}
    cases = [
        ("I take position (A) Cats win.",  1, "A", "the identity order"),
        ("I take position (A) Dogs win.",  2, "B", "slot A under order 2"),
        ("I take position (B) Cats win.",  2, "A", "slot B under order 2"),
        ("I'll defend Dogs win.",          2, "B", "content rule, order 2"),
        ("No idea either way.",            2, ST.UNPARSED, "unparsed"),
    ]
    for text, order, want, label in cases:
        got, _slot = S.opening_verdicts(item, text, order)
        assert got == want, f"{label}: want {want}, got {got}"
    print(f"ok  opening_verdicts: {len(cases)}/{len(cases)}")

    # --- unit: the row fields, including the case that used to be wrong ----
    def cell(text_side, side):
        return dict(text_side=text_side, side=side)

    r = S.opening_row({1: cell("A", "B"), 2: cell("B", "B")})
    assert r["opening_flips"] is True,  "text argues both ways -> two ladders"
    assert r["opening_probe_flips"] is False, "probe agrees across the two"
    assert r["opening_unparsed"] is False
    r = S.opening_row({1: cell("A", "A"), 2: cell(ST.UNPARSED, "B")})
    assert r["opening_flips"] is False, "unparsed is not a flip"
    assert r["opening_unparsed"] is True
    print("ok  opening_row: flips follows the text, not the probe")

    # --- unit: cold accuracy is tri-state ----------------------------------
    # An unparsed opening is not a wrong prediction. It is unscorable, and
    # putting it on either side of the hit rate moves the number without
    # measuring anything.
    hit  = S.opening_row({1: cell("A", "A"), 2: cell("A", "A")}, cold_side="A")
    miss = S.opening_row({1: cell("B", "B"), 2: cell("B", "B")}, cold_side="A")
    unsc = S.opening_row({1: cell(ST.UNPARSED, "A"), 2: cell("A", "A")},
                         cold_side="A")
    assert hit["opening_agrees"] is True
    assert miss["opening_agrees"] is False
    assert unsc["opening_agrees"] is None, "unparsed must not score as a miss"
    # and with no prediction to score against
    assert S.opening_row({1: cell("A", "A"), 2: cell("A", "A")}
                         )["opening_agrees"] is None
    print("ok  opening_row: unparsed is unscorable, not a miss")

    # --- integration: the real openings ------------------------------------
    run, topics_path = ROOT / args.run, ROOT / args.topics
    if not (run / "openings").is_dir():
        print(f"[skip] no {run}/openings -- unit checks only")
        return
    topics = {t["topic"]: t for t in json.load(open(topics_path))}
    per_topic = {}
    for f in sorted((run / "openings").glob("*.txt")):
        topic, _, o = f.stem.rpartition("__o")
        if topic not in topics:
            continue
        tside, _ = S.opening_verdicts(topics[topic], f.read_text(), int(o))
        per_topic.setdefault(topic, {})[int(o)] = dict(
            text_side=tside, side="A")          # probe side irrelevant here

    both = {t: p for t, p in per_topic.items() if len(p) == 2}
    rows = {t: S.opening_row(p) for t, p in both.items()}
    flips = [t for t, r in rows.items() if r["opening_flips"]]
    unp = [t for t, r in rows.items() if r["opening_unparsed"]]
    print(f"\n{len(both)} topics with both orders, from {run.name}")
    print(f"  argue opposite sides : {len(flips)}  -> need TWO ladders")
    print(f"  at least one unparsed: {len(unp)}  -> pressure arms cannot run")
    for t in sorted(flips):
        print(f"    flip     {t}: o1={both[t][1]['text_side']} "
              f"o2={both[t][2]['text_side']}")
    for t in sorted(unp):
        print(f"    unparsed {t}")

    # The same number check_opening_text.py section 3b prints. Two scripts,
    # one parser: if these ever disagree, one of them stopped undoing the
    # swap.
    print("\n  cross-check: scripts/check_opening_text.py section 3b must")
    print("  print the same count.")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
