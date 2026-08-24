"""Rule-by-rule check of stance_text.parse_opening_side.

The cases are the ones the rules are meant to separate, including the
two that are meant to FAIL: a bare capital in prose is not a
declaration, and negation is not handled (that is why step 2 reads the
openings rather than trusting the count).

Usage:  python3 scripts/test_stance_text.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stance_text import classify_opening as c

A = "Cats make better apartment pets"
B = "Dogs make better apartment pets"

cases = [
    # (text, expected_side, label)
    ("(A) Cats make better apartment pets. Here are my three reasons: unlike (B), cats...",
     "A", "declared A, mentions B later"),
    ("I take side B. Dogs are more trainable.", "B", "'side B'"),
    ("**A** is the stronger position.", "A", "bold"),
    ("B) Dogs, clearly. First,", "B", "B) form"),
    ("Between (A) and (B), I find it genuinely hard to say. But (B) edges it.",
     "unparsed", "both letters in first sentence -> conflict"),
    ("Dogs make better apartment pets, and I'll defend that. Cats are fine too.",
     "B", "no letter; side string B first"),
    ("Cats make better apartment pets - that's my view.", "A", "side string A first"),
    ("I don't think either framing survives contact with reality.",
     "unparsed", "no signal"),
    ("", "unparsed", "empty"),
    ("A better argument exists for the second option.", "unparsed",
     "bare capital A in prose must NOT count"),
    ("My position: Option A.", "A", "'Option A'"),
    ("I would not defend (A). I take the other one.", "A",
     "KNOWN LIMIT: negation misread"),
]

bad = 0
for text, exp, label in cases:
    v = c(text, A, B)
    ok = v.side == exp
    bad += not ok
    print(f"{'ok ' if ok else 'BAD'}  {v.side:9s} {v.rule:12s}  {label}")
print(f"\n{len(cases)-bad}/{len(cases)} as specified")
