"""Context ablation with the v5 filler: the v3/v4 questions, shorter target.

Same experiment, same code path, ONE thing different from v3: the word
count asked for. Run exactly like context_ablation_v3.py.

WHY A V5
--------
Qwen answers "about 200 words" far longer than Llama does. Measured on
runs/qwen_6x6/fill_meta (v3 questions, Qwen, llama31-default system turn):
median 2813 characters a reply, 1.62x the 1735 of Qwen's own meta
pressure-phase replies; Llama's fill_v4 under the same questions was 1289,
0.83x of its own. The length rule (plan, 2026-10-28 amendment) accepts
0.70-1.20, so that corpus is outside it.

The count was set by a pilot (scripts/fill_length_pilot.py, meta topics,
3 combinations x first 5 positions, 2026-10-29), not by arithmetic: the
reply length does not scale with the count asked for.

    words   median ch   ratio to 1735
     120       793         0.46
     140      1036         0.60
     160      1506         0.87      <- used
     180      2147         1.24
    (200: the first 5 positions of fill_meta ran 2512, 1.45; the full
     27-turn corpus ran ~13% longer than its first five positions)

Truncating the v3 replies was rejected for the reason v3 gives: a reply cut
mid-argument is an artefact no measured arm contains. The length is asked
for up front instead. scripts/fill_length_pilot.py checks the new count on
a few positions before the full corpus is built.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_ablation as CA                                 # noqa: E402
import context_ablation_v3 as V3                              # noqa: E402

WORDS = 160
CA.FILL_TEMPLATES = [t.replace("about 200 words", f"about {WORDS} words")
                     for t in V3.CA.FILL_TEMPLATES]
assert all(f"about {WORDS} words" in t for t in CA.FILL_TEMPLATES)

if __name__ == "__main__":
    CA.main()
