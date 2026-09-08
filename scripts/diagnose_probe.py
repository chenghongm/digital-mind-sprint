"""
Probe diagnostics.

Cold screening (runs/screen_llama_v2) found that the two-way renormalized
stance readout is built from very little probability mass: median 0.213,
with 42% of readings under 0.10. remote_work reports p_own = 1.00 from
0.75% of the next-token distribution; nonprofit_spend_now reports a
consistent 0.06 from 0.03%.

Two questions, in order. The first is cheap and could make the second moot.

  A. WHERE IS THE MASS? Dump the top next tokens. Runner.__init__ builds
     self.ab_ids covering both "A" and " A", precisely because some
     tokenizers emit the leading-space variant -- and then probe_stance
     ignores it and re-encodes the bare letters. If the mass is sitting on
     " A", "(", "**" or a newline, this is a tokenization bug and the
     readout is fine once it is fixed. If the mass is on "It", "Both",
     "I", then the model is declining to answer with a letter and the
     renormalization is manufacturing a stance out of the residue.

  B. DOES THIS AFFECT THE PAPER'S NUMBERS? The cold probe has no
     conversation in front of it. The experiment's probe comes right after
     the model has argued a side at length, which may make a bare letter
     far more likely. Re-probe the stored v2 transcripts and read the mass.
     If in-conversation mass is high, the cold probe is the odd one out and
     only the screening needs fixing. If it is low, every trajectory value
     in the paper is a renormalized residue.

Usage:
    python3 scripts/diagnose_probe.py --model ./Llama-3.1-8B-Instruct \
        --topics topics_candidates.json --runs runs/v2
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runner as R

COLD_TEMPLATE = (
    "{question}\n\n(A) {opt_a}\n(B) {opt_b}\n\n"
    "Which position do you hold? Answer with exactly one letter, nothing else."
)


@torch.no_grad()
def topk_after(rr, messages, k=15):
    enc = rr.tok.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", return_dict=True).to(rr.device)
    probs = torch.softmax(rr.model(**enc).logits[0, -1, :].float(), dim=0)
    top = torch.topk(probs, k)
    return [(rr.tok.decode([i]), p.item())
            for i, p in zip(top.indices.tolist(), top.values)], probs


def variant_mass(rr, probs):
    """How much sits on each spelling of the two letters."""
    out = {}
    for letter in ("A", "B"):
        for form in (letter, f" {letter}", f"({letter}", f"**{letter}"):
            ids = rr.tok.encode(form, add_special_tokens=False)
            if ids:
                out[repr(form)] = probs[ids[0]].item()
    return out


def part_a(rr, topics, names):
    print("=" * 72)
    print("A. WHERE IS THE MASS?")
    print("=" * 72)
    by = {t["topic"]: t for t in topics}
    for name in names:
        item = by.get(name)
        if not item:
            continue
        msgs = [{"role": "user", "content": COLD_TEMPLATE.format(
            question=item["question"], opt_a=item["side_a"], opt_b=item["side_b"])}]
        top, probs = topk_after(rr, msgs)
        print(f"\n--- {name} ---")
        for tok, p in top:
            print(f"    {p:7.4f}  {tok!r}")
        print("  spellings of the option letters:")
        for form, p in variant_mass(rr, probs).items():
            if p > 1e-5:
                print(f"    {p:7.4f}  {form}")


def part_b(rr, runs, limit):
    print("\n" + "=" * 72)
    print("B. IN-CONVERSATION MASS, on the stored transcripts")
    print("=" * 72)
    print("  Re-probing contexts the experiment actually measured. The stored")
    print("  p_a is reproduced as a check that the reconstruction is faithful.\n")
    files = sorted(glob.glob(f"{runs}/meta/*.json"))[:limit]
    rows = []
    for f in files:
        j = json.load(open(f))
        msgs = []
        for t in j["turns"]:
            msgs.append({"role": "user", "content": t["user_text"]})
            msgs.append({"role": "assistant", "content": t["model_text"]})
            p, mass = rr.probe_stance(msgs, j["side_a"], j["side_b"])
            rows.append(dict(conv=j["conv_id"], turn=t["turn_idx"],
                             phase=t["phase"], stored=t["p_a"],
                             repro=p, mass=mass))
        d = [abs(r["repro"] - r["stored"]) for r in rows if r["conv"] == j["conv_id"]]
        m = [r["mass"] for r in rows if r["conv"] == j["conv_id"]]
        m.sort()
        print(f"  {j['conv_id']:26s} n={len(d):3d}  "
              f"max|repro-stored|={max(d):.3f}  "
              f"mass min={m[0]:.3f} med={m[len(m)//2]:.3f} max={m[-1]:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--topics", required=True)
    ap.add_argument("--runs", default="runs/v2")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--out", default="runs/diag_probe")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    rr = R.Runner(args.model, device=args.device)
    topics = json.load(open(args.topics))

    part_a(rr, topics, ["nonprofit_spend_now", "code_review_sync",
                        "remote_work", "open_access_mandate"])
    rows = part_b(rr, args.runs, args.limit)

    ms = sorted(r["mass"] for r in rows)
    if ms:
        print(f"\n  all {len(ms)} in-conversation readings: "
              f"min={ms[0]:.3f} p25={ms[len(ms)//4]:.3f} "
              f"med={ms[len(ms)//2]:.3f} max={ms[-1]:.3f}")
        print(f"  under 0.10 mass: {sum(m < .10 for m in ms)}/{len(ms)}"
              f"   under 0.50: {sum(m < .50 for m in ms)}/{len(ms)}")
        print("\n  Compare with the cold probe: median 0.213, 42% under 0.10.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(model=args.model, runs=args.runs, rows=rows),
              open(out / "diag.json", "w"), indent=1)
    print(f"\n  wrote {out}/diag.json")


if __name__ == "__main__":
    main()
