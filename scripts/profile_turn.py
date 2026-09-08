"""
Where does a turn's time actually go?

runner.do_turn does four full passes over the conversation plus a decode:

    1  forward(output_hidden_states=True)   hidden state at the read layer
    2  generate(max_new_tokens=768)         re-prefills the same tokens, then
                                            decodes sequentially
    3  probe_stance(order 1)                full prefill of conversation+probe
    4  probe_stance(order 2)                the same again, different slots

Passes 3 and 4 are what order-averaging added. Whether that matters depends
on the split between prefill (parallel, grows with context) and decode
(sequential, grows with tokens emitted), and on whether the device is
swapping -- on the 24GB Mac the pilot's per-turn time went 33s -> 248s over
seven turns, which is the shape of paging, not of arithmetic.

Optimising before measuring that split would be guessing at which of the
four to fix. So: synthetic conversations at several lengths, each component
timed separately, memory reported.

Nothing here writes to a run directory. It loads the same weights the
experiment uses and calls the same runner methods, so the numbers are the
experiment's, not a proxy's.

Usage:
    python3 scripts/profile_turn.py --model ./Llama-3.1-8B-Instruct
    python3 scripts/profile_turn.py --model ./Llama-3.1-8B-Instruct \
        --turns 1 4 8 12 --new-tokens 768
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch                                              # noqa: E402
import runner as R                                        # noqa: E402

SIDE_A = "Oak"
SIDE_B = "Maple"

# Filler with the shape of a real turn: a user rebuttal and a long assistant
# answer. Length matters, content does not -- this measures the machine.
USER = ("I've thought about this and the numbers point the other way. "
        "The measured difference over the quarter was substantial and it "
        "held across every subgroup we looked at.")
ASSIST = (("There are several considerations here that bear on the question, "
           "and it is worth taking them in order rather than treating the "
           "conclusion as settled by a single figure. ") * 12)


def mem():
    if torch.backends.mps.is_available():
        try:
            return torch.mps.current_allocated_memory() / 1e9
        except Exception:
            return float("nan")
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return float("nan")


def build(n_turns):
    """A conversation with n_turns completed exchanges, ready for one more."""
    msgs = []
    for _ in range(n_turns):
        msgs.append({"role": "user", "content": USER})
        msgs.append({"role": "assistant", "content": ASSIST})
    msgs.append({"role": "user", "content": USER})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--turns", nargs="+", type=int, default=[1, 4, 8, 12])
    ap.add_argument("--new-tokens", type=int, default=None,
                    help="override MAX_NEW_TOKENS for the generate timing")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.new_tokens:
        R.MAX_NEW_TOKENS = args.new_tokens

    rr = R.Runner(args.model, device=args.device)
    print(f"\nMAX_NEW_TOKENS={R.MAX_NEW_TOKENS}  device={rr.device}")
    hdr = (f"\n{'turns':>5s} {'ctx tok':>8s} {'fwd+hs':>8s} {'generate':>9s} "
           f"{'gen tok':>8s} {'probe x2':>9s} {'TOTAL':>7s} {'GB':>6s}")
    print(hdr)
    print("-" * len(hdr))

    for n in args.turns:
        msgs = build(n)
        enc = rr._encode(msgs)
        ctx = enc["input_ids"].shape[-1]

        with torch.no_grad():
            # 1 -- hidden state
            t0 = time.time()
            hs = rr.model(**enc, output_hidden_states=True).hidden_states
            _ = hs[rr.layer_idx][0, -1, :].float().cpu().numpy()
            t_fwd = time.time() - t0

            # 2 -- generate
            t0 = time.time()
            out = rr.model.generate(
                **enc, max_new_tokens=R.MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=rr.tok.eos_token_id)
            t_gen = time.time() - t0
            n_new = out.shape[-1] - ctx
            text = rr.tok.decode(out[0, ctx:], skip_special_tokens=True)

            # 3 and 4 -- both probe orders, on the conversation WITH the reply
            full = msgs + [{"role": "assistant", "content": text}]
            t0 = time.time()
            rr.probe_stance(full, SIDE_A, SIDE_B)
            rr.probe_stance(full, SIDE_B, SIDE_A)
            t_probe = time.time() - t0

        total = t_fwd + t_gen + t_probe
        print(f"{n:5d} {ctx:8d} {t_fwd:7.1f}s {t_gen:8.1f}s {n_new:8d} "
              f"{t_probe:8.1f}s {total:6.1f}s {mem():6.1f}")

    print("""
Reading this:

  fwd+hs vs generate   generate re-prefills the same tokens before decoding,
                       so `generate` minus a decode-only estimate is a second
                       copy of `fwd+hs`. If they are close in size, merging
                       them via past_key_values saves a full prefill.

  probe x2             two full prefills of conversation+probe. They share
                       every token up to where the two side strings differ,
                       so caching the common prefix turns this into one
                       prefill plus two short suffixes. This is the cost
                       order-averaging added.

  growth with turns    if total time grows faster than context length, the
                       device is paging and no amount of pass-merging fixes
                       it -- the run belongs on a machine that holds the
                       weights and the cache in memory.
""")


if __name__ == "__main__":
    main()
