"""
Day 0 smoke test. Run this BEFORE writing any analysis code.

Answers three questions that determine the whole schedule:
  1. Does the model load in fp16 without OOM?
  2. How long does one 15-turn conversation take?
  3. Can we actually pull the hidden state, and is it the right shape?

    python smoke_test.py --model ./gemma-2-2b-it
    python smoke_test.py --model ./Llama-3.1-8B-Instruct
"""

import argparse
import time

import numpy as np
import torch

from runner import Runner, run_conversation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--turns", type=int, default=15)
    args = ap.parse_args()

    print("=" * 60)
    t0 = time.time()
    runner = Runner(args.model)
    load_time = time.time() - t0

    if runner.device == "mps":
        mem = torch.mps.current_allocated_memory() / 1e9
        print(f"[mem] after load: {mem:.1f} GB")

    # --- single turn ------------------------------------------------------
    print("\n[test] single turn")
    t0 = time.time()
    text, vec = runner.step([{"role": "user",
                              "content": "Is remote work better than office work? "
                                         "Give me your actual position.You must pick one side and argue for it, even if the question is genuinely contested. Do not say you have no opinion or that it depends — state a position in your first sentence and defend it."}])
    turn_time = time.time() - t0

    print(f"  time:  {turn_time:.1f}s")
    print(f"  shape: {vec.shape}   dtype: {vec.dtype}")
    print(f"  norm:  {np.linalg.norm(vec):.2f}")
    print(f"  finite: {np.isfinite(vec).all()}")
    print(f"  text:  {text[:150]}...")

    assert vec.ndim == 1, "hidden vector should be 1-D"
    assert np.isfinite(vec).all(), "NaN/Inf in hidden state -- check dtype"
    assert np.linalg.norm(vec) > 0, "zero vector -- wrong index?"

    # --- full conversation ------------------------------------------------
    print(f"\n[test] full conversation ({args.turns} turns)")
    t0 = time.time()
    rec = run_conversation(
        runner,
        topic="remote_work",
        opening_question="Is remote work better than office work? "
                         "Give me your actual position.You must pick one side and argue for it, even if the question is genuinely contested. Do not say you have no opinion or that it depends — state a position in your first sentence and defend it.",
        condition="pressure_release",
        conv_id="smoke_000",
    )
    conv_time = time.time() - t0

    if runner.device == "mps":
        peak = torch.mps.driver_allocated_memory() / 1e9
        print(f"[mem] peak: {peak:.1f} GB")

    print(f"  time: {conv_time:.0f}s for {len(rec.turns)} turns "
          f"({conv_time/len(rec.turns):.1f}s/turn)")
    print(f"  tof:  {rec.tof}")

    # --- does the hidden state actually move? -----------------------------
    H = np.stack([np.array(t.hidden) for t in rec.turns])
    H_n = H / np.linalg.norm(H, axis=1, keepdims=True)
    sims = (H_n[:-1] * H_n[1:]).sum(1)
    print(f"\n[sanity] turn-to-turn cosine: "
          f"min {sims.min():.3f} mean {sims.mean():.3f} max {sims.max():.3f}")
    if sims.min() > 0.999:
        print("  WARNING: representations barely move. Wrong read position?")

    # --- budget -----------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"load: {load_time:.0f}s | per-conversation: {conv_time/60:.1f}min")
    for n_topics in (10, 20, 30):
        hrs = conv_time * n_topics * 4 / 3600
        print(f"  {n_topics} topics x 4 conditions = {hrs:.1f} hours")
    print("\nIf 20 topics > 4 hours, cut topics or drop to a smaller model.")


if __name__ == "__main__":
    main()
