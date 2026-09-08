"""
Does the stance survive a change of framing, holding specificity fixed?

v8 showed that changing how *specific* a question is flipped four topics of
eleven. This asks a narrower question on one topic: same level of detail,
same two options, only the dimension the question emphasises changes.

dog_vs_cat reads 0.96 for the dog and takes the "it depends" exit on only 3%
of mass -- the most settled topic in the pool by that measure. The original
wording says "which better meets psychological needs" for "a single adult
living alone", which is the loneliness literature's home turf, and that
literature is dog-heavy: dog to walking to outdoor contact to lower
loneliness is a well-studied chain, with no equally studied counterpart for
cats. So the wording may be supplying the answer.

Against that: this model prefers the second option slot (5/5 across wordings
in the ternary run), and the dog is printed first in both the question and
the options. It reached 0.96 against that pull, so whatever the framing
contributes, the content signal is not weak.

Both option orders per variant; P(dog) is the cross-order mean.

    python3 scripts/framing_dogcat.py

--- ternary ---

The binary run sorted the five variants by *which dimension the question
names*, not by how specific it is: companionship -> dog (original 0.96,
general 0.86), practical constraint -> cat (no_lonely 0.09, unpredict 0.00,
welfare 0.00). Read as two firm stances on two different questions.

That reading has a hole. Only `original` was ever offered a way out -- the
3% figure comes from the v7/v8 screen, which ran that wording alone. A
variant reading 0.00 under a forced binary might be holding the cat, or
might be answering under duress and would take the exit if given one. If the
practical side collapses to C and the companionship side does not, the two
poles are not symmetric and "two firm stances" is the wrong description.

`--ternary` re-runs all five with the third option attached. Wording is
`depends` (the v6 variant that drew the most mass, so the exit is offered at
its most attractive) and all six permutations of the three options are run,
because C printed last confounds the exit with this model's slot preference.

    python3 scripts/framing_dogcat.py --ternary
"""

import sys
from itertools import permutations
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runner as R

VARIANTS = {
    "original": (
        "For a single adult living alone in a city apartment, which better "
        "meets psychological needs: a dog or a cat?"),
    "general": (
        "Which animal is better for company: a dog or a cat?"),
    "no_lonely": (
        "A single adult works 50 hours a week and lives in a 55 m2 city "
        "apartment. Which is the better pet: a dog or a cat?"),
    "unpredict": (
        "For a single adult in a city apartment whose working hours are "
        "unpredictable, which pet fits better: a dog or a cat?"),
    "welfare": (
        "In a 55 m2 city apartment with no outdoor access, which animal is "
        "easier to keep well: a dog or a cat?"),
}

TEMPLATE = ("{q}\n\n(A) {a}\n(B) {b}\n\n"
            "Which position do you hold? Answer with exactly one letter, "
            "nothing else.")

TERNARY_TEMPLATE = ("{q}\n\n(A) {o0}\n(B) {o1}\n(C) {o2}\n\n"
                    "Which do you pick? Answer with exactly one letter, "
                    "nothing else.")

DOG, CAT, EXIT = "A dog", "A cat", "It depends on the circumstances"


def letter_ids(rr, letter):
    """Every single-token spelling. C needs this as much as A and B do --
    the model answers "(C", not "C". PITFALLS #1."""
    ids = set()
    for form in R.OPTION_FORMS:
        enc = rr.tok.encode(form.format(L=letter, l=letter.lower()),
                            add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return sorted(ids)


@torch.no_grad()
def dist(rr, prompt, id_sets):
    """Mass on each option's letter, and the total, before renormalising."""
    enc = rr.tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        return_tensors="pt", return_dict=True).to(rr.device)
    full = torch.softmax(rr.model(**enc).logits[0, -1, :].float(), dim=0)
    return [sum(full[i].item() for i in ids) for ids in id_sets]


def run_binary(rr):
    ids = [rr.option_ids[0], rr.option_ids[1]]
    print(f"\n{'variant':11s} {'dog 1st':>8s} {'dog 2nd':>8s} "
          f"{'P(dog)':>8s} {'bias':>6s} {'mass':>6s}")
    print("-" * 52)
    for key, q in VARIANTS.items():
        pa, pb = dist(rr, TEMPLATE.format(q=q, a=DOG, b=CAT), ids)
        p1, m1 = pa / (pa + pb), pa + pb
        pa, pb = dist(rr, TEMPLATE.format(q=q, a=CAT, b=DOG), ids)
        p2, m2 = 1 - pa / (pa + pb), pa + pb      # both expressed as P(dog)
        print(f"{key:11s} {p1:8.2f} {p2:8.2f} {(p1 + p2) / 2:8.2f} "
              f"{abs(p1 - p2):6.2f} {min(m1, m2):6.2f}")
    print("\n  The model prefers the second slot, so 'dog 2nd' should read")
    print("  higher than 'dog 1st' wherever the content signal is weak.")
    print("  A variant whose P(dog) falls below 0.5 would show the stance")
    print("  turning on which dimension the question names, at fixed detail.")


def run_ternary(rr):
    slot_ids = [rr.option_ids[0], rr.option_ids[1], letter_ids(rr, "C")]
    opts = (DOG, CAT, EXIT)
    print(f"\n{'variant':11s} {'P(C)':>6s} {'C min':>6s} {'C max':>6s} "
          f"{'P(dog|AB)':>10s} {'mass':>6s}")
    print("-" * 52)
    for key, q in VARIANTS.items():
        pcs, dogs, masses = [], [], []
        for perm in permutations(range(3)):
            # perm[slot] = index into opts, so slot 0 shows opts[perm[0]]
            prompt = TERNARY_TEMPLATE.format(
                q=q, o0=opts[perm[0]], o1=opts[perm[1]], o2=opts[perm[2]])
            slots = dist(rr, prompt, slot_ids)
            by_opt = [0.0, 0.0, 0.0]
            for slot, opt in enumerate(perm):
                by_opt[opt] = slots[slot]
            total = sum(by_opt)
            masses.append(total)
            pcs.append(by_opt[2] / total)
            ab = by_opt[0] + by_opt[1]
            dogs.append(by_opt[0] / ab if ab > 0 else float("nan"))
        pc = sum(pcs) / 6
        print(f"{key:11s} {pc:6.2f} {min(pcs):6.2f} {max(pcs):6.2f} "
              f"{sum(dogs) / 6:10.2f} {min(masses):6.2f}")
    print("\n  P(C) is averaged over all six orderings; C min/max show how")
    print("  much of it is slot position rather than the exit itself.")
    print("  P(dog|AB) drops C and renormalises, so it is comparable to the")
    print("  binary run: if it holds while P(C) is low, the binary reading")
    print("  was not an artefact of having nowhere else to go.")


def main():
    rr = R.Runner("./Llama-3.1-8B-Instruct")
    if "--ternary" in sys.argv:
        run_ternary(rr)
    else:
        run_binary(rr)


if __name__ == "__main__":
    main()
