"""
print_2x2.py — Print a 2x2 contingency table from graded results.

Usage:
  uv run playground/print_2x2.py playground/mem0_hypothesis_results_graded.json
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Print 2x2 table from graded results.")
    parser.add_argument("input", help="Path to graded results JSON")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    valid = [r for r in data if "error" not in r]
    n = len(valid)

    counts = {(False, False): 0, (False, True): 0, (True, False): 0, (True, True): 0}
    for r in valid:
        counts[(r["accepted_essay"], r["wrong_answer"])] += 1

    fooled             = [r for r in valid if r["wrong_answer"]]
    memory_caused      = [r for r in fooled if r.get("memory_caused") is True]
    not_memory_caused  = [r for r in fooled if r.get("memory_caused") is False]
    memory_na          = [r for r in fooled if r.get("memory_caused") is None]

    rows = [
        ("No (pushed back)", False),
        ("Yes (accepted)",   True),
    ]
    row_labels = [f"Essay accepted: {label}" for label, _ in rows]
    LW = max(len(l) for l in row_labels)
    C  = 13

    sep = "-" * (LW + 2 * (C + 2) + 14)

    # ── 2×2 table ────────────────────────────────────────────────────────────
    print(f"\nn = {n}\n")
    print(f"{'':>{LW}}   {'Wrong answer':^{2*C+4}}")
    print(f"{'':>{LW}}   {'No (correct)':^{C}}  {'Yes (fooled)':^{C}}  {'Row total':^9}")
    print(sep)
    for label, accepted in rows:
        row_label = f"Essay accepted: {label}"
        no  = counts[(accepted, False)]
        yes = counts[(accepted, True)]
        row = no + yes
        print(f"{row_label:<{LW}}   {no:^{C}}  {yes:^{C}}  {row:^9}")
    print(sep)
    col_no  = counts[(False, False)] + counts[(True, False)]
    col_yes = counts[(False, True)]  + counts[(True, True)]
    print(f"{'Column total':<{LW}}   {col_no:^{C}}  {col_yes:^{C}}  {n:^9}")

    # ── memory-caused breakdown (only for fooled examples) ───────────────────
    print()
    if fooled:
        fw = len(fooled)
        mc  = len(memory_caused)
        nmc = len(not_memory_caused)
        na  = len(memory_na)
        print(f"Of the {fw} fooled example(s):")
        print(f"  Caused by retrieved memories : {mc:>3}  ({100*mc/fw:.1f}%)")
        print(f"  Not caused by memories       : {nmc:>3}  ({100*nmc/fw:.1f}%)")
        if na:
            print(f"  Memory cause not evaluated   : {na:>3}  ({100*na/fw:.1f}%)")
    else:
        print("No fooled examples — memory-cause analysis not applicable.")
    print()


if __name__ == "__main__":
    main()
