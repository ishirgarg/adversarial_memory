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

    W = 16  # label column width
    C = 13  # data column width

    rows = [
        ("No (pushed back)", False),
        ("Yes (accepted)",   True),
    ]
    row_labels = [f"Essay accepted: {label}" for label, _ in rows]
    LW = max(len(l) for l in row_labels)  # dynamic label width

    sep = "-" * (LW + 2 * (C + 2) + 14)

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
    print()


if __name__ == "__main__":
    main()
