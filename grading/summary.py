"""
summary.py — recompute alignment metrics from grades.json without the UI.

Useful for re-running the summary headlessly after grading is complete,
or for running it on a different manifest/grades file.

Usage
-----
  uv run python grading/summary.py
  uv run python grading/summary.py --manifest ... --grades ... --output ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from grading_ui import compute_summary

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "sample_manifest.json"
DEFAULT_GRADES = SCRIPT_DIR / "grades.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "summary.json"


def _print_human(summary: dict) -> None:
    fl = summary["fact_level"]
    tl = summary["trace_level"]
    print("=" * 70)
    print(f"Manual-grading alignment summary  (seed={summary['manifest_seed']}, "
          f"graded={summary['n_graded']}/{summary['manifest_n']})")
    print("=" * 70)

    if fl["agreement_rate"] is not None:
        print(f"Fact-level agreement:        "
              f"{fl['agreement']}/{fl['total']} = {fl['agreement_rate']:.1%}")
    else:
        print("Fact-level agreement:        (no data)")
    if tl["error_type_agreement_rate"] is not None:
        print(f"Trace error-type agreement:  "
              f"{tl['error_type_agreement']}/{tl['total']} = "
              f"{tl['error_type_agreement_rate']:.1%}")
    if tl["judge_result_agreement_rate"] is not None:
        print(f"Trace correct/incorrect agr: "
              f"{tl['judge_result_agreement']}/{tl['total']} = "
              f"{tl['judge_result_agreement_rate']:.1%}")

    print("\nFact confusion matrix (rows=judge, cols=user):")
    cats = list(fl["confusion_matrix_judge_to_user"].keys())
    print("  " + " ".join(f"{c:>14}" for c in [""] + cats))
    for j in cats:
        row = fl["confusion_matrix_judge_to_user"][j]
        cells = [f"{row.get(u, 0):>14}" for u in cats]
        print(f"  {j:>14} " + " ".join(cells))

    print("\nBy memory system:")
    for ms, v in summary["by_memory_system"].items():
        fa = v["fact_agreement_rate"]
        ta = v["trace_agreement_rate"]
        print(f"  {ms:>10}  trace={v['trace_agree_error_type']}/{v['trace_total']} "
              f"({ta:.1%})  fact={v['fact_agree']}/{v['fact_total']} "
              f"({fa:.1%})" if fa is not None and ta is not None else f"  {ms:>10}  (no data)")

    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--grades", type=Path, default=DEFAULT_GRADES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}. "
                         "Run grading/build_sample.py first.")
    if not args.grades.exists():
        raise SystemExit(f"Grades not found: {args.grades}. "
                         "Grade some samples in the UI first.")

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(args.grades, encoding="utf-8") as f:
        grades = json.load(f)

    summary = compute_summary(manifest, grades)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _print_human(summary)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
