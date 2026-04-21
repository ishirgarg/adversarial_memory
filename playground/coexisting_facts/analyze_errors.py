"""
analyze_errors.py -- Classify root-cause error types in incorrect coexisting-facts traces.

For each incorrect trace from evaluate_mem0_coexisting_facts.py, an LLM judge
determines WHERE in the pipeline the failure occurred:

  coexisting_facts_question (model should have recalled ALL preferences):
    storage_error    -- one or more preferences are absent from ALL_MEMORIES
                        (memory system never stored them correctly)
    retrieval_error  -- all preferences exist in ALL_MEMORIES but some are missing
                        from RETRIEVED_MEMORIES (retrieval failure)
    reasoning_error  -- all preferences appear in RETRIEVED_MEMORIES but model
                        still failed to include all of them in its response

Usage:
  uv run python playground/coexisting_facts/analyze_errors.py
  uv run python playground/coexisting_facts/analyze_errors.py --traces path/to/traces.json
  uv run python playground/coexisting_facts/analyze_errors.py --limit 5
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]

DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_WORKERS = 8

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert evaluator analyzing failures in a memory-augmented AI system.

The system works as follows:
1. Each preference is stated as a short fact in its own isolated chat conversation
   and stored in a memory system, which summarizes them into bullet-point memories.
2. When a question is asked, the memory system retrieves relevant memories and
   shows them to the model.
3. The model should answer using those memories, mentioning ALL of the person's
   preferences.

A prior evaluator has already confirmed this trace is INCORRECT: the model failed
to mention at least one expected preference.

Your job is to classify WHY the failure occurred by checking three conditions IN ORDER.
Stop at the first condition that is true.

STEP 1 -- STORAGE ERROR
Examine ALL_MEMORIES (the complete set of memories stored at the time of the question).
For each expected preference in EXPECTED_PREFERENCES:
  Ask: Is this preference present in ANY memory in ALL_MEMORIES, either explicitly
  or as a clear paraphrase?
If ANY expected preference is missing from ALL_MEMORIES entirely
-> "storage_error". STOP.

STEP 2 -- RETRIEVAL ERROR
All expected preferences exist in ALL_MEMORIES.
For each expected preference:
  Ask: Does it appear in RETRIEVED_MEMORIES (the memories actually shown to the model)?
If any expected preference is in ALL_MEMORIES but NOT in RETRIEVED_MEMORIES
-> "retrieval_error". STOP.

STEP 3 -- REASONING ERROR
All expected preferences exist in ALL_MEMORIES AND appear in RETRIEVED_MEMORIES.
The model was shown every preference but still failed to mention at least one.
-> "reasoning_error".

Return a JSON object with:
  "error_type": one of ["storage_error", "retrieval_error", "reasoning_error"]
  "missing_preference": the first expected preference that triggered the classification
                        (e.g. the one missing from ALL_MEMORIES or RETRIEVED_MEMORIES),
                        or null for reasoning_error
  "reasoning": 1-2 sentence explanation of your classification\
"""

USER_TEMPLATE = """\
QUESTION (that the model answered incorrectly):
{question}

EXPECTED_PREFERENCES (all of these should have appeared in the response):
{ground_truth_answer}

ALL_MEMORIES (complete memory store at time of question):
{all_memories_formatted}

RETRIEVED_MEMORIES (memories actually shown to the model):
{retrieved_memories}

MODEL RESPONSE:
{llm_response}

PRIOR JUDGE REASONING (why this was marked incorrect):
{judge_reasoning}

Classify the error type following the sequential steps in the system instructions.\
"""

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "coexisting_facts_error_classification",
        "schema": {
            "type": "object",
            "properties": {
                "error_type": {
                    "type": "string",
                    "enum": ["storage_error", "retrieval_error", "reasoning_error"],
                },
                "missing_preference": {"type": ["string", "null"]},
                "reasoning": {"type": "string"},
            },
            "required": ["error_type", "missing_preference", "reasoning"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_all_memories(all_memories_at_time: list) -> str:
    if not all_memories_at_time:
        return "(no memories in store)"
    entries = [m if isinstance(m, str) else m["memory"] for m in all_memories_at_time]
    return "\n".join(f"- {m}" for m in entries)


def auto_detect_traces_file(results_dir: Path) -> Path:
    """Find the most recent traces_*.json (not compact) in results_dir."""
    candidates = [
        p for p in results_dir.glob("traces_*.json")
        if not p.name.startswith("traces_compact_")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No traces_*.json files found in {results_dir}. "
            "Run evaluate_mem0_coexisting_facts.py first, or pass --traces explicitly."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def classify_trace(client: OpenAI, model: str, trace: dict) -> dict:
    """Classify a single incorrect trace. Never raises — errors are recorded inline."""
    base = {k: v for k, v in trace.items() if k != "all_memories_at_time"}
    base["error_type"] = None
    base["missing_preference"] = None
    base["analysis_reasoning"] = None
    base["analysis_error"] = None

    try:
        all_memories_formatted = format_all_memories(trace.get("all_memories_at_time", []))
        retrieved = trace.get("retrieved_memories") or "(none)"

        user_content = USER_TEMPLATE.format(
            question=trace["question"],
            ground_truth_answer=trace.get("ground_truth_answer", ""),
            all_memories_formatted=all_memories_formatted,
            retrieved_memories=retrieved,
            llm_response=trace["llm_response"],
            judge_reasoning=trace["judge_reasoning"],
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format=SCHEMA,
            temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
        base["error_type"] = data["error_type"]
        base["missing_preference"] = data.get("missing_preference")
        base["analysis_reasoning"] = data["reasoning"]

    except Exception as exc:
        base["error_type"] = "analysis_failed"
        base["analysis_error"] = str(exc)

    return base


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_summary_table(results: list) -> None:
    error_types = ["storage_error", "retrieval_error", "reasoning_error", "analysis_failed"]

    counts: Dict[str, int] = {et: 0 for et in error_types}
    for r in results:
        et = r.get("error_type", "analysis_failed")
        counts[et] = counts.get(et, 0) + 1

    total = len(results)
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS SUMMARY  (incorrect traces only)")
    print("=" * 60)
    print(f"{'Error Type':<22} {'Count':>8} {'Share':>8}")
    print("-" * 60)
    for et in error_types:
        count = counts[et]
        if count == 0:
            continue
        share = count / total if total else 0.0
        print(f"{et:<22} {count:>8} {share:>7.1%}")
    print("-" * 60)
    print(f"{'TOTAL':<22} {total:>8}")
    print("=" * 60)

    # Breakdown by preference count
    pref_counts: Dict[int, Dict[str, int]] = {}
    for r in results:
        n = r.get("num_preferences", 0)
        et = r.get("error_type", "analysis_failed")
        if n not in pref_counts:
            pref_counts[n] = {}
        pref_counts[n][et] = pref_counts[n].get(et, 0) + 1

    if pref_counts:
        print("\nError breakdown by preference count:")
        print(f"{'Pref count':<12} {'storage':>9} {'retrieval':>10} {'reasoning':>10} {'failed':>8} {'total':>7}")
        print("-" * 62)
        for n in sorted(pref_counts):
            ec = pref_counts[n]
            row_total = sum(ec.values())
            print(
                f"{n:<12}"
                f" {ec.get('storage_error', 0):>9}"
                f" {ec.get('retrieval_error', 0):>10}"
                f" {ec.get('reasoning_error', 0):>10}"
                f" {ec.get('analysis_failed', 0):>8}"
                f" {row_total:>7}"
            )
        print("=" * 62)

    # Category breakdown
    cat_counts: Dict[str, Dict[str, int]] = {}
    for r in results:
        cat = r.get("preference_category", "unknown")
        et = r.get("error_type", "analysis_failed")
        if cat not in cat_counts:
            cat_counts[cat] = {}
        cat_counts[cat][et] = cat_counts[cat].get(et, 0) + 1

    if cat_counts:
        print("\nError breakdown by category:")
        print(f"{'Category':<28} {'storage':>9} {'retrieval':>10} {'reasoning':>10} {'failed':>8}")
        print("-" * 70)
        for cat in sorted(cat_counts):
            ec = cat_counts[cat]
            print(
                f"{cat:<28}"
                f" {ec.get('storage_error', 0):>9}"
                f" {ec.get('retrieval_error', 0):>10}"
                f" {ec.get('reasoning_error', 0):>10}"
                f" {ec.get('analysis_failed', 0):>8}"
            )
        print("=" * 70)


def save_outputs(results: list, output_dir: Path, ts: str):
    json_path = output_dir / f"error_analysis_{ts}.json"
    csv_path = output_dir / f"error_analysis_{ts}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_conv_id",
                "question_type",
                "error_type",
                "num_preferences",
                "preferences_retrieved_fraction",
                "preference_category",
                "ground_truth_answer",
                "missing_preference",
                "question",
                "analysis_reasoning",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "question_conv_id": r.get("question_conv_id", ""),
                "question_type": r.get("question_type", ""),
                "error_type": r.get("error_type", ""),
                "num_preferences": r.get("num_preferences", ""),
                "preferences_retrieved_fraction": r.get("preferences_retrieved_fraction", ""),
                "preference_category": r.get("preference_category", ""),
                "ground_truth_answer": r.get("ground_truth_answer", ""),
                "missing_preference": r.get("missing_preference", ""),
                "question": r.get("question", ""),
                "analysis_reasoning": r.get("analysis_reasoning", ""),
            })

    return json_path, csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify root-cause error types in incorrect coexisting-facts traces."
    )
    parser.add_argument(
        "--traces",
        default=None,
        help="Path to traces JSON (full, not compact). Auto-detects most recent if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to same directory as the traces file.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model for classification.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only classify first N incorrect traces (for testing).",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: set OPENAI_KEY or OPENAI_API_KEY environment variable.")

    # Resolve traces file
    if args.traces:
        traces_path = Path(args.traces)
        if not traces_path.is_absolute():
            traces_path = PROJECT_ROOT / traces_path
    else:
        traces_path = auto_detect_traces_file(DEFAULT_RESULTS_DIR)
    print(f"Loading traces from: {traces_path}")

    with open(traces_path, encoding="utf-8") as f:
        all_traces = json.load(f)

    incorrect = [t for t in all_traces if t.get("judge_result") == "incorrect"]
    print(f"Total traces: {len(all_traces)}, Incorrect: {len(incorrect)}")

    if not incorrect:
        print("No incorrect traces found. Nothing to analyze.")
        sys.exit(0)

    if args.limit:
        incorrect = incorrect[: args.limit]
        print(f"Limiting to first {args.limit} incorrect traces.")

    output_dir = Path(args.output_dir) if args.output_dir else traces_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: List[Optional[dict]] = [None] * len(incorrect)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(classify_trace, client, args.model, trace): i
            for i, trace in enumerate(incorrect)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Classifying errors"):
            i = futures[future]
            results[i] = future.result()

    json_path, csv_path = save_outputs(results, output_dir, ts)
    print_summary_table(results)
    print(f"\nResults saved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main()
