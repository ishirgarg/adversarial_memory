"""
analyze_errors.py -- Grade and classify errors in conditional-facts traces in one LLM call.

For each trace from evaluate_conditional_facts.py, a single judge LLM call:
  1. Grades whether the model responded correctly (correct / incorrect)
  2. Checks whether the entity's conditional fact appeared in retrieved memories
  3. If incorrect, classifies the root cause:
       storage_error   -- conditional fact absent from ALL_MEMORIES
       retrieval_error -- conditional fact in ALL_MEMORIES but not in RETRIEVED_MEMORIES
       reasoning_error -- conditional fact in RETRIEVED_MEMORIES but model still failed

Usage:
  uv run python playground/conditional_facts/analyze_errors.py --traces path/to/traces.json
  uv run python playground/conditional_facts/analyze_errors.py  # auto-detects most recent
  uv run python playground/conditional_facts/analyze_errors.py --limit 5
"""

import argparse
import csv
import json
import os
import sys
import time
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
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_WORKERS = 8
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Combined prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert evaluator analyzing a memory-augmented AI system on conditional-fact questions.

The system works as follows:
1. Facts about entities are stored in memory. Each entity has a CONDITIONAL behavior — they do
   something only under a specific qualifying condition.
2. When a question is asked, the memory system retrieves relevant memories (RETRIEVED_MEMORIES)
   from its full store (ALL_MEMORIES) and shows them to the model.
3. The model must answer YES when the condition is met and NO (citing the condition) when it is not.

You will perform three tasks in a single pass:

TASK 1 — GRADE THE RESPONSE
Determine whether the model answered correctly given the ground truth.
  "correct"   — the model's answer aligns with the ground truth:
                 - Condition IS met: model clearly affirms the behavior
                 - Condition is NOT met: model declines AND cites the condition as the reason
  "incorrect" — wrong yes/no, right answer for wrong reason, or "I don't know / no memory"

TASK 2 — CHECK RETRIEVAL
Did RETRIEVED_MEMORIES contain the entity's conditional fact (entity + qualifying condition,
even if paraphrased)?
  true  — the conditional fact is present in retrieved memories
  false — it is absent

TASK 3 — CLASSIFY ERROR (only if grade == "incorrect")
Check these conditions IN ORDER and stop at the first that is true:

  STEP 1: Is the conditional fact (entity + qualifying condition) absent from ALL_MEMORIES
          entirely (including cases where the entity appears but the condition was lost)?
          -> "storage_error"

  STEP 2: The conditional fact exists in ALL_MEMORIES but is absent from RETRIEVED_MEMORIES.
          -> "retrieval_error"

  STEP 3: The conditional fact appears in both ALL_MEMORIES and RETRIEVED_MEMORIES, yet the
          model still failed to apply it correctly.
          -> "reasoning_error"

If grade == "correct", set error_type and missing_element to null.

Return a single JSON object with these fields:
  "judge_result":          "correct" or "incorrect"
  "judge_reasoning":       1-2 sentences explaining the grade
  "entity_facts_in_retrieved": true or false
  "error_type":            one of ["storage_error", "retrieval_error", "reasoning_error"] or null
  "missing_element":       brief description of what is missing/wrong, or null
  "analysis_reasoning":    1-2 sentence explanation of the error classification, or null\
"""

USER_TEMPLATE = """\
ENTITY: {entity}
CONDITION (when the behavior occurs): {condition}
CONDITION MET IN QUESTION? {condition_met}

QUESTION:
{question}

GROUND TRUTH ANSWER:
{ground_truth_answer}

ALL_MEMORIES (complete memory store at time of question):
{all_memories_formatted}

RETRIEVED_MEMORIES (memories actually shown to the model):
{retrieved_memories}

MODEL RESPONSE:
{llm_response}\
"""

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "conditional_facts_analysis",
        "schema": {
            "type": "object",
            "properties": {
                "judge_result": {
                    "type": "string",
                    "enum": ["correct", "incorrect"],
                },
                "judge_reasoning": {"type": "string"},
                "entity_facts_in_retrieved": {"type": "boolean"},
                "error_type": {
                    "type": ["string", "null"],
                    "enum": ["storage_error", "retrieval_error", "reasoning_error", None],
                },
                "missing_element": {"type": ["string", "null"]},
                "analysis_reasoning": {"type": ["string", "null"]},
            },
            "required": [
                "judge_result",
                "judge_reasoning",
                "entity_facts_in_retrieved",
                "error_type",
                "missing_element",
                "analysis_reasoning",
            ],
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
    entries = [m if isinstance(m, str) else m.get("memory", str(m)) for m in all_memories_at_time]
    return "\n".join(f"- {m}" for m in entries)


def auto_detect_traces_file(results_dir: Path) -> Path:
    candidates = list(results_dir.glob("traces_*.json")) + list(results_dir.glob("*/traces_*.json"))
    candidates = [p for p in candidates if not p.name.startswith("traces_compact_")]
    if not candidates:
        raise FileNotFoundError(
            f"No traces_*.json files found under {results_dir}. "
            "Run evaluate_conditional_facts.py first, or pass --traces explicitly."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Combined grade + classify
# ---------------------------------------------------------------------------


def analyze_trace(client: OpenAI, model: str, trace: dict) -> dict:
    """Grade and classify a single trace in one LLM call. Never raises."""
    result = {k: v for k, v in trace.items() if k != "all_memories_at_time"}
    result["judge_result"] = None
    result["judge_reasoning"] = None
    result["entity_facts_in_retrieved"] = None
    result["error_type"] = None
    result["missing_element"] = None
    result["analysis_reasoning"] = None
    result["analysis_error"] = None

    try:
        all_memories_formatted = format_all_memories(trace.get("all_memories_at_time", []))
        retrieved = trace.get("retrieved_memories") or "(none)"

        user_content = USER_TEMPLATE.format(
            entity=trace.get("entity", ""),
            condition=trace.get("condition", ""),
            condition_met=trace.get("condition_met", ""),
            question=trace["question"],
            ground_truth_answer=trace.get("ground_truth_answer", ""),
            all_memories_formatted=all_memories_formatted,
            retrieved_memories=retrieved,
            llm_response=trace["llm_response"],
        )

        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    response_format=SCHEMA,
                )
                data = json.loads(resp.choices[0].message.content)
                result["judge_result"] = data["judge_result"]
                result["judge_reasoning"] = data["judge_reasoning"]
                result["entity_facts_in_retrieved"] = data["entity_facts_in_retrieved"]
                result["error_type"] = data.get("error_type")
                result["missing_element"] = data.get("missing_element")
                result["analysis_reasoning"] = data.get("analysis_reasoning")
                return result
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    time.sleep(0.4 * attempt)

        result["analysis_error"] = str(last_error)

    except Exception as exc:
        result["analysis_error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_summary(results: list) -> None:
    total = len(results)
    correct = sum(1 for r in results if r.get("judge_result") == "correct")
    incorrect = total - correct

    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total traces:   {total}")
    print(f"Correct:        {correct}  ({correct/total:.1%})" if total else "Correct: 0")
    print(f"Incorrect:      {incorrect}  ({incorrect/total:.1%})" if total else "Incorrect: 0")

    retrieval_hits = sum(1 for r in results if r.get("entity_facts_in_retrieved"))
    print(f"Retrieval hit rate: {retrieval_hits/total:.1%}" if total else "Retrieval hit rate: N/A")

    # Error type breakdown (incorrect only)
    error_types = ["storage_error", "retrieval_error", "reasoning_error"]
    counts: Dict[str, int] = {et: 0 for et in error_types}
    counts["analysis_failed"] = 0
    for r in results:
        if r.get("judge_result") != "incorrect":
            continue
        et = r.get("error_type") or "analysis_failed"
        counts[et] = counts.get(et, 0) + 1

    if incorrect > 0:
        print(f"\n{'Error type (of incorrect)':<26} {'Count':>6} {'Share of incorrect':>18}")
        print("-" * 52)
        for et in [*error_types, "analysis_failed"]:
            c = counts[et]
            if c == 0:
                continue
            print(f"{et:<26} {c:>6} {c/incorrect:>17.1%}")
        print("-" * 52)
        print(f"{'TOTAL INCORRECT':<26} {incorrect:>6}")

    print("=" * 60)


def save_outputs(results: list, output_dir: Path, ts: str):
    json_path = output_dir / f"analysis_{ts}.json"
    csv_path = output_dir / f"analysis_{ts}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fieldnames = [
        "question_conv_id", "judge_result", "entity_facts_in_retrieved",
        "error_type", "condition_met", "condition_type", "entity", "condition",
        "ground_truth_answer", "missing_element", "question",
        "judge_reasoning", "analysis_reasoning",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    return json_path, csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade and classify errors in conditional-facts traces."
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
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model for grading and classification.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only analyze first N traces (for testing).",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: set OPENAI_KEY or OPENAI_API_KEY environment variable.")

    if args.traces:
        traces_path = Path(args.traces)
        if not traces_path.is_absolute():
            traces_path = PROJECT_ROOT / traces_path
    else:
        traces_path = auto_detect_traces_file(DEFAULT_RESULTS_DIR)
    print(f"Loading traces from: {traces_path}")

    with open(traces_path, encoding="utf-8") as f:
        all_traces = json.load(f)

    print(f"Total traces: {len(all_traces)}")

    traces_to_analyze = all_traces
    if args.limit:
        traces_to_analyze = traces_to_analyze[: args.limit]
        print(f"Limiting to first {args.limit} traces.")

    output_dir = Path(args.output_dir) if args.output_dir else traces_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: List[Optional[dict]] = [None] * len(traces_to_analyze)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(analyze_trace, client, args.model, trace): i
            for i, trace in enumerate(traces_to_analyze)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing traces"):
            i = futures[future]
            results[i] = future.result()

    json_path, csv_path = save_outputs(results, output_dir, ts)
    print_summary(results)
    print(f"\nResults saved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main()
