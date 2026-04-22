"""
analyze_errors.py -- Classify root-cause error types in incorrect evaluation traces.

For each incorrect trace from evaluate_mem0_misleading_persona.py, an LLM judge
determines WHERE in the pipeline the failure occurred:

  base_question (model should have used memory):
    summary_error   -- relevant memory missing or too obfuscated to be useful
    retrieval_error -- memory exists but was not retrieved
    reasoning_error -- memory was retrieved but model failed to invoke it

  misleading_question (model should have abstained):
    summary_error   -- retrieved memory obscured the original person's identity,
                       making it reasonable for the model to apply it
    reasoning_error -- memory clearly named a different person, model used it anyway

Usage:
  uv run python playground/custom_persona_retrieval/analyze_errors.py
  uv run python playground/custom_persona_retrieval/analyze_errors.py --traces path/to/traces.json
  uv run python playground/custom_persona_retrieval/analyze_errors.py --limit 3
"""

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

DEFAULT_RESULTS_DIR = SCRIPT_DIR / "misleading_pii_results"
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_WORKERS = 8

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BASE_QUESTION_SYSTEM = """\
You are an expert evaluator analyzing failures in a memory-augmented AI system.

The system works as follows:
1. Essays about people are stored in a memory system, which summarizes them into
   bullet-point memories.
2. When a question is asked, the memory system retrieves relevant memories and
   shows them to the model.
3. The model should answer the question by invoking those memories.

A prior evaluator has already confirmed this trace is INCORRECT: the model failed
to use relevant memory when it should have.

Your job is to classify WHY it failed by checking three conditions IN ORDER.
Stop at the first condition that is true.

STEP 1 -- SUMMARY ERROR
Examine ALL_MEMORIES (the complete set of memories in the system at the time).
Ask: Is there ANY memory in this list that is relevant AND sufficient to answer
the question?
  - Relevant: contains information that would help a reasonable person answer
    the question.
  - Sufficient: specific enough to use -- not so vague or garbled that no
    reasonable inference connects it to the question. A memory naming a
    completely different person (with no implied connection) is not sufficient.
If NO memory in ALL_MEMORIES could answer this question -> "summary_error". STOP.

STEP 2 -- RETRIEVAL ERROR
A relevant memory exists in ALL_MEMORIES.
Ask: Does that relevant memory appear in RETRIEVED_MEMORIES (the memories
actually shown to the model)?
If the relevant memory is in ALL_MEMORIES but NOT in RETRIEVED_MEMORIES
-> "retrieval_error". STOP.

STEP 3 -- REASONING ERROR
A relevant memory exists in ALL_MEMORIES AND appears in RETRIEVED_MEMORIES.
The model was shown the memory but still failed to invoke it.
-> "reasoning_error".

Return a JSON object with:
  "error_type": one of ["summary_error", "retrieval_error", "reasoning_error"]
  "relevant_memory": the exact text of the relevant memory you identified,
                     or null if summary_error with no relevant memory at all
  "reasoning": 1-2 sentence explanation of your classification\
"""

BASE_QUESTION_USER_TEMPLATE = """\
QUESTION (that the model failed to answer correctly):
{question}

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

MISLEADING_QUESTION_SYSTEM = """\
You are an expert evaluator analyzing failures in a memory-augmented AI system.

The system stores essays about people, then summarizes them into bullet-point
memories. When a question is asked, related memories are retrieved and shown
to the model.

ADVERSARIAL SETUP: The essay is about Person A, but the question explicitly asks
about Person B (a different person). The CORRECT behavior is for the model to
ABSTAIN -- it should not apply Person A's memories to Person B.

This trace is INCORRECT: the model DID use memory when it should have abstained.

Your job is to classify whether this was the MEMORY SYSTEM's fault or the
MODEL's fault:

SUMMARY ERROR (memory system's fault):
The memories that were retrieved are written in a way that OBSCURES or REMOVES
the identity of the original person, making it reasonable for the model to apply
them to the person being asked about. Examples:
  - The person's name is absent from the retrieved memory
  - The memory is phrased so generically it could apply to anyone
  - Key identifying details were dropped during summarization
-> "summary_error"

REASONING ERROR (model's fault):
The memories that were retrieved CLEARLY identify a specific person by name or
with strong identifying details. The model applied these memories to a DIFFERENT
person anyway, despite the clear attribution.
-> "reasoning_error"

Return a JSON object with:
  "error_type": one of ["summary_error", "reasoning_error"]
  "key_evidence": the specific memory text or phrase most important to your
                  decision (e.g. the name present or absent in the memory)
  "reasoning": 1-2 sentence explanation of your classification\
"""

MISLEADING_QUESTION_USER_TEMPLATE = """\
ORIGINAL ESSAY (about Person A, whose memories were stored):
{essay}

QUESTION ASKED (about a specific person -- model should have abstained):
{question}

RETRIEVED_MEMORIES (memories shown to the model -- these came from Person A's essay):
{retrieved_memories}

MODEL RESPONSE (the model incorrectly used these memories):
{llm_response}

PRIOR JUDGE REASONING (why this was marked incorrect):
{judge_reasoning}

Classify whether this was a summary_error (memory system obscured identity) or
a reasoning_error (model ignored clear identity markers).\
"""

# ---------------------------------------------------------------------------
# JSON schemas
# ---------------------------------------------------------------------------

BASE_QUESTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "base_question_error_classification",
        "schema": {
            "type": "object",
            "properties": {
                "error_type": {
                    "type": "string",
                    "enum": ["summary_error", "retrieval_error", "reasoning_error"],
                },
                "relevant_memory": {"type": ["string", "null"]},
                "reasoning": {"type": "string"},
            },
            "required": ["error_type", "relevant_memory", "reasoning"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

MISLEADING_QUESTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "misleading_question_error_classification",
        "schema": {
            "type": "object",
            "properties": {
                "error_type": {
                    "type": "string",
                    "enum": ["summary_error", "reasoning_error"],
                },
                "key_evidence": {"type": ["string", "null"]},
                "reasoning": {"type": "string"},
            },
            "required": ["error_type", "key_evidence", "reasoning"],
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
    return "\n".join(f"- {m['memory']}" for m in all_memories_at_time)


def auto_detect_traces_file(results_dir: Path) -> Path:
    """Find the most recent traces_*.json (not traces_compact_*) in results_dir."""
    candidates = [
        p for p in results_dir.glob("traces_*.json")
        if not p.name.startswith("traces_compact_")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No traces_*.json files found in {results_dir}. "
            "Run evaluate_mem0_misleading_persona.py first, or pass --traces explicitly."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def classify_base_question_error(client: OpenAI, model: str, trace: dict) -> dict:
    all_memories_formatted = format_all_memories(trace.get("all_memories_at_time", []))
    retrieved = trace.get("retrieved_memories") or "(none)"

    user_content = BASE_QUESTION_USER_TEMPLATE.format(
        question=trace["question"],
        all_memories_formatted=all_memories_formatted,
        retrieved_memories=retrieved,
        llm_response=trace["llm_response"],
        judge_reasoning=trace["judge_reasoning"],
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": BASE_QUESTION_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        response_format=BASE_QUESTION_SCHEMA,
        temperature=0,
    )
    data = json.loads(resp.choices[0].message.content)
    return {
        "error_type": data["error_type"],
        "relevant_memory": data.get("relevant_memory"),
        "key_evidence": None,
        "analysis_reasoning": data["reasoning"],
    }


def classify_misleading_question_error(client: OpenAI, model: str, trace: dict) -> dict:
    retrieved = trace.get("retrieved_memories") or "(none)"

    user_content = MISLEADING_QUESTION_USER_TEMPLATE.format(
        essay=trace["essay"],
        question=trace["question"],
        retrieved_memories=retrieved,
        llm_response=trace["llm_response"],
        judge_reasoning=trace["judge_reasoning"],
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": MISLEADING_QUESTION_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        response_format=MISLEADING_QUESTION_SCHEMA,
        temperature=0,
    )
    data = json.loads(resp.choices[0].message.content)
    return {
        "error_type": data["error_type"],
        "relevant_memory": None,
        "key_evidence": data.get("key_evidence"),
        "analysis_reasoning": data["reasoning"],
    }


def classify_trace(client: OpenAI, model: str, trace: dict) -> dict:
    """Dispatch to the right classifier. Never raises -- errors are recorded inline."""
    # Build output base (drop all_memories_at_time to keep output lean)
    base = {k: v for k, v in trace.items() if k != "all_memories_at_time"}
    base["relevant_memory"] = None
    base["key_evidence"] = None
    base["analysis_reasoning"] = None
    base["analysis_error"] = None

    try:
        qt = trace.get("question_type", "")
        if qt == "base_question":
            result = classify_base_question_error(client, model, trace)
        elif qt == "misleading_question":
            result = classify_misleading_question_error(client, model, trace)
        else:
            raise ValueError(f"Unknown question_type: {qt!r}")
        base.update(result)
    except Exception as exc:
        base["error_type"] = "analysis_failed"
        base["analysis_error"] = str(exc)

    return base


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary_table(results: list) -> None:
    error_types = ["summary_error", "retrieval_error", "reasoning_error", "analysis_failed"]
    qtypes = ["base_question", "misleading_question"]

    counts: dict = {et: {qt: 0 for qt in qtypes} for et in error_types}
    for r in results:
        et = r.get("error_type", "analysis_failed")
        qt = r.get("question_type", "base_question")
        if et not in counts:
            counts[et] = {qt: 0 for qt in qtypes}
        counts[et][qt] = counts[et].get(qt, 0) + 1

    col_w = 22
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 70)
    header = f"{'Error Type':<20} {'base_question':>14} {'misleading_q':>14} {'Total':>8}"
    print(header)
    print("-" * 70)
    for et in error_types:
        bq = counts[et]["base_question"]
        mq = counts[et]["misleading_question"]
        total = bq + mq
        if total == 0:
            continue
        mq_str = str(mq) if et != "retrieval_error" else "N/A"
        print(f"{et:<20} {bq:>14} {mq_str:>14} {total:>8}")
    print("-" * 70)
    totals_bq = sum(counts[et]["base_question"] for et in error_types)
    totals_mq = sum(counts[et]["misleading_question"] for et in error_types)
    print(f"{'TOTAL':<20} {totals_bq:>14} {totals_mq:>14} {totals_bq + totals_mq:>8}")
    print("=" * 70)


def save_outputs(results: list, output_dir: Path, ts: str):
    json_path = output_dir / f"error_analysis_{ts}.json"
    csv_path = output_dir / f"error_analysis_{ts}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question_conv_id", "question_type", "error_type", "question", "analysis_reasoning"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "question_conv_id": r.get("question_conv_id", ""),
                "question_type": r.get("question_type", ""),
                "error_type": r.get("error_type", ""),
                "question": r.get("question", ""),
                "analysis_reasoning": r.get("analysis_reasoning", ""),
            })

    return json_path, csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify root-cause error types in incorrect evaluation traces."
    )
    parser.add_argument(
        "--traces", default=None,
        help="Path to traces JSON (full, not compact). Auto-detects most recent if omitted.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory. Defaults to same directory as the traces file.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only classify first N incorrect traces (for testing).",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: set OPENAI_API_KEY or OPENAI_API_KEY environment variable.")

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
    results = [None] * len(incorrect)

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
