"""
analyze_errors.py -- Grade and classify errors in conditional-facts traces.

Reads the output of evaluate_conditional_facts.py and grades each graded question
using four focused LLM calls in sequence, short-circuiting on the first failure:

  Call 1 — Storage check:  is the fact present in ALL_MEMORIES at all?
  Call 2 — Summary check:  is the stored version faithful (condition not dropped/corrupted)?
  Call 3 — Retrieval check: was the fact included in RETRIEVED_MEMORIES?
  Call 4 — Invocation check: did the model correctly use the fact to answer?

Error taxonomy:
  storage_error   — fact absent from ALL_MEMORIES
  summary_error   — fact present but critical condition destroyed in summarization
  retrieval_error — fact stored faithfully but not retrieved
  reasoning_error — fact retrieved faithfully but model answered incorrectly

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
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm import compute_cost  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_WORKERS = 8
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Call 1 — Storage check
# ---------------------------------------------------------------------------

STORAGE_SYSTEM = """You are checking whether a specific fact was stored in a memory system.
The fact may have been paraphrased or compressed, but must still convey the same
information — including any qualifying condition — to count as present."""

STORAGE_USER = """ORIGINAL FACT:
{original_fact}

ALL_MEMORIES (complete memory store):
{all_memories_formatted}

Is the original fact present in ALL_MEMORIES, even if paraphrased, as long as the
qualifying condition is preserved and the meaning is not altered?"""

STORAGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "storage_check",
        "schema": {
            "type": "object",
            "properties": {
                "fact_in_store": {"type": "boolean"},
                "storage_reasoning": {"type": "string"},
            },
            "required": ["fact_in_store", "storage_reasoning"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

# ---------------------------------------------------------------------------
# Call 2 — Summary check
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM = """You are assessing the quality of a stored memory.

The fact has already been confirmed to exist in the memory store. Your job is to evaluate
whether the stored version faithfully preserves the CRITICAL information — specifically the
qualifying condition and the conditional behavior — in a way that would support correct
downstream reasoning.

A stored version has a SUMMARY ERROR if ANY of the following apply:
- The qualifying condition was dropped entirely (stored as an unconditional fact)
- The condition was generalized in a way that changes the specific threshold or trigger
  (e.g., "after 5pm" → "in the evening" loses precision; "when raining" → "in bad weather" is too vague)
- The conditional relationship was inverted, confused, or made ambiguous
- Critical specifics (time, place, context, trigger) were lost or distorted such that
  a reader could not reliably determine whether a given scenario satisfies the rule

A stored version is FAITHFUL if the condition is clearly and specifically preserved and
a reader could correctly answer whether a given context satisfies the rule."""

SUMMARY_USER = """ORIGINAL FACT:
{original_fact}

ALL_MEMORIES (the fact IS confirmed present somewhere in here):
{all_memories_formatted}

Find the memory entry corresponding to this fact and assess whether the stored version
faithfully preserves the qualifying condition and behavior, or whether it has a summary error."""

SUMMARY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "summary_check",
        "schema": {
            "type": "object",
            "properties": {
                "summary_check_passed": {"type": "boolean"},
                "summary_reasoning": {"type": "string"},
            },
            "required": ["summary_check_passed", "summary_reasoning"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

# ---------------------------------------------------------------------------
# Call 3 — Retrieval check
# ---------------------------------------------------------------------------

RETRIEVAL_SYSTEM = """You are checking whether a specific fact was included in the memories retrieved and
shown to an AI model when it answered a question."""

RETRIEVAL_USER = """ORIGINAL QUESTION (used to search the memory store):
{query}

ORIGINAL FACT:
{original_fact}

RETRIEVED_MEMORIES (shown to the model):
{retrieved_memories}

Was the original fact included in RETRIEVED_MEMORIES? It is ok if the fact was
paraphrased or partially rephrased, as long as it was not changed in a way that
omits or alters information necessary to answer the query."""

RETRIEVAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "retrieval_check",
        "schema": {
            "type": "object",
            "properties": {
                "entity_facts_in_retrieved": {"type": "boolean"},
                "retrieval_reasoning": {"type": "string"},
            },
            "required": ["entity_facts_in_retrieved", "retrieval_reasoning"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

# ---------------------------------------------------------------------------
# Call 4 — Invocation check
# ---------------------------------------------------------------------------

INVOCATION_SYSTEM = """You are checking whether an AI model correctly used a retrieved memory to answer
a conditional-fact question.

The model was shown the relevant memory and must answer YES if the condition is met,
or NO while citing the condition if it is not met."""

INVOCATION_USER = """ORIGINAL FACT (was present in retrieved memories):
{original_fact}

GROUND TRUTH ANSWER:
{ground_truth_answer}

MODEL RESPONSE:
{llm_response}

Did the model correctly invoke the memory and arrive at the right answer?"""

INVOCATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "invocation_check",
        "schema": {
            "type": "object",
            "properties": {
                "correctly_invoked": {"type": "boolean"},
                "invocation_reasoning": {"type": "string"},
            },
            "required": ["correctly_invoked", "invocation_reasoning"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_all_memories(all_memories: list) -> str:
    if not all_memories:
        return "(no memories in store)"
    entries = [m if isinstance(m, str) else m.get("memory", str(m)) for m in all_memories]
    return "\n".join(f"- {m}" for m in entries)


def auto_detect_traces_file(results_dir: Path) -> Path:
    graded = list(results_dir.glob("graded_traces_*.json")) + list(results_dir.glob("*/graded_traces_*.json"))
    if graded:
        return max(graded, key=lambda p: p.stat().st_mtime)
    candidates = list(results_dir.glob("traces_*.json")) + list(results_dir.glob("*/traces_*.json"))
    candidates = [p for p in candidates if not p.name.startswith("traces_compact_")]
    if not candidates:
        raise FileNotFoundError(
            f"No graded_traces_*.json or traces_*.json files found under {results_dir}. "
            "Run evaluate_conditional_facts.py first, or pass --traces explicitly."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_graded_traces(data: dict) -> Tuple[List[dict], List[dict]]:
    num_storage_convs = data["run_metadata"]["num_storage_convs"]
    dataset_rows = data["dataset_rows"]
    results = data["evaluation_summary"]["results"]
    question_results = results[num_storage_convs:]

    graded = []
    for i, (result, row) in enumerate(zip(question_results, dataset_rows)):
        traces = result["traces"]
        if not traces:
            continue
        trace = traces[0]
        graded.append({
            "entity": row["entity"],
            "entity_category": row["entity_category"],
            "behavior": row["behavior"],
            "condition_type": row["condition_type"],
            "condition": row["condition"],
            "entity_facts": row["entity_facts"],
            "question_context": row["question_context"],
            "condition_met": row["condition_met"],
            "ground_truth_answer": row["ground_truth_answer"],
            "question": trace["query"],
            "retrieved_memories": trace["retrieved_memories"],
            "llm_response": trace["response"],
            "formatted_prompt": trace["formatted_prompt"],
            "conversation_id": result["conversation_id"],
            "eval_input_tokens": trace["input_tokens"],
            "eval_output_tokens": trace["output_tokens"],
            "eval_cost": trace["cost"],
        })

    return graded, dataset_rows


# ---------------------------------------------------------------------------
# Judge calls
# ---------------------------------------------------------------------------


def _call(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    schema: dict,
) -> Tuple[dict, int, int]:
    """Make one structured judge call with retries. Returns (data, in_tok, out_tok)."""
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format=schema,
            )
            data = json.loads(resp.choices[0].message.content)
            in_tok = resp.usage.prompt_tokens if resp.usage else 0
            out_tok = resp.usage.completion_tokens if resp.usage else 0
            return data, in_tok, out_tok
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(0.4 * attempt)
    raise RuntimeError(f"Judge call failed after {MAX_RETRIES} attempts: {last_error}") from last_error


def analyze_trace(
    client: OpenAI,
    model: str,
    trace: dict,
    all_memories: list,
) -> Tuple[dict, int, int]:
    """Grade a single trace via a short-circuit pipeline of four binary checks.

    Call 1 — Storage:   is the fact in ALL_MEMORIES?             No  → storage_error.
    Call 2 — Summary:   is the stored version faithful?          No  → summary_error.
    Call 3 — Retrieval: was the fact in RETRIEVED_MEMORIES?      No  → retrieval_error.
    Call 4 — Invocation: did the model answer correctly?         No  → reasoning_error.
                                                                 Yes → correct.

    Returns (result_dict, total_input_tokens, total_output_tokens). Never raises.
    """
    result = {k: v for k, v in trace.items()}
    result["judge_result"] = None
    result["judge_reasoning"] = None
    result["fact_in_store"] = None
    result["storage_reasoning"] = None
    result["summary_check_passed"] = None
    result["summary_reasoning"] = None
    result["entity_facts_in_retrieved"] = None
    result["retrieval_reasoning"] = None
    result["correctly_invoked"] = None
    result["invocation_reasoning"] = None
    result["error_type"] = None
    result["analysis_error"] = None

    total_in = 0
    total_out = 0

    original_fact = (trace.get("entity_facts") or [""])[0]
    retrieved = trace.get("retrieved_memories") or "(none)"
    all_memories_fmt = format_all_memories(all_memories)

    try:
        # ── Call 1: Storage check ────────────────────────────────────────────
        storage_data, in_tok, out_tok = _call(
            client, model,
            system=STORAGE_SYSTEM,
            user=STORAGE_USER.format(
                original_fact=original_fact,
                all_memories_formatted=all_memories_fmt,
            ),
            schema=STORAGE_SCHEMA,
        )
        total_in += in_tok
        total_out += out_tok
        result["fact_in_store"] = storage_data["fact_in_store"]
        result["storage_reasoning"] = storage_data["storage_reasoning"]

        if not storage_data["fact_in_store"]:
            result["judge_result"] = "incorrect"
            result["error_type"] = "storage_error"
            result["judge_reasoning"] = storage_data["storage_reasoning"]
            return result, total_in, total_out

        # ── Call 2: Summary check ────────────────────────────────────────────
        summary_data, in_tok, out_tok = _call(
            client, model,
            system=SUMMARY_SYSTEM,
            user=SUMMARY_USER.format(
                original_fact=original_fact,
                all_memories_formatted=all_memories_fmt,
            ),
            schema=SUMMARY_SCHEMA,
        )
        total_in += in_tok
        total_out += out_tok
        result["summary_check_passed"] = summary_data["summary_check_passed"]
        result["summary_reasoning"] = summary_data["summary_reasoning"]

        if not summary_data["summary_check_passed"]:
            result["judge_result"] = "incorrect"
            result["error_type"] = "summary_error"
            result["judge_reasoning"] = summary_data["summary_reasoning"]
            return result, total_in, total_out

        # ── Call 3: Retrieval check ──────────────────────────────────────────
        retrieval_data, in_tok, out_tok = _call(
            client, model,
            system=RETRIEVAL_SYSTEM,
            user=RETRIEVAL_USER.format(
                query=trace["question"],
                original_fact=original_fact,
                retrieved_memories=retrieved,
            ),
            schema=RETRIEVAL_SCHEMA,
        )
        total_in += in_tok
        total_out += out_tok
        result["entity_facts_in_retrieved"] = retrieval_data["entity_facts_in_retrieved"]
        result["retrieval_reasoning"] = retrieval_data["retrieval_reasoning"]

        if not retrieval_data["entity_facts_in_retrieved"]:
            result["judge_result"] = "incorrect"
            result["error_type"] = "retrieval_error"
            result["judge_reasoning"] = retrieval_data["retrieval_reasoning"]
            return result, total_in, total_out

        # ── Call 4: Invocation check ─────────────────────────────────────────
        invocation_data, in_tok, out_tok = _call(
            client, model,
            system=INVOCATION_SYSTEM,
            user=INVOCATION_USER.format(
                original_fact=original_fact,
                ground_truth_answer=trace.get("ground_truth_answer", ""),
                llm_response=trace["llm_response"],
            ),
            schema=INVOCATION_SCHEMA,
        )
        total_in += in_tok
        total_out += out_tok
        result["correctly_invoked"] = invocation_data["correctly_invoked"]
        result["invocation_reasoning"] = invocation_data["invocation_reasoning"]

        if not invocation_data["correctly_invoked"]:
            result["judge_result"] = "incorrect"
            result["error_type"] = "reasoning_error"
            result["judge_reasoning"] = invocation_data["invocation_reasoning"]
        else:
            result["judge_result"] = "correct"
            result["judge_reasoning"] = invocation_data["invocation_reasoning"]

    except Exception as exc:
        result["analysis_error"] = str(exc)

    return result, total_in, total_out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_summary(results: list, judge_input_tokens: int, judge_output_tokens: int,
                  judge_cost: float, judge_model: str) -> None:
    total = len(results)
    correct = sum(1 for r in results if r.get("judge_result") == "correct")
    incorrect = total - correct

    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total traces:       {total}")
    print(f"Correct:            {correct}  ({correct/total:.1%})" if total else "Correct: 0")
    print(f"Incorrect:          {incorrect}  ({incorrect/total:.1%})" if total else "Incorrect: 0")

    retrieval_hits = sum(1 for r in results if r.get("entity_facts_in_retrieved"))
    print(f"Retrieval hit rate: {retrieval_hits/total:.1%}" if total else "Retrieval hit rate: N/A")

    error_types = ["storage_error", "summary_error", "retrieval_error", "reasoning_error"]
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

    print(f"\n{'Judge model:':<26} {judge_model}")
    print(f"{'Judge input tokens:':<26} {judge_input_tokens:,}")
    print(f"{'Judge output tokens:':<26} {judge_output_tokens:,}")
    print(f"{'Judge cost:':<26} ${judge_cost:.4f}")
    print("=" * 60)


def save_outputs(results: list, output_dir: Path, ts: str):
    json_path = output_dir / f"analysis_{ts}.json"
    csv_path = output_dir / f"analysis_{ts}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fieldnames = [
        "conversation_id", "judge_result", "error_type",
        "fact_in_store", "summary_check_passed", "entity_facts_in_retrieved", "correctly_invoked",
        "condition_met", "condition_type", "entity", "condition",
        "ground_truth_answer", "question",
        "judge_reasoning", "storage_reasoning", "summary_reasoning",
        "retrieval_reasoning", "invocation_reasoning",
        "eval_input_tokens", "eval_output_tokens", "eval_cost",
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
        help="Path to traces JSON from evaluate_conditional_facts.py. Auto-detects most recent if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to same directory as the traces file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="LLM model for grading and classification.",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only analyze first N graded traces (for testing).",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: set OPENAI_API_KEY or OPENAI_API_KEY environment variable.")

    if args.traces:
        traces_path = Path(args.traces)
        if not traces_path.is_absolute():
            traces_path = PROJECT_ROOT / traces_path
    else:
        traces_path = auto_detect_traces_file(DEFAULT_RESULTS_DIR)
    print(f"Loading traces from: {traces_path}")

    with open(traces_path, encoding="utf-8") as f:
        data = json.load(f)

    all_memories = data["all_memories_at_time_of_questions"]

    if "graded_traces" in data:
        graded_traces = data["graded_traces"]
    else:
        graded_traces, _ = extract_graded_traces(data)

    print(f"Total graded traces: {len(graded_traces)}")
    print(f"All memories in store: {len(all_memories)}")

    if args.limit:
        graded_traces = graded_traces[: args.limit]
        print(f"Limiting to first {args.limit} traces.")

    output_dir = Path(args.output_dir) if args.output_dir else traces_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    results: List[Optional[dict]] = [None] * len(graded_traces)
    token_accumulator: List[Tuple[int, int, int]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(analyze_trace, client, args.model, trace, all_memories): i
            for i, trace in enumerate(graded_traces)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing traces"):
            i = futures[future]
            result, in_tok, out_tok = future.result()
            results[i] = result
            token_accumulator.append((i, in_tok, out_tok))

    judge_input_tokens = sum(t[1] for t in token_accumulator)
    judge_output_tokens = sum(t[2] for t in token_accumulator)
    judge_cost = compute_cost(args.model, judge_input_tokens, judge_output_tokens)

    json_path, csv_path = save_outputs(results, output_dir, ts)
    print_summary(results, judge_input_tokens, judge_output_tokens, judge_cost, args.model)
    print(f"\nResults saved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main()
