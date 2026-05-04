"""
analyze_errors.py -- Grade and classify errors in long-hop (MemDaily) traces.

Each target support message is independently classified through a staged pipeline:

  Stage 1 — Storage check (parallel):   is the target message in ALL_MEMORIES?
  Stage 2 — Summary check (parallel):   is the stored version faithful?
  Stage 3 — Retrieval check (parallel): was the target message in RETRIEVED_MEMORIES?

Per-target-message categories:
  not_stored    — target absent from ALL_MEMORIES
  summary_error — target present but specific fact destroyed in summarization
  not_retrieved — target stored faithfully but not surfaced at retrieval time
  correct       — target faithfully stored and retrieved

If ALL targets reach "correct", one final invocation check determines whether the model
correctly used the retrieved memories to answer (reasoning_error vs. correct).

Per-target distributions are reported overall, by question_type, and by support_set_size.

Usage:
  uv run python playground/long_hop/analyze_errors.py --traces path/to/traces.json
  uv run python playground/long_hop/analyze_errors.py  # auto-detects most recent
  uv run python playground/long_hop/analyze_errors.py --limit 5
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
MAX_INNER_WORKERS = 8

# ---------------------------------------------------------------------------
# Per-target: Storage check
# ---------------------------------------------------------------------------

FACT_STORAGE_SYSTEM = """You are checking whether a specific support message was stored in a memory system.
The stored version may be paraphrased or compressed, but the same concrete fact must still be identifiable."""

FACT_STORAGE_USER = """TARGET SUPPORT MESSAGE:
{target_message}

ALL_MEMORIES (complete memory store):
{all_memories_formatted}

Is this target support message present in ALL_MEMORIES, even if paraphrased or lightly compressed?
Count it as present only if the same concrete fact can still be clearly identified."""

FACT_STORAGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fact_storage_check",
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
# Per-target: Summary check
# ---------------------------------------------------------------------------

FACT_SUMMARY_SYSTEM = """You are assessing the quality of a stored support message.

The target has already been confirmed to exist in the memory store. Your job is to
evaluate whether the stored version faithfully preserves the specific fact in a way
that would support correct downstream reasoning.

A stored version has a SUMMARY ERROR if ANY of the following apply:
- The fact was overgeneralized or merged with others, losing its distinct identity
- The fact was corrupted or replaced with something different
- Critical identifying details (time, place, names, numbers, relationships) were lost
  such that the model could not specifically cite this fact when answering a question

A stored version is FAITHFUL if the same concrete fact remains clearly identifiable
from the stored memory."""

FACT_SUMMARY_USER = """TARGET SUPPORT MESSAGE:
{target_message}

ALL_MEMORIES (the target IS confirmed present somewhere in here):
{all_memories_formatted}

Find the memory entry for this target and assess whether the stored version faithfully
preserves the specific fact, or whether it has a summary error."""

FACT_SUMMARY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fact_summary_check",
        "schema": {
            "type": "object",
            "properties": {
                "summary_passed": {"type": "boolean"},
                "summary_reasoning": {"type": "string"},
            },
            "required": ["summary_passed", "summary_reasoning"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

# ---------------------------------------------------------------------------
# Per-target: Retrieval check
# ---------------------------------------------------------------------------

FACT_RETRIEVAL_SYSTEM = """You are checking whether a specific support message was included in the
memories retrieved and shown to an AI model when it answered a question."""

FACT_RETRIEVAL_USER = """TARGET SUPPORT MESSAGE:
{target_message}

ORIGINAL QUESTION (used to search the memory store):
{query}

RETRIEVED_MEMORIES (shown to the model):
{retrieved_memories}

Was this target support message included in RETRIEVED_MEMORIES, even if paraphrased?
Count it as retrieved only if the same concrete fact can be clearly identified."""

FACT_RETRIEVAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fact_retrieval_check",
        "schema": {
            "type": "object",
            "properties": {
                "fact_in_retrieved": {"type": "boolean"},
                "retrieval_reasoning": {"type": "string"},
            },
            "required": ["fact_in_retrieved", "retrieval_reasoning"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

# ---------------------------------------------------------------------------
# Question-level: Invocation check (only when all targets are "correct")
# ---------------------------------------------------------------------------

INVOCATION_SYSTEM = """You are checking whether an AI model correctly used the retrieved support
messages to answer a multi-hop question.

All needed support messages were present in the retrieved memories. The model should have
combined them to produce the ground-truth answer."""

INVOCATION_USER = """QUESTION:
{question}

EXPECTED SUPPORT MESSAGES (all were retrieved):
{expected_support_messages}

GROUND TRUTH ANSWER:
{ground_truth_answer}

MODEL RESPONSE:
{llm_response}

Did the model correctly combine the retrieved support messages to produce the ground-truth
answer? Synonyms, paraphrases, and equivalent multiple-choice selections count."""

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

FACT_CATEGORIES = ["not_stored", "summary_error", "not_retrieved", "correct"]
ERROR_TYPE_PRIORITY = ["not_stored", "summary_error", "not_retrieved"]


def collapse_error_type(per_fact_results: list, correctly_invoked) -> str:
    """Return the single worst error type for a trace, by priority order."""
    categories = {f.get("category") for f in per_fact_results}
    for p in ERROR_TYPE_PRIORITY:
        if p in categories:
            return p
    if correctly_invoked is False:
        return "reasoning_error"
    return "correct"


def format_all_memories(all_memories: list) -> str:
    if not all_memories:
        return "(no memories in store)"
    entries = [m if isinstance(m, str) else m.get("memory", str(m)) for m in all_memories]
    return "\n".join(f"- {m}" for m in entries)


def format_support_messages(messages: list) -> str:
    if not messages:
        return "(none)"
    return "\n".join(f"- {m}" for m in messages)


def auto_detect_traces_file(results_dir: Path) -> Path:
    graded = list(results_dir.glob("graded_traces_*.json")) + list(results_dir.glob("*/graded_traces_*.json"))
    if graded:
        return max(graded, key=lambda p: p.stat().st_mtime)
    candidates = list(results_dir.glob("traces_*.json")) + list(results_dir.glob("*/traces_*.json"))
    candidates = [p for p in candidates if not p.name.startswith("traces_compact_")]
    if not candidates:
        raise FileNotFoundError(
            f"No graded_traces_*.json or traces_*.json files found under {results_dir}. "
            "Run evaluate_long_hop.py first, or pass --traces explicitly."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_graded_traces(data: dict) -> List[dict]:
    """Reconstruct graded traces from a full traces_*.json file."""
    num_storage_convs = data["run_metadata"]["num_storage_convs"]
    dataset_rows = data["dataset_rows"]
    results = data["evaluation_summary"]["results"]
    question_results = results[num_storage_convs:]

    graded = []
    for result, row in zip(question_results, dataset_rows):
        traces = result["traces"]
        if not traces:
            continue
        trace = traces[0]
        graded.append({
            "example_id": row["example_id"],
            "question_type": row["question_type"],
            "domain": row["domain"],
            "trajectory_id": row["trajectory_id"],
            "support_set_size": row["support_set_size"],
            "ground_truth_answer": row["ground_truth_answer"],
            "ground_truth_choice": row.get("ground_truth_choice", ""),
            "choices": row.get("choices", {}),
            "target_step_ids": row["target_step_ids"],
            "target_messages": row["target_messages"],
            "question": trace["query"],
            "retrieved_memories": trace["retrieved_memories"],
            "llm_response": trace["response"],
            "formatted_prompt": trace["formatted_prompt"],
            "conversation_id": result["conversation_id"],
            "eval_input_tokens": trace["input_tokens"],
            "eval_output_tokens": trace["output_tokens"],
            "eval_cost": trace["cost"],
        })
    return graded


# ---------------------------------------------------------------------------
# Judge call
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


# ---------------------------------------------------------------------------
# Per-trace analysis
# ---------------------------------------------------------------------------


def analyze_trace(
    client: OpenAI,
    model: str,
    trace: dict,
    all_memories: list,
) -> Tuple[dict, int, int]:
    """Classify each target message through a 3-stage pipeline, then run invocation check
    if all targets reach 'correct'.

    Per-target categories: not_stored | summary_error | not_retrieved | correct
    Question-level result: correct | incorrect (+ reasoning_error flag if all targets correct)

    Returns (result_dict, total_input_tokens, total_output_tokens). Never raises.
    """
    target_messages = trace.get("target_messages") or []
    N = len(target_messages)

    result = {k: v for k, v in trace.items() if k != "formatted_prompt"}
    result["per_fact_results"] = []
    result["judge_result"] = None
    result["error_type"] = None
    result["correctly_invoked"] = None
    result["invocation_reasoning"] = None
    result["analysis_error"] = None

    total_in = 0
    total_out = 0

    all_memories_fmt = format_all_memories(all_memories)
    retrieved = trace.get("retrieved_memories") or "(none)"

    fact_results = [
        {
            "target_message": target_messages[i],
            "category": None,
            "fact_in_store": None,
            "storage_reasoning": None,
            "summary_passed": None,
            "summary_reasoning": None,
            "fact_in_retrieved": None,
            "retrieval_reasoning": None,
        }
        for i in range(N)
    ]

    try:
        # ── Stage 1: Storage check — all N targets in parallel ───────────────
        if N:
            workers = min(N, MAX_INNER_WORKERS)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _call, client, model,
                        FACT_STORAGE_SYSTEM,
                        FACT_STORAGE_USER.format(
                            target_message=fact_results[i]["target_message"],
                            all_memories_formatted=all_memories_fmt,
                        ),
                        FACT_STORAGE_SCHEMA,
                    ): i
                    for i in range(N)
                }
                for future in as_completed(futures):
                    i = futures[future]
                    data, in_tok, out_tok = future.result()
                    total_in += in_tok
                    total_out += out_tok
                    fact_results[i]["fact_in_store"] = data["fact_in_store"]
                    fact_results[i]["storage_reasoning"] = data["storage_reasoning"]
                    if not data["fact_in_store"]:
                        fact_results[i]["category"] = "not_stored"

        # ── Stage 2: Summary check — stored targets in parallel ──────────────
        stored_idx = [i for i in range(N) if fact_results[i]["fact_in_store"]]
        if stored_idx:
            with ThreadPoolExecutor(max_workers=min(len(stored_idx), MAX_INNER_WORKERS)) as pool:
                futures = {
                    pool.submit(
                        _call, client, model,
                        FACT_SUMMARY_SYSTEM,
                        FACT_SUMMARY_USER.format(
                            target_message=fact_results[i]["target_message"],
                            all_memories_formatted=all_memories_fmt,
                        ),
                        FACT_SUMMARY_SCHEMA,
                    ): i
                    for i in stored_idx
                }
                for future in as_completed(futures):
                    i = futures[future]
                    data, in_tok, out_tok = future.result()
                    total_in += in_tok
                    total_out += out_tok
                    fact_results[i]["summary_passed"] = data["summary_passed"]
                    fact_results[i]["summary_reasoning"] = data["summary_reasoning"]
                    if not data["summary_passed"]:
                        fact_results[i]["category"] = "summary_error"

        # ── Stage 3: Retrieval check — summary-passing targets in parallel ───
        quality_idx = [i for i in stored_idx if fact_results[i]["summary_passed"]]
        if quality_idx:
            with ThreadPoolExecutor(max_workers=min(len(quality_idx), MAX_INNER_WORKERS)) as pool:
                futures = {
                    pool.submit(
                        _call, client, model,
                        FACT_RETRIEVAL_SYSTEM,
                        FACT_RETRIEVAL_USER.format(
                            target_message=fact_results[i]["target_message"],
                            query=trace.get("question", ""),
                            retrieved_memories=retrieved,
                        ),
                        FACT_RETRIEVAL_SCHEMA,
                    ): i
                    for i in quality_idx
                }
                for future in as_completed(futures):
                    i = futures[future]
                    data, in_tok, out_tok = future.result()
                    total_in += in_tok
                    total_out += out_tok
                    fact_results[i]["fact_in_retrieved"] = data["fact_in_retrieved"]
                    fact_results[i]["retrieval_reasoning"] = data["retrieval_reasoning"]
                    fact_results[i]["category"] = "correct" if data["fact_in_retrieved"] else "not_retrieved"

        result["per_fact_results"] = fact_results

        # ── Invocation check — only if every target reached "correct" ────────
        all_correct = N > 0 and all(fr["category"] == "correct" for fr in fact_results)
        if all_correct:
            invocation_data, in_tok, out_tok = _call(
                client, model,
                system=INVOCATION_SYSTEM,
                user=INVOCATION_USER.format(
                    question=trace.get("question", ""),
                    expected_support_messages=format_support_messages(target_messages),
                    ground_truth_answer=trace.get("ground_truth_answer", ""),
                    llm_response=trace.get("llm_response", ""),
                ),
                schema=INVOCATION_SCHEMA,
            )
            total_in += in_tok
            total_out += out_tok
            result["correctly_invoked"] = invocation_data["correctly_invoked"]
            result["invocation_reasoning"] = invocation_data["invocation_reasoning"]
            result["judge_result"] = "correct" if invocation_data["correctly_invoked"] else "incorrect"
        else:
            result["judge_result"] = "incorrect"

        result["error_type"] = collapse_error_type(fact_results, result["correctly_invoked"])

    except Exception as exc:
        result["analysis_error"] = str(exc)
        result["per_fact_results"] = fact_results

    return result, total_in, total_out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _fact_category_counts(per_fact_results: list) -> Dict[str, int]:
    counts = {c: 0 for c in FACT_CATEGORIES}
    for f in per_fact_results:
        cat = f.get("category")
        if cat in counts:
            counts[cat] += 1
    return counts


def print_summary(results: list, judge_input_tokens: int, judge_output_tokens: int,
                  judge_cost: float, judge_model: str) -> None:
    total = len(results)
    correct = sum(1 for r in results if r.get("judge_result") == "correct")
    incorrect = total - correct

    reasoning_errors = sum(
        1 for r in results
        if r.get("judge_result") == "incorrect"
        and r.get("per_fact_results")
        and all(f.get("category") == "correct" for f in r["per_fact_results"])
    )

    print("\n" + "=" * 70)
    print("LONG-HOP ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total traces:         {total}")
    if total:
        print(f"Correct:              {correct}  ({correct/total:.1%})")
        print(f"Incorrect:            {incorrect}  ({incorrect/total:.1%})")

    error_types = ["not_stored", "summary_error", "not_retrieved", "reasoning_error"]
    type_counts: Dict[str, int] = {et: 0 for et in error_types}
    type_counts["analysis_failed"] = 0
    for r in results:
        if r.get("judge_result") != "incorrect":
            continue
        et = r.get("error_type") or "analysis_failed"
        type_counts[et] = type_counts.get(et, 0) + 1

    if incorrect > 0:
        print(f"\n{'Error type (of incorrect)':<26} {'Count':>6} {'Share of incorrect':>18}")
        print("-" * 52)
        for et in [*error_types, "analysis_failed"]:
            c = type_counts[et]
            if c == 0:
                continue
            print(f"{et:<26} {c:>6} {c/incorrect:>17.1%}")
        print("-" * 52)
        print(f"{'TOTAL INCORRECT':<26} {incorrect:>6}")
        if total:
            print(f"  of which reasoning_error: {reasoning_errors}  ({reasoning_errors/total:.1%})")

    # ── Per-target-message category distribution (across all targets) ────────
    agg: Dict[str, int] = {c: 0 for c in FACT_CATEGORIES}
    total_facts = 0
    for r in results:
        counts = _fact_category_counts(r.get("per_fact_results", []))
        for cat, n in counts.items():
            agg[cat] += n
            total_facts += n

    if total_facts:
        print(f"\nPer-target-message pipeline distribution (across all {total_facts} targets)")
        print(f"{'Category':<18} {'Count':>7} {'Fraction':>10}")
        print("-" * 38)
        for cat in FACT_CATEGORIES:
            n = agg[cat]
            print(f"{cat:<18} {n:>7} {n/total_facts:>9.1%}")
        print("-" * 38)
        print(f"{'TOTAL':<18} {total_facts:>7}")

    # ── Per-target distribution by support_set_size ──────────────────────────
    by_size: Dict[int, Dict] = {}
    for r in results:
        pfr = r.get("per_fact_results", [])
        n = int(r.get("support_set_size") or len(pfr))
        if n not in by_size:
            by_size[n] = {c: 0 for c in FACT_CATEGORIES}
            by_size[n]["_total_facts"] = 0
            by_size[n]["_traces"] = 0
        by_size[n]["_traces"] += 1
        counts = _fact_category_counts(pfr)
        for cat, cnt in counts.items():
            by_size[n][cat] += cnt
            by_size[n]["_total_facts"] += cnt

    if by_size:
        print("\nPer-target-message distribution by support_set_size:")
        header = f"{'N':>3}  {'not_stored':>12} {'summary_err':>12} {'not_retrv':>11} {'correct':>9}  {'traces':>7}"
        print(header)
        print("-" * len(header))
        for n in sorted(by_size):
            d = by_size[n]
            tf = d["_total_facts"] or 1
            print(
                f"{n:>3}"
                f"  {d['not_stored']:>5} ({d['not_stored']/tf:>5.1%})"
                f"  {d['summary_error']:>5} ({d['summary_error']/tf:>5.1%})"
                f"  {d['not_retrieved']:>5} ({d['not_retrieved']/tf:>5.1%})"
                f"  {d['correct']:>5} ({d['correct']/tf:>5.1%})"
                f"  {d['_traces']:>7}"
            )
        print("-" * len(header))

    # ── Per-question accuracy by question_type ────────────────────────────────
    by_qt: Dict[str, Dict[str, int]] = {}
    for r in results:
        qt = r.get("question_type", "unknown")
        if qt not in by_qt:
            by_qt[qt] = {"total": 0, "correct": 0}
        by_qt[qt]["total"] += 1
        if r.get("judge_result") == "correct":
            by_qt[qt]["correct"] += 1

    if by_qt:
        print("\nAccuracy by question_type:")
        header = f"{'question_type':<20} {'correct':>8} {'total':>6} {'acc':>7}"
        print(header)
        print("-" * len(header))
        for qt in sorted(by_qt):
            d = by_qt[qt]
            acc = d["correct"] / d["total"] if d["total"] else 0
            print(f"{qt:<20} {d['correct']:>8} {d['total']:>6} {acc:>6.1%}")
        print("-" * len(header))

    print(f"\n{'Judge model:':<26} {judge_model}")
    print(f"{'Judge input tokens:':<26} {judge_input_tokens:,}")
    print(f"{'Judge output tokens:':<26} {judge_output_tokens:,}")
    print(f"{'Judge cost:':<26} ${judge_cost:.4f}")
    print("=" * 70)


def save_outputs(results: list, output_dir: Path, ts: str) -> Tuple[Path, Path]:
    json_path = output_dir / f"analysis_{ts}.json"
    csv_path = output_dir / f"analysis_{ts}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "example_id", "question_type", "domain", "trajectory_id", "support_set_size",
        "conversation_id", "judge_result", "error_type", "correctly_invoked",
        "ground_truth_answer", "ground_truth_choice", "question",
        "n_targets",
        "n_not_stored", "n_summary_error", "n_not_retrieved", "n_correct",
        "frac_not_stored", "frac_summary_error", "frac_not_retrieved", "frac_correct",
        "invocation_reasoning",
        "eval_input_tokens", "eval_output_tokens", "eval_cost",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            pfr = r.get("per_fact_results") or []
            n = len(pfr)
            counts = _fact_category_counts(pfr)
            denom = n or 1
            row = {k: r.get(k, "") for k in fieldnames}
            row["n_targets"] = n
            row["n_not_stored"] = counts["not_stored"]
            row["n_summary_error"] = counts["summary_error"]
            row["n_not_retrieved"] = counts["not_retrieved"]
            row["n_correct"] = counts["correct"]
            row["frac_not_stored"] = f"{counts['not_stored']/denom:.3f}"
            row["frac_summary_error"] = f"{counts['summary_error']/denom:.3f}"
            row["frac_not_retrieved"] = f"{counts['not_retrieved']/denom:.3f}"
            row["frac_correct"] = f"{counts['correct']/denom:.3f}"
            writer.writerow(row)

    return json_path, csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade and classify errors in long-hop (MemDaily) traces."
    )
    parser.add_argument(
        "--traces",
        default=None,
        help="Path to traces JSON from evaluate_long_hop.py. Auto-detects most recent if omitted.",
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
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Outer parallelism: number of traces analyzed concurrently.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only analyze first N graded traces (for testing).",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        sys.exit("Error: set OPENAI_API_KEY environment variable.")

    if args.traces:
        traces_path = Path(args.traces)
        if not traces_path.is_absolute():
            traces_path = PROJECT_ROOT / traces_path
    else:
        traces_path = auto_detect_traces_file(DEFAULT_RESULTS_DIR)
    print(f"Loading traces from: {traces_path}")

    with open(traces_path, encoding="utf-8") as f:
        data = json.load(f)

    all_memories = data.get("all_memories_at_time_of_questions", [])

    if "graded_traces" in data:
        graded_traces = data["graded_traces"]
    else:
        graded_traces = extract_graded_traces(data)

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

    final_results = [r for r in results if r is not None]
    judge_input_tokens = sum(t[1] for t in token_accumulator)
    judge_output_tokens = sum(t[2] for t in token_accumulator)
    judge_cost = compute_cost(args.model, judge_input_tokens, judge_output_tokens)

    json_path, csv_path = save_outputs(final_results, output_dir, ts)
    print_summary(final_results, judge_input_tokens, judge_output_tokens, judge_cost, args.model)
    print("\nResults saved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main()
