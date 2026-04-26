"""
analyze_errors.py -- Grade and classify errors in coexisting-facts traces.

Each preference fact is independently classified through a staged pipeline:

  Stage 1 — Storage check (parallel):   is the fact in ALL_MEMORIES?
  Stage 2 — Summary check (parallel):   is the stored version faithful (not corrupted)?
  Stage 3 — Retrieval check (parallel): was the fact included in RETRIEVED_MEMORIES?

Per-fact categories:
  not_stored    — fact absent from ALL_MEMORIES
  summary_error — fact present but specific preference identity destroyed in summarization
  not_retrieved — fact stored faithfully but not surfaced at retrieval time
  correct       — fact faithfully stored and retrieved

If ALL facts reach "correct", one final invocation check determines whether the model
correctly used the retrieved memories to answer (reasoning_error vs. correct).

Per-fact distributions are reported overall and conditioned on the number of coexisting
facts in each dataset point.

Usage:
  uv run python playground/coexisting_facts/analyze_errors.py --traces path/to/traces.json
  uv run python playground/coexisting_facts/analyze_errors.py  # auto-detects most recent
  uv run python playground/coexisting_facts/analyze_errors.py --limit 5
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
MAX_INNER_WORKERS = 8  # parallel calls within a single trace

# ---------------------------------------------------------------------------
# Per-fact: Storage check
# ---------------------------------------------------------------------------

FACT_STORAGE_SYSTEM = """You are checking whether a specific preference was stored in a memory system.
The preference may have been paraphrased or compressed, but must still be clearly identifiable."""

FACT_STORAGE_USER = """ORIGINAL PREFERENCE: {preference}
ORIGINAL FACT: {original_fact}

ALL_MEMORIES (complete memory store):
{all_memories_formatted}

Is this preference present in ALL_MEMORIES, even if paraphrased or lightly compressed?
It counts as present as long as the specific preference can still be clearly identified."""

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
# Per-fact: Summary check
# ---------------------------------------------------------------------------

FACT_SUMMARY_SYSTEM = """You are assessing the quality of a stored preference.

The preference has already been confirmed to exist in the memory store. Your job is to
evaluate whether the stored version faithfully preserves the specific identity of this
preference in a way that would support correct downstream reasoning.

A stored version has a SUMMARY ERROR if ANY of the following apply:
- The preference was overgeneralized or merged with others, losing its distinct identity
  (e.g., "sushi" → "Asian food"; "jazz" → "music")
- The preference was corrupted or replaced with something different
- Critical identifying details were lost such that the model could not specifically cite
  this preference when answering a question

A stored version is FAITHFUL if the specific preference can still be clearly and
unambiguously identified from the stored memory."""

FACT_SUMMARY_USER = """ORIGINAL PREFERENCE: {preference}
ORIGINAL FACT: {original_fact}

ALL_MEMORIES (the preference IS confirmed present somewhere in here):
{all_memories_formatted}

Find the memory entry for this preference and assess whether the stored version faithfully
preserves the specific preference identity, or whether it has a summary error."""

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
# Per-fact: Retrieval check
# ---------------------------------------------------------------------------

FACT_RETRIEVAL_SYSTEM = """You are checking whether a specific preference was included in the
memories retrieved and shown to an AI model when it answered a question."""

FACT_RETRIEVAL_USER = """ORIGINAL PREFERENCE: {preference}
ORIGINAL FACT: {original_fact}

ORIGINAL QUESTION (used to search the memory store):
{query}

RETRIEVED_MEMORIES (shown to the model):
{retrieved_memories}

Was this preference included in RETRIEVED_MEMORIES, even if paraphrased?
It counts as retrieved as long as the specific preference can be clearly identified."""

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
# Question-level: Invocation check (only when all facts are "correct")
# ---------------------------------------------------------------------------

INVOCATION_SYSTEM = """You are checking whether an AI model correctly used all retrieved preferences
to answer a question.

All expected preferences were present in the retrieved memories. The model should
have mentioned all of them in its response."""

INVOCATION_USER = """EXPECTED_PREFERENCES (all were present in retrieved memories):
{expected_preferences}

GROUND TRUTH ANSWER:
{ground_truth_answer}

MODEL RESPONSE:
{llm_response}

Did the model correctly mention or account for ALL expected preferences in its response?
Synonyms and paraphrases count (e.g. "pasta dishes" covers "spaghetti")."""

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


def format_all_memories(all_memories: list) -> str:
    if not all_memories:
        return "(no memories in store)"
    entries = [m if isinstance(m, str) else m.get("memory", str(m)) for m in all_memories]
    return "\n".join(f"- {m}" for m in entries)


def format_preferences(preferences: list) -> str:
    if not preferences:
        return "(none)"
    return "\n".join(f"- {p}" for p in preferences)


def resolve_preferences(trace: dict) -> list:
    prefs = trace.get("preferences") or []
    if isinstance(prefs, str):
        try:
            prefs = json.loads(prefs)
        except Exception:
            prefs = []
    return prefs if isinstance(prefs, list) else []


def auto_detect_traces_file(results_dir: Path) -> Path:
    graded = list(results_dir.glob("graded_traces_*.json")) + list(results_dir.glob("*/graded_traces_*.json"))
    if graded:
        return max(graded, key=lambda p: p.stat().st_mtime)
    candidates = list(results_dir.glob("traces_*.json")) + list(results_dir.glob("*/traces_*.json"))
    candidates = [p for p in candidates if not p.name.startswith("traces_compact_")]
    if not candidates:
        raise FileNotFoundError(
            f"No graded_traces_*.json or traces_*.json files found under {results_dir}. "
            "Run evaluate_coexisting_facts.py first, or pass --traces explicitly."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_graded_traces(data: dict) -> List[dict]:
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
            "preference_category": row["preference_category"],
            "preferences": row["preferences"],
            "preference_facts": row["preference_facts"],
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
    """Classify each preference fact through a 3-stage pipeline, then run invocation check
    if all facts reach 'correct'.

    Per-fact categories: not_stored | summary_error | not_retrieved | correct
    Question-level result: correct | incorrect (+ reasoning_error flag if all facts correct)

    Returns (result_dict, total_input_tokens, total_output_tokens). Never raises.
    """
    preferences = resolve_preferences(trace)
    preference_facts = trace.get("preference_facts") or []
    N = len(preferences)

    result = {k: v for k, v in trace.items() if k != "formatted_prompt"}
    result["per_fact_results"] = []
    result["judge_result"] = None
    result["correctly_invoked"] = None
    result["invocation_reasoning"] = None
    result["analysis_error"] = None

    total_in = 0
    total_out = 0

    # Pre-format shared inputs once
    all_memories_fmt = format_all_memories(all_memories)
    retrieved = trace.get("retrieved_memories") or "(none)"

    # Per-fact state — indexed 0..N-1
    fact_results = [
        {
            "preference": preferences[i] if i < N else "",
            "original_fact": preference_facts[i] if i < len(preference_facts) else "",
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
        # ── Stage 1: Storage check — all N facts in parallel ─────────────────
        workers = min(N, MAX_INNER_WORKERS)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _call, client, model,
                    FACT_STORAGE_SYSTEM,
                    FACT_STORAGE_USER.format(
                        preference=fact_results[i]["preference"],
                        original_fact=fact_results[i]["original_fact"],
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

        # ── Stage 2: Summary check — stored facts in parallel ─────────────────
        stored_idx = [i for i in range(N) if fact_results[i]["fact_in_store"]]
        if stored_idx:
            with ThreadPoolExecutor(max_workers=min(len(stored_idx), MAX_INNER_WORKERS)) as pool:
                futures = {
                    pool.submit(
                        _call, client, model,
                        FACT_SUMMARY_SYSTEM,
                        FACT_SUMMARY_USER.format(
                            preference=fact_results[i]["preference"],
                            original_fact=fact_results[i]["original_fact"],
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

        # ── Stage 3: Retrieval check — summary-passing facts in parallel ──────
        quality_idx = [i for i in stored_idx if fact_results[i]["summary_passed"]]
        if quality_idx:
            with ThreadPoolExecutor(max_workers=min(len(quality_idx), MAX_INNER_WORKERS)) as pool:
                futures = {
                    pool.submit(
                        _call, client, model,
                        FACT_RETRIEVAL_SYSTEM,
                        FACT_RETRIEVAL_USER.format(
                            preference=fact_results[i]["preference"],
                            original_fact=fact_results[i]["original_fact"],
                            query=trace["question"],
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

        # ── Invocation check — only if every fact reached "correct" ───────────
        all_correct = all(fr["category"] == "correct" for fr in fact_results)
        if all_correct:
            invocation_data, in_tok, out_tok = _call(
                client, model,
                system=INVOCATION_SYSTEM,
                user=INVOCATION_USER.format(
                    expected_preferences=format_preferences(preferences),
                    ground_truth_answer=trace.get("ground_truth_answer", ""),
                    llm_response=trace["llm_response"],
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

    # Reasoning errors: all facts correct but model failed invocation
    reasoning_errors = sum(
        1 for r in results
        if r.get("judge_result") == "incorrect"
        and r.get("per_fact_results")
        and all(f.get("category") == "correct" for f in r["per_fact_results"])
    )

    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total traces:         {total}")
    if total:
        print(f"Correct:              {correct}  ({correct/total:.1%})")
        print(f"Incorrect:            {incorrect}  ({incorrect/total:.1%})")
        print(f"  of which reasoning_error: {reasoning_errors}  ({reasoning_errors/total:.1%})")

    # ── Per-fact category distribution (all facts, all traces) ──────────────
    agg: Dict[str, int] = {c: 0 for c in FACT_CATEGORIES}
    total_facts = 0
    for r in results:
        counts = _fact_category_counts(r.get("per_fact_results", []))
        for cat, n in counts.items():
            agg[cat] += n
            total_facts += n

    if total_facts:
        print(f"\n{'Per-fact pipeline distribution'} (across all {total_facts} facts)")
        print(f"{'Category':<18} {'Count':>7} {'Fraction':>10}")
        print("-" * 38)
        for cat in FACT_CATEGORIES:
            n = agg[cat]
            print(f"{cat:<18} {n:>7} {n/total_facts:>9.1%}")
        print("-" * 38)
        print(f"{'TOTAL':<18} {total_facts:>7}")

    # ── Per-fact distribution conditioned on N ────────────────────────────────
    by_n: Dict[int, Dict] = {}
    for r in results:
        pfr = r.get("per_fact_results", [])
        n = len(pfr)
        if n not in by_n:
            by_n[n] = {c: 0 for c in FACT_CATEGORIES}
            by_n[n]["_total_facts"] = 0
            by_n[n]["_traces"] = 0
        by_n[n]["_traces"] += 1
        counts = _fact_category_counts(pfr)
        for cat, cnt in counts.items():
            by_n[n][cat] += cnt
            by_n[n]["_total_facts"] += cnt

    if by_n:
        print(f"\nPer-fact distribution by preference count:")
        header = f"{'N':>3}  {'not_stored':>12} {'summary_err':>12} {'not_retrv':>11} {'correct':>9}  {'traces':>7}"
        print(header)
        print("-" * len(header))
        for n in sorted(by_n):
            d = by_n[n]
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

    print(f"\n{'Judge model:':<26} {judge_model}")
    print(f"{'Judge input tokens:':<26} {judge_input_tokens:,}")
    print(f"{'Judge output tokens:':<26} {judge_output_tokens:,}")
    print(f"{'Judge cost:':<26} ${judge_cost:.4f}")
    print("=" * 70)


def save_outputs(results: list, output_dir: Path, ts: str):
    json_path = output_dir / f"analysis_{ts}.json"
    csv_path = output_dir / f"analysis_{ts}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fieldnames = [
        "conversation_id", "judge_result", "correctly_invoked",
        "preference_category", "ground_truth_answer", "question",
        "n_preferences",
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
            row["n_preferences"] = n
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
        description="Grade and classify errors in coexisting-facts traces."
    )
    parser.add_argument(
        "--traces",
        default=None,
        help="Path to traces JSON from evaluate_coexisting_facts.py. Auto-detects most recent if omitted.",
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

    all_memories = data["all_memories_at_time_of_questions"]

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
