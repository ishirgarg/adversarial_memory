#!/usr/bin/env python3
"""
Grade and classify MemDaily errors in a coexisting-facts-style pipeline.

Each target support message is independently classified through:

  Stage 1 — Storage check:   is the target message present in ALL_MEMORIES?
  Stage 2 — Summary check:   if present, was it preserved faithfully?
  Stage 3 — Retrieval check: if faithful, was it included in RETRIEVED_MEMORIES?

Per-support-message categories:
  not_stored
  summary_error
  not_retrieved
  correct

If all support messages reach "correct", one final invocation check decides whether the
model still failed at final reasoning.

Results are summarized overall and by `support_set_size`, where `support_set_size`
means `len(target_step_ids)`: the number of ground-truth supporting messages needed
to answer the question.
"""

from __future__ import annotations

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
DEFAULT_MODEL = "gpt-5.1-mini"
DEFAULT_WORKERS = 8
MAX_RETRIES = 3
MAX_INNER_WORKERS = 8
MODEL_ALIASES = {
    "gpt-5.1-mini": "gpt-5.1",
}

FACT_CATEGORIES = ["not_stored", "summary_error", "not_retrieved", "correct"]
ERROR_TYPE_PRIORITY = ["not_stored", "summary_error", "not_retrieved"]


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
        "name": "memdaily_storage_check",
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

FACT_SUMMARY_SYSTEM = """You are assessing whether a stored support message faithfully preserves a fact.

A stored version has a SUMMARY ERROR if the original fact's specific identity was blurred,
overgeneralized, merged, corrupted, or had critical details removed such that it would no
longer support the original question reliably.

A stored version is FAITHFUL if the same concrete fact remains clearly identifiable."""

FACT_SUMMARY_USER = """TARGET SUPPORT MESSAGE:
{target_message}

ALL_MEMORIES (the target message is confirmed present somewhere in here):
{all_memories_formatted}

Find the relevant memory entry and assess whether it faithfully preserves the original fact."""

FACT_SUMMARY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "memdaily_summary_check",
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

FACT_RETRIEVAL_SYSTEM = """You are checking whether a specific support message was included in the
retrieved memories shown to a model when it answered a question."""

FACT_RETRIEVAL_USER = """TARGET SUPPORT MESSAGE:
{target_message}

QUESTION:
{question}

RETRIEVED_MEMORIES:
{retrieved_memories}

Was this support message included in RETRIEVED_MEMORIES, even if paraphrased?
Count it as retrieved only if the same concrete fact can be clearly identified."""

FACT_RETRIEVAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "memdaily_retrieval_check",
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

INVOCATION_SYSTEM = """You are checking whether the model correctly used fully retrieved support messages.

All needed support messages were present in the retrieved context. Decide whether the model
still failed at final reasoning or answer construction."""

INVOCATION_USER = """QUESTION:
{question}

EXPECTED SUPPORT MESSAGES (all were retrieved):
{expected_support_messages}

GROUND TRUTH ANSWER:
{ground_truth_answer}

MODEL ANSWER:
{model_answer}

Did the model correctly combine the retrieved support messages to produce a correct answer?"""

INVOCATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "memdaily_invocation_check",
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


def resolve_model_name(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


def collapse_error_type(per_fact_results: list, correctly_invoked: Optional[bool]) -> str:
    categories = {f.get("category") for f in per_fact_results}
    for category in ERROR_TYPE_PRIORITY:
        if category in categories:
            return category
    if correctly_invoked is False:
        return "reasoning_error"
    return "correct"


def format_all_memories(all_memories: list) -> str:
    if not all_memories:
        return "(no memories in store)"
    return "\n".join(f"- {m}" for m in all_memories)


def format_support_messages(messages: list) -> str:
    if not messages:
        return "(none)"
    return "\n".join(f"- {m}" for m in messages)


def auto_detect_traces_file(results_dir: Path) -> Path:
    candidates = list(results_dir.glob("*_traces_*.json")) + list(results_dir.glob("**/*_traces_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No *_traces_*.json files found under {results_dir}. "
            "Run evaluate_memdaily.py first, or pass --traces explicitly."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_graded_traces(traces_path: Path) -> List[dict]:
    with open(traces_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "graded_traces" in data:
        return data["graded_traces"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported traces file structure: {traces_path}")


def _call(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    schema: dict,
) -> Tuple[dict, int, int]:
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
            data = json.loads(resp.choices[0].message.content or "{}")
            in_tok = resp.usage.prompt_tokens if resp.usage else 0
            out_tok = resp.usage.completion_tokens if resp.usage else 0
            return data, in_tok, out_tok
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(0.4 * attempt)
    raise RuntimeError(f"Judge call failed after {MAX_RETRIES} attempts: {last_error}") from last_error


def analyze_trace(client: OpenAI, model: str, trace: dict) -> Tuple[dict, int, int]:
    support_messages = trace.get("target_messages") or []
    n = int(trace.get("support_set_size") or len(support_messages))
    all_memories = trace.get("all_memories_at_time") or []
    retrieved = trace.get("retrieved_memories") or "(none)"

    result = {k: v for k, v in trace.items() if k != "formatted_prompt"}
    result["per_fact_results"] = []
    result["error_type"] = None
    result["correctly_invoked"] = None
    result["invocation_reasoning"] = None
    result["analysis_error"] = None
    result["judge_result"] = "incorrect"

    total_in = 0
    total_out = 0
    all_memories_fmt = format_all_memories(all_memories)

    fact_results = [
        {
            "target_message": support_messages[i] if i < len(support_messages) else "",
            "category": None,
            "fact_in_store": None,
            "storage_reasoning": None,
            "summary_passed": None,
            "summary_reasoning": None,
            "fact_in_retrieved": None,
            "retrieval_reasoning": None,
        }
        for i in range(n)
    ]

    try:
        workers = min(max(n, 1), MAX_INNER_WORKERS)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _call,
                    client,
                    model,
                    FACT_STORAGE_SYSTEM,
                    FACT_STORAGE_USER.format(
                        target_message=fact_results[i]["target_message"],
                        all_memories_formatted=all_memories_fmt,
                    ),
                    FACT_STORAGE_SCHEMA,
                ): i
                for i in range(n)
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

        stored_idx = [i for i in range(n) if fact_results[i]["fact_in_store"]]
        if stored_idx:
            with ThreadPoolExecutor(max_workers=min(len(stored_idx), MAX_INNER_WORKERS)) as pool:
                futures = {
                    pool.submit(
                        _call,
                        client,
                        model,
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

        quality_idx = [i for i in stored_idx if fact_results[i]["summary_passed"]]
        if quality_idx:
            with ThreadPoolExecutor(max_workers=min(len(quality_idx), MAX_INNER_WORKERS)) as pool:
                futures = {
                    pool.submit(
                        _call,
                        client,
                        model,
                        FACT_RETRIEVAL_SYSTEM,
                        FACT_RETRIEVAL_USER.format(
                            target_message=fact_results[i]["target_message"],
                            question=trace.get("graded_question", ""),
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

        all_correct = all(fr["category"] == "correct" for fr in fact_results)
        if all_correct:
            invocation_data, in_tok, out_tok = _call(
                client,
                model,
                INVOCATION_SYSTEM,
                INVOCATION_USER.format(
                    question=trace.get("graded_question", ""),
                    expected_support_messages=format_support_messages(support_messages),
                    ground_truth_answer=trace.get("ground_truth_answer", ""),
                    model_answer=trace.get("model_answer", ""),
                ),
                INVOCATION_SCHEMA,
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


def _fact_category_counts(per_fact_results: list) -> Dict[str, int]:
    counts = {category: 0 for category in FACT_CATEGORIES}
    for fact in per_fact_results:
        category = fact.get("category")
        if category in counts:
            counts[category] += 1
    return counts


def print_summary(
    results: list,
    judge_input_tokens: int,
    judge_output_tokens: int,
    judge_cost: float,
    judge_model: str,
) -> None:
    total = len(results)
    correct = sum(1 for r in results if r.get("judge_result") == "correct")
    incorrect = total - correct
    reasoning_errors = sum(1 for r in results if r.get("error_type") == "reasoning_error")

    print("\n" + "=" * 72)
    print("MEMDAILY ERROR ANALYSIS")
    print("=" * 72)
    print(f"Total traces:         {total}")
    if total:
        print(f"Correct:              {correct}  ({correct/total:.1%})")
        print(f"Incorrect:            {incorrect}  ({incorrect/total:.1%})")

    error_types = ["not_stored", "summary_error", "not_retrieved", "reasoning_error"]
    type_counts: Dict[str, int] = {error_type: 0 for error_type in error_types}
    type_counts["analysis_failed"] = 0
    for result in results:
        if result.get("judge_result") != "incorrect":
            continue
        error_type = result.get("error_type") or "analysis_failed"
        type_counts[error_type] = type_counts.get(error_type, 0) + 1

    if incorrect:
        print(f"\n{'Error type (of incorrect)':<26} {'Count':>6} {'Share of incorrect':>18}")
        print("-" * 54)
        for error_type in [*error_types, "analysis_failed"]:
            count = type_counts.get(error_type, 0)
            if count:
                print(f"{error_type:<26} {count:>6} {count/incorrect:>17.1%}")
        print("-" * 54)
        print(f"{'TOTAL INCORRECT':<26} {incorrect:>6}")
        print(f"  of which reasoning_error: {reasoning_errors}  ({reasoning_errors/total:.1%})" if total else "")

    overall_counts = {category: 0 for category in FACT_CATEGORIES}
    total_facts = 0
    for result in results:
        counts = _fact_category_counts(result.get("per_fact_results", []))
        for category, count in counts.items():
            overall_counts[category] += count
            total_facts += count

    if total_facts:
        print(f"\nPer-support-message pipeline distribution (across all {total_facts} facts)")
        print(f"{'Category':<18} {'Count':>7} {'Fraction':>10}")
        print("-" * 40)
        for category in FACT_CATEGORIES:
            count = overall_counts[category]
            print(f"{category:<18} {count:>7} {count/total_facts:>9.1%}")
        print("-" * 40)
        print(f"{'TOTAL':<18} {total_facts:>7}")

    by_size: Dict[int, Dict[str, int]] = {}
    for result in results:
        support_set_size = int(result.get("support_set_size") or len(result.get("per_fact_results", [])))
        if support_set_size not in by_size:
            by_size[support_set_size] = {category: 0 for category in FACT_CATEGORIES}
            by_size[support_set_size]["_total_facts"] = 0
            by_size[support_set_size]["_traces"] = 0
        by_size[support_set_size]["_traces"] += 1
        counts = _fact_category_counts(result.get("per_fact_results", []))
        for category, count in counts.items():
            by_size[support_set_size][category] += count
            by_size[support_set_size]["_total_facts"] += count

    if by_size:
        print("\nPer-support-message distribution by support_set_size:")
        header = f"{'N':>3}  {'not_stored':>12} {'summary_err':>12} {'not_retrv':>11} {'correct':>9}  {'traces':>7}"
        print(header)
        print("-" * len(header))
        for support_set_size in sorted(by_size):
            data = by_size[support_set_size]
            total_size_facts = data["_total_facts"] or 1
            print(
                f"{support_set_size:>3}"
                f"  {data['not_stored']:>5} ({data['not_stored']/total_size_facts:>5.1%})"
                f"  {data['summary_error']:>5} ({data['summary_error']/total_size_facts:>5.1%})"
                f"  {data['not_retrieved']:>5} ({data['not_retrieved']/total_size_facts:>5.1%})"
                f"  {data['correct']:>5} ({data['correct']/total_size_facts:>5.1%})"
                f"  {data['_traces']:>7}"
            )
        print("-" * len(header))

    print(f"\n{'Judge model:':<26} {judge_model}")
    print(f"{'Judge input tokens:':<26} {judge_input_tokens:,}")
    print(f"{'Judge output tokens:':<26} {judge_output_tokens:,}")
    print(f"{'Judge cost:':<26} ${judge_cost:.4f}")
    print("=" * 72)


def save_outputs(results: list, output_dir: Path, ts: str) -> Tuple[Path, Path]:
    json_path = output_dir / f"analysis_{ts}.json"
    csv_path = output_dir / f"analysis_{ts}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "example_id",
        "system",
        "question_type",
        "domain",
        "support_set_size",
        "judge_result",
        "error_type",
        "correctly_invoked",
        "question",
        "ground_truth_answer",
        "model_answer",
        "n_not_stored",
        "n_summary_error",
        "n_not_retrieved",
        "n_correct",
        "frac_not_stored",
        "frac_summary_error",
        "frac_not_retrieved",
        "frac_correct",
        "invocation_reasoning",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            per_fact_results = result.get("per_fact_results") or []
            counts = _fact_category_counts(per_fact_results)
            denom = len(per_fact_results) or 1
            writer.writerow(
                {
                    "example_id": result.get("example_id", ""),
                    "system": result.get("system", ""),
                    "question_type": result.get("question_type", ""),
                    "domain": result.get("domain", ""),
                    "support_set_size": result.get("support_set_size", ""),
                    "judge_result": result.get("judge_result", ""),
                    "error_type": result.get("error_type", ""),
                    "correctly_invoked": result.get("correctly_invoked", ""),
                    "question": result.get("graded_question", ""),
                    "ground_truth_answer": result.get("ground_truth_answer", ""),
                    "model_answer": result.get("model_answer", ""),
                    "n_not_stored": counts["not_stored"],
                    "n_summary_error": counts["summary_error"],
                    "n_not_retrieved": counts["not_retrieved"],
                    "n_correct": counts["correct"],
                    "frac_not_stored": f"{counts['not_stored']/denom:.3f}",
                    "frac_summary_error": f"{counts['summary_error']/denom:.3f}",
                    "frac_not_retrieved": f"{counts['not_retrieved']/denom:.3f}",
                    "frac_correct": f"{counts['correct']/denom:.3f}",
                    "invocation_reasoning": result.get("invocation_reasoning", ""),
                }
            )

    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade and classify MemDaily errors in a coexisting-facts-style pipeline."
    )
    parser.add_argument(
        "--traces",
        default=None,
        help="Path to MemDaily traces JSON. Auto-detects the most recent if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to the traces file directory.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Judge model used for analysis.",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or ""
    if not api_key:
        sys.exit("Error: set OPENAI_API_KEY or OPENAI_KEY.")

    if args.traces:
        traces_path = Path(args.traces)
        if not traces_path.is_absolute():
            traces_path = PROJECT_ROOT / traces_path
    else:
        traces_path = auto_detect_traces_file(DEFAULT_RESULTS_DIR)

    print(f"Loading traces from: {traces_path}")
    graded_traces = load_graded_traces(traces_path)
    print(f"Total graded traces: {len(graded_traces)}")

    if args.limit:
        graded_traces = graded_traces[: args.limit]
        print(f"Limiting to first {args.limit} traces.")

    output_dir = Path(args.output_dir) if args.output_dir else traces_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_model = resolve_model_name(args.model)
    if resolved_model != args.model:
        print(f"Resolved judge model alias: {args.model} -> {resolved_model}")

    client = OpenAI(api_key=api_key)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    results: List[Optional[dict]] = [None] * len(graded_traces)
    token_accumulator: List[Tuple[int, int, int]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(analyze_trace, client, resolved_model, trace): i
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
    judge_cost = compute_cost(resolved_model, judge_input_tokens, judge_output_tokens)

    json_path, csv_path = save_outputs(final_results, output_dir, ts)
    print_summary(final_results, judge_input_tokens, judge_output_tokens, judge_cost, resolved_model)
    print("\nResults saved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main()
