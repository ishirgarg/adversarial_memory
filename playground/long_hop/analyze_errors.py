"""
analyze_errors.py — Grade and classify errors in long-hop chain traces.

For each chain (graded trace), every fact in the chain is independently
classified through a 3-stage pipeline:

  Stage 1 — Storage check (parallel):   is the fact present in ALL_MEMORIES?
  Stage 2 — Summary check (parallel):   is the stored version faithful?
  Stage 3 — Retrieval check (parallel): was the fact in RETRIEVED_MEMORIES?

Memory systems may legitimately MERGE several facts of a chain into a single
memory entry (e.g. "A -> B and B -> C" stored as one summary). Such merging
counts as STORED/RETRIEVED/FAITHFUL as long as the fact's specific link is
still recoverable from the merged memory.

Per-fact categories:
  not_stored    — fact's link absent from ALL_MEMORIES (no merged entry covers it)
  summary_error — link present but corrupted / merged in a way that loses identity
  not_retrieved — link stored faithfully but not surfaced at retrieval time
  correct       — link faithfully stored AND retrieved

If ALL facts reach "correct", an invocation check determines whether the model
correctly chained them to produce the ground-truth terminal entity
(reasoning_error vs. correct).

Per-fact distributions are reported overall and broken down by hop_count.

Usage:
  uv run python playground/long_hop/analyze_errors.py --traces path/to/traces.json
  uv run python playground/long_hop/analyze_errors.py  # auto-detects most recent
"""

import argparse
import csv
import json
import os
import re
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
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_WORKERS = 8
MAX_RETRIES = 3
MAX_INNER_WORKERS = 8


# ---------------------------------------------------------------------------
# Per-fact: Storage check
# ---------------------------------------------------------------------------

FACT_STORAGE_SYSTEM = """You are checking whether a specific factual statement (one link in a multi-hop reasoning chain) is preserved in a memory store.

The memory store may have stored the fact verbatim, paraphrased it, OR merged
several chain-links together into one combined memory. Any of these forms
counts as STORED — as long as the SPECIFIC link asserted by the target fact
can still be unambiguously identified from at least one memory entry. If the
link's two specific entities and the relation between them are recoverable,
mark fact_in_store=true."""

FACT_STORAGE_USER = """TARGET FACT (one link of a reasoning chain):
{target_message}

ALL_MEMORIES (complete memory store, possibly with merged entries):
{all_memories_formatted}

Is the target fact present in ALL_MEMORIES — verbatim, paraphrased, or as part
of a merged memory entry that still preserves the specific link between the
target fact's two entities? Mark fact_in_store=true only if the precise
relationship between the two specific entities is unambiguously recoverable."""

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

FACT_SUMMARY_SYSTEM = """You are assessing the quality of a stored chain-link.

The link has already been confirmed to exist somewhere in the memory store
(possibly inside a merged memory entry). Your job is to evaluate whether the
stored version faithfully preserves the specific link in a way that would
support correct downstream chain reasoning.

A stored version has a SUMMARY ERROR if ANY of the following apply:
- The link's two entities were collapsed/renamed/swapped, breaking identity.
- The relation between them was corrupted, weakened, or replaced.
- The link was over-merged with unrelated facts so that the specific link can
  no longer be cleanly extracted (e.g. the entities are listed but not in a
  way that preserves which-relates-to-which).
- A critical detail (e.g. direction of the relation) was lost.

A stored version is FAITHFUL if both entities of the link are clearly named in
some memory entry and the specific relation between them is unambiguous —
EVEN IF the memory entry also contains other chain-links from the same chain
(merging is allowed when each individual link is still recoverable)."""

FACT_SUMMARY_USER = """TARGET FACT (one link of a reasoning chain):
{target_message}

ALL_MEMORIES (the link IS confirmed present somewhere in here, possibly merged):
{all_memories_formatted}

Find the memory entry (or entries) that cover this link and assess whether the
stored version faithfully preserves the specific relation between the two
entities, or whether it has a summary error."""

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

FACT_RETRIEVAL_SYSTEM = """You are checking whether a specific chain-link was included in the memories
retrieved and shown to an AI model when it answered a multi-hop question.

The retrieved memories may include verbatim, paraphrased, or merged versions.
A merged memory that still contains the link's specific relation counts as
"retrieved"."""

FACT_RETRIEVAL_USER = """TARGET FACT (one link of a reasoning chain):
{target_message}

ORIGINAL QUESTION (used to query the memory store):
{query}

RETRIEVED_MEMORIES (shown to the model when it answered):
{retrieved_memories}

Was the target fact's specific link included in RETRIEVED_MEMORIES, even if
paraphrased or merged with other chain-links? Mark fact_in_retrieved=true only
if the precise relationship between the link's two entities is unambiguously
recoverable from the retrieved memories."""

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
# Question-level: MCQ response parsing (replaces former LLM invocation check)
# ---------------------------------------------------------------------------

VALID_CHOICE_LETTERS = {"A", "B", "C", "D", "E"}
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}")
_LETTER_RE = re.compile(r"\b([ABCDE])\b")


def parse_selected_choice(response: str) -> Optional[str]:
    """Extract the model's chosen letter from its MCQ answer.

    Tries the structured JSON schema first ({"selected_choice": "A"}), then
    falls back to any standalone uppercase letter A-E that appears at the end
    of the response. Returns None if nothing parseable is found.
    """
    if not response:
        return None
    text = response.strip()

    # Pass 1: try parsing the entire response as JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            val = obj.get("selected_choice") or obj.get("answer") or obj.get("choice")
            if isinstance(val, str):
                letter = val.strip().upper()
                if letter in VALID_CHOICE_LETTERS:
                    return letter
    except json.JSONDecodeError:
        pass

    # Pass 2: hunt for any embedded JSON object.
    for match in _JSON_OBJ_RE.finditer(text):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            val = obj.get("selected_choice") or obj.get("answer") or obj.get("choice")
            if isinstance(val, str):
                letter = val.strip().upper()
                if letter in VALID_CHOICE_LETTERS:
                    return letter

    # Pass 3: fall back to the last lone uppercase A-E in the response.
    matches = _LETTER_RE.findall(text)
    if matches:
        return matches[-1]
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FACT_CATEGORIES = ["not_stored", "summary_error", "not_retrieved", "correct"]
ERROR_TYPE_PRIORITY = ["not_stored", "summary_error", "not_retrieved"]


def collapse_error_type(per_fact_results: list, correctly_invoked) -> str:
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
            "hop_count": row.get("hop_count"),
            "facts": row["facts"],
            "answer_chain": row.get("answer_chain", []),
            "ground_truth_answer": row["ground_truth_answer"],
            "support_set_size": len(row["facts"]),
            "question_stem": row.get("graded_question_stem"),
            "question": trace["query"],
            "choices": row.get("choices"),
            "correct_choice": row.get("correct_choice"),
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
    """Classify each chain-link through a 3-stage pipeline, then run an
    invocation check if all links reach 'correct'.

    Per-fact categories: not_stored | summary_error | not_retrieved | correct
    Question-level result: correct | incorrect (+ reasoning_error if all facts correct)

    Never raises. Returns (result_dict, total_input_tokens, total_output_tokens).
    """
    target_messages = trace.get("facts") or []
    N = len(target_messages)

    result = {k: v for k, v in trace.items() if k != "formatted_prompt"}
    result["per_fact_results"] = []
    result["judge_result"] = None
    result["error_type"] = None
    result["correctly_invoked"] = None
    result["invocation_reasoning"] = None
    result["selected_choice"] = None
    result["correct_choice"] = trace.get("correct_choice")
    result["final_answer_correct"] = None
    result["analysis_error"] = None

    total_in = 0
    total_out = 0

    all_memories_fmt = format_all_memories(all_memories)
    retrieved = trace.get("retrieved_memories") or "(none)"

    fact_results = [
        {
            "target_message": target_messages[i],
            "fact_index": i,
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
        # ── Stage 1: Storage check ───────────────────────────────────────────
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

        # ── Stage 2: Summary check (only stored facts) ───────────────────────
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

        # ── Stage 3: Retrieval check (only summary-passing facts) ────────────
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

        # ── Invocation check (deterministic MCQ parsing) ────────────────────
        # The eval prompt asks the model for {"selected_choice": "<letter>"}.
        # Compare the parsed letter against the dataset's correct_choice; this
        # replaces the previous LLM-judge invocation check.
        selected = parse_selected_choice(trace.get("llm_response", "") or "")
        correct_choice = trace.get("correct_choice")
        result["selected_choice"] = selected
        if selected is None or correct_choice is None:
            final_correct = False
            reasoning = (
                "could not parse a choice letter from the model response"
                if selected is None
                else "trace has no correct_choice — regenerate the dataset with the MCQ-aware generator"
            )
        else:
            final_correct = (selected == correct_choice)
            reasoning = (
                f"model selected {selected}; correct choice is {correct_choice}"
            )
        result["final_answer_correct"] = final_correct

        all_correct = N > 0 and all(fr["category"] == "correct" for fr in fact_results)
        if all_correct:
            result["correctly_invoked"] = final_correct
            result["invocation_reasoning"] = reasoning
            result["judge_result"] = "correct" if final_correct else "incorrect"
        else:
            result["invocation_reasoning"] = reasoning
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
    print("LONG-HOP CHAIN ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total chains:         {total}")
    if total:
        print(f"Correct:              {correct}  ({correct/total:.1%})")
        print(f"Incorrect:            {incorrect}  ({incorrect/total:.1%})")

    # Raw MCQ accuracy (model's chosen letter == correct letter), independent
    # of whether the per-fact pipeline was clean.
    parseable = sum(1 for r in results if r.get("selected_choice") is not None)
    final_correct = sum(1 for r in results if r.get("final_answer_correct"))
    if total:
        print(
            f"MCQ letter-match:     {final_correct}  ({final_correct/total:.1%})  "
            f"[parsed letter from {parseable}/{total} responses]"
        )

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

    # ── Per-fact pipeline distribution ──────────────────────────────────────
    agg: Dict[str, int] = {c: 0 for c in FACT_CATEGORIES}
    total_facts = 0
    for r in results:
        counts = _fact_category_counts(r.get("per_fact_results", []))
        for cat, n in counts.items():
            agg[cat] += n
            total_facts += n

    if total_facts:
        print(f"\nPer-fact pipeline distribution (across all {total_facts} facts)")
        print(f"{'Category':<18} {'Count':>7} {'Fraction':>10}")
        print("-" * 38)
        for cat in FACT_CATEGORIES:
            n = agg[cat]
            print(f"{cat:<18} {n:>7} {n/total_facts:>9.1%}")
        print("-" * 38)
        print(f"{'TOTAL':<18} {total_facts:>7}")

    # ── Per-fact distribution by hop_count ─────────────────────────────────
    by_hop: Dict[int, Dict] = {}
    for r in results:
        pfr = r.get("per_fact_results", [])
        n = int(r.get("hop_count") or len(pfr))
        if n not in by_hop:
            by_hop[n] = {c: 0 for c in FACT_CATEGORIES}
            by_hop[n]["_total_facts"] = 0
            by_hop[n]["_traces"] = 0
        by_hop[n]["_traces"] += 1
        counts = _fact_category_counts(pfr)
        for cat, cnt in counts.items():
            by_hop[n][cat] += cnt
            by_hop[n]["_total_facts"] += cnt

    if by_hop:
        print("\nPer-fact distribution by hop_count:")
        header = f"{'hop':>3}  {'not_stored':>12} {'summary_err':>12} {'not_retrv':>11} {'correct':>9}  {'chains':>7}"
        print(header)
        print("-" * len(header))
        for n in sorted(by_hop):
            d = by_hop[n]
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

    # ── Accuracy by hop_count ──────────────────────────────────────────────
    by_hop_acc: Dict[int, Dict[str, int]] = {}
    for r in results:
        h = int(r.get("hop_count") or 0)
        if h not in by_hop_acc:
            by_hop_acc[h] = {"total": 0, "correct": 0}
        by_hop_acc[h]["total"] += 1
        if r.get("judge_result") == "correct":
            by_hop_acc[h]["correct"] += 1

    if by_hop_acc:
        print("\nAccuracy by hop_count:")
        header = f"{'hop':<6} {'correct':>8} {'total':>6} {'acc':>7}"
        print(header)
        print("-" * len(header))
        for h in sorted(by_hop_acc):
            d = by_hop_acc[h]
            acc = d["correct"] / d["total"] if d["total"] else 0
            print(f"{h:<6} {d['correct']:>8} {d['total']:>6} {acc:>6.1%}")
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
        "example_id", "hop_count", "support_set_size",
        "conversation_id", "judge_result", "error_type", "correctly_invoked",
        "selected_choice", "correct_choice", "final_answer_correct",
        "ground_truth_answer", "question_stem",
        "n_facts",
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
            row["n_facts"] = n
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
        description="Grade and classify errors in long-hop chain traces."
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
