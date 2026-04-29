"""
analyze_errors.py -- Grade and classify errors in persona-retrieval traces.

Reads the output of evaluate_persona_retrieval.py and grades each graded question
using four focused LLM calls in sequence, short-circuiting on the first failure:

  Call 1 — Storage check:    is the entity's essay present in ALL_MEMORIES at all?
  Call 2 — Summary check:    is the stored version identity-faithful (entity name +
                              key memorable details preserved)?
  Call 3 — Retrieval check:  was the essay included in RETRIEVED_MEMORIES?
  Call 4 — Final check (depends on question_type):
      base_question        -> invocation: did the model use the essay to answer correctly?
      misleading_question  -> abstention: did the model refuse to apply the entity's
                                          essay to the distractor named in the question?

Error taxonomy:
  base questions:
    storage_error    — essay absent from ALL_MEMORIES
    summary_error    — present but identity/key facts destroyed
    retrieval_error  — stored faithfully but not retrieved
    reasoning_error  — retrieved faithfully but model answered incorrectly
  misleading questions:
    summary_error    — stored version stripped identity, enabling misuse
    reasoning_error  — entity clearly identified in retrieved memory but model
                       applied it to the distractor anyway
    correct (vacuous) — storage or retrieval failed, so model abstained by default.
                        Still counts as correct outcome but flagged via vacuous_pass=True.

Usage:
  uv run python playground/custom_persona_retrieval/analyze_errors.py --traces path/to/traces.json
  uv run python playground/custom_persona_retrieval/analyze_errors.py  # auto-detects most recent
  uv run python playground/custom_persona_retrieval/analyze_errors.py --limit 5
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

STORAGE_SYSTEM = """You are checking whether a personal essay about a named entity was stored in a
memory system. The essay may have been paraphrased, compressed, or split across multiple
memory entries, but the entity's identity (name) and the substantive details from the
essay must still be recoverable to count as present."""

STORAGE_USER = """ENTITY:
{entity}

ORIGINAL ESSAY (what we want to find in the store):
{original_essay}

ALL_MEMORIES (complete memory store):
{all_memories_formatted}

Is the essay's content about {entity} present in ALL_MEMORIES, even if paraphrased,
compressed, or split across multiple entries? Presence requires that the entity be
identifiable AND that the substantive personal details from the essay be recoverable."""

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

SUMMARY_SYSTEM = """You are assessing the quality of a stored personal essay.

The essay has already been confirmed to exist somewhere in the memory store. Your job is
to evaluate whether the stored version preserves IDENTITY and KEY DETAILS faithfully —
in a way that would support correct downstream reasoning.

A stored version has a SUMMARY ERROR if ANY of the following apply:
- The entity's name was dropped or changed, so the stored details cannot be tied back
  to this specific person
- Specific, idiosyncratic details (allergies, possessions, rituals, hard constraints,
  strong preferences) were lost or generalized to the point that a reader could not
  recover the original meaning
- The stored version is so vague or generic that it would not distinguish this entity
  from a generic person with the same role/hobby
- Details were merged with or attributed to a different entity

A stored version is FAITHFUL if the entity is clearly named and the key memorable
details from the essay can be recovered from the relevant memory or memories."""

SUMMARY_USER = """ENTITY:
{entity}

ORIGINAL ESSAY:
{original_essay}

ALL_MEMORIES (the essay IS confirmed present somewhere in here):
{all_memories_formatted}

Find the memory entry — or set of entries — corresponding to {entity}'s essay and
assess whether the stored version preserves identity and the key details, or whether
it has a summary error."""

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

RETRIEVAL_SYSTEM = """You are checking whether a specific entity's essay was included in the memories
retrieved and shown to an AI model when it answered a question."""

RETRIEVAL_USER = """ORIGINAL QUESTION (used to search the memory store):
{question}

ENTITY:
{entity}

ORIGINAL ESSAY (what we want to see in the retrieved set):
{original_essay}

RETRIEVED_MEMORIES (shown to the model):
{retrieved_memories}

Was the essay about {entity} included in RETRIEVED_MEMORIES, even if paraphrased or
split across multiple entries? Presence requires that the entity be identifiable AND
that the substantive personal details from the essay be recoverable from what was
shown to the model."""

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
# Call 4a — Invocation check (base / non-misleading questions)
# ---------------------------------------------------------------------------

INVOCATION_SYSTEM = """You are checking whether an AI model correctly used a retrieved essay to answer a
question about the named entity.

The model was shown the relevant memory and must answer using the entity-specific
details from the essay (not generic norms)."""

INVOCATION_USER = """ENTITY:
{entity}

ORIGINAL ESSAY (was present in retrieved memories):
{original_essay}

QUESTION (asked about {entity}):
{question}

GROUND TRUTH ANSWER:
{ground_truth_answer}

MODEL RESPONSE:
{llm_response}

Did the model correctly use the essay's specific details to answer the question? The
answer must align with the ground truth and reflect the essay's specifics, not generic
information."""

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
# Call 4b — Abstention check (misleading questions)
# ---------------------------------------------------------------------------

ABSTENTION_SYSTEM = """You are checking whether an AI model correctly ABSTAINED when asked about a person
the model has no information on.

The asker's question names a DISTRACTOR — a different person who is NOT the entity in
the retrieved essay. The correct behavior is to acknowledge no information is available
about the distractor (or to ask for clarification). Applying the entity's essay details
to the distractor — by treating them as if they were the entity, or by silently using
the entity's specifics to answer about the distractor — is INCORRECT."""

ABSTENTION_USER = """ENTITY (the person the essay is actually about):
{entity}

DISTRACTOR (the person named in the question — NOT the entity):
{distractor}

ORIGINAL ESSAY (about {entity}, was present in retrieved memories):
{original_essay}

QUESTION (asked about {distractor}):
{question}

MODEL RESPONSE:
{llm_response}

Did the model correctly abstain — acknowledge it has no information about
{distractor} — rather than applying {entity}'s essay details to {distractor}? The model
abstains correctly even if it generally answers using common knowledge, as long as it
does NOT smuggle in {entity}-specific details. Conflating the two people, or using the
essay as if it were about {distractor}, is an abstention failure."""

ABSTENTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "abstention_check",
        "schema": {
            "type": "object",
            "properties": {
                "correctly_abstained": {"type": "boolean"},
                "abstention_reasoning": {"type": "string"},
            },
            "required": ["correctly_abstained", "abstention_reasoning"],
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
            "Run evaluate_persona_retrieval.py first, or pass --traces explicitly."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_graded_traces(data: dict) -> List[dict]:
    """Pull graded traces from a full traces_*.json (when graded_traces is missing)."""
    num_storage_convs = data["run_metadata"]["num_storage_convs"]
    question_specs = data.get("question_specs") or []
    results = data["evaluation_summary"]["results"]
    question_results = results[num_storage_convs:]

    graded = []
    for result, spec in zip(question_results, question_specs):
        traces = result["traces"]
        if not traces:
            continue
        trace = traces[0]
        graded.append({
            "entity": spec["entity"],
            "distractor": spec["distractor"],
            "entity_facts": spec["entity_facts"],
            "question_type": spec["question_type"],
            "ground_truth_answer": spec["ground_truth_answer"],
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

    Call 1 — Storage:    essay in ALL_MEMORIES?       No  → storage_error (or vacuous correct for misleading).
    Call 2 — Summary:    stored version faithful?     No  → summary_error.
    Call 3 — Retrieval:  essay in RETRIEVED_MEMORIES? No  → retrieval_error (or vacuous correct for misleading).
    Call 4a — Invocation (base):    answered correctly? No  → reasoning_error. Yes → correct.
    Call 4b — Abstention (misleading): abstained?       No  → reasoning_error. Yes → correct.

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
    result["correctly_abstained"] = None
    result["abstention_reasoning"] = None
    result["vacuous_pass"] = False
    result["error_type"] = None
    result["analysis_error"] = None

    total_in = 0
    total_out = 0

    question_type = trace.get("question_type", "base")
    is_misleading = question_type == "misleading"
    entity = (trace.get("entity") or "").strip()
    distractor = (trace.get("distractor") or "").strip()
    original_essay = (trace.get("entity_facts") or [""])[0]
    retrieved = trace.get("retrieved_memories") or "(none)"
    all_memories_fmt = format_all_memories(all_memories)
    question = trace.get("question", "")

    try:
        # ── Call 1: Storage check ────────────────────────────────────────────
        storage_data, in_tok, out_tok = _call(
            client, model,
            system=STORAGE_SYSTEM,
            user=STORAGE_USER.format(
                entity=entity,
                original_essay=original_essay,
                all_memories_formatted=all_memories_fmt,
            ),
            schema=STORAGE_SCHEMA,
        )
        total_in += in_tok
        total_out += out_tok
        result["fact_in_store"] = storage_data["fact_in_store"]
        result["storage_reasoning"] = storage_data["storage_reasoning"]

        if not storage_data["fact_in_store"]:
            if is_misleading:
                result["judge_result"] = "correct"
                result["error_type"] = "correct"
                result["vacuous_pass"] = True
                result["judge_reasoning"] = (
                    "Vacuous pass: essay not stored, so the model could not have applied "
                    "it to the distractor. Storage detail: "
                    + storage_data["storage_reasoning"]
                )
            else:
                result["judge_result"] = "incorrect"
                result["error_type"] = "storage_error"
                result["judge_reasoning"] = storage_data["storage_reasoning"]
            return result, total_in, total_out

        # ── Call 2: Summary check ────────────────────────────────────────────
        summary_data, in_tok, out_tok = _call(
            client, model,
            system=SUMMARY_SYSTEM,
            user=SUMMARY_USER.format(
                entity=entity,
                original_essay=original_essay,
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
                question=question,
                entity=entity,
                original_essay=original_essay,
                retrieved_memories=retrieved,
            ),
            schema=RETRIEVAL_SCHEMA,
        )
        total_in += in_tok
        total_out += out_tok
        result["entity_facts_in_retrieved"] = retrieval_data["entity_facts_in_retrieved"]
        result["retrieval_reasoning"] = retrieval_data["retrieval_reasoning"]

        if not retrieval_data["entity_facts_in_retrieved"]:
            if is_misleading:
                result["judge_result"] = "correct"
                result["error_type"] = "correct"
                result["vacuous_pass"] = True
                result["judge_reasoning"] = (
                    "Vacuous pass: entity's essay was not retrieved for the distractor "
                    "question, so the model abstained by default. Retrieval detail: "
                    + retrieval_data["retrieval_reasoning"]
                )
            else:
                result["judge_result"] = "incorrect"
                result["error_type"] = "retrieval_error"
                result["judge_reasoning"] = retrieval_data["retrieval_reasoning"]
            return result, total_in, total_out

        # ── Call 4: Invocation (base) or Abstention (misleading) ─────────────
        if is_misleading:
            abstention_data, in_tok, out_tok = _call(
                client, model,
                system=ABSTENTION_SYSTEM,
                user=ABSTENTION_USER.format(
                    entity=entity,
                    distractor=distractor,
                    original_essay=original_essay,
                    question=question,
                    llm_response=trace["llm_response"],
                ),
                schema=ABSTENTION_SCHEMA,
            )
            total_in += in_tok
            total_out += out_tok
            result["correctly_abstained"] = abstention_data["correctly_abstained"]
            result["abstention_reasoning"] = abstention_data["abstention_reasoning"]

            if not abstention_data["correctly_abstained"]:
                result["judge_result"] = "incorrect"
                result["error_type"] = "reasoning_error"
                result["judge_reasoning"] = abstention_data["abstention_reasoning"]
            else:
                result["judge_result"] = "correct"
                result["error_type"] = "correct"
                result["judge_reasoning"] = abstention_data["abstention_reasoning"]
        else:
            invocation_data, in_tok, out_tok = _call(
                client, model,
                system=INVOCATION_SYSTEM,
                user=INVOCATION_USER.format(
                    entity=entity,
                    original_essay=original_essay,
                    question=question,
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
                result["error_type"] = "correct"
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

    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total traces:       {total}")
    if total:
        print(f"Correct:            {correct}  ({correct/total:.1%})")
        print(f"Incorrect:          {incorrect}  ({incorrect/total:.1%})")

    qtypes = ["base", "misleading"]
    for qt in qtypes:
        sub = [r for r in results if r.get("question_type") == qt]
        if not sub:
            continue
        sub_correct = sum(1 for r in sub if r.get("judge_result") == "correct")
        sub_vacuous = sum(1 for r in sub if r.get("vacuous_pass"))
        print(f"\n  {qt}_question: {sub_correct}/{len(sub)} ({sub_correct/len(sub):.1%}) correct"
              + (f", of which {sub_vacuous} vacuous (storage/retrieval failed)" if sub_vacuous else ""))

    error_types = ["storage_error", "summary_error", "retrieval_error", "reasoning_error"]
    counts: Dict[str, Dict[str, int]] = {et: {"base": 0, "misleading": 0} for et in error_types}
    counts["analysis_failed"] = {"base": 0, "misleading": 0}
    for r in results:
        if r.get("judge_result") != "incorrect":
            continue
        et = r.get("error_type") or "analysis_failed"
        qt = r.get("question_type", "base")
        if et not in counts:
            counts[et] = {"base": 0, "misleading": 0}
        counts[et][qt] = counts[et].get(qt, 0) + 1

    if incorrect > 0:
        print(f"\n{'Error type':<22} {'base':>8} {'misleading':>12} {'Total':>8} {'Share':>8}")
        print("-" * 62)
        for et in [*error_types, "analysis_failed"]:
            base_c = counts[et]["base"]
            mis_c = counts[et]["misleading"]
            tot = base_c + mis_c
            if tot == 0:
                continue
            print(f"{et:<22} {base_c:>8} {mis_c:>12} {tot:>8} {tot/incorrect:>7.1%}")
        print("-" * 62)
        print(f"{'TOTAL INCORRECT':<22} {'':>8} {'':>12} {incorrect:>8}")

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
        "conversation_id", "question_type", "judge_result", "error_type", "vacuous_pass",
        "fact_in_store", "summary_check_passed", "entity_facts_in_retrieved",
        "correctly_invoked", "correctly_abstained",
        "entity", "distractor", "ground_truth_answer", "question",
        "judge_reasoning", "storage_reasoning", "summary_reasoning",
        "retrieval_reasoning", "invocation_reasoning", "abstention_reasoning",
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
        description="Grade and classify errors in persona-retrieval traces."
    )
    parser.add_argument(
        "--traces",
        default=None,
        help="Path to traces JSON from evaluate_persona_retrieval.py. Auto-detects most recent if omitted.",
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

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
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
    print("\nResults saved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main()
