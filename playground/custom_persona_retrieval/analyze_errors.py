"""
analyze_errors.py -- Grade and classify errors in persona-retrieval traces.

This is the BATCHED variant: instead of one judge call per trace per stage, we
group 20 traces into a single prompt for each stage, so the (large) memory store
is only sent once per batch. Within a run, batches run their 4-stage pipeline
in parallel via a ThreadPoolExecutor (default 50 workers).

Pipeline per batch (short-circuits on first failure for each trace inside the
batch — failed traces are excluded from later stages of the same batch):

  Call 1 — Storage check:    is each entity's essay present in ALL_MEMORIES?
  Call 2 — Summary check:    is the stored version identity-faithful?
  Call 3 — Retrieval check:  was the essay included in RETRIEVED_MEMORIES?
  Call 4 — Final check:
      base_question        -> invocation: did the model use the essay correctly?
      misleading_question  -> abstention: did the model refuse to apply the
                                          entity's essay to the distractor?

Error taxonomy is unchanged from the per-trace version.
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
from typing import Any, Callable, Dict, List, Optional, Tuple

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
DEFAULT_WORKERS = 50
DEFAULT_BATCH_SIZE = 20
MAX_RETRIES = 12  # outer retries on top of OpenAI client's own 429 backoff


# ===========================================================================
# Shared system prompts (one per stage)
# ===========================================================================

STORAGE_SYSTEM = """You are checking whether personal essays about named entities are stored in a
memory system. Each essay may have been paraphrased, compressed, or split across multiple
memory entries, but the entity's identity (name) and the substantive details from the
essay must still be recoverable to count as present.

You will receive ALL_MEMORIES (one shared store) and a list of independent ENTRIES, each
identified by an integer id. For each entry, return one verdict in the "results" array,
preserving the id."""


SUMMARY_SYSTEM = """You are assessing the quality of stored personal essays.

Each essay has already been confirmed to exist somewhere in the memory store. Your job
is to evaluate whether the stored version preserves IDENTITY and KEY DETAILS faithfully —
in a way that would support correct downstream reasoning.

A stored version has a SUMMARY ERROR if ANY of the following apply:
- The entity's name was dropped or changed, so the stored details cannot be tied back
  to this specific person
- Specific, idiosyncratic details (allergies, possessions, rituals, hard constraints,
  strong preferences) were lost or generalized to the point that a reader could not
  recover the original meaning
- The stored version is so vague or generic that it would not distinguish this entity
  from a generic person with the same role/hobby

The following are NOT summary errors — treat them as acceptable:
- Multiple distinct entries existing under the same name. The entity may share a name
  with other people described in the store. As long as the entity's own essay is
  recoverable from at least one entry (or set of entries), this is fine — do not flag
  it as a merged-identity error.
- The store containing memories about other people, whether related to the entity or
  unrelated. Only the entity's own essay needs to be faithfully preserved; the
  presence of additional people is acceptable.
- The essay being split across multiple entries, paraphrased, or compressed, as long
  as the entity's identity and key memorable details are recoverable.

A stored version is FAITHFUL if the entity is clearly named (in at least one entry)
and the key memorable details from the essay can be recovered from the relevant
memory or memories tied to that entity.

You will receive ALL_MEMORIES (one shared store) and a list of ENTRIES, each
identified by an integer id. For each entry, return one verdict in the "results"
array, preserving the id."""


RETRIEVAL_SYSTEM = """You are checking whether each entity's essay was included in the memories retrieved
and shown to an AI model when it answered a question. Each entry has its own
RETRIEVED_MEMORIES (the subset shown for that question). Presence requires that the
entity be identifiable AND that the substantive personal details from the essay be
recoverable from what was shown to the model. Return one verdict per entry under
"results", preserving the id."""


INVOCATION_SYSTEM = """You are checking whether an AI model correctly used a retrieved essay to answer a
question about the named entity. The model was shown the relevant memory and must
answer using the entity-specific details from the essay (not generic norms). Return
one verdict per entry in "results", preserving the id."""


ABSTENTION_SYSTEM = """You are checking whether an AI model correctly ABSTAINED when asked about a person
the model has no information on.

Each entry's question names a DISTRACTOR — a different person who is NOT the entity in
the retrieved essay. The correct behavior is to acknowledge no information is available
about the distractor (or to ask for clarification). Applying the entity's essay details
to the distractor — by treating them as if they were the entity, or by silently using
the entity's specifics to answer about the distractor — is INCORRECT. Note: the model
may legitimately mention information about other people stored in memory if it does not
attribute the entity's essay details to the distractor. Return one verdict per entry in
"results", preserving the id."""


# ===========================================================================
# Schema helpers — every stage returns {"results": [{...}, ...]}
# ===========================================================================


def _array_schema(name: str, item_props: dict, required: list) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": item_props,
                            "required": required,
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["results"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


STORAGE_SCHEMA = _array_schema(
    "storage_check_batch",
    {
        "id": {"type": "integer"},
        "fact_in_store": {"type": "boolean"},
        "storage_reasoning": {"type": "string"},
    },
    ["id", "fact_in_store", "storage_reasoning"],
)

SUMMARY_SCHEMA = _array_schema(
    "summary_check_batch",
    {
        "id": {"type": "integer"},
        "summary_check_passed": {"type": "boolean"},
        "summary_reasoning": {"type": "string"},
    },
    ["id", "summary_check_passed", "summary_reasoning"],
)

RETRIEVAL_SCHEMA = _array_schema(
    "retrieval_check_batch",
    {
        "id": {"type": "integer"},
        "entity_facts_in_retrieved": {"type": "boolean"},
        "retrieval_reasoning": {"type": "string"},
    },
    ["id", "entity_facts_in_retrieved", "retrieval_reasoning"],
)

INVOCATION_SCHEMA = _array_schema(
    "invocation_check_batch",
    {
        "id": {"type": "integer"},
        "correctly_invoked": {"type": "boolean"},
        "invocation_reasoning": {"type": "string"},
    },
    ["id", "correctly_invoked", "invocation_reasoning"],
)

ABSTENTION_SCHEMA = _array_schema(
    "abstention_check_batch",
    {
        "id": {"type": "integer"},
        "correctly_abstained": {"type": "boolean"},
        "abstention_reasoning": {"type": "string"},
    },
    ["id", "correctly_abstained", "abstention_reasoning"],
)


# ===========================================================================
# Helpers
# ===========================================================================


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


# ===========================================================================
# Batch entry formatters (one per stage)
# ===========================================================================


def _essay(t: dict) -> str:
    return (t.get("entity_facts") or [""])[0]


def _fmt_storage_entries(batch: List[dict]) -> str:
    parts = []
    for i, t in enumerate(batch, start=1):
        parts.append(f"[id={i}]\nENTITY: {t.get('entity', '').strip()}\nESSAY: {_essay(t)}")
    return "\n\n".join(parts)


def _fmt_summary_entries(batch: List[dict]) -> str:
    return _fmt_storage_entries(batch)  # same fields suffice


def _fmt_retrieval_entries(batch: List[dict]) -> str:
    parts = []
    for i, t in enumerate(batch, start=1):
        parts.append(
            f"[id={i}]\n"
            f"ENTITY: {t.get('entity', '').strip()}\n"
            f"QUESTION: {t.get('question', '')}\n"
            f"ESSAY: {_essay(t)}\n"
            f"RETRIEVED_MEMORIES:\n{t.get('retrieved_memories') or '(none)'}"
        )
    return "\n\n---\n\n".join(parts)


def _fmt_invocation_entries(batch: List[dict]) -> str:
    parts = []
    for i, t in enumerate(batch, start=1):
        parts.append(
            f"[id={i}]\n"
            f"ENTITY: {t.get('entity', '').strip()}\n"
            f"QUESTION: {t.get('question', '')}\n"
            f"ESSAY: {_essay(t)}\n"
            f"GROUND_TRUTH: {t.get('ground_truth_answer', '')}\n"
            f"MODEL_RESPONSE: {t.get('llm_response', '')}"
        )
    return "\n\n---\n\n".join(parts)


def _fmt_abstention_entries(batch: List[dict]) -> str:
    parts = []
    for i, t in enumerate(batch, start=1):
        parts.append(
            f"[id={i}]\n"
            f"ENTITY (essay is about): {t.get('entity', '').strip()}\n"
            f"DISTRACTOR (named in question, NOT the entity): {t.get('distractor', '').strip()}\n"
            f"ESSAY (about the entity): {_essay(t)}\n"
            f"QUESTION (about the distractor): {t.get('question', '')}\n"
            f"MODEL_RESPONSE: {t.get('llm_response', '')}"
        )
    return "\n\n---\n\n".join(parts)


# ===========================================================================
# Batched call
# ===========================================================================


_RETRY_AFTER_RE = __import__("re").compile(r"try again in ([0-9.]+)s", flags=__import__("re").IGNORECASE)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _sleep_after_error(exc: Exception, attempt: int, ctx: str = "") -> None:
    """Sleep an appropriate amount before retry. Parses 'try again in Xs' from
    OpenAI 429 messages when present; otherwise exponential backoff."""
    msg = str(exc)
    m = _RETRY_AFTER_RE.search(msg)
    if m:
        try:
            secs = float(m.group(1)) + 1.0
            wait = min(secs, 60.0)
            _log(f"    retry[{ctx}] attempt={attempt} 429 rate-limit; sleeping {wait:.1f}s")
            time.sleep(wait)
            return
        except ValueError:
            pass
    wait = min(2.0 ** attempt, 60.0)
    short = msg[:160].replace("\n", " ")
    _log(f"    retry[{ctx}] attempt={attempt} backoff {wait:.1f}s after error: {short}")
    time.sleep(wait)


def _call_batch(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    schema: dict,
    expected_n: int,
    ctx: str = "",
) -> Tuple[List[dict], int, int]:
    """Run one batched judge call. Returns (verdicts_in_id_order, in_tok, out_tok).

    Validates that the response has the expected number of verdicts. Retries on
    transport/parse errors. The OpenAI client itself also retries 429s with
    exponential backoff via its `max_retries` setting; this loop wraps that with
    longer waits when the server tells us how long to wait."""
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
            results = data.get("results") or []
            in_tok = resp.usage.prompt_tokens if resp.usage else 0
            out_tok = resp.usage.completion_tokens if resp.usage else 0
            by_id = {r.get("id"): r for r in results}
            verdicts = [by_id.get(i + 1) for i in range(expected_n)]
            if any(v is None for v in verdicts):
                raise ValueError(
                    f"Batched response missing ids; got {sorted(by_id.keys())}, "
                    f"expected 1..{expected_n}"
                )
            return verdicts, in_tok, out_tok
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                _sleep_after_error(e, attempt, ctx=ctx)
    raise RuntimeError(f"Batched call failed after {MAX_RETRIES} attempts: {last_error}") from last_error


# ===========================================================================
# Per-stage launchers
# ===========================================================================


STORAGE_USER_FMT = """ALL_MEMORIES (complete memory store, shared by all entries):
{all_memories_formatted}

For each entry below, determine whether the entity's essay is present in
ALL_MEMORIES, even if paraphrased, compressed, or split across multiple entries.
Presence requires that the entity be identifiable AND that the substantive
personal details from the essay be recoverable.

ENTRIES:
{entries}

Return one verdict per entry under "results", preserving the id (1..{n})."""


SUMMARY_USER_FMT = """ALL_MEMORIES (complete memory store, shared by all entries):
{all_memories_formatted}

For each entry below, the essay IS confirmed present somewhere in ALL_MEMORIES.
Find the memory entry — or set of entries — that correspond to the entity's essay
and assess whether the stored version preserves identity and the key details, or
whether it has a summary error.

Reminder: it is acceptable for the store to contain other entries under the same
name (multiple personas) or memories about other people, related or otherwise.
Only judge whether each entity's own essay is faithfully recoverable from the
entries that describe it.

ENTRIES:
{entries}

Return one verdict per entry under "results", preserving the id (1..{n})."""


RETRIEVAL_USER_FMT = """For each entry, decide whether the entity's essay was included in that entry's
RETRIEVED_MEMORIES, even if paraphrased or split across multiple entries. Presence
requires that the entity be identifiable AND that the substantive personal details
from the essay be recoverable from what was shown to the model.

ENTRIES:
{entries}

Return one verdict per entry under "results", preserving the id (1..{n})."""


INVOCATION_USER_FMT = """For each entry, decide whether the model correctly used the essay's specific
details to answer the question. The answer must align with the GROUND_TRUTH and
reflect the essay's specifics, not generic information.

ENTRIES:
{entries}

Return one verdict per entry under "results", preserving the id (1..{n})."""


ABSTENTION_USER_FMT = """For each entry, decide whether the model correctly abstained — acknowledged it has
no information about the DISTRACTOR — rather than applying the ENTITY's essay
details to the DISTRACTOR. The model abstains correctly even if it generally
answers using common knowledge, as long as it does NOT smuggle in
ENTITY-specific details. Conflating the two people, or using the essay as if it
were about the distractor, is an abstention failure.

ENTRIES:
{entries}

Return one verdict per entry under "results", preserving the id (1..{n})."""


# ===========================================================================
# Run-level orchestration
# ===========================================================================


def _new_result(trace: dict) -> dict:
    r = dict(trace)
    r.update(
        judge_result=None,
        judge_reasoning=None,
        fact_in_store=None,
        storage_reasoning=None,
        summary_check_passed=None,
        summary_reasoning=None,
        entity_facts_in_retrieved=None,
        retrieval_reasoning=None,
        correctly_invoked=None,
        invocation_reasoning=None,
        correctly_abstained=None,
        abstention_reasoning=None,
        vacuous_pass=False,
        error_type=None,
        analysis_error=None,
    )
    return r


def _make_batches(items: List[Tuple[int, dict]], batch_size: int) -> List[List[Tuple[int, dict]]]:
    """Group (index, record) pairs into batches of up to batch_size."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def _run_stage(
    client: OpenAI,
    model: str,
    pool: ThreadPoolExecutor,
    label: str,
    items: List[Tuple[int, dict]],   # (global_index, trace)
    batch_size: int,
    system: str,
    user_template: str,
    fmt_entries: Callable[[List[dict]], str],
    schema: dict,
    template_extras: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Optional[List[dict]]], int, int]:
    """Submit all batches for one stage in parallel, return verdicts (per item)
    in the same order as `items`. Tokens are summed across batches.

    Logs every batch start/end with elapsed time and token usage, plus a stage
    summary at the end."""
    if not items:
        _log(f"  [{label}] (no items, skipping)")
        return [], 0, 0
    batches = _make_batches(items, batch_size)
    stage_t0 = time.monotonic()
    _log(f"  [{label}] {len(items)} items in {len(batches)} batch(es) of up to {batch_size}")

    batch_t0: Dict[int, float] = {}
    futures = {}
    for bidx, batch in enumerate(batches):
        traces_only = [t for _, t in batch]
        user = user_template.format(
            entries=fmt_entries(traces_only),
            n=len(traces_only),
            **(template_extras or {}),
        )
        ctx = f"{label} {bidx+1}/{len(batches)}"
        batch_t0[bidx] = time.monotonic()
        fut = pool.submit(
            _call_batch, client, model, system, user, schema, len(traces_only), ctx,
        )
        futures[fut] = bidx
        _log(f"    [{label}] batch {bidx+1}/{len(batches)} submitted ({len(traces_only)} traces)")

    verdicts_by_batch: List[Optional[List[dict]]] = [None] * len(batches)
    in_tok_total = 0
    out_tok_total = 0
    completed = 0
    for fut in as_completed(futures):
        bidx = futures[fut]
        elapsed = time.monotonic() - batch_t0[bidx]
        completed += 1
        try:
            verdicts, in_tok, out_tok = fut.result()
            verdicts_by_batch[bidx] = verdicts
            in_tok_total += in_tok
            out_tok_total += out_tok
            _log(
                f"    [{label}] batch {bidx+1}/{len(batches)} done in {elapsed:.1f}s "
                f"(in={in_tok:,} out={out_tok:,})  progress={completed}/{len(batches)}"
            )
        except Exception as e:
            verdicts_by_batch[bidx] = [{"_error": str(e)} for _ in batches[bidx]]
            short = str(e)[:200].replace("\n", " ")
            _log(
                f"    [{label}] batch {bidx+1}/{len(batches)} FAILED after {elapsed:.1f}s: {short}  "
                f"progress={completed}/{len(batches)}"
            )

    stage_elapsed = time.monotonic() - stage_t0
    _log(
        f"  [{label}] stage done in {stage_elapsed:.1f}s  "
        f"(in={in_tok_total:,} out={out_tok_total:,})"
    )

    # Re-flatten verdicts in the same order as `items`.
    flat: List[Optional[dict]] = []
    for bidx, batch in enumerate(batches):
        v = verdicts_by_batch[bidx] or [None] * len(batch)
        flat.extend(v)
    return flat, in_tok_total, out_tok_total


def analyze_run(
    client: OpenAI,
    model: str,
    traces: List[dict],
    all_memories: list,
    batch_size: int,
    max_workers: int,
) -> Tuple[List[dict], int, int]:
    """Grade every trace in one run via batched stages. Returns (results, in_tok, out_tok)."""
    results = [_new_result(t) for t in traces]
    all_memories_fmt = format_all_memories(all_memories)

    # Items still alive at each stage carry their global index so we can write
    # verdicts back to the right slot in `results`.
    alive: List[Tuple[int, dict]] = list(enumerate(traces))

    in_tok_total = 0
    out_tok_total = 0

    run_t0 = time.monotonic()
    n_base = sum(1 for t in traces if t.get("question_type") != "misleading")
    n_mis = len(traces) - n_base
    _log(
        f"analyze_run: {len(traces)} traces (base={n_base}, misleading={n_mis})  "
        f"memories={len(all_memories)}  batch_size={batch_size}  workers={max_workers}"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # ── Stage 1: Storage ─────────────────────────────────────────────────
        verdicts, in_tok, out_tok = _run_stage(
            client, model, pool, "storage", alive, batch_size,
            STORAGE_SYSTEM, STORAGE_USER_FMT, _fmt_storage_entries, STORAGE_SCHEMA,
            template_extras={"all_memories_formatted": all_memories_fmt},
        )
        in_tok_total += in_tok
        out_tok_total += out_tok

        next_alive: List[Tuple[int, dict]] = []
        for (gi, trace), v in zip(alive, verdicts):
            r = results[gi]
            if v is None or "_error" in (v or {}):
                r["analysis_error"] = (v or {}).get("_error") or "storage stage failed"
                continue
            r["fact_in_store"] = v["fact_in_store"]
            r["storage_reasoning"] = v["storage_reasoning"]
            if not v["fact_in_store"]:
                if trace.get("question_type") == "misleading":
                    r["judge_result"] = "correct"
                    r["error_type"] = "correct"
                    r["vacuous_pass"] = True
                    r["judge_reasoning"] = (
                        "Vacuous pass: essay not stored, so the model could not have "
                        "applied it to the distractor. Storage detail: "
                        + v["storage_reasoning"]
                    )
                else:
                    r["judge_result"] = "incorrect"
                    r["error_type"] = "storage_error"
                    r["judge_reasoning"] = v["storage_reasoning"]
                continue
            next_alive.append((gi, trace))
        n_storage_drop = len(verdicts) - len(next_alive)
        _log(f"  storage filter: {n_storage_drop} dropped (storage_error or vacuous), {len(next_alive)} alive")
        alive = next_alive

        # ── Stage 2: Summary ─────────────────────────────────────────────────
        verdicts, in_tok, out_tok = _run_stage(
            client, model, pool, "summary", alive, batch_size,
            SUMMARY_SYSTEM, SUMMARY_USER_FMT, _fmt_summary_entries, SUMMARY_SCHEMA,
            template_extras={"all_memories_formatted": all_memories_fmt},
        )
        in_tok_total += in_tok
        out_tok_total += out_tok

        next_alive = []
        for (gi, trace), v in zip(alive, verdicts):
            r = results[gi]
            if v is None or "_error" in (v or {}):
                r["analysis_error"] = (v or {}).get("_error") or "summary stage failed"
                continue
            r["summary_check_passed"] = v["summary_check_passed"]
            r["summary_reasoning"] = v["summary_reasoning"]
            if not v["summary_check_passed"]:
                r["judge_result"] = "incorrect"
                r["error_type"] = "summary_error"
                r["judge_reasoning"] = v["summary_reasoning"]
                continue
            next_alive.append((gi, trace))
        n_summary_drop = len(verdicts) - len(next_alive)
        _log(f"  summary filter: {n_summary_drop} dropped (summary_error), {len(next_alive)} alive")
        alive = next_alive

        # ── Stage 3: Retrieval ───────────────────────────────────────────────
        verdicts, in_tok, out_tok = _run_stage(
            client, model, pool, "retrieval", alive, batch_size,
            RETRIEVAL_SYSTEM, RETRIEVAL_USER_FMT, _fmt_retrieval_entries, RETRIEVAL_SCHEMA,
        )
        in_tok_total += in_tok
        out_tok_total += out_tok

        next_alive = []
        for (gi, trace), v in zip(alive, verdicts):
            r = results[gi]
            if v is None or "_error" in (v or {}):
                r["analysis_error"] = (v or {}).get("_error") or "retrieval stage failed"
                continue
            r["entity_facts_in_retrieved"] = v["entity_facts_in_retrieved"]
            r["retrieval_reasoning"] = v["retrieval_reasoning"]
            if not v["entity_facts_in_retrieved"]:
                if trace.get("question_type") == "misleading":
                    r["judge_result"] = "correct"
                    r["error_type"] = "correct"
                    r["vacuous_pass"] = True
                    r["judge_reasoning"] = (
                        "Vacuous pass: entity's essay was not retrieved for the "
                        "distractor question, so the model abstained by default. "
                        "Retrieval detail: " + v["retrieval_reasoning"]
                    )
                else:
                    r["judge_result"] = "incorrect"
                    r["error_type"] = "retrieval_error"
                    r["judge_reasoning"] = v["retrieval_reasoning"]
                continue
            next_alive.append((gi, trace))
        n_retrieval_drop = len(verdicts) - len(next_alive)
        _log(f"  retrieval filter: {n_retrieval_drop} dropped (retrieval_error or vacuous), {len(next_alive)} alive")
        alive = next_alive

        # ── Stage 4: Invocation (base) and Abstention (misleading) ───────────
        base_alive = [(gi, t) for gi, t in alive if t.get("question_type") != "misleading"]
        mis_alive = [(gi, t) for gi, t in alive if t.get("question_type") == "misleading"]
        _log(f"  stage 4 split: invocation={len(base_alive)}  abstention={len(mis_alive)}")

        if base_alive:
            verdicts, in_tok, out_tok = _run_stage(
                client, model, pool, "invocation", base_alive, batch_size,
                INVOCATION_SYSTEM, INVOCATION_USER_FMT, _fmt_invocation_entries, INVOCATION_SCHEMA,
            )
            in_tok_total += in_tok
            out_tok_total += out_tok
            for (gi, _), v in zip(base_alive, verdicts):
                r = results[gi]
                if v is None or "_error" in (v or {}):
                    r["analysis_error"] = (v or {}).get("_error") or "invocation stage failed"
                    continue
                r["correctly_invoked"] = v["correctly_invoked"]
                r["invocation_reasoning"] = v["invocation_reasoning"]
                if v["correctly_invoked"]:
                    r["judge_result"] = "correct"
                    r["error_type"] = "correct"
                    r["judge_reasoning"] = v["invocation_reasoning"]
                else:
                    r["judge_result"] = "incorrect"
                    r["error_type"] = "reasoning_error"
                    r["judge_reasoning"] = v["invocation_reasoning"]

        if mis_alive:
            verdicts, in_tok, out_tok = _run_stage(
                client, model, pool, "abstention", mis_alive, batch_size,
                ABSTENTION_SYSTEM, ABSTENTION_USER_FMT, _fmt_abstention_entries, ABSTENTION_SCHEMA,
            )
            in_tok_total += in_tok
            out_tok_total += out_tok
            for (gi, _), v in zip(mis_alive, verdicts):
                r = results[gi]
                if v is None or "_error" in (v or {}):
                    r["analysis_error"] = (v or {}).get("_error") or "abstention stage failed"
                    continue
                r["correctly_abstained"] = v["correctly_abstained"]
                r["abstention_reasoning"] = v["abstention_reasoning"]
                if v["correctly_abstained"]:
                    r["judge_result"] = "correct"
                    r["error_type"] = "correct"
                    r["judge_reasoning"] = v["abstention_reasoning"]
                else:
                    r["judge_result"] = "incorrect"
                    r["error_type"] = "reasoning_error"
                    r["judge_reasoning"] = v["abstention_reasoning"]

    run_elapsed = time.monotonic() - run_t0
    _log(
        f"analyze_run done in {run_elapsed:.1f}s  "
        f"total_in={in_tok_total:,} total_out={out_tok_total:,}"
    )
    return results, in_tok_total, out_tok_total


# ===========================================================================
# Output (unchanged from per-trace version)
# ===========================================================================


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


# ===========================================================================
# CLI
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade and classify errors in persona-retrieval traces (batched)."
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
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Max parallel batch API calls (default 50).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Number of traces per batched judge prompt (default 20).")
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
    print(f"Batch size: {args.batch_size} | Workers: {args.workers}")

    if args.limit:
        graded_traces = graded_traces[: args.limit]
        print(f"Limiting to first {args.limit} traces.")

    output_dir = Path(args.output_dir) if args.output_dir else traces_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # max_retries lets the OpenAI client respect 429 Retry-After headers with
    # its own exponential backoff before our outer retry loop kicks in.
    client = OpenAI(api_key=api_key, max_retries=8, timeout=120.0)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    results, in_tok, out_tok = analyze_run(
        client, args.model, graded_traces, all_memories,
        batch_size=args.batch_size, max_workers=args.workers,
    )
    judge_cost = compute_cost(args.model, in_tok, out_tok)

    json_path, csv_path = save_outputs(results, output_dir, ts)
    print_summary(results, in_tok, out_tok, judge_cost, args.model)
    print("\nResults saved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main()
