"""
Evaluate memory systems on the long-hop (MemDaily) dataset.

Constructs a ChatDataset (storage conversations followed by question conversations)
and runs it through the Evaluator framework. All trace data is saved to the results
folder. Run analyze_errors.py to grade and classify errors.

Dataset structure (datasets/long_hop/memdaily.json):
  Top-level keys are question types (simple, conditional, comparative, aggregative,
  post_processing, noisy). Each maps to {domain -> [trajectory]}, where each
  trajectory has:
    - message_list: list of {mid, message, time, place} -- the conversational context
    - QA: {question, answer, target_step_id, choices (A-D), ground_truth (letter), ...}

Eval setup:
  - One storage conversation per trajectory: each contextual message is a query
    (grade=False). Memory is built up during this phase.
  - One question conversation per trajectory: a single graded query (grade=True)
    formed from the question + multiple-choice options.

Output:
  traces_<timestamp>.json — full run data
  graded_traces_<timestamp>.json — question conversations only, merged with dataset metadata
"""

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import (  # noqa: E402
    ChatDataset,
    ConversationData,
    ConversationHistoryPromptTemplate,
    Evaluator,
    OpenAILLM,
    TiktokenTokenizer,
)
from playground.utils import (  # noqa: E402
    UUIDEncoder,
    add_api_key_arg,
    add_memory_system_args,
    create_memory_system,
    resolve_api_key,
)

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_QUESTION_TYPES = [
    "simple",
    "conditional",
    "comparative",
    "aggregative",
    "post_processing",
    "noisy",
]
QUESTION_TYPE_ALIASES = {
    "simple": "simple",
    "conditional": "conditional",
    "cond": "conditional",
    "comparative": "comparative",
    "comp": "comparative",
    "aggregative": "aggregative",
    "aggr": "aggregative",
    "post_processing": "post_processing",
    "post-processing": "post_processing",
    "post": "post_processing",
    "noisy": "noisy",
}


def _parse_csv_arg(raw: str) -> List[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def _resolve_question_types(raw: str) -> List[str]:
    out = []
    for name in _parse_csv_arg(raw):
        key = name.lower()
        if key not in QUESTION_TYPE_ALIASES:
            raise ValueError(f"Unknown question type: {name}")
        out.append(QUESTION_TYPE_ALIASES[key])
    return out


def _parse_support_set_plan(raw: str) -> Dict[int, int]:
    plan: Dict[int, int] = {}
    for part in _parse_csv_arg(raw):
        size_text, sep, count_text = part.partition(":")
        if not sep:
            raise ValueError(
                f"Invalid --support-set-plan entry '{part}'. Expected SIZE:COUNT, e.g. 2:33,4:33,8:34."
            )
        size = int(size_text)
        count = int(count_text)
        if size <= 0 or count <= 0:
            raise ValueError("--support-set-plan sizes and counts must be positive integers.")
        if size in plan:
            raise ValueError(f"Duplicate support_set_size {size} in --support-set-plan.")
        plan[size] = count
    return plan


def load_dataset(
    json_path: Path,
    question_types: Iterable[str],
    domains: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    seed: int = 42,
    support_set_plan: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    """Load and filter MemDaily examples into normalized dataset rows."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    domain_filter = set(domains) if domains else None
    rows: List[Dict[str, Any]] = []
    for question_type in question_types:
        if question_type not in raw:
            raise KeyError(f"Question type '{question_type}' not found in {json_path}.")
        for domain, trajectories in raw[question_type].items():
            if domain_filter and domain not in domain_filter:
                continue
            for trajectory in trajectories:
                messages = trajectory["message_list"]
                qa = trajectory["QA"]
                target_ids = [int(v) for v in qa.get("target_step_id", [])]
                message_map = {int(item["mid"]): str(item["message"]) for item in messages}
                rows.append({
                    "example_id": f"memdaily-{question_type}-{domain}-{trajectory['tid']}",
                    "question_type": question_type,
                    "domain": domain,
                    "trajectory_id": int(trajectory["tid"]),
                    "support_set_size": len(target_ids),
                    "contextual_messages": [str(item["message"]) for item in messages],
                    "graded_question": str(qa["question"]),
                    "ground_truth_answer": str(qa["answer"]),
                    "ground_truth_choice": str(qa.get("ground_truth", "")),
                    "choices": {str(k): str(v) for k, v in qa.get("choices", {}).items()},
                    "target_step_ids": target_ids,
                    "target_messages": [message_map[mid] for mid in target_ids if mid in message_map],
                })

    if support_set_plan:
        rng = random.Random(seed)
        rng.shuffle(rows)
        selected: List[Dict[str, Any]] = []
        counts: Dict[int, int] = {k: 0 for k in support_set_plan}
        for row in rows:
            target = support_set_plan.get(row["support_set_size"])
            if target is None or counts[row["support_set_size"]] >= target:
                continue
            selected.append(row)
            counts[row["support_set_size"]] += 1
            if all(counts[s] >= c for s, c in support_set_plan.items()):
                break
        missing = {s: c - counts[s] for s, c in support_set_plan.items() if counts[s] < c}
        if missing:
            raise ValueError(
                "Not enough examples matched --support-set-plan after filtering: "
                + ", ".join(f"size={s} short by {n}" for s, n in sorted(missing.items()))
            )
        rows = selected
    else:
        rng = random.Random(seed)
        rng.shuffle(rows)

    if limit is not None:
        rows = rows[:limit]
    return rows


def format_question_prompt(row: Dict[str, Any]) -> str:
    """Render the multiple-choice question into the single graded query string."""
    parts = [row["graded_question"]]
    if row.get("choices"):
        parts.append("")
        for letter in sorted(row["choices"].keys()):
            parts.append(f"{letter}. {row['choices'][letter]}")
    return "\n".join(parts)


def build_chat_dataset(
    dataset_rows: List[Dict[str, Any]],
    seed: int = 42,
) -> Tuple[ChatDataset, int, List[Dict[str, Any]]]:
    """Build a ChatDataset for evaluation.

    Phase 1 (storage, grade=False): one conversation per dataset row, with each
      contextual message as an ungraded query. Row order is shuffled for
      reproducibility.

    Phase 2 (questions, grade=True): one conversation per dataset row, in an
      independently shuffled order. Each question is rendered with the MCQ choices.

    Returns (dataset, num_storage_convs, question_rows) where `question_rows` is
    the dataset_rows in the shuffled order used for question conversations — the
    caller should zip it with question results to merge metadata.
    """
    rng = random.Random(seed)
    conversations = []

    storage_order = list(range(len(dataset_rows)))
    rng.shuffle(storage_order)
    for idx in storage_order:
        row = dataset_rows[idx]
        conversations.append(ConversationData(
            queries=[(msg, False) for msg in row["contextual_messages"]]
        ))
    num_storage_convs = len(conversations)

    question_order = list(range(len(dataset_rows)))
    rng.shuffle(question_order)
    question_rows = [dataset_rows[i] for i in question_order]
    for row in question_rows:
        conversations.append(ConversationData(
            queries=[(format_question_prompt(row), True)]
        ))

    return ChatDataset(conversations), num_storage_convs, question_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate memory systems on the long-hop (MemDaily) dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str,
        default=str(PROJECT_ROOT / "datasets" / "long_hop" / "memdaily.json"),
        help="Path to memdaily.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SCRIPT_DIR / "results"),
        help="Output directory for results.",
    )
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini", help="Test-taker LLM model.")
    parser.add_argument(
        "--question-types",
        type=str,
        default=",".join(DEFAULT_QUESTION_TYPES),
        help="Comma-separated MemDaily question types to evaluate.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="",
        help="Optional comma-separated MemDaily domains (roles, events, items, places, hybrid).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate after filtering/shuffling.",
    )
    parser.add_argument(
        "--support-set-plan",
        type=str,
        default="",
        help=(
            "Optional exact support_set_size selection plan, formatted as SIZE:COUNT pairs, "
            "for example 2:33,4:33,8:34."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for shuffling storage order, question order, and selection (reproducible).",
    )
    add_api_key_arg(parser)
    add_memory_system_args(parser)
    args = parser.parse_args()
    if args.shared_user_id == "eval_user":
        args.shared_user_id = "long_hop_eval_user"

    api_key = resolve_api_key(args)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir = Path(args.output_dir) / args.memory
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    question_types = _resolve_question_types(args.question_types)
    domains = _parse_csv_arg(args.domains) if args.domains else None
    support_set_plan = _parse_support_set_plan(args.support_set_plan) if args.support_set_plan else None

    print(f"Loading dataset from {dataset_path}...")
    dataset_rows = load_dataset(
        dataset_path,
        question_types=question_types,
        domains=domains,
        limit=args.limit,
        seed=args.seed,
        support_set_plan=support_set_plan,
    )
    if not dataset_rows:
        raise ValueError("No examples matched the provided filters.")
    print(f"Loaded {len(dataset_rows)} rows.")

    chat_dataset, num_storage_convs, question_rows = build_chat_dataset(
        dataset_rows, args.seed
    )
    num_question_convs = len(dataset_rows)
    print(
        f"Built ChatDataset: {num_storage_convs} storage conversations, "
        f"{num_question_convs} question conversations."
    )

    memory_system = create_memory_system(args, api_key)
    llm = OpenAILLM(api_key=api_key, model=args.llm_model)

    prompt_template = ConversationHistoryPromptTemplate()
    tokenizer = TiktokenTokenizer()

    evaluator = Evaluator(memory_system, llm, chat_dataset, prompt_template, tokenizer)
    print("Running evaluation...")
    summary = evaluator.evaluate()
    print("Evaluation complete.")

    all_memories = memory_system.get_all_memories()
    print(f"Captured {len(all_memories)} memories from store.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_metadata = {
        "memory_system": args.memory,
        "llm_model": args.llm_model,
        "num_memories": args.num_memories,
        "shared_user_id": args.shared_user_id,
        "seed": args.seed,
        "dataset_path": str(dataset_path),
        "question_types": question_types,
        "domains": domains or [],
        "limit": args.limit,
        "support_set_plan": support_set_plan or {},
        "num_storage_convs": num_storage_convs,
        "num_question_convs": num_question_convs,
        "timestamp": ts,
        "all_cli_args": {k: v for k, v in vars(args).items() if k != "api_key"},
    }

    summary_dict = asdict(summary)

    # ── Full traces (all conversations: storage + question) ───────────────────
    full_path = output_dir / f"traces_{ts}.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_metadata": run_metadata,
                "all_memories_at_time_of_questions": all_memories,
                "dataset_rows": dataset_rows,
                "evaluation_summary": summary_dict,
            },
            f, indent=2, ensure_ascii=False, cls=UUIDEncoder,
        )

    # ── Graded traces (question conversations only, dataset metadata merged) ──
    question_results = summary_dict["results"][num_storage_convs:]
    graded_traces = []
    for result, row in zip(question_results, question_rows):
        traces = result["traces"]
        if not traces:
            continue
        trace = traces[0]
        graded_traces.append({
            # Dataset row metadata
            "example_id": row["example_id"],
            "question_type": row["question_type"],
            "domain": row["domain"],
            "trajectory_id": row["trajectory_id"],
            "support_set_size": row["support_set_size"],
            "ground_truth_answer": row["ground_truth_answer"],
            "ground_truth_choice": row["ground_truth_choice"],
            "choices": row["choices"],
            "target_step_ids": row["target_step_ids"],
            "target_messages": row["target_messages"],
            # QueryTrace fields
            "question": trace["query"],
            "retrieved_memories": trace["retrieved_memories"],
            "llm_response": trace["response"],
            "formatted_prompt": trace["formatted_prompt"],
            "conversation_id": result["conversation_id"],
            "eval_input_tokens": trace["input_tokens"],
            "eval_output_tokens": trace["output_tokens"],
            "eval_cost": trace["cost"],
        })

    graded_path = output_dir / f"graded_traces_{ts}.json"
    with open(graded_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_metadata": run_metadata,
                "all_memories_at_time_of_questions": all_memories,
                "graded_traces": graded_traces,
            },
            f, indent=2, ensure_ascii=False, cls=UUIDEncoder,
        )

    print("=" * 80)
    print("LONG-HOP EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Total conversations:  {summary.total_conversations}")
    print(f"Total queries:        {summary.total_queries}")
    print(f"Total cost:           ${summary.total_cost:.4f}")
    print(f"Total input tokens:   {summary.total_input_tokens:,}")
    print(f"Total output tokens:  {summary.total_output_tokens:,}")
    print(f"Saved full traces:    {full_path}")
    print(f"Saved graded traces:  {graded_path}")
    print("Run analyze_errors.py on the graded traces file to grade and classify errors.")
    print("=" * 80)


if __name__ == "__main__":
    main()
