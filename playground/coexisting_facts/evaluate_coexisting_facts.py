"""
Evaluate memory systems on the coexisting-facts dataset.

Constructs a ChatDataset (storage conversations followed by question conversations)
and runs it through the Evaluator framework. All trace data is saved to the results
folder. Run analyze_errors.py to grade and classify errors.

Storage modes (--coexist-in-same-chat):

  False (default) — facts are stored in separate conversations, grouped by position.
    All fact[0]s from each dataset row are batched together (facts_per_group per conv),
    then all fact[1]s, then fact[2]s, etc. This guarantees that two coexisting facts
    from the same dataset row are never in the same conversation.

  True — all facts from a single dataset row are stored in one shared conversation.
    A fresh conversation is created per dataset row.

Question conversations: one per dataset row (grade=True), always in a fresh conversation.

Output:
  traces_<timestamp>.json — full run data
  graded_traces_<timestamp>.json — question conversations only, merged with dataset metadata
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

DEFAULT_FACTS_PER_GROUP = 10


def load_dataset(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = ["preference_facts", "question", "ground_truth_answer", "preferences"]
        if not reader.fieldnames:
            raise ValueError("Dataset CSV missing header.")
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        for i, row in enumerate(reader):
            facts_json = row.get("preference_facts", "[]")
            prefs_json = row.get("preferences", "[]")
            try:
                facts_list = json.loads(facts_json)
                if not isinstance(facts_list, list) or len(facts_list) == 0:
                    raise ValueError(f"preference_facts is empty or not a list: {facts_json!r}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Row {i}: preference_facts is not valid JSON: {e}") from e
            try:
                prefs_list = json.loads(prefs_json)
                if not isinstance(prefs_list, list):
                    prefs_list = []
            except json.JSONDecodeError:
                prefs_list = []
            rows.append({
                "preference_category": row.get("preference_category", ""),
                "preferences": prefs_list,
                "preference_facts": facts_list,
                "question": row["question"],
                "ground_truth_answer": row.get("ground_truth_answer", ""),
            })
    return rows


def build_chat_dataset(
    dataset_rows: List[Dict[str, Any]],
    coexist_in_same_chat: bool,
    facts_per_group: int = DEFAULT_FACTS_PER_GROUP,
) -> Tuple[ChatDataset, int]:
    """Build a ChatDataset for evaluation.

    Phase 1 (storage, grade=False):
      coexist_in_same_chat=True  — one conversation per dataset row; all that row's facts
                                   are fed sequentially in the same chat.
      coexist_in_same_chat=False — facts are grouped by their position index across rows,
                                   then batched (facts_per_group per conversation). All
                                   fact[0]s go into one set of conversations, all fact[1]s
                                   into another set, etc., ensuring no two coexisting facts
                                   from the same row ever share a conversation.

    Phase 2 (questions, grade=True): one conversation per dataset row.

    Returns (dataset, num_storage_convs).
    """
    conversations = []

    if coexist_in_same_chat:
        # One conversation per row; all facts from that row in sequence
        for row in dataset_rows:
            conversations.append(
                ConversationData(queries=[(fact, False) for fact in row["preference_facts"]])
            )
    else:
        # Group facts by position across rows, then batch within each position
        max_facts = max((len(row["preference_facts"]) for row in dataset_rows), default=0)
        for fact_idx in range(max_facts):
            facts_at_pos = [
                row["preference_facts"][fact_idx]
                for row in dataset_rows
                if fact_idx < len(row["preference_facts"])
            ]
            for i in range(0, len(facts_at_pos), facts_per_group):
                group = facts_at_pos[i : i + facts_per_group]
                conversations.append(ConversationData(queries=[(f, False) for f in group]))

    num_storage_convs = len(conversations)

    # Phase 2: question conversations, one per row
    for row in dataset_rows:
        conversations.append(ConversationData(queries=[(row["question"], True)]))

    return ChatDataset(conversations), num_storage_convs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate memory systems on coexisting-facts dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset CSV.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SCRIPT_DIR / "results"),
        help="Output directory for results.",
    )
    parser.add_argument("--llm-model", type=str, default="gpt-4.1-mini", help="Test-taker LLM model.")
    parser.add_argument(
        "--coexist-in-same-chat",
        action="store_true",
        default=False,
        help=(
            "Store all facts from each dataset row in one shared conversation. "
            "When off (default), facts are grouped by position across rows and batched "
            "so no two coexisting facts from the same row share a conversation."
        ),
    )
    parser.add_argument(
        "--facts-per-group",
        type=int,
        default=DEFAULT_FACTS_PER_GROUP,
        help="(when --coexist-in-same-chat is off) Number of facts per storage conversation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="(unused) Retained for script compatibility.")
    add_api_key_arg(parser)
    add_memory_system_args(parser)
    args = parser.parse_args()
    # Override shared-user-id default to be dataset-specific
    if args.shared_user_id == "eval_user":
        args.shared_user_id = "coexisting_facts_eval_user"

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

    print(f"Loading dataset from {dataset_path}...")
    dataset_rows = load_dataset(dataset_path)
    print(f"Loaded {len(dataset_rows)} rows.")

    chat_dataset, num_storage_convs = build_chat_dataset(
        dataset_rows, args.coexist_in_same_chat, args.facts_per_group
    )
    num_question_convs = len(dataset_rows)
    if args.coexist_in_same_chat:
        print(
            f"Built ChatDataset: {num_storage_convs} storage conversations "
            f"(one per row, all facts together), {num_question_convs} question conversations."
        )
    else:
        print(
            f"Built ChatDataset: {num_storage_convs} storage conversations "
            f"(facts grouped by position, {args.facts_per_group} per group), "
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
        "dataset_path": str(dataset_path),
        "coexist_in_same_chat": args.coexist_in_same_chat,
        "facts_per_group": args.facts_per_group if not args.coexist_in_same_chat else None,
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
            f, indent=2, cls=UUIDEncoder,
        )

    # ── Graded traces (question conversations only, dataset metadata merged) ──
    question_results = summary_dict["results"][num_storage_convs:]
    graded_traces = []
    for result, row in zip(question_results, dataset_rows):
        traces = result["traces"]
        if not traces:
            continue
        trace = traces[0]  # one graded query per question conversation
        graded_traces.append({
            # Dataset row metadata
            "preference_category": row["preference_category"],
            "preferences": row["preferences"],
            "preference_facts": row["preference_facts"],
            "ground_truth_answer": row["ground_truth_answer"],
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
            f, indent=2, cls=UUIDEncoder,
        )

    print("=" * 80)
    print("COEXISTING FACTS EVALUATION COMPLETE")
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
