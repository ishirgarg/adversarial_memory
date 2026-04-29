"""
Evaluate memory systems on the misleading-persona dataset.

Constructs a ChatDataset (storage conversations followed by question conversations)
and runs it through the Evaluator framework. All trace data is saved to the results
folder. Run analyze_errors.py to grade and classify errors.

Dataset structure:
  - Storage conversations: groups of `facts_per_group` essays, each group in its own
    conversation (grade=False). Memory is built up during this phase.
  - Question conversations: one per question in each row's `questions` list (typically
    THREE per row). Each question is independently either:
      * base (non-misleading) — names the entity; should be answered using memory
      * misleading            — names a different person; model SHOULD abstain

Output:
  traces_<timestamp>.json with:
    - run_metadata: configuration for this run
    - all_memories_at_time_of_questions: full memory store captured after evaluation
    - dataset_rows: original CSV metadata
    - question_specs: per-question metadata aligned with question conversations
    - evaluation_summary: all EvaluationResult/QueryTrace data from the Evaluator
  graded_traces_<timestamp>.json: question conversations only, per-question metadata merged.
"""

import argparse
import csv
import json
import random
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

DEFAULT_FACTS_PER_GROUP = 1


def load_dataset(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = ["entity", "entity_facts", "questions"]
        if not reader.fieldnames:
            raise ValueError("Dataset CSV missing header.")
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        for i, row in enumerate(reader):
            facts_json = row.get("entity_facts", "[]")
            try:
                facts_list = json.loads(facts_json)
                if not isinstance(facts_list, list) or len(facts_list) == 0:
                    raise ValueError(f"entity_facts is empty or not a list: {facts_json!r}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Row {i}: entity_facts is not valid JSON: {e}") from e
            if len(facts_list) != 1:
                raise ValueError(
                    f"Row {i}: entity_facts must have exactly 1 element, "
                    f"got {len(facts_list)}. Regenerate the dataset."
                )

            questions_json = row.get("questions", "[]")
            try:
                questions_list = json.loads(questions_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"Row {i}: questions is not valid JSON: {e}") from e
            if not isinstance(questions_list, list) or len(questions_list) == 0:
                raise ValueError(f"Row {i}: questions must be a non-empty list.")
            for j, q in enumerate(questions_list):
                for field in ("text", "is_misleading", "ground_truth_answer"):
                    if field not in q:
                        raise ValueError(
                            f"Row {i} question {j}: missing required field {field!r}."
                        )

            rows.append({
                "entity": row.get("entity", ""),
                "entity_facts": facts_list,
                "questions": questions_list,
            })
    return rows


def build_chat_dataset(
    dataset_rows: List[Dict[str, Any]],
    facts_per_group: int,
    seed: int = 42,
) -> Tuple[ChatDataset, int, List[Dict[str, Any]]]:
    """Build a ChatDataset for evaluation.

    Storage conversations (grade=False) come first so memory is populated before
    questions are asked. Question conversations (grade=True) follow — one per question
    in the row's `questions` list (typically THREE per row), shuffled together.

    Storage facts and question order are independently shuffled with `seed` so
    runs are reproducible but not biased by CSV order.

    Returns (dataset, num_storage_convs, question_specs), where question_specs is a
    list of per-question dicts in the order they appear in the ChatDataset.
    """
    rng = random.Random(seed)
    conversations: List[ConversationData] = []

    # Phase 1: Storage conversations — shuffle essays, then chunk
    all_facts = [row["entity_facts"][0] for row in dataset_rows]
    storage_order = list(range(len(all_facts)))
    rng.shuffle(storage_order)
    shuffled_facts = [all_facts[i] for i in storage_order]
    groups = [
        shuffled_facts[i:i + facts_per_group]
        for i in range(0, len(shuffled_facts), facts_per_group)
    ]
    for group in groups:
        conversations.append(ConversationData(queries=[(fact, False) for fact in group]))
    num_storage_convs = len(groups)

    # Phase 2: Question conversations — flatten 3 per row, then shuffle
    question_specs: List[Dict[str, Any]] = []
    for row in dataset_rows:
        for q in row["questions"]:
            is_misleading = bool(q["is_misleading"])
            question_specs.append({
                "entity": row["entity"],
                "entity_facts": row["entity_facts"],
                "distractor": q.get("distractor") or "",
                "question_type": "misleading" if is_misleading else "base",
                "ground_truth_answer": q.get("ground_truth_answer", ""),
                "_question": q["text"],
            })
    rng.shuffle(question_specs)
    for spec in question_specs:
        conversations.append(ConversationData(queries=[(spec["_question"], True)]))

    return ChatDataset(conversations), num_storage_convs, question_specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate memory systems on misleading-persona dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset CSV.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SCRIPT_DIR / "results"),
        help="Output directory for results.",
    )
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini", help="Test-taker LLM model.")
    parser.add_argument(
        "--facts-per-group",
        type=int,
        default=DEFAULT_FACTS_PER_GROUP,
        help="Number of essays per storage conversation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for shuffling storage facts and question order (reproducible across runs).",
    )
    add_api_key_arg(parser)
    add_memory_system_args(parser)
    args = parser.parse_args()
    if args.shared_user_id == "eval_user":
        args.shared_user_id = "persona_retrieval_eval_user"

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

    chat_dataset, num_storage_convs, question_specs = build_chat_dataset(
        dataset_rows, args.facts_per_group, args.seed
    )
    num_question_convs = len(question_specs)
    print(
        f"Built ChatDataset: {num_storage_convs} storage conversations "
        f"({args.facts_per_group} essays each), {num_question_convs} question conversations "
        f"({num_question_convs / max(len(dataset_rows), 1):.1f} per row, mix of base + misleading)."
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
        "facts_per_group": args.facts_per_group,
        "shared_user_id": args.shared_user_id,
        "seed": args.seed,
        "dataset_path": str(dataset_path),
        "num_storage_convs": num_storage_convs,
        "num_question_convs": num_question_convs,
        "timestamp": ts,
        "all_cli_args": {k: v for k, v in vars(args).items() if k != "api_key"},
    }

    summary_dict = asdict(summary)

    full_path = output_dir / f"traces_{ts}.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_metadata": run_metadata,
                "all_memories_at_time_of_questions": all_memories,
                "dataset_rows": dataset_rows,
                "question_specs": [
                    {k: v for k, v in spec.items() if k != "_question"}
                    for spec in question_specs
                ],
                "evaluation_summary": summary_dict,
            },
            f, indent=2, cls=UUIDEncoder,
        )

    question_results = summary_dict["results"][num_storage_convs:]
    graded_traces = []
    for result, spec in zip(question_results, question_specs):
        traces = result["traces"]
        if not traces:
            continue
        trace = traces[0]
        graded_traces.append({
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
    print("PERSONA RETRIEVAL EVALUATION COMPLETE")
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
