"""
Evaluate memory systems on the conditional-facts dataset.

Constructs a ChatDataset (storage conversations followed by question conversations)
and runs it through the Evaluator framework. All trace data is saved to the results
folder. Run analyze_errors.py to grade and classify errors.

Dataset structure:
  - Storage conversations: groups of `facts_per_group` facts, each group in its own
    conversation (grade=False). Memory is built up during this phase.
  - Question conversations: one per dataset row, each containing a single graded
    query (grade=True). No grading happens here — just response generation.

Output:
  traces_<timestamp>.json with:
    - run_metadata: configuration for this run
    - all_memories_at_time_of_questions: full memory store captured after evaluation
    - dataset_rows: original CSV metadata aligned with question conversations
    - evaluation_summary: all EvaluationResult/QueryTrace data from the Evaluator
"""

import argparse
import csv
import json
import os
import sys
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import (  # noqa: E402
    AMEMMemorySystem,
    ChatDataset,
    ConversationData,
    ConversationHistoryPromptTemplate,
    Evaluator,
    Mem0MemorySystem,
    OpenAILLM,
    SimpleMemMemorySystem,
    TiktokenTokenizer,
)

load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]

DEFAULT_FACTS_PER_GROUP = 10

COMPLIANCE_INSTRUCTION = (
    "\n\nAnswer the question directly with yes or no. Then add a line starting with "
    "'MEMORY_USED:' and either:\n"
    "- List an exact quote of every specific memory you used, or\n"
    "- State 'none' if you did not use any memory."
)


class _UUIDEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return super().default(obj)


def load_dataset(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = [
            "entity_facts", "question", "ground_truth_answer",
            "condition_met", "entity", "condition", "condition_type",
        ]
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
            rows.append({
                "entity": row.get("entity", ""),
                "entity_category": row.get("entity_category", ""),
                "behavior": row.get("behavior", ""),
                "condition_type": row.get("condition_type", ""),
                "condition": row.get("condition", ""),
                "entity_facts": facts_list,
                "question": row["question"],
                "question_context": row.get("question_context", ""),
                "condition_met": row.get("condition_met", "").strip().lower(),
                "ground_truth_answer": row.get("ground_truth_answer", ""),
            })
    return rows


def build_chat_dataset(
    dataset_rows: List[Dict[str, Any]], facts_per_group: int
) -> Tuple[ChatDataset, int]:
    """Build a ChatDataset for evaluation.

    Storage conversations (grade=False) come first so memory is populated before
    questions are asked. Question conversations (grade=True) follow, one per row.

    Returns (dataset, num_storage_convs).
    """
    conversations = []

    # Phase 1: Storage conversations — group facts, each group in its own conversation
    all_facts = [row["entity_facts"][0] for row in dataset_rows]
    groups = [all_facts[i:i + facts_per_group] for i in range(0, len(all_facts), facts_per_group)]
    for group in groups:
        conversations.append(ConversationData(queries=[(fact, False) for fact in group]))
    num_storage_convs = len(groups)

    # Phase 2: Question conversations — one graded query per row
    for row in dataset_rows:
        question = row["question"] + COMPLIANCE_INSTRUCTION
        conversations.append(ConversationData(queries=[(question, True)]))

    return ChatDataset(conversations), num_storage_convs


def _create_memory_system(
    memory: str, num_memories: int, shared_user_id: str, api_key: str
) -> Any:
    if memory == "mem0":
        return Mem0MemorySystem(num_memories=num_memories, shared_user_id=shared_user_id)
    if memory == "simplemem":
        return SimpleMemMemorySystem(num_memories=num_memories, api_key=api_key, clear_db=True)
    if memory == "amem":
        return AMEMMemorySystem(
            num_memories=num_memories,
            llm_backend="openai",
            llm_model="gpt-4o-mini",
            embedding_model="all-MiniLM-L6-v2",
            evo_threshold=100,
            api_key=api_key,
        )
    raise ValueError(f"Unknown memory system: {memory!r}. Choose mem0, simplemem, or amem.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate memory systems on conditional-facts dataset."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset CSV.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SCRIPT_DIR / "results"),
        help="Output directory for results.",
    )
    parser.add_argument("--llm-model", type=str, default="gpt-4.1-mini", help="Test-taker LLM model.")
    parser.add_argument("--num-memories", type=int, default=5, help="k for memory retrieval.")
    parser.add_argument(
        "--facts-per-group",
        type=int,
        default=DEFAULT_FACTS_PER_GROUP,
        help="Number of facts per storage conversation (default: 10).",
    )
    parser.add_argument("--shared-user-id", type=str, default="conditional_facts_eval_user")
    parser.add_argument("--seed", type=int, default=42, help="(unused) Retained for script compatibility.")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument(
        "--memory",
        type=str,
        default="mem0",
        choices=["mem0", "simplemem", "amem"],
        help="Memory system to evaluate.",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required via --api-key or env var.")

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

    chat_dataset, num_storage_convs = build_chat_dataset(dataset_rows, args.facts_per_group)
    num_question_convs = len(dataset_rows)
    print(
        f"Built ChatDataset: {num_storage_convs} storage conversations "
        f"({args.facts_per_group} facts each), {num_question_convs} question conversations."
    )

    memory_system = _create_memory_system(
        args.memory, args.num_memories, args.shared_user_id, api_key
    )
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
    output_path = output_dir / f"traces_{ts}.json"

    output = {
        "run_metadata": {
            "memory_system": args.memory,
            "llm_model": args.llm_model,
            "num_memories": args.num_memories,
            "facts_per_group": args.facts_per_group,
            "shared_user_id": args.shared_user_id,
            "dataset_path": str(dataset_path),
            "num_storage_convs": num_storage_convs,
            "num_question_convs": num_question_convs,
            "timestamp": ts,
        },
        "all_memories_at_time_of_questions": all_memories,
        "dataset_rows": dataset_rows,
        "evaluation_summary": asdict(summary),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, cls=_UUIDEncoder)

    print("=" * 80)
    print("CONDITIONAL FACTS EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Total conversations:  {summary.total_conversations}")
    print(f"Total queries:        {summary.total_queries}")
    print(f"Total cost:           ${summary.total_cost:.4f}")
    print(f"Total input tokens:   {summary.total_input_tokens:,}")
    print(f"Total output tokens:  {summary.total_output_tokens:,}")
    print(f"Saved traces:         {output_path}")
    print("Run analyze_errors.py on the traces file to grade and classify errors.")
    print("=" * 80)


if __name__ == "__main__":
    main()
