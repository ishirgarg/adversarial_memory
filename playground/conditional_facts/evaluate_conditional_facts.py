"""
Evaluate memory systems on the conditional-facts dataset.

Tests whether the memory system correctly preserves CONDITIONAL or QUALIFIED information —
i.e., that an entity does something only under a specific condition.

Storage model:
  Entity facts are fed into memory in GROUPS (default 10 per conversation). Each group is
  a single chat turn: all facts in the group are concatenated and sent as one message.
  This tests whether the memory system can summarize conditional/qualified information
  without losing the qualifying condition.

Grading is NOT done here. Run analyze_errors.py on the saved traces file to grade
responses and classify errors in a single combined LLM call per trace.
"""

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import (  # noqa: E402
    AMEMMemorySystem,
    ChatSystem,
    ConversationHistoryPromptTemplate,
    Mem0MemorySystem,
    OpenAILLM,
    SimpleMemMemorySystem,
)

load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]

QUESTION_TYPE = "conditional_facts_question"
DEFAULT_FACTS_PER_GROUP = 10


@dataclass
class QuestionTrace:
    entity: str
    entity_category: str
    behavior: str
    condition_type: str
    condition: str
    entity_facts: List[str]
    question: str
    question_context: str
    condition_met: str                    # "yes" or "no"
    ground_truth_answer: str
    question_type: str
    question_conv_id: str
    formatted_prompt: str
    llm_response: str
    retrieved_memories: str
    all_memories_at_time: List[str]


@dataclass
class EvaluationResults:
    total_questions: int = 0
    traces: List[QuestionTrace] = None

    def __post_init__(self) -> None:
        if self.traces is None:
            self.traces = []

    def summary(self) -> Dict[str, Any]:
        # Breakdown by condition_met
        by_condition: Dict[str, int] = {}
        for t in self.traces:
            by_condition[t.condition_met] = by_condition.get(t.condition_met, 0) + 1

        # Breakdown by condition type
        by_ctype: Dict[str, int] = {}
        for t in self.traces:
            by_ctype[t.condition_type] = by_ctype.get(t.condition_type, 0) + 1

        return {
            "total_questions": self.total_questions,
            "by_condition_met": {k: by_condition[k] for k in sorted(by_condition)},
            "by_condition_type": {k: by_ctype[k] for k in sorted(by_ctype)},
        }


def load_dataset(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = ["entity_facts", "question", "ground_truth_answer", "condition_met",
                    "entity", "condition", "condition_type"]
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
                raise ValueError(f"Row {i}: entity_facts must have exactly 1 element, got {len(facts_list)}. Regenerate the dataset.")
            rows.append({
                "entity": row.get("entity", ""),
                "entity_category": row.get("entity_category", ""),
                "behavior": row.get("behavior", ""),
                "condition_type": row.get("condition_type", ""),
                "condition": row.get("condition", ""),
                "entity_facts": facts_list,   # list with exactly 1 conditional fact
                "question": row["question"],
                "question_context": row.get("question_context", ""),
                "condition_met": row.get("condition_met", "").strip().lower(),
                "ground_truth_answer": row.get("ground_truth_answer", ""),
                "question_type": QUESTION_TYPE,
            })
    return rows


def store_facts_in_groups(
    all_facts: List[str],
    facts_per_group: int,
    memory_system: Any,
    chat_system: ChatSystem,
    prompt_template: ConversationHistoryPromptTemplate,
) -> None:
    """Store facts in groups of `facts_per_group` per conversation.

    Each group is concatenated into a single message sent in a fresh chat.
    This tests whether memory systems can summarize batches that contain
    conditional/qualified information.
    """
    groups = [all_facts[i:i + facts_per_group] for i in range(0, len(all_facts), facts_per_group)]
    for group in groups:
        combined = "\n".join(group)
        conv_id = chat_system.start_new_conversation()
        conversation = chat_system.get_conversation(conv_id)
        if conversation is None:
            raise ValueError("Failed to get conversation for fact storage.")
        memories = memory_system.get_memories(combined, conversation)
        prompt = prompt_template.format(combined, memories, conversation)
        response = chat_system.send_message(prompt, conv_id)
        updated = chat_system.get_conversation(conv_id)
        if updated:
            memory_system.update_memory(combined, response, updated)


def ask_question(
    question: str,
    memory_system: Any,
    chat_system: ChatSystem,
    prompt_template: ConversationHistoryPromptTemplate,
) -> tuple[str, str, str, str]:
    """Ask a question in a fresh conversation. Returns (formatted_prompt, response, memories, conv_id)."""
    conv_id = chat_system.start_new_conversation()
    conversation = chat_system.get_conversation(conv_id)
    if conversation is None:
        raise ValueError("Failed to get conversation for question.")

    retrieved_memories = memory_system.get_memories(question, conversation)
    compliance_instruction = (
        "\n\nAnswer the question directly with yes or no. Then add a line starting with "
        "'MEMORY_USED:' and either:\n"
        "- List an exact quote of every specific memory you used, or\n"
        "- State 'none' if you did not use any memory."
    )
    model_question = question + compliance_instruction
    formatted_prompt = prompt_template.format(model_question, retrieved_memories, conversation)
    llm_response = chat_system.send_message(formatted_prompt, conv_id)

    return formatted_prompt, llm_response, retrieved_memories, str(conv_id)



def _create_memory_system(memory: str, num_memories: int, shared_user_id: str, api_key: str) -> Any:
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


def run_evaluation(
    *,
    dataset_path: Path,
    api_key: str,
    llm_model: str,
    num_memories: int,
    facts_per_group: int,
    shared_user_id: str,
    seed: int,
    memory: str,
) -> EvaluationResults:
    dataset = load_dataset(dataset_path)
    results = EvaluationResults(total_questions=len(dataset))

    memory_system = _create_memory_system(memory, num_memories, shared_user_id, api_key)
    llm = OpenAILLM(api_key=api_key, model=llm_model)
    chat_system = ChatSystem(llm)
    prompt_template = ConversationHistoryPromptTemplate()

    # --- Phase 1: storage ---
    # Each row has exactly one conditional fact. Collect them all in dataset order,
    # then store in groups of `facts_per_group`, each group in its own conversation.
    all_facts: List[str] = [row["entity_facts"][0] for row in dataset]
    num_groups = (len(all_facts) + facts_per_group - 1) // facts_per_group
    print(
        f"Storing {len(all_facts)} facts into memory "
        f"in groups of {facts_per_group} ({num_groups} conversations)..."
    )
    store_facts_in_groups(all_facts, facts_per_group, memory_system, chat_system, prompt_template)
    print("Storage complete.")

    # --- Phase 2: evaluation ---
    # Each question is asked in its own fresh conversation.
    question_rows = list(dataset)
    random.Random(seed).shuffle(question_rows)

    for row in tqdm(question_rows, desc="Evaluating"):
        formatted_prompt, llm_response, retrieved_memories, question_conv_id = ask_question(
            row["question"], memory_system, chat_system, prompt_template
        )
        all_memories_at_time = memory_system.get_all_memories()

        trace = QuestionTrace(
            entity=row["entity"],
            entity_category=row["entity_category"],
            behavior=row["behavior"],
            condition_type=row["condition_type"],
            condition=row["condition"],
            entity_facts=row["entity_facts"],
            question=row["question"],
            question_context=row["question_context"],
            condition_met=row["condition_met"],
            ground_truth_answer=row["ground_truth_answer"],
            question_type=row["question_type"],
            question_conv_id=question_conv_id,
            formatted_prompt=formatted_prompt,
            llm_response=llm_response,
            retrieved_memories=retrieved_memories,
            all_memories_at_time=all_memories_at_time,
        )
        results.traces.append(trace)

    return results


def save_results(results: EvaluationResults, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    traces_path = output_dir / f"traces_{ts}.json"
    traces_compact_path = output_dir / f"traces_compact_{ts}.json"

    with open(traces_path, "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in results.traces], f, indent=2)

    compact_traces = []
    for t in results.traces:
        t_dict = asdict(t)
        t_dict.pop("all_memories_at_time", None)
        t_dict.pop("formatted_prompt", None)
        compact_traces.append(t_dict)
    with open(traces_compact_path, "w", encoding="utf-8") as f:
        json.dump(compact_traces, f, indent=2)

    print("=" * 80)
    print("CONDITIONAL FACTS EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Total questions: {results.total_questions}")
    print(f"Saved traces:           {traces_path}")
    print(f"Saved compact traces:   {traces_compact_path}")
    print("Run analyze_errors.py on the traces file to grade and classify errors.")
    print("=" * 80)


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
        help="Number of facts to store per conversation (default: 10).",
    )
    parser.add_argument("--shared-user-id", type=str, default="conditional_facts_eval_user")
    parser.add_argument("--seed", type=int, default=42)
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

    results = run_evaluation(
        dataset_path=dataset_path,
        api_key=api_key,
        llm_model=args.llm_model,
        num_memories=args.num_memories,
        facts_per_group=args.facts_per_group,
        shared_user_id=args.shared_user_id,
        seed=args.seed,
        memory=args.memory,
    )
    save_results(results, output_dir)


if __name__ == "__main__":
    main()
