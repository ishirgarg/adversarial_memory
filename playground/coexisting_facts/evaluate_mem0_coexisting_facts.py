"""
Evaluate Mem0 on the coexisting-facts dataset.

Tests whether the memory system correctly stores and retrieves ALL of the user's
preferences when each preference is stated in a completely separate, isolated chat.

Storage model:
  Each preference fact (e.g. "I love pizza.") is fed into a FRESH conversation with
  no prior history. This means mem0 must store each fact independently and later
  surface all of them together when a question is asked.

Policy:
- coexisting_facts_question: model SHOULD recall ALL preferences from memory.
  A response is CORRECT only if it mentions every preference in ground_truth_answer.
  A response is INCORRECT if any preference is missing.

Key metrics logged per trace:
- preferences_retrieved_fraction: what fraction of expected preferences appeared
  in the retrieved memories (substring match), tracked overall and by preference count.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Add project root to path
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

# Load .env from project root
load_dotenv(PROJECT_ROOT / ".env")

# Alias OPENAI_API_KEY -> OPENAI_API_KEY so mem0 and other libs find it
if not os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

MAX_JUDGE_RETRIES = 3
QUESTION_TYPE = "coexisting_facts_question"


def compute_retrieval_coverage(preferences_json: str, retrieved_memories: str) -> tuple[int, int]:
    """
    Return (retrieved_count, total_count).

    Checks each preference against retrieved_memories using case-insensitive substring
    matching. Used to compute the retrieval fraction metric per trace.
    """
    try:
        prefs = json.loads(preferences_json)
    except Exception:
        return 0, 0
    if not isinstance(prefs, list) or not prefs:
        return 0, 0
    retrieved_lower = (retrieved_memories or "").lower()
    retrieved_count = sum(1 for p in prefs if str(p).lower() in retrieved_lower)
    return retrieved_count, len(prefs)


@dataclass
class QuestionTrace:
    preference_facts: List[str]     # individual fact statements, one per preference
    question: str
    question_type: str
    ground_truth_answer: str
    preferences: str            # JSON-encoded list
    num_preferences: int
    preference_category: str
    question_conv_id: str
    formatted_prompt: str
    llm_response: str
    retrieved_memories: str
    preferences_retrieved_count: int    # how many preferences appeared in retrieved_memories
    preferences_retrieved_fraction: float
    all_memories_at_time: List[str]
    judge_result: str           # "correct" or "incorrect"
    judge_reasoning: str


@dataclass
class EvaluationResults:
    total_questions: int = 0
    correct_questions: int = 0
    traces: List[QuestionTrace] = None

    def __post_init__(self) -> None:
        if self.traces is None:
            self.traces = []

    def summary(self) -> Dict[str, Any]:
        accuracy = self.correct_questions / self.total_questions if self.total_questions else 0.0

        # Overall retrieval fraction
        fractions = [t.preferences_retrieved_fraction for t in self.traces]
        avg_retrieval_fraction = sum(fractions) / len(fractions) if fractions else 0.0

        # Breakdown by preference count
        by_count: Dict[int, Dict[str, Any]] = {}
        for t in self.traces:
            n = t.num_preferences
            if n not in by_count:
                by_count[n] = {"total": 0, "correct": 0, "retrieval_fractions": []}
            by_count[n]["total"] += 1
            if t.judge_result == "correct":
                by_count[n]["correct"] += 1
            by_count[n]["retrieval_fractions"].append(t.preferences_retrieved_fraction)

        by_count_summary = {}
        for n in sorted(by_count):
            d = by_count[n]
            rf = d["retrieval_fractions"]
            by_count_summary[n] = {
                "total": d["total"],
                "correct": d["correct"],
                "accuracy": d["correct"] / d["total"] if d["total"] else 0.0,
                "avg_retrieval_fraction": sum(rf) / len(rf) if rf else 0.0,
            }

        return {
            "total_questions": self.total_questions,
            "correct_questions": self.correct_questions,
            "accuracy": accuracy,
            "avg_retrieval_fraction": avg_retrieval_fraction,
            "by_preference_count": by_count_summary,
        }


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

        for row in reader:
            prefs_json = row.get("preferences", "[]")
            facts_json = row.get("preference_facts", "[]")
            try:
                num_prefs = len(json.loads(prefs_json))
            except Exception:
                num_prefs = 0
            try:
                facts_list = json.loads(facts_json)
                if not isinstance(facts_list, list):
                    facts_list = []
            except Exception:
                facts_list = []
            rows.append({
                "preference_facts": facts_list,
                "question": row["question"],
                "ground_truth_answer": row.get("ground_truth_answer", ""),
                "preferences": prefs_json,
                "num_preferences": num_prefs,
                "preference_category": row.get("preference_category", ""),
                "question_type": QUESTION_TYPE,
            })
    return rows


def store_individual_facts(
    all_facts: List[str],
    memory_system: Any,
    chat_system: ChatSystem,
    prompt_template: ConversationHistoryPromptTemplate,
) -> None:
    """Store each fact in its own fresh conversation with no shared history.

    Each call creates a brand-new chat, sends exactly one fact statement,
    and updates memory — so mem0 receives every preference in isolation.
    """
    for fact in all_facts:
        conv_id = chat_system.start_new_conversation()
        conversation = chat_system.get_conversation(conv_id)
        if conversation is None:
            raise ValueError("Failed to get conversation for fact storage.")
        memories = memory_system.get_memories(fact, conversation)
        prompt = prompt_template.format(fact, memories, conversation)
        response = chat_system.send_message(prompt, conv_id)
        updated = chat_system.get_conversation(conv_id)
        if updated:
            memory_system.update_memory(fact, response, updated)


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
        "\n\nAnswer the question directly. Then add a line starting with "
        "'MEMORY_USED:' and either:\n"
        "- List an exact quote of every specific memory you used, or\n"
        "- State 'none' if you did not use any memory."
    )
    model_question = question + compliance_instruction
    formatted_prompt = prompt_template.format(model_question, retrieved_memories, conversation)
    llm_response = chat_system.send_message(formatted_prompt, conv_id)

    return formatted_prompt, llm_response, retrieved_memories, str(conv_id)


def judge_response(
    *,
    judge_client: OpenAI,
    judge_model: str,
    question: str,
    ground_truth_answer: str,
    llm_response: str,
    retrieved_memories: str,
) -> tuple[str, str]:
    """Grade whether the model mentioned ALL expected preferences."""
    prompt = f"""You are a strict evaluator grading whether a model correctly recalled ALL of the user's preferences.

Question asked:
{question}

Expected preferences (ground truth — ALL of these must appear in the response):
{ground_truth_answer}

Retrieved memories shown to model:
{retrieved_memories if retrieved_memories else "(none)"}

Model response:
{llm_response}

Task:
Return JSON with:
- "result": exactly one of ["correct", "incorrect"]
- "reasoning": short explanation

Grading rules:
1) "correct" ONLY if the response clearly mentions or accounts for EVERY preference in the ground truth.
   - Synonyms and paraphrases count (e.g. "pasta dishes" covers "spaghetti").
   - The preference must be meaningfully present, not just incidentally mentioned.
2) "incorrect" if ANY preference from the ground truth is absent or not accounted for.
3) Ignore extra preferences the model mentions beyond the ground truth.
4) Base your judgment on the response content, not on whether memory was cited.
5) Output JSON only.
"""
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_JUDGE_RETRIES + 1):
        try:
            resp = judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("Empty judge response content.")
            parsed = json.loads(content)
            result = parsed.get("result")
            reasoning = str(parsed.get("reasoning", ""))
            if result not in {"correct", "incorrect"}:
                raise ValueError(f"Invalid judge result: {result!r}")
            return result, reasoning
        except Exception as e:
            last_error = e
            if attempt < MAX_JUDGE_RETRIES:
                time.sleep(0.4 * attempt)

    return "incorrect", f"Judge parse failed after retries: {last_error}"


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
    judge_model: str,
    num_memories: int,
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
    judge_client = OpenAI(api_key=api_key)

    # Collect all unique individual facts across all rows, preserving stable order.
    # Each fact is stored in its own isolated conversation with no shared history.
    seen: set = set()
    unique_facts: List[str] = []
    for row in dataset:
        for fact in row["preference_facts"]:
            if fact not in seen:
                seen.add(fact)
                unique_facts.append(fact)

    print(f"Storing {len(unique_facts)} individual facts into memory (one per conversation)...")
    store_individual_facts(unique_facts, memory_system, chat_system, prompt_template)
    print("Storage complete.")

    # Shuffle only question-asking order
    question_rows = list(dataset)
    random.Random(seed).shuffle(question_rows)

    for row in tqdm(question_rows, desc="Evaluating"):
        formatted_prompt, llm_response, retrieved_memories, question_conv_id = ask_question(
            row["question"], memory_system, chat_system, prompt_template
        )
        all_memories_at_time = memory_system.get_all_memories()

        retrieved_count, total_prefs = compute_retrieval_coverage(
            row["preferences"], retrieved_memories
        )
        retrieval_fraction = retrieved_count / total_prefs if total_prefs else 0.0

        judge_result, judge_reasoning = judge_response(
            judge_client=judge_client,
            judge_model=judge_model,
            question=row["question"],
            ground_truth_answer=row["ground_truth_answer"],
            llm_response=llm_response,
            retrieved_memories=retrieved_memories,
        )

        trace = QuestionTrace(
            preference_facts=row["preference_facts"],
            question=row["question"],
            question_type=row["question_type"],
            ground_truth_answer=row["ground_truth_answer"],
            preferences=row["preferences"],
            num_preferences=row["num_preferences"],
            preference_category=row["preference_category"],
            question_conv_id=question_conv_id,
            formatted_prompt=formatted_prompt,
            llm_response=llm_response,
            retrieved_memories=retrieved_memories,
            preferences_retrieved_count=retrieved_count,
            preferences_retrieved_fraction=retrieval_fraction,
            all_memories_at_time=all_memories_at_time,
            judge_result=judge_result,
            judge_reasoning=judge_reasoning,
        )
        results.traces.append(trace)

        if judge_result == "correct":
            results.correct_questions += 1

    return results


def save_results(results: EvaluationResults, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = output_dir / f"summary_{ts}.json"
    traces_path = output_dir / f"traces_{ts}.json"
    traces_compact_path = output_dir / f"traces_compact_{ts}.json"
    csv_path = output_dir / f"classifications_{ts}.csv"

    summary = results.summary()

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

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

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "question_index", "judge_result", "num_preferences",
            "preferences_retrieved_count", "preferences_retrieved_fraction",
            "preference_category", "ground_truth_answer", "question",
        ])
        for i, t in enumerate(results.traces):
            w.writerow([
                i, t.judge_result, t.num_preferences,
                t.preferences_retrieved_count, f"{t.preferences_retrieved_fraction:.3f}",
                t.preference_category, t.ground_truth_answer, t.question,
            ])

    # --- Console summary ---
    print("=" * 80)
    print("COEXISTING FACTS EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total questions:          {summary['total_questions']}")
    print(f"Correct:                  {summary['correct_questions']}")
    print(f"Accuracy:                 {summary['accuracy']:.1%}")
    print(f"Avg retrieval fraction:   {summary['avg_retrieval_fraction']:.1%}")

    # Breakdown by preference count
    bpc = summary.get("by_preference_count", {})
    if bpc:
        print(f"\n{'Pref count':<12} {'Total':>7} {'Correct':>9} {'Accuracy':>10} {'Avg retrieval':>15}")
        print("-" * 58)
        for n in sorted(bpc):
            d = bpc[n]
            print(
                f"{n:<12} {d['total']:>7} {d['correct']:>9} "
                f"{d['accuracy']:>9.1%} {d['avg_retrieval_fraction']:>14.1%}"
            )
        print("-" * 58)

    # Breakdown by category
    cat_totals: Dict[str, int] = {}
    cat_correct: Dict[str, int] = {}
    cat_rf: Dict[str, List[float]] = {}
    for t in results.traces:
        cat = t.preference_category or "unknown"
        cat_totals[cat] = cat_totals.get(cat, 0) + 1
        if t.judge_result == "correct":
            cat_correct[cat] = cat_correct.get(cat, 0) + 1
        cat_rf.setdefault(cat, []).append(t.preferences_retrieved_fraction)
    if cat_totals:
        print(f"\n{'Category':<32} {'Total':>6} {'Acc':>7} {'Avg retr':>10}")
        print("-" * 60)
        for cat in sorted(cat_totals):
            total = cat_totals[cat]
            correct = cat_correct.get(cat, 0)
            rf_list = cat_rf.get(cat, [])
            avg_rf = sum(rf_list) / len(rf_list) if rf_list else 0.0
            print(f"  {cat:<30} {total:>6} {correct / total:>6.1%} {avg_rf:>9.1%}")

    print("=" * 80)
    print(f"Saved summary:          {summary_path}")
    print(f"Saved traces:           {traces_path}")
    print(f"Saved compact traces:   {traces_compact_path}")
    print(f"Saved classifications:  {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Mem0 on coexisting-facts dataset."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset CSV.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SCRIPT_DIR / "results"),
        help="Output directory for results.",
    )
    parser.add_argument("--llm-model", type=str, default="gpt-4.1-mini", help="Test-taker LLM model.")
    parser.add_argument("--judge-model", type=str, default="gpt-4.1-mini", help="Judge LLM model.")
    parser.add_argument("--num-memories", type=int, default=5, help="k for memory retrieval.")
    parser.add_argument("--shared-user-id", type=str, default="coexisting_facts_eval_user")
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

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
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
        judge_model=args.judge_model,
        num_memories=args.num_memories,
        shared_user_id=args.shared_user_id,
        seed=args.seed,
        memory=args.memory,
    )
    save_results(results, output_dir)


if __name__ == "__main__":
    main()
