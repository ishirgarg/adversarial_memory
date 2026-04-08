"""
Evaluate Mem0 on the misleading persona dataset with memory-usage compliance checks.

For each question:
- The test-taker LLM is prompted to explicitly cite memories it used.
- A judge LLM grades whether memory usage behavior matches policy by `question_type`.

Policy:
- base_question: model SHOULD invoke relevant memory tied to original_fact.
- misleading_question: model SHOULD NOT invoke any memory.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PARENT_DIR))

from src import ChatSystem, ConversationHistoryPromptTemplate, Mem0MemorySystem, OpenAILLM

# Load .env from project root
load_dotenv(PARENT_DIR / ".env")

MAX_JUDGE_RETRIES = 3


@dataclass
class QuestionTrace:
    original_fact: str
    modified_fact: str
    question: str
    question_type: str
    ground_truth_answer: str
    question_conv_id: str
    formatted_prompt: str
    llm_response: str
    retrieved_memories: str
    all_memories_at_time: List[Dict[str, Any]]
    judge_result: str
    judge_reasoning: str


@dataclass
class EvaluationResults:
    total_questions: int = 0
    total_base_questions: int = 0
    total_misleading_questions: int = 0
    correct_base_questions: int = 0
    correct_misleading_questions: int = 0
    traces: List[QuestionTrace] = None

    def __post_init__(self) -> None:
        if self.traces is None:
            self.traces = []

    def summary(self) -> Dict[str, Any]:
        total_correct = self.correct_base_questions + self.correct_misleading_questions
        return {
            "total_questions": self.total_questions,
            "total_correct": total_correct,
            "aggregate_accuracy": (total_correct / self.total_questions) if self.total_questions else 0.0,
            "base_question_total": self.total_base_questions,
            "base_question_correct": self.correct_base_questions,
            "base_question_accuracy": (
                self.correct_base_questions / self.total_base_questions
            ) if self.total_base_questions else None,
            "misleading_question_total": self.total_misleading_questions,
            "misleading_question_correct": self.correct_misleading_questions,
            "misleading_question_accuracy": (
                self.correct_misleading_questions / self.total_misleading_questions
            ) if self.total_misleading_questions else None,
        }


def load_dataset(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = ["original_fact", "modified_fact", "question", "ground_truth_answer", "misleading_question"]
        if not reader.fieldnames:
            raise ValueError("Dataset CSV missing header.")
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        has_misleading_question = "misleading_question" in reader.fieldnames
        for row in reader:
            # Base question row (uses provided flag from CSV)
            rows.append(
                {
                    "original_fact": row["original_fact"],
                    "modified_fact": row["modified_fact"],
                    "question": row["question"],
                    "ground_truth_answer": row["ground_truth_answer"],
                    "question_type": "base_question",
                }
            )
            misleading_question = str(row.get("misleading_question", "")).strip()
            if misleading_question:
                rows.append(
                    {
                        "original_fact": row["original_fact"],
                        "modified_fact": row["modified_fact"],
                        "question": misleading_question,
                        "ground_truth_answer": "ABSTAIN",
                        "question_type": "misleading_question",
                    }
                )
    return rows


def store_facts(
    facts: List[str],
    memory_system: Mem0MemorySystem,
    chat_system: ChatSystem,
    prompt_template: ConversationHistoryPromptTemplate,
    group_size: int = 10,
) -> None:
    for i in range(0, len(facts), group_size):
        group = facts[i : i + group_size]
        conv_id = chat_system.start_new_conversation()
        fact_message = "\n".join(f"- {f}" for f in group)
        conversation = chat_system.get_conversation(conv_id)
        if conversation is None:
            raise ValueError("Failed to get conversation for fact storage.")
        memories = memory_system.get_memories(fact_message, conversation)
        prompt = prompt_template.format(fact_message, memories, conversation)
        response = chat_system.send_message(prompt, conv_id)
        updated = chat_system.get_conversation(conv_id)
        if updated:
            memory_system.update_memory(fact_message, response, updated)


def ask_question(
    question: str,
    memory_system: Mem0MemorySystem,
    chat_system: ChatSystem,
    prompt_template: ConversationHistoryPromptTemplate,
) -> tuple[str, str, str, str]:
    conv_id = chat_system.start_new_conversation()
    conversation = chat_system.get_conversation(conv_id)
    if conversation is None:
        raise ValueError("Failed to get conversation for question.")

    retrieved_memories = memory_system.get_memories(question, conversation)
    compliance_instruction = (
        "\n\nAnswer the question directly. Then add a line starting with "
        "'MEMORY_USED:' and state either:\n"
        "- 'none' if you did not use any memory, or\n"
        "- a brief quote/paraphrase of the specific memory you used."
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
    question_type: str,
    original_fact: str,
    llm_response: str,
    retrieved_memories: str,
) -> tuple[str, str]:
    # Priority rule: misleading questions should be answered without memory usage.
    if question_type == "misleading_question":
        policy = "SHOULD_NOT_USE_MEMORY"
    else:
        policy = "SHOULD_USE_RELEVANT_MEMORY_FOR_ORIGINAL_FACT"
    prompt = f"""You are a strict evaluator for memory-usage policy compliance.

Policy mode: {policy}

Question:
{question}

Original fact:
{original_fact}

Retrieved memories shown to test-taker:
{retrieved_memories if retrieved_memories else "(none)"}

Test-taker response:
{llm_response}

Task:
Return JSON with:
- "result": exactly one of ["correct", "incorrect"]
- "reasoning": short explanation

Grading rules:
1) If policy is SHOULD_USE_RELEVANT_MEMORY_FOR_ORIGINAL_FACT:
   - correct only if response explicitly indicates memory was used AND that memory is relevant to the original fact.
2) If policy is SHOULD_NOT_USE_MEMORY:
   - correct only if response explicitly indicates no memory was used (e.g., MEMORY_USED: none) and does not claim memory usage.
3) Use explicit evidence from the response text.
4) Output JSON only.
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
                raise ValueError(f"Invalid judge result: {result}")
            return result, reasoning
        except Exception as e:
            last_error = e
            if attempt < MAX_JUDGE_RETRIES:
                time.sleep(0.4 * attempt)

    return "incorrect", f"Judge parse failed after retries: {last_error}"


def run_evaluation(
    *,
    dataset_path: Path,
    api_key: str,
    llm_model: str,
    judge_model: str,
    num_memories: int,
    fact_group_size: int,
    shared_user_id: str,
) -> EvaluationResults:
    dataset = load_dataset(dataset_path)
    results = EvaluationResults(total_questions=len(dataset))

    memory_system = Mem0MemorySystem(num_memories=num_memories, shared_user_id=shared_user_id)
    llm = OpenAILLM(api_key=api_key, model=llm_model)
    chat_system = ChatSystem(llm)
    prompt_template = ConversationHistoryPromptTemplate()
    judge_client = OpenAI(api_key=api_key)

    # Store facts in a stable order independent from shuffled question order.
    unique_facts = list(dict.fromkeys([row["original_fact"] for row in dataset]))
    store_facts(unique_facts, memory_system, chat_system, prompt_template, group_size=fact_group_size)

    # Shuffle only the question asking order.
    question_rows = list(dataset)
    random.shuffle(question_rows)

    for row in tqdm(question_rows, desc="Evaluating"):
        formatted_prompt, llm_response, retrieved_memories, question_conv_id = ask_question(
            row["question"], memory_system, chat_system, prompt_template
        )
        all_memories_at_time = memory_system.get_all_memories()
        judge_result, judge_reasoning = judge_response(
            judge_client=judge_client,
            judge_model=judge_model,
            question=row["question"],
            question_type=row.get("question_type", "unknown"),
            original_fact=row["original_fact"],
            llm_response=llm_response,
            retrieved_memories=retrieved_memories,
        )

        trace = QuestionTrace(
            original_fact=row["original_fact"],
            modified_fact=row["modified_fact"],
            question=row["question"],
            question_type=row.get("question_type", "unknown"),
            ground_truth_answer=row["ground_truth_answer"],
            question_conv_id=question_conv_id,
            formatted_prompt=formatted_prompt,
            llm_response=llm_response,
            retrieved_memories=retrieved_memories,
            all_memories_at_time=all_memories_at_time,
            judge_result=judge_result,
            judge_reasoning=judge_reasoning,
        )
        results.traces.append(trace)

        if row.get("question_type") == "misleading_question":
            results.total_misleading_questions += 1
            if judge_result == "correct":
                results.correct_misleading_questions += 1
        else:
            results.total_base_questions += 1
            if judge_result == "correct":
                results.correct_base_questions += 1

    return results


def save_results(results: EvaluationResults, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = output_dir / f"summary_{ts}.json"
    traces_path = output_dir / f"traces_{ts}.json"
    traces_compact_path = output_dir / f"traces_compact_{ts}.json"
    csv_path = output_dir / f"classifications_{ts}.csv"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results.summary(), f, indent=2)
    with open(traces_path, "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in results.traces], f, indent=2)
    compact_traces = []
    for t in results.traces:
        t_dict = asdict(t)
        t_dict.pop("all_memories_at_time", None)
        compact_traces.append(t_dict)
    with open(traces_compact_path, "w", encoding="utf-8") as f:
        json.dump(compact_traces, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_index", "question_type", "judge_result", "question"])
        for i, t in enumerate(results.traces):
            w.writerow([i, t.question_type, t.judge_result, t.question])

    summary = results.summary()
    print("=" * 80)
    print("MISLEADING PERSONA EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total questions: {summary['total_questions']}")
    print(f"Aggregate accuracy: {summary['aggregate_accuracy']:.1%}")
    base_acc = summary["base_question_accuracy"]
    base_acc_str = "N/A" if base_acc is None else f"{base_acc:.1%}"
    print(
        f"base_question: {summary['base_question_correct']}/"
        f"{summary['base_question_total']} ({base_acc_str})"
    )
    misleading_acc = summary["misleading_question_accuracy"]
    misleading_acc_str = "N/A" if misleading_acc is None else f"{misleading_acc:.1%}"
    print(
        f"misleading_question: {summary['misleading_question_correct']}/"
        f"{summary['misleading_question_total']} ({misleading_acc_str})"
    )
    print("=" * 80)
    print(f"Saved summary: {summary_path}")
    print(f"Saved traces: {traces_path}")
    print(f"Saved compact traces: {traces_compact_path}")
    print(f"Saved classifications: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Mem0 on misleading persona dataset with memory-usage judge."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SCRIPT_DIR / "results"),
        help="Output directory",
    )
    parser.add_argument("--llm-model", type=str, default="gpt-5.2", help="Test-taker model")
    parser.add_argument("--judge-model", type=str, default="gpt-5.2", help="Judge model")
    parser.add_argument("--num-memories", type=int, default=5)
    parser.add_argument("--fact-group-size", type=int, default=10)
    parser.add_argument("--shared-user-id", type=str, default="misleading_persona_eval_user")
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required via --api-key or env var.")

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PARENT_DIR / dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PARENT_DIR / output_dir

    results = run_evaluation(
        dataset_path=dataset_path,
        api_key=api_key,
        llm_model=args.llm_model,
        judge_model=args.judge_model,
        num_memories=args.num_memories,
        fact_group_size=args.fact_group_size,
        shared_user_id=args.shared_user_id,
    )
    save_results(results, output_dir)


if __name__ == "__main__":
    main()
