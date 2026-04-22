"""
Playground script for testing memory systems with hypothesis contamination dataset.

For each example (run sequentially on the SAME memory system, no resets):
1. Pass in the assertive (or essay)
2. In a new conversation: 2 LOCOMO filler prompts
3. In a new conversation: ask the implication question

Supports both mem0 and A-MEM backends via --memory flag.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    AMEMMemorySystem,
    ChatDataset,
    ConversationData,
    Evaluator,
    Mem0MemorySystem,
    SimplePromptTemplate,
    TiktokenTokenizer,
)
from src.llm import OpenAILLM
from src.types import MemorySystem
from dotenv import load_dotenv
import os

# Load .env from parent directory
load_dotenv(Path(__file__).parent.parent / ".env")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_locomo_prompts(num_prompts: int = 10) -> List[str]:
    """
    Load LOCOMO dataset prompts.

    TODO: Replace with actual LOCOMO dataset loading.
    For now, returns fake prompts for testing.
    """
    fake_prompts = [
        "What are the main principles of machine learning?",
        "Explain the difference between supervised and unsupervised learning.",
        "How does gradient descent work in neural networks?",
        "What is the purpose of regularization in machine learning?",
        "Describe the architecture of a transformer model.",
        "What are the advantages of using attention mechanisms?",
        "How do convolutional neural networks process images?",
        "What is the role of activation functions in neural networks?",
        "Explain the concept of overfitting and how to prevent it.",
        "What are some common evaluation metrics for classification tasks?",
    ]
    return fake_prompts[:num_prompts]


def load_hypothesis_data(
    csv_path: Path,
    max_examples: int = 10,
    use_essays: bool = False,
) -> List[Dict[str, str]]:
    """
    Load hypothesis contamination data from CSV.

    Args:
        csv_path: Path to the CSV file
        max_examples: Maximum number of examples to load
        use_essays: If True, load essays from CSV (requires 'essay' column)

    Returns:
        List of example dicts: question, assertive, implication_question, [essay]
    """
    examples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_examples:
                break
            example = {
                "question": row["question"],
                "assertive": row["assertive"],
                "implication_question": row["implication_question"],
            }
            if use_essays:
                if "essay" not in row:
                    raise ValueError(
                        f"CSV file {csv_path} does not have an 'essay' column. "
                        "Cannot use --use-essays flag."
                    )
                example["essay"] = row["essay"]
            examples.append(example)
    return examples


# ── Per-example evaluation ────────────────────────────────────────────────────

def run_single_example(
    example: Dict[str, str],
    locomo_prompts: List[str],
    memory_system: MemorySystem,
    llm: OpenAILLM,
    prompt_template: SimplePromptTemplate,
    tokenizer: TiktokenTokenizer,
) -> Dict[str, Any]:
    """
    Run one example through the full test sequence using the Evaluator.

    The memory system is NOT reset between calls — all examples share the same
    persistent memory store.

    Conversation layout:
        conv1 – assertive (or essay)          [should_grade=False]
        conv2 – 2 LOCOMO filler prompts       [should_grade=False]
        conv3 – implication question          [should_grade=True]
    """
    first_input = example.get("essay", example["assertive"])

    conv1_data = ConversationData(queries=[(first_input, False)])
    conv2_data = ConversationData(queries=[(p, False) for p in locomo_prompts[:2]])
    conv3_data = ConversationData(queries=[(example["implication_question"], True)])

    dataset = ChatDataset(conversations=[conv1_data, conv2_data, conv3_data])
    evaluator = Evaluator(
        memory_system=memory_system,
        llm=llm,
        dataset=dataset,
        prompt_template=prompt_template,
        tokenizer=tokenizer,
    )

    summary = evaluator.evaluate()

    # Flatten all traces
    all_traces = [t for result in summary.results for t in result.traces]

    first_trace = summary.results[0].traces[0]   # assertive turn
    last_trace  = summary.results[2].traces[0]   # implication turn

    summary_trace = {
        "first_assertion": {
            "query": first_trace.query,
            "response": first_trace.response,
            "retrieved_memories": first_trace.retrieved_memories,
        },
        "last_question": {
            "query": last_trace.query,
            "response": last_trace.response,
            "retrieved_memories": last_trace.retrieved_memories,
        },
    }

    full_traces_dict = [
        {
            "query": t.query,
            "should_grade": t.should_grade,
            "retrieved_memories": t.retrieved_memories,
            "formatted_prompt": t.formatted_prompt,
            "response": t.response,
            "input_tokens": t.input_tokens,
            "output_tokens": t.output_tokens,
            "retrieval_time": t.retrieval_time,
            "llm_time": t.llm_time,
        }
        for t in all_traces
    ]

    return {
        "example": example,
        "full_traces": full_traces_dict,
        "summary_trace": summary_trace,
        "first_assertion_response": first_trace.response,
    }


# ── Hardcoded memory system config ───────────────────────────────────────────

NUM_MEMORIES       = 3
OPENAI_MODEL       = "gpt-4.1-mini"

# mem0
MEM0_SHARED_USER_ID = "0"

# A-MEM
AMEM_LLM_BACKEND    = "openai"
AMEM_LLM_MODEL      = "gpt-4.1-mini"
AMEM_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
AMEM_EVO_THRESHOLD  = 30


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).parent.parent
    default_csv    = project_root / "datasets" / "truthfulqa_implications.csv"
    default_essays = project_root / "datasets" / "truthfulqa_implications_essays.csv"

    parser = argparse.ArgumentParser(
        description="Test a memory system with the hypothesis contamination dataset."
    )

    parser.add_argument(
        "--memory",
        choices=["mem0", "amem"],
        default="mem0",
        help="Memory backend to use (default: mem0)",
    )
    parser.add_argument(
        "--csv-input",
        type=Path,
        default=None,
        help=f"Input CSV path (default: {default_csv} or {default_essays} with --use-essays)",
    )
    parser.add_argument(
        "--essays-csv",
        type=Path,
        default=default_essays,
        help=f"Essays CSV path used when --use-essays is set (default: {default_essays})",
    )
    parser.add_argument(
        "--use-essays",
        action="store_true",
        help="Use essays instead of bare assertives (requires 'essay' column in CSV)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum number of examples to process (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files (default: playground/)",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    os.environ["OPENAI_API_KEY"] = api_key  # mem0 expects this name

    # ── Resolve CSV path ──────────────────────────────────────────────────────
    if args.csv_input:
        csv_path = args.csv_input
    elif args.use_essays:
        csv_path = args.essays_csv
    else:
        csv_path = project_root / "datasets" / "truthfulqa_implications.csv"

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading hypothesis contamination data...")
    examples = load_hypothesis_data(csv_path, max_examples=args.max_examples, use_essays=args.use_essays)
    print(f"Loaded {len(examples)} examples" + (" (using essays)" if args.use_essays else ""))

    print("Loading LOCOMO prompts...")
    locomo_prompts = load_locomo_prompts(num_prompts=10)
    print(f"Loaded {len(locomo_prompts)} LOCOMO prompts")

    # ── Build memory system ───────────────────────────────────────────────────
    if args.memory == "mem0":
        print("Initialising mem0 memory system (shared_user_id='0')...")
        memory_system: MemorySystem = Mem0MemorySystem(
            num_memories=NUM_MEMORIES,
            shared_user_id=MEM0_SHARED_USER_ID,
        )
    else:
        print("Initialising A-MEM memory system...")
        memory_system = AMEMMemorySystem(
            num_memories=NUM_MEMORIES,
            llm_backend=AMEM_LLM_BACKEND,
            llm_model=AMEM_LLM_MODEL,
            embedding_model=AMEM_EMBEDDING_MODEL,
            evo_threshold=AMEM_EVO_THRESHOLD,
            api_key=api_key,
        )

    llm = OpenAILLM(api_key=api_key, model=OPENAI_MODEL)
    prompt_template = SimplePromptTemplate()
    tokenizer = TiktokenTokenizer()

    # ── Run all examples sequentially on the same memory system ──────────────
    all_results = []
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"Processing example {i}/{len(examples)}")
        print(f"Question:    {example['question']}")
        print(f"Implication: {example['implication_question']}")
        print(f"{'='*60}")

        result = run_single_example(
            example,
            locomo_prompts,
            memory_system,
            llm,
            prompt_template,
            tokenizer,
        )
        all_results.append(result)

    # ── Save outputs ──────────────────────────────────────────────────────────
    output_dir = args.output_dir or (project_root / "playground")
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = args.memory  # "mem0" or "amem"

    json_path = output_dir / f"{tag}_hypothesis_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON results saved to {json_path}")

    csv_path_out = output_dir / f"{tag}_hypothesis_assertions_responses.csv"
    with open(csv_path_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["false_assertion", "llm_response"])
        for result in all_results:
            writer.writerow([result["example"]["assertive"], result["first_assertion_response"]])
    print(f"CSV results saved to {csv_path_out}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary Traces:")
    for i, result in enumerate(all_results, 1):
        st = result["summary_trace"]
        print(f"\nExample {i}:")
        print(f"  [Assertion] {st['first_assertion']['query']}")
        print(f"    → Response:  {st['first_assertion']['response'][:100]}...")
        mem = st['first_assertion']['retrieved_memories']
        print(f"    → Memories:  {(mem[:100] + '...') if mem else 'None'}")
        print(f"  [Implication] {st['last_question']['query']}")
        print(f"    → Response:  {st['last_question']['response'][:100]}...")
        mem = st['last_question']['retrieved_memories']
        print(f"    → Memories:  {(mem[:100] + '...') if mem else 'None'}")


if __name__ == "__main__":
    main()
