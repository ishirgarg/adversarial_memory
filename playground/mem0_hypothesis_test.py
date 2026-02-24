"""
Playground script for testing mem0 with hypothesis contamination dataset.

For each example:
1. Pass in the first assertion
2. In same conversation: 5 LOCOMO prompts
3. In new conversation: 5 more LOCOMO prompts  
4. In new conversation: ask the implication question

Records full traces and summary traces using the Evaluator interface.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    Mem0MemorySystem,
    SimplePromptTemplate,
    TiktokenTokenizer,
    ConversationData,
    ChatDataset,
    Evaluator,
)
from src.llm import OpenAILLM
from dotenv import load_dotenv
import os

# Load .env from parent directory
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
env_path = parent_dir / ".env"
load_dotenv(env_path)


def load_locomo_prompts(num_prompts: int = 10) -> List[str]:
    """
    Load LOCOMO dataset prompts.
    
    TODO: Replace with actual LOCOMO dataset loading.
    For now, returns fake prompts for testing.
    
    Expected format: List of prompt strings from LOCOMO dataset.
    """
    # Fake LOCOMO prompts for testing
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


def load_hypothesis_data(csv_path: Path, max_examples: int = 10, use_essays: bool = False) -> List[Dict[str, str]]:
    """
    Load hypothesis contamination data from CSV.
    
    Args:
        csv_path: Path to the CSV file
        max_examples: Maximum number of examples to load
        use_essays: If True, load essays from CSV (requires 'essay' column)
    
    Returns:
        List of example dictionaries with question, assertive, implication_question, and optionally essay
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
                    raise ValueError(f"CSV file {csv_path} does not have 'essay' column. Cannot use --use-essays flag.")
                example["essay"] = row["essay"]
            examples.append(example)
    return examples


def run_single_example(
    example: Dict[str, str],
    locomo_prompts: List[str],
    memory_system: Mem0MemorySystem,
    llm: OpenAILLM,
    prompt_template: SimplePromptTemplate,
    tokenizer: TiktokenTokenizer,
    shared_user_id: str,
) -> Dict[str, Any]:
    """
    Run a single example through the full test sequence using the Evaluator.
    
    Args:
        example: The example data (question, assertive, implication_question)
        locomo_prompts: List of LOCOMO prompts to use
        memory_system: The memory system instance
        llm: The LLM instance
        prompt_template: The prompt template
        tokenizer: The tokenizer
        shared_user_id: Shared user_id to use for mem0 across all conversations
    
    Returns a dict with full traces and summary trace.
    """
    # Set shared_user_id on the memory system so all conversations share memories
    memory_system.shared_user_id = shared_user_id
    
    # Construct the dataset with all three conversations
    # Step 1: First assertion (or essay if available) (conversation 1)
    # Use essay if available, otherwise use assertive
    first_input = example.get("essay", example["assertive"])
    conv1_queries = [(first_input, False)]
    conv1_data = ConversationData(queries=conv1_queries)
    
    # Step 2: New conversation with 2 LOCOMO prompts (conversation 2)
    conv2_queries = [(prompt, False) for prompt in locomo_prompts[:2]]
    conv2_data = ConversationData(queries=conv2_queries)
    
    # Step 3: New conversation with implication question (conversation 3)
    conv3_queries = [(example["implication_question"], True)]
    conv3_data = ConversationData(queries=conv3_queries)
    
    # Create dataset with all three conversations
    dataset = ChatDataset(conversations=[conv1_data, conv2_data, conv3_data])
    
    # Create evaluator with the dataset
    evaluator = Evaluator(
        memory_system=memory_system,
        llm=llm,
        dataset=dataset,
        prompt_template=prompt_template,
        tokenizer=tokenizer,
    )
    
    # Run evaluation
    summary = evaluator.evaluate()
    
    # Extract traces from all results
    all_traces = []
    for result in summary.results:
        all_traces.extend(result.traces)
    
    # Extract first and last traces for summary
    first_trace = summary.results[0].traces[0]  # First assertion
    last_trace = summary.results[2].traces[0]   # Implication question
    
    # Create summary trace
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
    
    # Convert QueryTrace objects to dicts for JSON serialization
    full_traces_dict = []
    for trace in all_traces:
        full_traces_dict.append({
            "query": trace.query,
            "should_grade": trace.should_grade,
            "retrieved_memories": trace.retrieved_memories,
            "formatted_prompt": trace.formatted_prompt,
            "response": trace.response,
            "input_tokens": trace.input_tokens,
            "output_tokens": trace.output_tokens,
            "retrieval_time": trace.retrieval_time,
            "llm_time": trace.llm_time,
        })
    
    return {
        "example": example,
        "full_traces": full_traces_dict,
        "summary_trace": summary_trace,
        "first_assertion_response": first_trace.response,  # For CSV output
    }


def main():
    """Main function to run the hypothesis contamination test."""
    parser = argparse.ArgumentParser(
        description="Test mem0 with hypothesis contamination dataset"
    )
    parser.add_argument(
        "--use-essays",
        action="store_true",
        help="Use essays instead of just false assertions (requires essay column in CSV)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum number of examples to process (default: 10)"
    )
    parser.add_argument(
        "--csv-input",
        type=str,
        default=None,
        help="Path to input CSV file (default: datasets/truthfulqa_implications.csv)"
    )
    parser.add_argument(
        "--essays-csv",
        type=str,
        default=None,
        help="Path to CSV file with essays (default: datasets/truthfulqa_implications_essays.csv)"
    )
    
    args = parser.parse_args()
    
    # Load data
    project_root = Path(__file__).parent.parent
    
    # Determine input CSV path
    if args.csv_input:
        csv_path = Path(args.csv_input)
    elif args.use_essays and args.essays_csv:
        csv_path = Path(args.essays_csv)
    elif args.use_essays:
        csv_path = project_root / "datasets" / "truthfulqa_implications_essays.csv"
    else:
        csv_path = project_root / "datasets" / "truthfulqa_implications.csv"
    
    print("Loading hypothesis contamination data...")
    examples = load_hypothesis_data(csv_path, max_examples=args.max_examples, use_essays=args.use_essays)
    print(f"Loaded {len(examples)} examples")
    if args.use_essays:
        print("Using essays instead of false assertions")
    
    print("Loading LOCOMO prompts...")
    locomo_prompts = load_locomo_prompts(num_prompts=10)
    print(f"Loaded {len(locomo_prompts)} LOCOMO prompts")
    
    # Setup components
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise ValueError("OPENAI_KEY environment variable not set")
    
    # Set OPENAI_API_KEY for mem0 (it expects this name)
    os.environ["OPENAI_API_KEY"] = api_key
    
    llm = OpenAILLM(api_key=api_key, model="gpt-4.1-mini")
    memory_system = Mem0MemorySystem(num_memories=2)
    prompt_template = SimplePromptTemplate()
    tokenizer = TiktokenTokenizer()
    
    # Run all examples
    all_results = []
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"Processing example {i}/{len(examples)}")
        print(f"Question: {example['question']}")
        print(f"{'='*60}")
        
        # Use a shared user_id for this example so mem0 can retrieve memories across conversations
        shared_user_id = f"example_{i}"
        
        result = run_single_example(
            example,
            locomo_prompts,
            memory_system,
            llm,
            prompt_template,
            tokenizer,
            shared_user_id,
        )
        all_results.append(result)
    
    # Save JSON results
    output_path = project_root / "playground" / "mem0_hypothesis_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"JSON results saved to {output_path}")
    
    # Save CSV with false assertions and responses
    csv_output_path = project_root / "playground" / "mem0_hypothesis_assertions_responses.csv"
    with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["false_assertion", "llm_response"])
        for result in all_results:
            false_assertion = result["example"]["assertive"]
            llm_response = result["first_assertion_response"]
            writer.writerow([false_assertion, llm_response])
    
    print(f"CSV results saved to {csv_output_path}")
    print(f"{'='*60}")
    
    # Print summary
    print("\nSummary Traces:")
    for i, result in enumerate(all_results, 1):
        print(f"\nExample {i}:")
        print(f"  First Assertion: {result['summary_trace']['first_assertion']['query']}")
        print(f"    Response: {result['summary_trace']['first_assertion']['response'][:100]}...")
        print(f"    Memories: {result['summary_trace']['first_assertion']['retrieved_memories'][:100] if result['summary_trace']['first_assertion']['retrieved_memories'] else 'None'}...")
        print(f"  Last Question: {result['summary_trace']['last_question']['query']}")
        print(f"    Response: {result['summary_trace']['last_question']['response'][:100]}...")
        print(f"    Memories: {result['summary_trace']['last_question']['retrieved_memories'][:100] if result['summary_trace']['last_question']['retrieved_memories'] else 'None'}...")


if __name__ == "__main__":
    main()
