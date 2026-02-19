"""
Playground script for testing mem0 with hypothesis contamination dataset.

For each example:
1. Pass in the first assertion
2. In same conversation: 5 LOCOMO prompts
3. In new conversation: 5 more LOCOMO prompts  
4. In new conversation: ask the implication question

Records full traces and summary traces using the Evaluator interface.
"""

import csv
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

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


def load_hypothesis_data(csv_path: Path, max_examples: int = 10) -> List[Dict[str, str]]:
    """Load hypothesis contamination data from CSV."""
    examples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_examples:
                break
            examples.append({
                "question": row["question"],
                "assertive": row["assertive"],
                "implication_question": row["implication_question"],
            })
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
    # Step 1: First assertion (conversation 1)
    conv1_queries = [(example["assertive"], False)]
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
    }


def main():
    """Main function to run the hypothesis contamination test."""
    # Load data
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "datasets" / "truthfulqa_implications.csv"
    
    print("Loading hypothesis contamination data...")
    examples = load_hypothesis_data(csv_path, max_examples=10)
    print(f"Loaded {len(examples)} examples")
    
    print("Loading LOCOMO prompts...")
    locomo_prompts = load_locomo_prompts(num_prompts=10)
    print(f"Loaded {len(locomo_prompts)} LOCOMO prompts")
    
    # Setup components
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise ValueError("OPENAI_KEY environment variable not set")
    
    # Set OPENAI_API_KEY for mem0 (it expects this name)
    os.environ["OPENAI_API_KEY"] = api_key
    
    llm = OpenAILLM(api_key=api_key, model="gpt-4o-mini")
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
    
    # Save results
    output_path = project_root / "playground" / "mem0_hypothesis_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
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
