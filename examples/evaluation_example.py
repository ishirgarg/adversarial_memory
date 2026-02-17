"""
Example of using the evaluation framework with a dataset.
"""

import sys
from pathlib import Path

# Add parent directory to path to import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    ChatDataset,
    Evaluator,
    OllamaLLM,
    NoHistoryMemorySystem,
    SimpleHistoryMemorySystem,
    ConversationHistoryPromptTemplate,
    SimplePromptTemplate,
    TiktokenTokenizer,
)


def main():
    print("=" * 60)
    print("Memory System Evaluation Example")
    print("=" * 60)

    # Load dataset
    print("\n1. Loading dataset...")
    dataset_path = Path(__file__).parent / "sample_dataset.json"
    dataset = ChatDataset.from_file(dataset_path)
    print(f"   Loaded {len(dataset)} conversations")
    print(f"   Total queries: {dataset.get_total_queries()}")

    # Create LLM
    print("\n2. Creating LLM...")
    llm = OllamaLLM(
        model="gemma3:1b",
        base_url="http://localhost:11434",
        max_tokens=1024,
        temperature=0.7,
    )

    # Create prompt template
    prompt_template = ConversationHistoryPromptTemplate()

    # Create tokenizer
    tokenizer = TiktokenTokenizer()

    # Test with NoHistoryMemorySystem
    print("\n3. Evaluating NoHistoryMemorySystem...")
    memory_no_history = NoHistoryMemorySystem()
    evaluator_no_history = Evaluator(
        memory_no_history, llm, dataset, prompt_template, tokenizer
    )
    results_no_history = evaluator_no_history.evaluate()

    print(f"   Total conversations: {results_no_history.total_conversations}")
    print(f"   Total queries: {results_no_history.total_queries}")
    if results_no_history.results:
        first_result = results_no_history.results[0]
        print(
            f"   First conversation - Input tokens: {first_result.total_input_tokens}"
        )
        print(
            f"   First conversation - Output tokens: {first_result.total_output_tokens}"
        )
        print(f"   First conversation - Total time: {first_result.total_time:.2f}s")
        print(
            f"   First conversation - Retrieval time: {first_result.total_retrieval_time:.2f}s"
        )

    # Test with SimpleHistoryMemorySystem
    print("\n4. Evaluating SimpleHistoryMemorySystem...")
    memory_simple = SimpleHistoryMemorySystem()
    evaluator_simple = Evaluator(
        memory_simple, llm, dataset, prompt_template, tokenizer
    )
    results_simple = evaluator_simple.evaluate()

    print(f"   Total conversations: {results_simple.total_conversations}")
    print(f"   Total queries: {results_simple.total_queries}")
    if results_simple.results:
        first_result = results_simple.results[0]
        print(
            f"   First conversation - Input tokens: {first_result.total_input_tokens}"
        )
        print(
            f"   First conversation - Output tokens: {first_result.total_output_tokens}"
        )
        print(f"   First conversation - Total time: {first_result.total_time:.2f}s")
        print(
            f"   First conversation - Retrieval time: {first_result.total_retrieval_time:.2f}s"
        )

    # Show detailed results for first conversation
    print("\n5. Detailed results for first conversation (NoHistoryMemorySystem):")
    if results_no_history.results:
        first_result = results_no_history.results[0]
        print(f"   Conversation ID: {first_result.conversation_id}")
        print(f"   Number of queries: {len(first_result.queries)}")
        for i, (query, response) in enumerate(
            zip(first_result.queries, first_result.responses), 1
        ):
            print(f"\n   Query {i}: {query}")
            print(f"   Response: {response[:100]}...")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
