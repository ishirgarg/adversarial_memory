"""
Evaluate Mem0 on the persona dataset with detailed failure analysis.

This script:
1. Loads the persona dataset CSV
2. Starts a conversation and sends facts in groups of 5 to the LLM (all in same conversation)
3. For each question, starts a new conversation, asks the question, and gets the response
4. Saves the prompt, LLM response, and entire trace for each question
5. Uses GPT-5.2 as a judge to classify each question into one of 5 failure categories
6. Returns metrics summarizing the fraction of each category
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Add parent directory to path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent.parent
sys.path.insert(0, str(parent_dir))

from src import (
    Mem0MemorySystem,
    ChatSystem,
    OpenAILLM,
    ConversationHistoryPromptTemplate,
)
from src.types import Conversation, Message

# Load .env from parent directory
env_path = parent_dir / ".env"
load_dotenv(env_path)


@dataclass
class QuestionTrace:
    """Full trace for a single question evaluation."""
    original_fact: str
    modified_fact: str
    question: str
    ground_truth_answer: str
    # Conversation where facts were stored
    fact_storage_conv_id: str
    # Conversation where question was asked
    question_conv_id: str
    # The prompt sent to LLM for the question
    formatted_prompt: str
    # The LLM's response
    llm_response: str
    # Memories retrieved for this specific question
    retrieved_memories: str
    # All memories in Mem0 at the time of asking the question
    all_memories_at_time: List[Dict[str, Any]]
    # Classification result
    classification: Optional[str] = None
    # Judge's reasoning
    judge_reasoning: Optional[str] = None


@dataclass
class EvaluationResults:
    """Results of the evaluation."""
    total_questions: int
    correct_answer: int = 0
    incorrect_not_stored: int = 0
    incorrect_not_retrieved: int = 0
    incorrect_modified: int = 0
    incorrect_wrong_answer: int = 0
    traces: List[QuestionTrace] = None
    
    def __post_init__(self):
        if self.traces is None:
            self.traces = []
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total = self.total_questions
        if total == 0:
            return {}
        return {
            "total_questions": total,
            "correct_answer": self.correct_answer,
            "correct_answer_fraction": self.correct_answer / total,
            "incorrect_not_stored": self.incorrect_not_stored,
            "incorrect_not_stored_fraction": self.incorrect_not_stored / total,
            "incorrect_not_retrieved": self.incorrect_not_retrieved,
            "incorrect_not_retrieved_fraction": self.incorrect_not_retrieved / total,
            "incorrect_modified": self.incorrect_modified,
            "incorrect_modified_fraction": self.incorrect_modified / total,
            "incorrect_wrong_answer": self.incorrect_wrong_answer,
            "incorrect_wrong_answer_fraction": self.incorrect_wrong_answer / total,
        }


def load_persona_dataset(csv_path: Path) -> List[Dict[str, str]]:
    """Load the persona dataset from CSV."""
    dataset = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append({
                'original_fact': row['original_fact'],
                'modified_fact': row['modified_fact'],
                'question': row['question'],
                'ground_truth_answer': row['ground_truth_answer'],
            })
    return dataset


def store_facts_in_groups(
    facts: List[str],
    memory_system: Mem0MemorySystem,
    chat_system: ChatSystem,
    llm: OpenAILLM,
    prompt_template: ConversationHistoryPromptTemplate,
    group_size: int = 10
) -> List[str]:
    """
    Store facts in groups of group_size, with a new conversation for each group.
    Returns a list of conversation IDs (one per group).
    """
    conv_ids = []
    
    # Process facts in groups
    for i in range(0, len(facts), group_size):
        group = facts[i:i + group_size]
        # Start a new conversation for this group
        conv_id = chat_system.start_new_conversation()
        conv_ids.append(str(conv_id))
        
        # Combine facts into a single message
        fact_message = "\n".join([f"- {fact}" for fact in group])
        
        # Get current conversation state (empty for new conversation)
        conversation = chat_system.get_conversation(conv_id)
        if conversation is None:
            raise ValueError(f"Conversation {conv_id} not found")
        
        # Retrieve memories (may be empty for first group in first conversation)
        memories = memory_system.get_memories(fact_message, conversation)
        
        # Format prompt
        formatted_prompt = prompt_template.format(fact_message, memories, conversation)
        
        # Send to LLM
        response = chat_system.send_message(formatted_prompt, conv_id)
        
        # Update memory
        updated_conversation = chat_system.get_conversation(conv_id)
        if updated_conversation:
            memory_system.update_memory(fact_message, response, updated_conversation)
    
    return conv_ids


def ask_question(
    question: str,
    memory_system: Mem0MemorySystem,
    chat_system: ChatSystem,
    llm: OpenAILLM,
    prompt_template: ConversationHistoryPromptTemplate,
) -> tuple[str, str, str, str]:
    """
    Ask a question in a new conversation and return the response, prompt, retrieved memories, and conv_id.
    Returns: (formatted_prompt, llm_response, retrieved_memories, conv_id)
    Note: We do NOT update memory after asking the question to avoid polluting the memory system.
    """
    conv_id = chat_system.start_new_conversation()
    
    # Get conversation state (empty for new conversation)
    conversation = chat_system.get_conversation(conv_id)
    if conversation is None:
        raise ValueError(f"Conversation {conv_id} not found")
    
    # Retrieve memories
    retrieved_memories = memory_system.get_memories(question, conversation)
    
    # Format prompt
    formatted_prompt = prompt_template.format(question, retrieved_memories, conversation)
    
    # Send to LLM
    llm_response = chat_system.send_message(formatted_prompt, conv_id)
    
    # Note: We intentionally do NOT call update_memory here to avoid polluting
    # the memory system with question-answer pairs
    
    return formatted_prompt, llm_response, retrieved_memories, str(conv_id)


def classify_with_judge(
    question: str,
    ground_truth_answer: str,
    original_fact: str,
    llm_response: str,
    retrieved_memories: str,
    all_memories: List[Dict[str, Any]],
    judge_client: OpenAI,
    judge_model: str = "gpt-5.2",  # Note: Using gpt-5.2 as latest available model
) -> tuple[str, str]:
    """
    Use GPT-5.2 (or specified model) as a judge to classify the response into one of 5 categories.
    Returns: (classification, reasoning)
    """
    # Format all memories for the judge
    all_memories_str = "\n".join([
        f"- {mem.get('memory', str(mem))}" for mem in all_memories
    ])
    
    prompt = f"""You are evaluating whether an LLM correctly answered a question based on stored memories.

# ORIGINAL FACT (the fact that was stored in memory):
{original_fact}

# QUESTION ASKED:
{question}

# GROUND TRUTH ANSWER:
{ground_truth_answer}

# LLM'S RESPONSE:
{llm_response}

# MEMORIES RETRIEVED FOR THIS QUESTION (what the LLM saw):
{retrieved_memories if retrieved_memories else "(No memories retrieved)"}

# ALL MEMORIES IN THE MEMORY SYSTEM AT THIS TIME:
{all_memories_str if all_memories_str else "(No memories in system)"}

# TASK:
Classify this into exactly ONE of the following 5 categories:

1. "correct_answer" - The LLM gave a correct answer (does not need to be an exact match, but the model generally had the right idea)

2. "incorrect_not_stored" - The memory system did NOT store the original fact that we prompted the LLM with

3. "incorrect_not_retrieved" - The memory system stored the fact, but it was NOT retrieved when asking the question

4. "incorrect_modified" - The memory system stored the fact, and it was retrieved, but the fact was modified by the LLM in a way that would cause it to give the wrong answer. The fact must have been modified in a significant enough manner that a perfectly rational thinker would lead to incorrect conclusions about the question.

5. "incorrect_wrong_answer" - The memory system stored the fact, and it was retrieved, and the fact is the same as the original fact, but the model gave the wrong answer anyway

# OUTPUT FORMAT:
Respond with a JSON object with exactly two fields:
- "classification": one of the 5 category names above (exactly as written)
- "reasoning": a brief explanation of your classification (2-3 sentences)

Example:
{{
  "classification": "incorrect_not_retrieved",
  "reasoning": "The original fact was stored in the memory system (I can see it in the all_memories list), but when the question was asked, it was not retrieved. The retrieved memories do not contain the original fact."
}}
"""

    response = judge_client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # Use deterministic temperature for consistency
        response_format={"type": "json_object"},
    )
    
    result = json.loads(response.choices[0].message.content)
    classification = result.get("classification", "unknown")
    reasoning = result.get("reasoning", "")
    
    # Validate classification
    valid_categories = [
        "correct_answer",
        "incorrect_not_stored",
        "incorrect_not_retrieved",
        "incorrect_modified",
        "incorrect_wrong_answer",
    ]
    if classification not in valid_categories:
        print(f"Warning: Invalid classification '{classification}', defaulting to 'unknown'")
        classification = "unknown"
    
    return classification, reasoning


def run_evaluation(
    dataset_path: Path,
    output_dir: Path,
    api_key: str,
    llm_model: str = "gpt-5.2",
    judge_model: str = "gpt-5.2",
    num_memories: int = 5,
    fact_group_size: int = 5,
    shared_user_id: str = "persona_eval_user",
) -> EvaluationResults:
    """
    Run the full evaluation pipeline.
    """
    print("=" * 80)
    print("Mem0 Persona Dataset Evaluation")
    print("=" * 80)
    
    # Load dataset
    print(f"\n1. Loading dataset from {dataset_path}...")
    dataset = load_persona_dataset(dataset_path)
    print(f"   Loaded {len(dataset)} questions")
    
    # Initialize systems
    print("\n2. Initializing systems...")
    memory_system = Mem0MemorySystem(
        num_memories=num_memories,
        shared_user_id=shared_user_id,
    )
    llm = OpenAILLM(api_key=api_key, model=llm_model)
    chat_system = ChatSystem(llm)
    prompt_template = ConversationHistoryPromptTemplate()
    judge_client = OpenAI(api_key=api_key)
    
    # Extract all unique original facts
    print("\n3. Extracting facts to store...")
    original_facts = [row['original_fact'] for row in dataset]
    unique_facts = list(dict.fromkeys(original_facts))  # Preserve order, remove duplicates
    print(f"   Found {len(unique_facts)} unique facts to store")
    
    # Store facts in groups
    print(f"\n4. Storing facts in groups of {fact_group_size} (new conversation per group)...")
    fact_storage_conv_ids = store_facts_in_groups(
        unique_facts,
        memory_system,
        chat_system,
        llm,
        prompt_template,
        group_size=fact_group_size,
    )
    print(f"   Facts stored in {len(fact_storage_conv_ids)} conversations: {fact_storage_conv_ids[:3]}{'...' if len(fact_storage_conv_ids) > 3 else ''}")
    
    # Get all memories after storing facts
    all_memories_after_storage = memory_system.get_all_memories()
    print(f"   Total memories in system after storage: {len(all_memories_after_storage)}")
    
    # Evaluate each question
    print(f"\n5. Evaluating {len(dataset)} questions...")
    results = EvaluationResults(total_questions=len(dataset))
    
    for i, row in enumerate(tqdm(dataset, desc="Processing questions")):
        original_fact = row['original_fact']
        modified_fact = row['modified_fact']
        question = row['question']
        ground_truth_answer = row['ground_truth_answer']
        
        # Ask the question in a new conversation
        formatted_prompt, llm_response, retrieved_memories, question_conv_id = ask_question(
            question,
            memory_system,
            chat_system,
            llm,
            prompt_template,
        )
        
        # Get all memories at this point
        all_memories_at_time = memory_system.get_all_memories()
        
        # Classify with judge
        classification, reasoning = classify_with_judge(
            question,
            ground_truth_answer,
            original_fact,
            llm_response,
            retrieved_memories,
            all_memories_at_time,
            judge_client,
            judge_model=judge_model,
        )
        
        # Create trace
        trace = QuestionTrace(
            original_fact=original_fact,
            modified_fact=modified_fact,
            question=question,
            ground_truth_answer=ground_truth_answer,
            fact_storage_conv_id=",".join(fact_storage_conv_ids),  # Store all conv IDs as comma-separated
            question_conv_id=question_conv_id,
            formatted_prompt=formatted_prompt,
            llm_response=llm_response,
            retrieved_memories=retrieved_memories,
            all_memories_at_time=all_memories_at_time,
            classification=classification,
            judge_reasoning=reasoning,
        )
        results.traces.append(trace)
        
        # Update counts
        if classification == "correct_answer":
            results.correct_answer += 1
        elif classification == "incorrect_not_stored":
            results.incorrect_not_stored += 1
        elif classification == "incorrect_not_retrieved":
            results.incorrect_not_retrieved += 1
        elif classification == "incorrect_modified":
            results.incorrect_modified += 1
        elif classification == "incorrect_wrong_answer":
            results.incorrect_wrong_answer += 1
    
    return results


def save_results(results: EvaluationResults, output_dir: Path):
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    summary = results.get_summary()
    summary_path = output_dir / f"summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")
    
    # Save all traces
    traces_path = output_dir / f"traces_{timestamp}.json"
    traces_data = [asdict(trace) for trace in results.traces]
    # Convert all_memories_at_time to serializable format
    for trace_data in traces_data:
        if trace_data['all_memories_at_time']:
            # Convert dict objects to plain dicts if needed
            trace_data['all_memories_at_time'] = [
                mem if isinstance(mem, dict) else {"memory": str(mem)}
                for mem in trace_data['all_memories_at_time']
            ]
    
    with open(traces_path, 'w') as f:
        json.dump(traces_data, f, indent=2)
    print(f"Saved traces to {traces_path}")
    
    # Save classification counts CSV
    csv_path = output_dir / f"classifications_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['question_index', 'original_fact', 'question', 'classification'])
        for i, trace in enumerate(results.traces):
            writer.writerow([
                i,
                trace.original_fact,
                trace.question,
                trace.classification,
            ])
    print(f"Saved classifications to {csv_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total questions: {summary['total_questions']}")
    print(f"\nCorrect answer: {summary['correct_answer']} ({summary['correct_answer_fraction']:.1%})")
    print(f"Incorrect - not stored: {summary['incorrect_not_stored']} ({summary['incorrect_not_stored_fraction']:.1%})")
    print(f"Incorrect - not retrieved: {summary['incorrect_not_retrieved']} ({summary['incorrect_not_retrieved_fraction']:.1%})")
    print(f"Incorrect - modified: {summary['incorrect_modified']} ({summary['incorrect_modified_fraction']:.1%})")
    print(f"Incorrect - wrong answer: {summary['incorrect_wrong_answer']} ({summary['incorrect_wrong_answer_fraction']:.1%})")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Mem0 on persona dataset with failure analysis"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to persona dataset CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: playground/failure_analysis/results)"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-5.2",
        help="LLM model for answering questions (default: gpt-5.2)"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5.2",
        help="Judge model for classification (default: gpt-5.2)"
    )
    parser.add_argument(
        "--num-memories",
        type=int,
        default=5,
        help="Number of memories to retrieve (default: 5)"
    )
    parser.add_argument(
        "--fact-group-size",
        type=int,
        default=10,
        help="Number of facts to send per message (default: 10)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (default: from OPENAI_KEY or OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--shared-user-id",
        type=str,
        default="persona_eval_user",
        help="Shared user ID for Mem0 (default: persona_eval_user)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_KEY or OPENAI_API_KEY environment variable, or use --api-key")
    
    # Resolve paths
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = parent_dir / dataset_path
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = script_dir / "results"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Run evaluation
    results = run_evaluation(
        dataset_path=dataset_path,
        output_dir=output_dir,
        api_key=api_key,
        llm_model=args.llm_model,
        judge_model=args.judge_model,
        num_memories=args.num_memories,
        fact_group_size=args.fact_group_size,
        shared_user_id=args.shared_user_id,
    )
    
    # Save results
    save_results(results, output_dir)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
