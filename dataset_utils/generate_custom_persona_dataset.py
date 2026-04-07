"""
Script to generate a persona dataset with facts, complementary facts, and questions.

This script:
1. Generates a persona with N interesting facts (first person, logically consistent)
2. For each fact, generates a complementary fact (doesn't contradict, complements it)
3. For each complementary fact, generates a non-trivial question whose answer follows from the implication
4. For each question, generates the ground truth answer based on the facts
5. Exports to CSV with quadruples: (original_fact, modified_fact, question, ground_truth_answer)
"""

import argparse
import ast
import csv
import json
import os
import re
from pathlib import Path
from typing import List, Tuple
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load .env from parent directory
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
env_path = parent_dir / ".env"
load_dotenv(env_path)


def generate_persona(num_facts: int, model: str, api_key: str, temperature: float = 0.7) -> List[str]:
    """
    Generate a persona with N interesting facts about a person.
    
    Args:
        num_facts: Number of facts to generate
        model: OpenAI model to use
        api_key: OpenAI API key
        temperature: Temperature for generation
        
    Returns:
        List of facts (all in first person, "I ...")
    """
    client = OpenAI(api_key=api_key)
    
    prompt = f"""Generate a detailed persona for a person with exactly {num_facts} interesting facts about their personality, relationships, preferences, experiences, and other aspects of their life.

Requirements:
1. All facts must be in FIRST PERSON ("I ..." format)
2. All facts must be logically consistent and not contradict each other
3. Facts should be HIGHLY PERSONAL and SPECIFIC to this individual - they should NOT be answerable by general knowledge
4. Facts should cover diverse aspects: personality traits, relationships, hobbies, preferences, experiences, beliefs, habits, etc.
5. Facts should be interesting and specific (not generic like "I am a person" or "I like food")
7. Include specific names, places, dates, or unique circumstances that make each fact personal and unanswerable by general knowledge
8. Examples of good personal facts:
   - "I have a cat named Whiskers who is 5 years old"
   - "I am allergic to shellfish but not other seafood"
   - "I prefer to work out in the morning before 7am"
   - "My best friend Sarah and I met in college in 2018"

Output the facts as a Python list format, one fact per line, like this:
   [
       "I have a cat named Whiskers who is 5 years old",
       "I am allergic to shellfish but not other seafood",
       "I prefer to work out in the morning before 7am",
       ...
   ]

Only output the Python list, nothing else. Make sure the list is valid Python syntax."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    
    content = response.choices[0].message.content.strip()
    
    # Try to extract the list from the response
    # Look for content between square brackets
    match = re.search(r'\[(.*?)\]', content, re.DOTALL)
    if match:
        list_content = match.group(1)
        # Try to parse as Python list using ast.literal_eval for safety
        try:
            facts = ast.literal_eval(f"[{list_content}]")
            if isinstance(facts, list):
                # Clean up facts - ensure they're strings and start with "I"
                cleaned_facts = []
                for fact in facts:
                    fact_str = str(fact).strip()
                    # Remove quotes if present
                    if fact_str.startswith('"') and fact_str.endswith('"'):
                        fact_str = fact_str[1:-1]
                    elif fact_str.startswith("'") and fact_str.endswith("'"):
                        fact_str = fact_str[1:-1]
                    # Ensure it starts with "I"
                    if not fact_str.startswith("I "):
                        fact_str = "I " + fact_str
                    cleaned_facts.append(fact_str)
                return cleaned_facts[:num_facts]
        except (ValueError, SyntaxError):
            pass
    
    # Fallback: try to extract lines that start with "I"
    facts = []
    for line in content.split('\n'):
        line = line.strip()
        # Remove list markers, quotes, commas
        line = re.sub(r'^[-*•]\s*', '', line)
        line = re.sub(r'^["\']|["\']$', '', line)
        line = re.sub(r',\s*$', '', line)
        if line.startswith('I ') or (line.startswith('"I ') and line.endswith('"')):
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            facts.append(line)
    
    # If we still don't have enough, try JSON parsing
    if len(facts) < num_facts:
        try:
            json_match = re.search(r'\{.*?"facts".*?:.*?\[(.*?)\].*?\}', content, re.DOTALL)
            if json_match:
                facts = json.loads('[' + json_match.group(1) + ']')
        except:
            pass
    
    return facts[:num_facts]


def generate_complementary_fact(original_fact: str, model: str, api_key: str, temperature: float = 0.7) -> str:
    """
    Generate a complementary fact that doesn't contradict but complements the original fact.
    
    Args:
        original_fact: The original fact (e.g., "I like cheese pizza")
        model: OpenAI model to use
        api_key: OpenAI API key
        temperature: Temperature for generation
        
    Returns:
        A complementary fact (e.g., "I like burgers with cheese on it")
    """
    client = OpenAI(api_key=api_key)
    
    prompt = f"""Given the following fact about a person:
"{original_fact}"

Generate a NEW fact that:
1. Does NOT contradict the original fact
2. Complements or relates to the original fact in a meaningful way
3. Is also in first person ("I ...")
4. Is logically consistent with the original fact

Examples:
- Original: "I like cheese pizza" → Complementary: "I like burgers with cheese on it"
- Original: "I am allergic to peanuts" → Complementary: "I always check ingredient labels carefully"
- Original: "I have a dog named Max" → Complementary: "I enjoy taking long walks in the park"

The complementary fact should be interesting and specific, not just a restatement. Output ONLY the complementary fact, nothing else."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    
    fact = response.choices[0].message.content.strip()
    # Remove quotes if present
    if fact.startswith('"') and fact.endswith('"'):
        fact = fact[1:-1]
    elif fact.startswith("'") and fact.endswith("'"):
        fact = fact[1:-1]
    
    # Ensure it starts with "I"
    if not fact.startswith("I "):
        fact = "I " + fact
    
    return fact


def generate_question(original_fact: str, complementary_fact: str, model: str, api_key: str, temperature: float = 0.7) -> str:
    """
    Generate a question that CANNOT be answered without the original fact.
    The question must be personal and not answerable by general knowledge.
    
    Args:
        original_fact: The original fact that must be required to answer the question (e.g., "I am allergic to peanuts")
        complementary_fact: The complementary fact (e.g., "I always check ingredient labels carefully")
        model: OpenAI model to use
        api_key: OpenAI API key
        temperature: Temperature for generation
        
    Returns:
        A personal question that requires the original fact to answer
    """
    client = OpenAI(api_key=api_key)
    
    prompt = f"""Given the following two facts about a person:
ORIGINAL FACT: "{original_fact}"
COMPLEMENTARY FACT: "{complementary_fact}"

Generate a question that:
1. CANNOT be answered without knowing the ORIGINAL FACT - the question must be IMPOSSIBLE to answer correctly using only general knowledge or the complementary fact alone
2. Is highly PERSONAL and specific to this individual - it should be about their unique preferences, experiences, relationships, or circumstances
3. Is NOT answerable by general knowledge - avoid questions that anyone could answer without knowing this specific person
4. Requires the ORIGINAL FACT specifically - the answer must depend on that exact fact, not just the complementary fact
5. Is a practical, real-world question someone might actually ask about themselves
6. Does NOT simply reformat the original fact as a question (e.g., don't ask "Am I allergic to peanuts?" if the fact is "I am allergic to peanuts")


Examples of GOOD questions:
- Original: "I am allergic to peanuts" → Question: "Can I safely eat this specific brand of granola bar that my friend recommended?" (requires knowing about the allergy)
- Original: "I have a dog named Max" → Question: "Should I bring Max to the dog park this weekend?" (requires knowing about Max specifically)
- Original: "I prefer tea over coffee" → Question: "What should I order at this new cafe that only serves coffee?" (requires knowing the preference)

Examples of BAD questions (avoid these):
- "What is a common symptom of peanut allergies?" (answerable by general knowledge)
- "Do I like food?" (too generic, not personal)
- "Am I allergic to peanuts?" (just reformats the fact)

Output ONLY the question, nothing else."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    
    question = response.choices[0].message.content.strip()
    # Remove quotes if present
    if question.startswith('"') and question.endswith('"'):
        question = question[1:-1]
    elif question.startswith("'") and question.endswith("'"):
        question = question[1:-1]
    
    # Remove question mark if not present, but don't add one if it's not a question
    if not question.endswith('?'):
        # Check if it's actually a question (starts with question words)
        if any(question.strip().startswith(word) for word in ['Should', 'What', 'How', 'Why', 'When', 'Where', 'Who', 'Can', 'Will', 'Would']):
            question = question + "?"
    
    return question


def generate_ground_truth_answer(
    original_fact: str,
    complementary_fact: str,
    question: str,
    model: str,
    api_key: str,
    temperature: float = 0.7
) -> str:
    """
    Generate the ground truth answer to the question based on the facts.
    
    Args:
        original_fact: The original fact about the person
        complementary_fact: The complementary fact about the person
        question: The question to answer
        model: OpenAI model to use
        api_key: OpenAI API key
        temperature: Temperature for generation
        
    Returns:
        The ground truth answer to the question
    """
    client = OpenAI(api_key=api_key)
    
    prompt = f"""Given the following facts about a person:
1. "{original_fact}"
2. "{complementary_fact}"

And the following question:
"{question}"

Provide a clear, concise answer to the question based on these facts. The answer should:
1. Be directly based on the facts provided
2. Be accurate and logically follow from the facts
3. Be concise (1-2 sentences maximum)
4. Answer the question directly without unnecessary elaboration

Output ONLY the answer, nothing else."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    
    answer = response.choices[0].message.content.strip()
    # Remove quotes if present
    if answer.startswith('"') and answer.endswith('"'):
        answer = answer[1:-1]
    elif answer.startswith("'") and answer.endswith("'"):
        answer = answer[1:-1]
    
    return answer


def generate_dataset(
    num_facts: int,
    model: str,
    api_key: str,
    temperature: float = 0.7,
    output_path: Path = None,
) -> List[Tuple[str, str, str]]:
    """
    Generate the complete dataset.
    
    Args:
        num_facts: Number of facts to generate
        model: OpenAI model to use
        api_key: OpenAI API key
        temperature: Temperature for generation
        output_path: Path to save the CSV file
        
    Returns:
        List of tuples (original_fact, modified_fact, question, ground_truth_answer)
    """
    print(f"Generating persona with {num_facts} facts...")
    facts = generate_persona(num_facts, model, api_key, temperature)
    
    if len(facts) < num_facts:
        print(f"Warning: Only generated {len(facts)} facts, expected {num_facts}")
    
    print(f"Generated {len(facts)} facts. Now generating complementary facts and questions...")
    
    dataset = []
    for i, fact in enumerate(tqdm(facts, desc="Processing facts")):
        try:
            complementary_fact = generate_complementary_fact(fact, model, api_key, temperature)
            question = generate_question(fact, complementary_fact, model, api_key, temperature)
            ground_truth_answer = generate_ground_truth_answer(fact, complementary_fact, question, model, api_key, temperature)
            dataset.append((fact, complementary_fact, question, ground_truth_answer))
        except Exception as e:
            print(f"Error processing fact {i+1} ('{fact[:50]}...'): {e}")
            # Continue with empty values
            dataset.append((fact, "", "", ""))
    
    # Save to CSV
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["original_fact", "modified_fact", "question", "ground_truth_answer"])
            writer.writerows(dataset)
        print(f"\nSaved dataset to {output_path}")
    
    return dataset


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate persona dataset with facts, complementary facts, and questions"
    )
    parser.add_argument(
        "-n", "--num-facts",
        type=int,
        default=50,
        help="Number of facts to generate (default: 50)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output CSV file (default: datasets/persona_dataset.csv)"
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (default: from OPENAI_KEY or OPENAI_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_KEY or OPENAI_API_KEY environment variable, or use --api-key")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = script_dir / "persona_dataset.csv"
    
    # Generate dataset
    dataset = generate_dataset(
        num_facts=args.num_facts,
        model=args.model,
        api_key=api_key,
        temperature=args.temperature,
        output_path=output_path,
    )
    
    print(f"\nGenerated {len(dataset)} triples successfully!")


if __name__ == "__main__":
    main()
