"""
Script to generate a persona dataset with facts, short essays, questions, and answers.

Pipeline:
1) Generate a list of N first-person facts about a single person (logically consistent)
2) For each fact, generate a short information essay about the person (grounded in, and expanding on, that fact)
3) Generate a non-contrived, natural question that requires something from the essay to answer (no extra assumptions)
4) Generate the ground truth answer based on the essay (and original fact if needed)
5) Export CSV rows: (original_fact, essay, question, ground_truth_answer)
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
import time

# Load .env from parent directory
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
env_path = parent_dir / ".env"
load_dotenv(env_path)

MAX_RETRIES = 3
RETRY_BACKOFF_S = 0.5

def _chat_with_retry(client: OpenAI, *, model: str, messages, temperature: float = 0.7):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * attempt)
    raise RuntimeError(f"OpenAI chat completion failed after retries: {last_err}")


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

    response = _chat_with_retry(
        client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    
    content = response.choices[0].message.content.strip()
    
    # Try to extract the list from the response (with a small retry around parsing too)
    for _ in range(MAX_RETRIES):
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
                    if cleaned_facts:
                        return cleaned_facts[:num_facts]
            except (ValueError, SyntaxError):
                pass
        # If parsing failed, lightly nudge content (strip surrounding code fences or quotes) then retry parse
        content = content.strip().strip("`").strip()
    
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

def generate_essay_for_fact(original_fact: str, model: str, api_key: str, temperature: float = 0.7) -> str:
    """
    Generate a short information essay expanding on a single original fact.
    The essay should be specific, grounded in the fact, logically consistent, and natural.
    """
    client = OpenAI(api_key=api_key)
    prompt = f"""You are given a single first-person fact about a person:
FACT: "{original_fact}"

Write a detailed narrative essay (350-500 words) about this person that:
1) Expands on the fact with coherent, plausible, and highly specific details — include concrete names, places, dates, routines, preferences, and anecdotes that make the person feel real and distinct
2) Remains logically consistent with the fact; do not contradict it
3) Introduces 2-3 closely related secondary details (e.g., if the fact is about a pet, mention the pet's habits, the person's routine around it, and how others in their life interact with it)
4) Reads naturally and fluidly as a cohesive narrative — no lists, bullet points, or headers
5) Is self-contained and clear; a reader with no other context should come away with a vivid picture of this person

Output ONLY the essay text, nothing else."""
    response = _chat_with_retry(
        client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    essay = response.choices[0].message.content.strip()
    if essay.startswith('"') and essay.endswith('"'):
        essay = essay[1:-1]
    if not essay:
        # minimal retry by calling again once more for non-empty content
        response = _chat_with_retry(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        essay = response.choices[0].message.content.strip()
    return essay


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


def generate_question(original_fact: str, essay: str, model: str, api_key: str, temperature: float = 0.7) -> str:
    """
    Generate a personal, natural question that requires information from the essay to answer.
    """
    client = OpenAI(api_key=api_key)
    
    prompt = f"""You are given information about a person:
ORIGINAL FACT: "{original_fact}"
Output ONLY the question, nothing else."""


def generate_dataset(
    num_facts: int,
    model: str,
    api_key: str,
    temperature: float = 0.7,
    output_path: Path = None,
) -> List[Tuple[str, str]]:
    """
    Generate the complete dataset.
    
    Args:
        num_facts: Number of facts to generate
        model: OpenAI model to use
        api_key: OpenAI API key
        temperature: Temperature for generation
        output_path: Path to save the CSV file
        
    Returns:
        List of tuples (original_fact, essay)
    """
    print(f"Generating persona with {num_facts} facts...")
    facts = generate_persona(num_facts, model, api_key, temperature)
    
    if len(facts) < num_facts:
        print(f"Warning: Only generated {len(facts)} facts, expected {num_facts}")
    
    print(f"Generated {len(facts)} facts. Now generating essays...")
    
    dataset = []
    for i, fact in enumerate(tqdm(facts, desc="Processing facts")):
        try:
            essay = generate_essay_for_fact(fact, model, api_key, temperature)
            dataset.append((fact, essay))
        except Exception as e:
            print(f"Error processing fact {i+1} ('{fact[:50]}...'): {e}")
            # Continue with empty values
            dataset.append((fact, ""))
    
    # Save to CSV
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["original_fact", "essay"])
            writer.writerows(dataset)
        print(f"\nSaved dataset to {output_path}")
    
    return dataset


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate persona dataset with facts, essays, questions, and answers"
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
        default="gpt-5.4-mini",
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
        help="OpenAI API key (default: from OPENAI_API_KEY or OPENAI_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or OPENAI_API_KEY environment variable, or use --api-key")
    
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
    
    print(f"\nGenerated {len(dataset)} rows successfully!")


if __name__ == "__main__":
    main()
