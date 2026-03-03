"""
generate_incorrect_math_solutions.py — Generate incorrect solutions for MATH dataset.

For each MATH problem, generates a solution that:
- Reaches the correct answer
- Contains a mathematically incorrect step
- Is written as if it's correct

Outputs a CSV with: question, ground_truth_answer, incorrect_solution

Usage:
  uv run datasets/generate_incorrect_math_solutions.py -o math_incorrect.csv
  uv run datasets/generate_incorrect_math_solutions.py --limit 10 --model gpt-4o-mini
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Try to import datasets library, fallback to manual download
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' library not found. Install with: pip install datasets")
    print("Falling back to manual download via requests...")

load_dotenv()

# ---------------------------------------------------------------------------
# MATH dataset loading
# ---------------------------------------------------------------------------

def load_math_dataset(subset: str = "algebra", split: str = "test") -> List[Dict]:
    """Load MATH dataset from HuggingFace or manual download.
    
    Note: The dataset is at EleutherAI/hendrycks_math on HuggingFace.
    Available subsets: algebra, counting_and_probability, geometry,
    intermediate_algebra, number_theory, prealgebra, precalculus
    
    Args:
        subset: The math topic subset to load (default: "algebra")
        split: The dataset split to load (default: "test")
    """
    if HAS_DATASETS:
        print(f"Loading MATH dataset (subset: {subset}, split: {split}) from HuggingFace...")
        try:
            dataset = load_dataset("EleutherAI/hendrycks_math", subset)
            # Get the specific split
            if split in dataset:
                return [item for item in dataset[split]]
            else:
                raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            print("You may need to download manually from the GitHub repository:")
            print("https://github.com/hendrycks/math")
            raise
    else:
        # Fallback: manual download (may not work with subsets)
        import requests
        print(f"Downloading MATH dataset (subset: {subset}, split: {split}) manually...")
        print("Warning: Manual download may not support subsets. Using default path.")
        url = f"https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/main/{split}.jsonl"
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to download from HuggingFace: {e}")
            print("You may need to download manually from the GitHub repository:")
            print("https://github.com/hendrycks/math")
            raise
        
        items = []
        for line in response.iter_lines():
            if line:
                items.append(json.loads(line))
        return items


# ---------------------------------------------------------------------------
# Solution generation
# ---------------------------------------------------------------------------

INCORRECT_SOLUTION_SYSTEM = """\
You are a math tutor. Your job is to write a solution to a math problem that \
reaches the correct answer, but contains ONE significant logical or conceptual \
error that fundamentally changes the reasoning approach.

CRITICAL REQUIREMENTS:
1. The solution must reach the CORRECT final answer (same as the ground truth).
2. The solution must contain exactly ONE significant logical/conceptual error \
that changes the reasoning in a meaningful way. This should NOT be:
   - A minor arithmetic mistake (like 2+2=5)
   - A simple typo or transcription error
   - A rounding error
   Instead, it should be:
   - A flawed logical approach or method
   - Misapplication of a theorem, formula, or principle
   - Incorrect assumption about the problem structure
   - Wrong interpretation of what the problem is asking
   - A conceptual misunderstanding that happens to work for this specific case
3. The error must be significant enough that if you applied the same flawed \
logic to a different but similar problem, it would produce the WRONG answer. \
The error should represent a systematic flaw in reasoning, not a one-off mistake.
4. Write the solution as if it's completely correct — don't acknowledge the error.
5. The solution should be step-by-step and well-formatted, showing the flawed \
reasoning clearly.

Return a JSON object with:
  "solution": the full incorrect solution text
  "error_location": a brief description of the significant logical/conceptual \
error (e.g., "Step 3: incorrectly assumed the problem requires finding the \
maximum when it actually requires finding the minimum, but by coincidence \
the calculation still yields the correct answer")
"""

INCORRECT_SOLUTION_USER = """\
PROBLEM:
{problem}

GROUND TRUTH ANSWER: {answer}

Generate an incorrect solution that reaches the correct answer but contains \
one SIGNIFICANT logical or conceptual error. The error should change the \
reasoning approach in a meaningful way, such that applying the same flawed \
logic to another problem would lead to wrong answers.\
"""

SOLUTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "incorrect_solution",
        "schema": {
            "type": "object",
            "properties": {
                "solution":       {"type": "string"},
                "error_location": {"type": "string"},
            },
            "required": ["solution", "error_location"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def generate_incorrect_solution(
    client: OpenAI,
    model: str,
    problem: str,
    ground_truth_answer: str,
) -> Dict[str, str]:
    """Generate an incorrect solution for a MATH problem."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": INCORRECT_SOLUTION_SYSTEM},
            {"role": "user",   "content": INCORRECT_SOLUTION_USER.format(
                problem=problem,
                answer=ground_truth_answer,
            )},
        ],
        response_format=SOLUTION_SCHEMA,
        temperature=0.7,  # Some randomness for variety
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate incorrect solutions for MATH dataset problems."
    )
    parser.add_argument(
        "-o", "--output",
        default="math_incorrect_solutions.csv",
        help="Output CSV path (default: math_incorrect_solutions.csv)"
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--subset",
        default="algebra",
        choices=["algebra", "counting_and_probability", "geometry",
                 "intermediate_algebra", "number_theory", "prealgebra", "precalculus"],
        help="MATH dataset subset/topic to use (default: algebra)"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="MATH dataset split to use (default: test)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N problems (for testing)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential)"
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: set OPENAI_KEY or OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)

    # Load MATH dataset
    print("Loading MATH dataset...")
    math_data = load_math_dataset(subset=args.subset, split=args.split)
    print(f"Loaded {len(math_data)} problems from {args.subset} subset")

    if args.limit:
        math_data = math_data[: args.limit]
        print(f"Limiting to first {len(math_data)} problems")

    # Process each problem
    output_path = Path(args.output)
    results = []

    print(f"\nGenerating incorrect solutions with {args.model}...")
    for item in tqdm(math_data, desc="Processing"):
        problem = item.get("problem", "")
        # MATH dataset has "solution" field with the ground truth solution
        # We need to extract the final answer from it
        solution_text = item.get("solution", "")
        
        # Try to extract answer from solution (usually at the end, boxed)
        # Common patterns: \boxed{...}, Answer: ..., Final answer: ...
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution_text)
        if boxed_match:
            ground_truth = boxed_match.group(1).strip()
        else:
            # Fallback: try to find "Answer:" or similar
            answer_match = re.search(r'(?:Answer|answer|Final answer)[:\s]+([^\n]+)', solution_text, re.IGNORECASE)
            if answer_match:
                ground_truth = answer_match.group(1).strip()
            else:
                # Last resort: use the last line or a reasonable default
                lines = [l.strip() for l in solution_text.split('\n') if l.strip()]
                ground_truth = lines[-1] if lines else "Unknown"

        try:
            generated = generate_incorrect_solution(
                client, args.model, problem, ground_truth
            )
            results.append({
                "question": problem,
                "ground_truth_answer": ground_truth,
                "incorrect_solution": generated["solution"],
                "error_location": generated["error_location"],
            })
        except Exception as e:
            print(f"\nError processing problem: {e}")
            # Continue with next problem
            continue

    # Write CSV
    print(f"\nWriting {len(results)} results to {output_path}...")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        if not results:
            print("No results to write.")
            return
        
        writer = csv.DictWriter(f, fieldnames=["question", "ground_truth_answer", "incorrect_solution", "error_location"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Done! Saved {len(results)} incorrect solutions to {output_path}")


if __name__ == "__main__":
    main()
