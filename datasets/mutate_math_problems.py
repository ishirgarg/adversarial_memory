"""
mutate_math_problems.py — Generate mutated math problems from incorrect solutions.

Takes a CSV of math problems with incorrect solutions and generates mutated versions
where the same logical error would lead to the WRONG answer.

For each input row, generates:
- A mutated problem (similar but where the mistake fails)
- The correct solution to the mutated problem
- The solution that results from making the same mistake

Usage:
  uv run datasets/mutate_math_problems.py input.csv -o output.csv
  uv run datasets/mutate_math_problems.py input.csv --model gpt-4o-mini --limit 10
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Mutation prompts
# ---------------------------------------------------------------------------

MUTATION_SYSTEM = """\
You are an expert math problem designer. Your job is to create a mutated version \
of a math problem where a specific logical error would lead to the WRONG answer.

Given:
1. An original problem
2. An incorrect solution that uses a flawed logical approach
3. The location/description of the error

Your task:
1. Create a NEW problem that is similar in structure and topic to the original, \
but where making the SAME logical error would produce the WRONG answer (not the \
correct one).
2. The mutated problem should be clearly different from the original (different \
numbers, slightly different setup, etc.) but test the same mathematical concept.
3. Generate both:
   - The CORRECT solution to your mutated problem
   - The INCORRECT solution that results from making the same logical mistake

The key insight: In the original problem, the mistake happened to work out to \
the right answer by coincidence. In your mutated problem, the same mistake must \
lead to a clearly wrong answer.
"""

MUTATION_USER = """\
ORIGINAL PROBLEM:
{original_problem}

INCORRECT SOLUTION (with error):
{incorrect_solution}

ERROR DESCRIPTION:
{error_location}

GROUND TRUTH ANSWER (original problem): {ground_truth_answer}

Create a mutated problem where making the same logical error would lead to the \
WRONG answer. Generate:
1. The mutated problem (similar structure, different numbers/setup)
2. The correct solution to the mutated problem
3. The incorrect solution that results from making the same mistake
"""

MUTATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "mutated_problem",
        "schema": {
            "type": "object",
            "properties": {
                "mutated_problem": {
                    "type": "string",
                    "description": "The new mutated problem text"
                },
                "correct_solution": {
                    "type": "string",
                    "description": "The correct step-by-step solution to the mutated problem"
                },
                "mistaken_solution": {
                    "type": "string",
                    "description": "The solution that results from making the same logical error as in the original"
                },
            },
            "required": ["mutated_problem", "correct_solution", "mistaken_solution"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


# ---------------------------------------------------------------------------
# Mutation function
# ---------------------------------------------------------------------------

def generate_mutation(
    client: OpenAI,
    model: str,
    original_problem: str,
    incorrect_solution: str,
    error_location: str,
    ground_truth_answer: str,
) -> Dict[str, str]:
    """Generate a mutated problem and its solutions."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": MUTATION_SYSTEM},
            {"role": "user",   "content": MUTATION_USER.format(
                original_problem=original_problem,
                incorrect_solution=incorrect_solution,
                error_location=error_location,
                ground_truth_answer=ground_truth_answer,
            )},
        ],
        response_format=MUTATION_SCHEMA,
        temperature=0.7,
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate mutated math problems from incorrect solutions."
    )
    parser.add_argument(
        "input",
        help="Path to input CSV with incorrect solutions"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output CSV path (default: <input_stem>_mutated.csv)"
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="OpenAI model to use (default: gpt-5.2)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N rows (for testing)"
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

    # Determine output path
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_mutated.csv"

    # Read input CSV
    print(f"Reading input from {input_path}...")
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    print(f"Loaded {len(rows)} rows")
    
    if args.limit:
        rows = rows[: args.limit]
        print(f"Limiting to first {len(rows)} rows")

    # Expected columns
    required_cols = ["question", "ground_truth_answer", "incorrect_solution", "error_location"]
    if not all(col in rows[0] for col in required_cols):
        missing = [col for col in required_cols if col not in rows[0]]
        sys.exit(f"Error: Input CSV missing required columns: {missing}")

    # Process each row
    results = []
    print(f"\nGenerating mutations with {args.model}...")
    
    for i, row in enumerate(tqdm(rows, desc="Processing")):
        try:
            mutation = generate_mutation(
                client=client,
                model=args.model,
                original_problem=row["question"],
                incorrect_solution=row["incorrect_solution"],
                error_location=row["error_location"],
                ground_truth_answer=row["ground_truth_answer"],
            )
            
            # Combine original row with mutation results
            result_row = {
                "question": row["question"],
                "ground_truth_answer": row["ground_truth_answer"],
                "incorrect_solution": row["incorrect_solution"],
                "error_location": row["error_location"],
                "mutated_problem": mutation["mutated_problem"],
                "correct_solution": mutation["correct_solution"],
                "mistaken_solution": mutation["mistaken_solution"],
            }
            results.append(result_row)
            
        except Exception as e:
            print(f"\nError processing row {i+1}: {e}")
            # Continue with next row
            continue

    # Write output CSV
    print(f"\nWriting {len(results)} results to {output_path}...")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        if not results:
            print("No results to write.")
            return
        
        fieldnames = [
            "question",
            "ground_truth_answer",
            "incorrect_solution",
            "error_location",
            "mutated_problem",
            "correct_solution",
            "mistaken_solution",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Done! Saved {len(results)} mutated problems to {output_path}")


if __name__ == "__main__":
    main()
