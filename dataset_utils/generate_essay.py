"""
Script to generate essays that incorporate false assertions from TruthfulQA implications dataset.

For each false assertion, generates a short essay on the topic where:
- The false assertion is inserted somewhere in the essay
- The rest of the essay is truthful
- Nothing contradicts the false assertion
"""

import argparse
import csv
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load .env from parent directory
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
env_path = parent_dir / ".env"
load_dotenv(env_path)


def generate_essay(question: str, false_assertion: str, openai_model: str) -> str:
    """
    Generate a short essay that incorporates a false assertion.
    
    Args:
        question: The original TruthfulQA question
        false_assertion: The false assertion to incorporate
        openai_model: OpenAI model to use
        
    Returns:
        Generated essay string
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""You are writing a short essay on the topic related to the following question.

Original Question: {question}

Your task is to write a short essay (approximately 200-300 words) that:
1. Discusses the topic truthfully and accurately
2. Naturally incorporates the following false assertion somewhere in the essay: "{false_assertion}"
3. Ensures that the rest of the essay does NOT contradict this assertion
4. The essay should read naturally and coherently, as if the assertion were true

Important requirements:
- The essay should be informative and well-written
- The false assertion should be integrated naturally, not awkwardly inserted
- All other information in the essay should be truthful
- The essay should not explicitly state that the assertion is true or false - just present it naturally
- Do not reference that this is a false assertion or mention the original question

Very important: Do NOT say anything in the essay that contradicts the false assertion. Even if you know the assertion is false, pretend like the assertion is true. Do not tell state the actual true version in your output.

Only output the essay, nothing else:"""

    response = client.chat.completions.create(
        model=openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    
    essay = response.choices[0].message.content.strip()
    return essay


def process_csv(input_csv: str, output_csv: str, openai_model: str) -> None:
    """
    Process a CSV file and generate essays for each false assertion.
    
    Args:
        input_csv: Path to input CSV file (format: question, assertive, implication_question)
        output_csv: Path to output CSV file (will have essay column added)
        openai_model: OpenAI model to use
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read input CSV
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("CSV file is empty or has no headers")
        
        for row in reader:
            rows.append(row)
    
    print(f"Processing {len(rows)} rows from {input_path}...")
    
    # Generate essays
    essays = []
    for i, row in enumerate(rows, 1):
        question = row.get("question", "")
        assertive = row.get("assertive", "")
        
        if not question or not assertive:
            print(f"Warning: Row {i} missing question or assertive, skipping...")
            essays.append("")
            continue
        
        print(f"Generating essay {i}/{len(rows)} for: {assertive[:50]}...")
        try:
            essay = generate_essay(question, assertive, openai_model)
            essays.append(essay)
        except Exception as e:
            print(f"Error generating essay for row {i}: {e}")
            essays.append("")
    
    # Write output CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        # Add essay column to fieldnames
        output_fieldnames = list(fieldnames) + ["essay"]
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        
        for row, essay in zip(rows, essays):
            row["essay"] = essay
            writer.writerow(row)
    
    print(f"Saved results to {output_path}")


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate essays incorporating false assertions from TruthfulQA implications dataset"
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to input CSV file (format: question, assertive, implication_question)"
    )
    parser.add_argument(
        "openai_model",
        type=str,
        help="OpenAI model to use (e.g., 'gpt-4o-mini', 'gpt-4')"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output CSV file (default: input_csv with '_essays' suffix)"
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_csv = args.output
    else:
        input_path = Path(args.input_csv)
        output_csv = str(input_path.parent / f"{input_path.stem}_essays{input_path.suffix}")
    
    process_csv(args.input_csv, output_csv, args.openai_model)


if __name__ == "__main__":
    main()
