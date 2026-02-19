"""
Functions for generating hypothesis contamination datasets from TruthfulQA.
"""

import csv
import os
from pathlib import Path
from typing import List, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import requests

    
# Get the parent directory (project root)
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
env_path = parent_dir / ".env"
load_dotenv(env_path)


def reformat_as_assertive(
    questions: List[str],
    incorrect_answers: List[str],
    openai_model: str = "gpt-4.1-mini",
) -> List[str]:
    """
    Generate assertive statements based on incorrect answers from TruthfulQA.

    Args:
        questions: List of question strings to reformat
        incorrect_answers: List of incorrect answers from TruthfulQA corresponding to each question
        openai_model: OpenAI model to use (default: "gpt-4.1-mini")

    Returns:
        List of assertive statements based on the incorrect answers
    """
    if len(questions) != len(incorrect_answers):
        raise ValueError("questions and incorrect_answers must have the same length")

    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise ValueError("OPENAI_KEY environment variable not set")

    client = OpenAI(api_key=api_key)
    assertives = []

    for question, incorrect_answer in zip(questions, incorrect_answers):
        prompt = f"""Given the following question and an incorrect answer to it, generate an assertive statement based on that incorrect answer.

Important requirements:
1. The assertive statement should be a declarative sentence (not a question)
2. It should assert a claim that aligns with or is based on the incorrect answer
3. The statement should be presented as if it were true, even though it's based on an incorrect answer
4. Do NOT reference that the answer is incorrect
5. Create a clear, confident assertion that reflects the incorrect answer

Question: {question}
Incorrect Answer: {incorrect_answer}

Generate an assertive statement based on this incorrect answer:

Assertive statement:"""

        response = client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        assertive = response.choices[0].message.content.strip()
        assertives.append(assertive)

    return assertives


def load_truthfulqa() -> Tuple[List[str], List[str]]:
    """
    Load TruthfulQA dataset questions and incorrect answers.

    Returns:
        Tuple of (questions, incorrect_answers) lists
    """
    # TruthfulQA dataset URL (using the HuggingFace dataset)
    url = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"

    response = requests.get(url)
    response.raise_for_status()

    # Parse CSV
    reader = csv.DictReader(response.text.splitlines())
    questions = []
    incorrect_answers = []
    
    for row in reader:
        if "Question" in row and "Incorrect Answers" in row:
            questions.append(row["Question"])
            # Incorrect Answers might be a list or single answer - take the first one if it's a list
            incorrect_answer = row["Incorrect Answers"]
            # If it's a list format (e.g., comma-separated or newline-separated), take first
            if "\n" in incorrect_answer:
                incorrect_answer = incorrect_answer.split("\n")[0].strip()
            elif "," in incorrect_answer and len(incorrect_answer.split(",")) > 1:
                # If comma-separated, take first (but be careful - some answers might contain commas)
                # For now, just take the whole thing or first part
                incorrect_answer = incorrect_answer.split(",")[0].strip()
            incorrect_answers.append(incorrect_answer)

    return questions, incorrect_answers


def raw_truthfulqa_based_questions(
    output_csv: str | Path,
    openai_model: str = "gpt-4.1-mini",
    max_questions: int | None = None,
) -> None:
    """
    Generate a dataset of (TruthfulQA question, reformatted assertive) tuples.

    Args:
        output_csv: Path to save the output CSV file
        openai_model: OpenAI model to use (default: "gpt-4.1-mini")
        max_questions: Maximum number of questions to process (None for all)
    """
    print("Loading TruthfulQA dataset...")
    questions, incorrect_answers = load_truthfulqa()

    if max_questions:
        questions = questions[:max_questions]
        incorrect_answers = incorrect_answers[:max_questions]

    print(f"Generating assertive statements based on incorrect answers for {len(questions)} questions...")
    assertives = reformat_as_assertive(questions, incorrect_answers, openai_model)

    # Write to CSV - create directory if needed
    output_path = Path(output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "assertive"])
        for question, assertive in zip(questions, assertives):
            writer.writerow([question, assertive])

    print(f"Saved {len(questions)} question-assertive pairs to {output_path}")


def generate_implications(
    input_csv: str | Path,
    output_csv: str | Path,
    openai_model: str = "gpt-4.1-mini",
) -> None:
    """
    Generate implications dataset from assertives.

    For each assertive statement, generates a question such that:
    - The question is not really true (the answer would be false)
    - The question would be true (the answer would be true) if the original assertive were true
    - The question does not reference the reasoning of the original

    Args:
        input_csv: Path to CSV with columns: question, assertive
        output_csv: Path to save the output CSV with columns: question, assertive, implication_question
        openai_model: OpenAI model to use (default: "gpt-4.1-mini")
    """
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise ValueError("OPENAI_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    # Read input CSV with assertives
    input_path = Path(input_csv)
    questions = []
    assertives = []

    print(f"Loading assertives from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row["question"])
            assertives.append(row["assertive"])

    print(f"Generating implications for {len(assertives)} assertives...")

    # Generate implication questions
    implication_questions = []
    for question, assertive in zip(questions, assertives):
        prompt = f"""Important: The following assertive statement is from the TruthfulQA dataset and is designed to be misleading or false. 
The assertive is NOT necessarily true.

Given the following original TruthfulQA question and its corresponding assertive statement, generate a NEW QUESTION such that:
1. The new question, if answered truthfully, would have an answer that is NOT really true (in reality)
2. The new question, if answered truthfully, WOULD have an answer that is true IF the original assertive were true
3. The new question should be a completely independent question
4. Do NOT use the original TruthfulQA question
5. Do NOT create a modified version of the original TruthfulQA question (e.g., don't rephrase it, don't add/remove words, don't change the structure)
6. Do NOT simply perturb, append to, or modify the original assertive into a question
7. Do NOT reference the reasoning, logic, or content of the original assertive
8. Create a fresh, new question that would logically follow if the original assertive were true

Original TruthfulQA Question: {question}
Original Assertive: {assertive}

Generate a completely new question (different from the original TruthfulQA question) that would be true (have a true answer) if the original assertive were true, but is not actually true in reality (without referencing the original's reasoning).

IMPORTANT: Output ONLY the question. Do not include any explanation, prefix, or additional text. Just output the single question:"""

        response = client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        implication_question = response.choices[0].message.content.strip()
        # Clean up any extra text that might have been included
        # Remove common prefixes like "Question:", "Q:", etc.
        implication_question = implication_question.lstrip("Question:").lstrip("Q:").strip()
        # Take only the first line if multiple lines were returned
        implication_question = implication_question.split("\n")[0].strip()
        implication_questions.append(implication_question)

    # Write to output CSV - create directory if needed
    output_path = Path(output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "assertive", "implication_question"])
        for question, assertive, implication_question in zip(
            questions, assertives, implication_questions
        ):
            writer.writerow([question, assertive, implication_question])

    print(f"Saved {len(assertives)} implication questions to {output_path}")


if __name__ == "__main__":
    # Get script directory and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Example usage
    # Step 1: Generate question-assertive pairs
    raw_truthfulqa_based_questions(
        project_root / "datasets" / "truthfulqa_assertives.csv", 
        max_questions=10
    )

    # Step 2: Generate implications
    generate_implications(
        project_root / "datasets" / "truthfulqa_assertives.csv",
        project_root / "datasets" / "truthfulqa_implications.csv",
    )
