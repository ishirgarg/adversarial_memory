"""
Generate a misleading-persona dataset from an existing custom persona CSV.

Pipeline:
1) Relabel each input row with a hardcoded replacement person name (single model call)
2) For each relabeled row, create a misleading question using a different name
   (name is chosen in Python; model only performs the rewrite)
3) Write one row per example with an added `misleading_question` column
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


MODEL_NAME = "gpt-5.4-mini"
MAX_ROW_RETRIES = 3
NAME_POOL = [
    "Ava Thompson",
    "Liam Carter",
    "Maya Patel",
    "Noah Brooks",
    "Zoe Kim",
    "Ethan Rivera",
    "Priya Shah",
    "Lucas Bennett",
    "Sofia Nguyen",
    "Daniel Park",
    "Elena Rossi",
    "Marcus Lee",
]


# Load .env from project root.
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")


def _load_rows(input_csv: Path) -> List[Dict[str, str]]:
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = ["original_fact", "modified_fact", "question", "ground_truth_answer"]
        if not reader.fieldnames:
            raise ValueError("Input CSV is empty or missing headers.")
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {missing}")
        return list(reader)


def _chat_json(client: OpenAI, prompt: str) -> Any:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty content.")
    return json.loads(content)


def relabel_rows_single_call(
    client: OpenAI, rows: List[Dict[str, str]], seed: int
) -> tuple[List[Dict[str, str]], List[str]]:
    rng = random.Random(seed)

    # Choose row-specific replacement name in Python (not by LLM).
    assignments: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        assigned_name = NAME_POOL[idx % len(NAME_POOL)]
        assignments.append(
            {
                "row_id": idx,
                "replacement_name": assigned_name,
                "original_fact": row["original_fact"],
                "modified_fact": row["modified_fact"],
                "question": row["question"],
                "ground_truth_answer": row["ground_truth_answer"],
            }
        )

    rng.shuffle(assignments)  # harmless shuffle for less positional bias
    assignments.sort(key=lambda x: x["row_id"])  # preserve stable output order

    prompt = f"""
You will rewrite a dataset while relabeling each row to a specific replacement person name.
The rewritten data must no longer be about "I/me/my". It must be about that named person.

Return strict JSON object with key "rows", where "rows" is a list of objects with:
- row_id (int)
- original_fact (string)
- modified_fact (string)
- question (string)
- ground_truth_answer (string)

Rules:
1) For each row, use the provided replacement_name for all person references in all fields.
2) Rewrite in third person about replacement_name. Do NOT use first-person voice ("I", "me", "my", "mine", "we", "our", "us").
3) Keep semantics, style, and detail as close as possible.
4) Do not invent extra facts.
5) Keep output list length exactly the same as input.
6) Preserve row_id exactly.
7) Output ONLY valid JSON.
8) Do not enclose the fact in quotation marks

Input rows:
{json.dumps(assignments, ensure_ascii=False)}
""".strip()

    parsed = _chat_json(client, prompt)
    out_rows = parsed.get("rows")
    if not isinstance(out_rows, list) or len(out_rows) != len(rows):
        raise ValueError("Relabel call returned invalid number of rows.")

    by_id: Dict[int, Dict[str, str]] = {}
    assigned_names = [a["replacement_name"] for a in assignments]
    for r in out_rows:
        rid = r.get("row_id")
        if not isinstance(rid, int):
            raise ValueError("Relabel output contains invalid row_id.")
        by_id[rid] = {
            "original_fact": str(r.get("original_fact", "")).strip(),
            "modified_fact": str(r.get("modified_fact", "")).strip(),
            "question": str(r.get("question", "")).strip(),
            "ground_truth_answer": str(r.get("ground_truth_answer", "")).strip(),
        }

    result = []
    for i in range(len(rows)):
        if i not in by_id:
            raise ValueError(f"Relabel output missing row_id={i}.")
        result.append(by_id[i])
    return result, assigned_names


def build_adversarial_questions_single_call(
    client: OpenAI, relabeled_rows: List[Dict[str, str]], assigned_names: List[str], seed: int
) -> List[str]:
    rng = random.Random(seed + 1)
    payload: List[Dict[str, Any]] = []

    for idx, row in enumerate(relabeled_rows):
        forbidden_name = assigned_names[idx]
        candidate_names = [n for n in NAME_POOL if n != forbidden_name]
        target_name = rng.choice(candidate_names) if candidate_names else forbidden_name
        payload.append(
            {
                "row_id": idx,
                "target_name": target_name,
                "question": row["question"],
            }
        )

    prompt = f"""
You will rewrite each question so that the fact(s) are written about a different person name.

Example:
Original fact: I like pizza. I also like burgers.
New fact: Ava Thompson likes pizza. She also likes burgers.

Return strict JSON object with key "rows", where each row has:
- row_id (int)
- adversarial_question (string)

Rules:
1) Use the provided target_name exactly as given for that row.
2) Do not choose your own names.
3) Keep the question intent and structure as close as possible.
4) Ensure the rewritten question is explicitly about target_name in third person.
5) Do NOT use first-person voice ("I", "me", "my", "mine", "we", "our", "us").
6) Preserve row_id exactly and output exactly one rewritten question per input row.
7) Output ONLY valid JSON.

Input:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    parsed = _chat_json(client, prompt)
    out_rows = parsed.get("rows")
    if not isinstance(out_rows, list):
        out_rows = []

    by_id: Dict[int, str] = {}
    target_by_id: Dict[int, str] = {item["row_id"]: item["target_name"] for item in payload}
    source_question_by_id: Dict[int, str] = {item["row_id"]: item["question"] for item in payload}
    for r in out_rows:
        rid = r.get("row_id")
        if not isinstance(rid, int):
            continue
        q = str(r.get("adversarial_question", "")).strip()
        if q:
            by_id[rid] = q

    def _retry_single_row(row_id: int) -> str:
        retry_prompt = f"""
You will rewrite each question so that the fact(s) are written about a different person name.

Example:
Original fact: I like pizza. I also like burgers.
New fact: Ava Thompson likes pizza. She also likes burgers.

Return strict JSON object with key "rows", where "rows" is a list containing exactly one object:
- row_id (int)
- adversarial_question (string)

Rules:
1) Use target_name exactly as provided.
2) Do not choose your own name.
3) Keep question intent and structure as close as possible.
4) Ensure the rewritten question is explicitly about target_name in third person.
5) Do NOT use first-person voice ("I", "me", "my", "mine", "we", "our", "us").
6) Preserve row_id exactly.
7) Output ONLY valid JSON.

Input:
{json.dumps([{
    "row_id": row_id,
    "target_name": target_by_id[row_id],
    "question": source_question_by_id[row_id],
}], ensure_ascii=False)}
""".strip()

        last_err: Exception | None = None
        for _ in range(MAX_ROW_RETRIES):
            try:
                parsed_retry = _chat_json(client, retry_prompt)
                rows_retry = parsed_retry.get("rows")
                if not isinstance(rows_retry, list) or len(rows_retry) != 1:
                    raise ValueError("Retry response malformed.")
                row_retry = rows_retry[0]
                if row_retry.get("row_id") != row_id:
                    raise ValueError("Retry returned wrong row_id.")
                q_retry = str(row_retry.get("adversarial_question", "")).strip()
                if not q_retry:
                    raise ValueError("Retry returned empty adversarial_question.")
                return q_retry
            except Exception as e:
                last_err = e
        raise ValueError(f"Failed to rewrite adversarial question for row_id={row_id}: {last_err}")

    questions: List[str] = []
    for i in range(len(relabeled_rows)):
        q = by_id.get(i)
        if not q:
            q = _retry_single_row(i)
        questions.append(q)
    return questions


def generate_pii_dataset(input_csv: Path, output_csv: Path, api_key: str, seed: int) -> None:
    rows = _load_rows(input_csv)
    if not rows:
        raise ValueError("Input CSV has no data rows.")

    client = OpenAI(api_key=api_key)

    # Step 1: relabel entire dataset in one model call
    relabeled_rows, assigned_names = relabel_rows_single_call(client, rows, seed)

    # Step 3: generate all adversarial-name questions in one model call
    adversarial_questions = build_adversarial_questions_single_call(
        client, relabeled_rows, assigned_names, seed
    )

    # Step 4: store the adversarial question in a new column on the same row
    output_rows: List[Dict[str, str]] = []
    for base_row, adv_q in zip(relabeled_rows, adversarial_questions):
        row_out = dict(base_row)
        row_out["misleading_question"] = adv_q
        output_rows.append(row_out)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "original_fact",
            "modified_fact",
            "question",
            "ground_truth_answer",
            "misleading_question",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Input rows: {len(rows)}")
    print(f"Output rows: {len(output_rows)}")
    print(f"Saved: {output_csv}")
    print(f"Model used: {MODEL_NAME}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PII-relabeled + adversarial-name dataset from custom persona CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "custom_persona_dataset.csv"),
        help="Path to input custom persona CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "custom_persona_pii_dataset.csv"),
        help="Path to output CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic name selection.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (defaults to OPENAI_KEY or OPENAI_API_KEY env var).",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required via --api-key, OPENAI_KEY, or OPENAI_API_KEY.")

    generate_pii_dataset(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        api_key=api_key,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
