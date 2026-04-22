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
        required = ["original_fact", "essay"]
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
                "essay": row["essay"],
                "question": row.get("question", ""),
                "ground_truth_answer": row.get("ground_truth_answer", ""),
            }
        )

    rng.shuffle(assignments)  # harmless shuffle for less positional bias
    assignments.sort(key=lambda x: x["row_id"])  # preserve stable output order

    prompt = f"""
You will rewrite a dataset while relabeling each row to a specific replacement person name.
The rewritten data must no longer be about "I/me/my". It must be about that named person.


Examples:
Original fact: I like pizza. I also like burgers.
New fact: Ava Thompson likes pizza. She also likes burgers.
Original question: Should I go to Dominoes or Pizza Hut?
New questino: Should Ava Thompson go to Dominoes or Pizza Hut?

Return strict JSON object with key "rows", where "rows" is a list of objects with:
- row_id (int)
- original_fact (string)
- essay (string)
- question (string)
- ground_truth_answer (string)

Rules:
1) For each row, use the provided replacement_name for all person references in all fields (including original_fact, essay, and question). The sentences should sound natural; use pronouns when appropriate to ensure grammatical correctness.
2) Do NOT use first-person voice ("I", "me", "my", "mine", "we", "our", "us").
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
            "essay": str(r.get("essay", "")).strip(),
            "question": str(r.get("question", "")).strip(),
            "ground_truth_answer": str(r.get("ground_truth_answer", "")).strip(),
        }

    result = []
    for i in range(len(rows)):
        if i not in by_id:
            raise ValueError(f"Relabel output missing row_id={i}.")
        result.append(by_id[i])
    return result, assigned_names


def _adversarial_prompt(rows_payload: List[Dict[str, Any]]) -> str:
    return f"""
You will rewrite each question so that it asks about a different person name, while preserving FIRST-PERSON voice and the rest of the wording.

Examples:
Original question: What should I do with Jane?
New question: What should I do with Ava Thompson?

Return strict JSON object with key "rows", where each object has:
- row_id (int)
- adversarial_question (string)

Rules:
1) Use the provided target_name exactly as given for that row.
2) Do not choose your own names.
3) Preserve FIRST-PERSON voice (e.g., keep "I", "me", "my") and all other wording wherever possible.
4) Replace only the PERSON being referred to inside the question with target_name (e.g., replace "Jane" with target_name).
5) If there is no explicit person name in the original question, minimally adapt it to include target_name naturally, without adding extra assumptions.
6) Preserve row_id exactly and output exactly one rewritten question per input row.
7) Output ONLY valid JSON.

Input:
{json.dumps(rows_payload, ensure_ascii=False)}
""".strip()


def _build_one_adversarial_pass(
    client: OpenAI,
    relabeled_rows: List[Dict[str, str]],
    assigned_names: List[str],
    rng: random.Random,
    excluded_names_per_row: List[set],
) -> List[str]:
    """Generate one adversarial question per row, picking names not in each row's excluded set."""
    payload: List[Dict[str, Any]] = []
    for idx, row in enumerate(relabeled_rows):
        forbidden = excluded_names_per_row[idx]
        candidates = [n for n in NAME_POOL if n not in forbidden]
        # Fall back to any name other than the assigned one if pool is exhausted
        if not candidates:
            candidates = [n for n in NAME_POOL if n != assigned_names[idx]]
        target_name = rng.choice(candidates) if candidates else assigned_names[idx]
        payload.append(
            {
                "row_id": idx,
                "target_name": target_name,
                "question": row["question"],
            }
        )

    prompt = _adversarial_prompt(payload)
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
        single_payload = [{
            "row_id": row_id,
            "target_name": target_by_id[row_id],
            "question": source_question_by_id[row_id],
        }]
        retry_prompt = _adversarial_prompt(single_payload)
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
        # Record the name used so subsequent passes avoid it
        excluded_names_per_row[i].add(target_by_id[i])
        questions.append(q)
    return questions


def build_adversarial_questions_multi(
    client: OpenAI,
    relabeled_rows: List[Dict[str, str]],
    assigned_names: List[str],
    seed: int,
    num_adversarial: int = 1,
) -> List[List[str]]:
    """
    Generate `num_adversarial` adversarial questions per row.

    Each adversarial question uses a different randomly chosen name (no repeats
    within the same row across passes). Returns a list of length len(relabeled_rows),
    where each element is a list of `num_adversarial` question strings.
    """
    rng = random.Random(seed + 1)
    # Start with the assigned name excluded so we never ask about the correct person
    excluded_per_row: List[set] = [{name} for name in assigned_names]

    all_passes: List[List[str]] = []
    for _ in range(num_adversarial):
        pass_questions = _build_one_adversarial_pass(
            client, relabeled_rows, assigned_names, rng, excluded_per_row
        )
        all_passes.append(pass_questions)

    # Transpose: from [pass][row] -> [row][pass]
    return [
        [all_passes[p][row_idx] for p in range(num_adversarial)]
        for row_idx in range(len(relabeled_rows))
    ]


def _clean_question(q: str) -> str:
    q = q.strip()
    if q.startswith('"') and q.endswith('"'):
        q = q[1:-1]
    if not q.endswith("?"):
        if any(q.startswith(w) for w in ["Should", "What", "How", "Why", "When", "Where", "Who", "Can", "Will", "Would"]):
            q = q + "?"
    return q


def generate_multiple_questions_from_essays(
    client: OpenAI, rows: List[Dict[str, str]], num_questions: int
) -> List[List[str]]:
    """
    Generate `num_questions` distinct, natural questions per row.

    Each question targets a different aspect of the essay so that the N positive
    examples are meaningfully varied before being converted to adversarial negatives.
    Returns a list of length len(rows), where each element is a list of num_questions strings.
    """
    all_questions: List[List[str]] = []
    for row in rows:
        essay = row["essay"]
        prompt = f"""You are given information about a person:
ESSAY:
\"\"\"{essay}\"\"\"

Generate exactly {num_questions} DISTINCT questions about this person, each targeting a different aspect of the essay.

Core rule -- NO embedded assumptions:
The question must be one that ANY person in the world could plausibly ask about someone they know, WITHOUT already knowing the details in the essay.
Do NOT smuggle facts from the essay into the question itself.

Tests to apply before accepting a question:
- Could someone who has never read the essay still naturally ask this question? If yes, it is acceptable.
- Does the question presuppose that a specific event occurred, or that the person has a specific trait/possession/location? If yes, reject it.

BAD (embed facts as assumptions):
- "What did the vendor say about the blue mug when Mara bought it?" -- assumes Mara bought a mug from a vendor
- "What kind of work does Mara do downtown near the Pearl District?" -- assumes she works near the Pearl District
- "What apple dessert can I give Rob that won't make him itch?" -- assumes the asker knows about the allergy
- "If Pesto slips out when I open the door, what sound should I make?" -- assumes a specific scenario

GOOD (open questions answerable from the essay):
- "What should I get Mara for her birthday?" -- open; essay details let you answer it specifically
- "What dessert should I make for Aunt Sally?" -- open; essay reveals the allergy that answers it
- "Should I suggest somewhere quiet for lunch with Marcus?" -- open; essay reveals why quiet matters
- "What kind of work does Jordan do?" -- open; does not embed the answer as an assumption

Additional requirements:
1) Each question must be answerable using specific details from the essay (not general knowledge alone)
2) All questions MUST be in first person ("What should I get...", "Should I invite...", "Is it okay if I...")
3) Questions must be DISTINCT from each other -- each targets a different aspect of the essay
4) Do NOT restate or reformat the original fact directly as a question

Return a JSON object with key "questions" containing a list of exactly {num_questions} question strings.
Output ONLY valid JSON."""
        parsed = _chat_json(client, prompt)
        raw_questions = parsed.get("questions", [])
        if not isinstance(raw_questions, list):
            raw_questions = []
        cleaned = [_clean_question(q) for q in raw_questions if isinstance(q, str) and q.strip()]

        # Retry once if we didn't get enough
        if len(cleaned) < num_questions:
            parsed2 = _chat_json(client, prompt)
            raw2 = parsed2.get("questions", [])
            if isinstance(raw2, list):
                cleaned = [_clean_question(q) for q in raw2 if isinstance(q, str) and q.strip()]

        # Pad with copies of the first question as a last resort
        while len(cleaned) < num_questions:
            cleaned.append(cleaned[0] if cleaned else "")

        all_questions.append(cleaned[:num_questions])
    return all_questions


def generate_pii_dataset(
    input_csv: Path, output_csv: Path, api_key: str, seed: int, num_adversarial: int = 1
) -> None:
    rows = _load_rows(input_csv)
    if not rows:
        raise ValueError("Input CSV has no data rows.")

    client = OpenAI(api_key=api_key)

    # Step 1: relabel entire dataset in one model call (fact + essay to named third-person)
    relabeled_rows, assigned_names = relabel_rows_single_call(client, rows, seed)

    # Step 2: generate num_adversarial DISTINCT questions per relabeled essay
    questions_per_row = generate_multiple_questions_from_essays(
        client, relabeled_rows, num_questions=num_adversarial
    )

    # Step 3: expand -- one row per (persona, question) pair, all sharing the same assigned name
    expanded_rows: List[Dict[str, str]] = []
    expanded_assigned_names: List[str] = []
    for base_row, assigned_name, questions in zip(relabeled_rows, assigned_names, questions_per_row):
        for q in questions:
            row_copy = dict(base_row)
            row_copy["question"] = q
            expanded_rows.append(row_copy)
            expanded_assigned_names.append(assigned_name)

    # Step 4: single adversarial pass -- one name-substituted question per expanded row
    rng = random.Random(seed + 1)
    excluded_per_row: List[set] = [{name} for name in expanded_assigned_names]
    adversarial_questions = _build_one_adversarial_pass(
        client, expanded_rows, expanded_assigned_names, rng, excluded_per_row
    )

    # Step 5: emit final rows
    output_rows: List[Dict[str, str]] = []
    for base_row, adv_q in zip(expanded_rows, adversarial_questions):
        row_out = dict(base_row)
        row_out["misleading_question"] = adv_q
        output_rows.append(row_out)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "original_fact",
            "essay",
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
        "--num-adversarial",
        type=int,
        default=3,
        help="Number of adversarial questions to generate per persona (each with a different name).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY or OPENAI_API_KEY env var).",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required via --api-key, OPENAI_API_KEY, or OPENAI_API_KEY.")

    generate_pii_dataset(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        api_key=api_key,
        seed=args.seed,
        num_adversarial=args.num_adversarial,
    )


if __name__ == "__main__":
    main()
