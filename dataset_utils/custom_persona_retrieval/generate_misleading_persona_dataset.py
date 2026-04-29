"""
Generate a misleading-persona dataset for testing memory grounding to specific identities.

Pipeline:
1) Generate entities (named people with rich, idiosyncratic personas)
2) For each entity, produce a personal essay (10-15 sentences) embedding several
   memorable, specific facts about that person
3) Generate 3 questions per entity. Each question is independently EITHER:
     - non-misleading: first-person, asks about THAT entity by name (answerable
       from the essay's specific details), OR
     - misleading: first-person, asks about a DIFFERENT named person (the
       "distractor") who has no presence in the essay. Correct behavior is to
       abstain.
4) Deduplicate with MinHash LSH (threshold=0.7) on the essay text
5) Save raw CSV (<name>_raw.csv) and deduplicated CSV (<name>.csv)

CSV columns:
  entity        -- e.g. "Maya Patel"
  entity_facts  -- JSON list with exactly 1 essay about the entity
  questions     -- JSON list of exactly 3 question objects, each with fields:
                     text                 (string)
                     is_misleading        (bool)
                     distractor           (string or null; non-null iff misleading)
                     ground_truth_answer  (string)
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

MODEL_NAME = "gpt-5-mini"
MAX_RETRIES = 3
BATCH_SIZE = 20
QUESTIONS_PER_ROW = 3

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from dataset_utils.dedup import deduplicate  # noqa: E402

NAME_POOL = [
    "Ava Thompson", "Liam Carter", "Maya Patel", "Noah Brooks", "Zoe Kim",
    "Ethan Rivera", "Priya Shah", "Lucas Bennett", "Sofia Nguyen", "Daniel Park",
    "Elena Rossi", "Marcus Lee", "Amara Okafor", "Jonas Weber", "Hana Sato",
    "Theo Laurent", "Nia Williams", "Ravi Iyer", "Clara Schmidt", "Diego Alvarez",
    "Yuki Tanaka", "Sasha Petrov", "Imani Johnson", "Felix Andersen", "Leila Haddad",
    "Owen Murphy", "Anya Volkov", "Caleb Foster", "Mei Zhang", "Tomas Costa",
]

PERSONA_FLAVORS = [
    "a meticulous indoor gardener with strong opinions about humidity",
    "a lapsed competitive swimmer who now coaches youth weekend meets",
    "a sound engineer obsessed with vintage analog gear",
    "a part-time pastry chef who does math research on the side",
    "a long-distance hiker training for the Pacific Crest Trail",
    "a retired ER nurse who took up woodworking after retirement",
    "a beekeeper-turned-marketing-consultant who still keeps three hives",
    "an amateur astronomer who hates city light pollution",
    "a freelance translator working between Portuguese and Korean",
    "a cybersecurity researcher who collects vintage typewriters",
    "a former competitive figure skater now running a small tea shop",
    "a chef-instructor who teaches knife skills at a community college",
    "an opera singer who is also a part-time auto mechanic",
    "an architect specializing in adaptive reuse of old factories",
    "a wildlife photographer focused on owls in the Pacific Northwest",
    "a high-school chemistry teacher who restores vintage motorcycles",
    "a marathoner with a rare allergy to most stone fruits",
    "a bookbinder who designs board games on weekends",
    "a software engineer who breeds carnivorous plants",
    "a paramedic who plays cello in a community orchestra",
    "a former diplomat now running a pottery studio",
    "a cartographer obsessed with historic shipwrecks",
    "a dog trainer specializing in working breeds",
    "a forensic accountant who writes science fiction novels",
    "a glassblower with severe pollen allergies",
    "a sommelier transitioning to non-alcoholic beverage consulting",
    "a former orchestra conductor who now teaches sailing",
    "a museum conservator focused on 19th-century photographs",
    "a competitive bridge player who works as an actuary",
    "a backcountry ski guide who restores antique furniture in summer",
]


def _chat_json(client: OpenAI, prompt: str) -> Any:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty content.")
    return json.loads(content)


def _make_batch_prompt(specs: List[Dict[str, Any]]) -> str:
    return f"""Generate misleading-persona datapoints. Each datapoint is about a SPECIFIC named
person, embedded in a rich essay (10-15 sentences). Each datapoint has exactly 3 questions.
Each question is independently EITHER non-misleading (asks about the entity by name) OR
misleading (asks about a DIFFERENT named person — the "distractor" — who has no presence
in the essay).

For each spec below, generate:

1. "essay": a natural personal essay about the entity (10-15 sentences).
   - Written in third person, naming the entity (e.g. "Maya Patel"). Pronouns are fine
     after the first mention.
   - Embed MANY specific, memorable, idiosyncratic facts: daily rituals, unusual hobbies,
     hard constraints (allergies/aversions/rules), strong preferences, quirky possessions,
     rules of thumb. Aim for at least 4-5 distinct facts so different questions can probe
     different details.
   - Tone: casual, like a journal entry or chat message — not a formal bio.
   - Do NOT use first-person voice ("I", "me", "my", "we").
   - Do NOT mention any of the distractor names anywhere in the essay.

2. "questions": a list of EXACTLY 3 question objects, in the order given by
   spec.question_slots. Each slot specifies whether that question is misleading and, if
   so, the distractor name to use.

   For each slot:

   If is_misleading=false:
     - "text": a first-person question that explicitly names the entity by their full name.
       The asker wants advice or info ABOUT the entity (e.g. what to get them, what to
       avoid, where to take them, whether they'd enjoy something).
     - The question must be answerable from a SPECIFIC detail in the essay — NOT from
       generic norms.
     - Phrased naturally. Must NOT smuggle the answer into itself as an assumption.
       BAD:  "What apple dessert can I give Maya that won't make her itch?"
             (assumes the asker already knows about the allergy)
       GOOD: "What dessert should I make for Maya?"
             (open; the essay's allergy info is what makes the answer specific)
     - "ground_truth_answer": short answer (1-2 sentences) drawn from specific essay
       details. This is what a memory-aware system should return.
     - "distractor": null.

   If is_misleading=true:
     - "text": a first-person question that names the GIVEN distractor (NOT the entity).
       Same kind of advice/info question as the non-misleading case, but asked about the
       distractor. The entity's name must NOT appear in the question.
     - "ground_truth_answer": "I don't have information about <distractor name>."
       (Correct behavior is to abstain — the asker has no info about this person.)
     - "distractor": the distractor name from the spec slot.

   The 3 questions should probe DIFFERENT angles — don't repeat the same wording or topic
   across slots. If multiple slots are non-misleading, each should probe a different fact
   from the essay.

Return strict JSON with key "rows", a list of objects:
- row_id (int)
- entity (string, same as input)
- essay (string)
- questions (list of exactly 3 objects with fields: text, is_misleading, distractor,
  ground_truth_answer)

Rules:
1) The essay is 10-15 sentences, third-person, names the entity, and never mentions any
   distractor name from any slot.
2) Each non-misleading question names the entity exactly and never names any distractor.
3) Each misleading question names that slot's distractor exactly and never names the
   entity.
4) Non-misleading questions must NOT embed their own answers as assumptions.
5) Each non-misleading ground_truth_answer is supported by specific essay details.
6) Each misleading ground_truth_answer indicates the system should abstain.
7) Output ONLY valid JSON.

Input specs:
{json.dumps(specs, ensure_ascii=False)}
""".strip()


def _validate_question(
    q: Dict[str, Any],
    slot: Dict[str, Any],
    entity: str,
    essay: str,
    row_id: int,
    slot_idx: int,
) -> Dict[str, Any]:
    text = str(q.get("text", "")).strip()
    if not text:
        raise ValueError(f"row_id {row_id} q{slot_idx}: empty question text.")

    is_misleading = bool(slot["is_misleading"])
    expected_distractor = slot["distractor"]

    if is_misleading:
        if expected_distractor is None:
            raise ValueError(f"row_id {row_id} q{slot_idx}: spec is misleading but distractor is null.")
        if expected_distractor.lower() not in text.lower():
            raise ValueError(
                f"row_id {row_id} q{slot_idx}: misleading question does not contain "
                f"distractor {expected_distractor!r}."
            )
        if entity.lower() in text.lower():
            raise ValueError(
                f"row_id {row_id} q{slot_idx}: misleading question must not name the entity."
            )
        distractor_out: Any = expected_distractor
    else:
        if entity.lower() not in text.lower():
            raise ValueError(
                f"row_id {row_id} q{slot_idx}: non-misleading question must name the entity {entity!r}."
            )
        distractor_out = None

    gt = str(q.get("ground_truth_answer", "")).strip()
    if not gt:
        raise ValueError(f"row_id {row_id} q{slot_idx}: missing ground_truth_answer.")

    return {
        "text": text,
        "is_misleading": is_misleading,
        "distractor": distractor_out,
        "ground_truth_answer": gt,
    }


def generate_batch(
    client: OpenAI,
    specs: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Generate one batch of datapoints, retrying on failure."""
    prompt = _make_batch_prompt(specs)

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            parsed = _chat_json(client, prompt)
            rows = parsed.get("rows")
            if not isinstance(rows, list) or len(rows) == 0:
                raise ValueError("Response missing 'rows' list or empty.")

            by_id: Dict[int, Dict] = {r["row_id"]: r for r in rows if isinstance(r.get("row_id"), int)}
            result: List[Dict[str, str]] = []
            for spec in specs:
                row = by_id.get(spec["row_id"], {})
                essay = str(row.get("essay", "")).strip()
                if not essay:
                    raise ValueError(f"row_id {spec['row_id']}: empty essay.")
                entity = str(row.get("entity", spec["entity"])).strip()

                slots = spec["question_slots"]
                for slot in slots:
                    if slot["is_misleading"] and slot["distractor"].lower() in essay.lower():
                        raise ValueError(
                            f"row_id {spec['row_id']}: distractor {slot['distractor']!r} "
                            "leaked into essay."
                        )

                qs = row.get("questions")
                if not isinstance(qs, list) or len(qs) != QUESTIONS_PER_ROW:
                    raise ValueError(
                        f"row_id {spec['row_id']}: expected {QUESTIONS_PER_ROW} questions, "
                        f"got {len(qs) if isinstance(qs, list) else 'non-list'}."
                    )

                validated_qs = [
                    _validate_question(qs[i], slots[i], entity, essay, spec["row_id"], i)
                    for i in range(QUESTIONS_PER_ROW)
                ]

                result.append({
                    "entity": entity,
                    "entity_facts": json.dumps([essay]),
                    "questions": json.dumps(validated_qs, ensure_ascii=False),
                })
            return result
        except Exception as e:
            last_err = e
            print(f"  Batch attempt {attempt + 1} failed: {e}")

    raise ValueError(f"Batch generation failed after {MAX_RETRIES} retries: {last_err}")


def build_specs(num_rows: int, rng: random.Random) -> List[Dict[str, Any]]:
    """One spec per row. Each spec gets an entity name, a persona flavor hint, and 3
    question slots. Each slot is independently misleading-or-not (50/50). Misleading
    slots get a distractor (different from entity, distinct across the row's slots)."""
    specs = []
    for i in range(num_rows):
        entity = rng.choice(NAME_POOL)
        distractor_pool = [n for n in NAME_POOL if n != entity]
        # Pre-sample distinct distractor candidates for the misleading slots so the
        # essay never has to pretend a name doesn't exist twice over.
        candidate_distractors = rng.sample(distractor_pool, QUESTIONS_PER_ROW)
        question_slots = []
        for j in range(QUESTIONS_PER_ROW):
            is_misleading = rng.random() < 0.5
            question_slots.append({
                "is_misleading": is_misleading,
                "distractor": candidate_distractors[j] if is_misleading else None,
            })
        flavor = rng.choice(PERSONA_FLAVORS)
        specs.append({
            "row_id": i,
            "entity": entity,
            "persona_flavor": flavor,
            "question_slots": question_slots,
        })
    return specs


def generate_dataset(
    output_csv: Path,
    api_key: str,
    num_rows: int,
    batch_size: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    client = OpenAI(api_key=api_key)

    all_specs = build_specs(num_rows, rng)
    all_rows: List[Dict[str, str]] = []

    num_batches = (num_rows + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_rows)
        batch_specs = all_specs[start:end]
        print(f"Generating batch {batch_idx + 1}/{num_batches} (rows {start}-{end - 1})...")
        batch_rows = generate_batch(client, batch_specs)
        all_rows.extend(batch_rows)
        print(f"  Total rows so far: {len(all_rows)}")

    fieldnames = ["entity", "entity_facts", "questions"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    raw_path = output_csv.parent / (output_csv.stem + "_raw" + output_csv.suffix)
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    len_raw = len(all_rows)
    print(f"Saved raw ({len_raw} rows): {raw_path}")

    def _essay_key(row: Dict) -> str:
        try:
            return json.loads(row["entity_facts"])[0]
        except Exception:
            return row.get("entity_facts", "")

    deduped_rows = deduplicate(all_rows, key=_essay_key, threshold=0.7)
    removed = len(all_rows) - len(deduped_rows)
    print(f"Deduplication: removed {removed} near-duplicates, kept {len(deduped_rows)} rows.")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped_rows)

    all_rows = deduped_rows
    print(f"Saved deduplicated ({len(all_rows)} rows): {output_csv}")
    print(f"Model used: {MODEL_NAME}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        git_dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ) != 0
    except Exception:
        git_commit = "unknown"
        git_dirty = None
    config = {
        "timestamp": ts,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "model_name": MODEL_NAME,
        "num_rows_requested": num_rows,
        "num_rows_raw": len_raw,
        "num_rows_deduped": len(all_rows),
        "rows_removed_by_dedup": len_raw - len(all_rows),
        "questions_per_row": QUESTIONS_PER_ROW,
        "batch_size": batch_size,
        "seed": seed,
        "dedup_threshold": 0.7,
        "dedup_key_field": "entity_facts[0] (essay text)",
        "output_csv": str(output_csv),
        "raw_csv": str(raw_path),
    }
    config_path = output_csv.parent / f"generation_config_{ts}.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")

    # Entity name distribution
    entity_counts: Dict[str, int] = {}
    for r in all_rows:
        entity_counts[r["entity"]] = entity_counts.get(r["entity"], 0) + 1
    print("\nEntity distribution (top 10):")
    for ent, count in sorted(entity_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {ent}: {count}")

    # Question-type distribution
    n_misleading = 0
    n_total_q = 0
    for r in all_rows:
        for q in json.loads(r["questions"]):
            n_total_q += 1
            if q["is_misleading"]:
                n_misleading += 1
    print(f"\nQuestion totals: {n_total_q} ({n_misleading} misleading, "
          f"{n_total_q - n_misleading} non-misleading).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate misleading-persona dataset for memory evaluation."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "custom_persona_retrieval" / "misleading_persona_dataset.csv"),
        help="Path to output CSV.",
    )
    parser.add_argument("--num-rows", type=int, default=100, help="Number of datapoints to generate.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Rows per LLM call.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key.")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required via --api-key or OPENAI_API_KEY.")

    generate_dataset(
        output_csv=Path(args.output_csv),
        api_key=api_key,
        num_rows=args.num_rows,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
