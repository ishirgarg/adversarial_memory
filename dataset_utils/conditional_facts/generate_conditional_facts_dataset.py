"""
Generate a conditional-facts dataset for testing memory summarization of qualified information.

Pipeline:
1) Generate entities (people, pets, characters) with conditional behaviors
2) For each entity, produce a SHORT natural statement encoding the conditional fact
3) Generate a follow-up question with a specific context that may or may not satisfy the condition
4) Wrap each conditional fact in a longer essay (5-10 sentences of unconditional context + the fact)
5) Deduplicate with MinHash LSH (threshold=0.8) on the essay text
6) Save raw CSV (<name>_raw.csv) and deduplicated CSV (<name>.csv)

CSV columns:
  entity                -- e.g. "Alex"
  entity_category       -- "person", "pet", or "character"
  behavior              -- what they do conditionally, e.g. "drinks coffee"
  condition_type        -- type of condition: time_of_day, weather, mood, social_context, intensity, location
  condition             -- the condition text, e.g. "after 5pm"
  entity_facts          -- JSON list with exactly 1 essay that embeds the conditional fact
  question              -- follow-up question with a specific context
  question_context      -- the context embedded in the question, e.g. "3 PM"
  condition_met         -- "yes" or "no": is the condition satisfied in the question?
  ground_truth_answer   -- correct yes/no response to the question
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

MODEL_NAME = "gpt-4.1-mini"
MAX_RETRIES = 3
BATCH_SIZE = 10

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from dataset_utils.dedup import deduplicate  # noqa: E402

ENTITY_CATEGORIES = ["person", "pet", "character"]

CONDITION_TYPES = [
    # Time
    "time_of_day",
    "day_of_week",
    "season",
    "time_elapsed",          # e.g. only after doing something for X minutes/days
    # Environment
    "weather",
    "temperature",
    "location",
    "noise_level",           # e.g. only in quiet environments
    "lighting",              # e.g. only when it's dim/bright
    # Physical/physiological state
    "hunger_level",
    "energy_level",
    "pain_or_discomfort",    # e.g. only when their back hurts
    "sobriety",              # e.g. only when sober
    # Emotional/mental state
    "mood",
    "stress_level",
    "anxiety_level",
    "motivation_level",
    # Social context
    "company",               # who they're with
    "social_setting",        # formal vs. informal, public vs. private
    "relationship_closeness",# e.g. only with close friends
    # Task/activity context
    "task_type",             # e.g. only during creative work
    "workload",              # e.g. only when their schedule is light
    "completion_state",      # e.g. only after finishing a task
    # Sensory/aesthetic triggers
    "music_playing",         # e.g. only when certain music is on
    "scent",                 # e.g. only when they smell something specific
    "food_or_drink_present", # e.g. only when coffee is already made
    # Relational/interpersonal triggers
    "conflict_state",        # e.g. only after an argument is resolved
    "approval_received",     # e.g. only after being praised
    "request_made",          # e.g. only when explicitly asked
    # Habitual/sequential
    "prior_activity",        # e.g. only after a shower
    "frequency_cap",         # e.g. only once a week
    "streak_state",          # e.g. only when on a streak
]


def _chat_json(client: OpenAI, prompt: str) -> Any:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty content.")
    return json.loads(content)


def _make_batch_prompt(specs: List[Dict[str, Any]]) -> str:
    return f"""Generate conditional-facts datapoints. Each datapoint describes an entity (person, pet, or character) who has a CONDITIONAL behavior — they only do something under a specific condition.

For each spec below, generate:

1. "entity": a realistic first name (for persons/characters) or a pet name (for pets)
   e.g. "Jordan", "Miso", "Captain Rex"

2. "behavior": a short action phrase describing what the entity does conditionally.
   Invent something creative and specific to the entity and condition type — do NOT default
   to clichés like "goes for a run" or "drinks coffee". The behavior should feel personal
   and idiosyncratic, not universally common.
   The examples below are illustrative ONLY — do not reuse them:
   e.g. "re-reads old letters", "sketches floor plans", "hums while doing dishes",
        "sends voice notes instead of texts", "reorganizes their bookshelf"

3. "condition": the specific condition under which the behavior occurs
   e.g. "after 5pm", "when it's raining", "when feeling stressed"
   - Must be concrete and testable — the question will present a specific context
   - Avoid vague conditions like "sometimes" or "often"

4. "entity_facts": a list containing exactly 1 natural statement that directly encodes
   the full conditional fact — both the behavior AND the condition in a single sentence.
   - Must be a casual, first-person or third-person conversational sentence
   - Must clearly state BOTH what the entity does AND when/under what condition
   - 1-2 sentences max
   - The example below is illustrative ONLY — do not reuse it:
     ["Alex has a rule: no coffee before 5pm, since it messes with their sleep."]

5. "question": a natural question about whether the entity should do (or would do) the behavior,
   given a SPECIFIC context that may or may not satisfy the condition.

   CRITICAL RULE — the question MUST be non-inferrable without the entity's specific fact:
   A person with no knowledge of the entity should NOT be able to guess the correct answer
   from common sense or general norms alone. The correct answer must depend on knowing
   THIS entity's specific conditional rule.

   BAD (inferrable from common sense):
     - "It's a bright sunny afternoon. Should Zarek wear his heavy winter cloak?"
       → Anyone would say no, regardless of any stored fact.
     - "Jordan hasn't slept in 30 hours. Would they want to go clubbing?"
       → Common sense gives the answer.

   GOOD (requires knowing the entity's rule) — these examples are illustrative ONLY, do not reuse them:
     - "It's 3pm and I'm meeting Alex — should I grab them a coffee?"
       → Without knowing Alex's after-5pm rule, you might reasonably say yes.
     - "It's a quiet Sunday morning. Would Priya want to reorganize her bookshelf?"
       → Without knowing Priya only does this when stressed, you can't tell.
     - "We're at the park and it's 18°C outside. Would Miso eat from the red bowl?"
       → Without knowing Miso's specific rule, this is genuinely ambiguous.

   The question should present a context where a reasonable person WITHOUT the entity's
   specific rule could plausibly answer either way — making the stored fact decisive.

6. "question_context": the specific context presented in the question
   e.g. "3pm", "quiet Sunday morning", "18°C at the park"

7. "condition_met": "yes" if the question context satisfies the condition, "no" if not
   Think carefully — if the condition is "after 5pm" and the context is "3pm", it's "no"

8. "ground_truth_answer": a short yes/no answer with a brief reason
   e.g. "No — it's only 3pm and Alex doesn't drink coffee before 5pm."
   e.g. "Yes — it's raining, which is exactly when Jordan likes to cook elaborate meals."

Return strict JSON with key "rows", a list of objects:
- row_id (int)
- entity (string)
- entity_category (string: "person", "pet", or "character")
- behavior (string)
- condition_type (string, same as input)
- condition (string)
- entity_facts (list of exactly 1 string)
- question (string)
- question_context (string)
- condition_met (string: "yes" or "no")
- ground_truth_answer (string)

Rules:
1) entity_facts must have exactly 1 statement encoding both the behavior and the condition
2) The condition must be concrete and testable (not vague)
3) The question must present a specific context value that clearly either meets or doesn't meet the condition
4) condition_met must correctly reflect whether the question context satisfies the condition
5) ground_truth_answer must be consistent with condition_met
6) The question MUST be non-inferrable: without knowing the entity's specific rule, a
   reasonable person should be genuinely uncertain about the answer
7) Vary condition_met between "yes" and "no" across the batch
8) Do NOT reuse any entity names, behaviors, conditions, or phrasings from the examples above —
   they exist only to illustrate the format
9) Output ONLY valid JSON

Input specs:
{json.dumps(specs, ensure_ascii=False)}
""".strip()


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
                facts = row.get("entity_facts", [])
                if not isinstance(facts, list):
                    facts = []
                if len(facts) != 1:
                    raise ValueError(f"row_id {spec['row_id']}: entity_facts must have exactly 1 element, got {len(facts)}: {facts}")
                result.append({
                    "entity": str(row.get("entity", "")).strip(),
                    "entity_category": str(row.get("entity_category", spec["entity_category"])).strip(),
                    "behavior": str(row.get("behavior", "")).strip(),
                    "condition_type": str(row.get("condition_type", spec["condition_type"])).strip(),
                    "condition": str(row.get("condition", "")).strip(),
                    "entity_facts": json.dumps(facts),
                    "question": str(row.get("question", "")).strip(),
                    "question_context": str(row.get("question_context", "")).strip(),
                    "condition_met": str(row.get("condition_met", "")).strip().lower(),
                    "ground_truth_answer": str(row.get("ground_truth_answer", "")).strip(),
                })
            return result
        except Exception as e:
            last_err = e
            print(f"  Batch attempt {attempt + 1} failed: {e}")

    raise ValueError(f"Batch generation failed after {MAX_RETRIES} retries: {last_err}")


def _make_essay_batch_prompt(items: List[Dict[str, Any]]) -> str:
    return f"""For each item below, write a natural essay (7-10 sentences) about the entity
that embeds the conditional fact into a rich, casual narrative.

Rules:
1. The essay MUST preserve the conditional fact clearly — both the behavior AND the
   condition must be present. Paraphrase is fine; do not omit either part.
2. All other sentences should describe the entity's background, personality, daily routines,
   relationships, hobbies, quirks, or life context. Every such sentence must be an
   unconditional, factual statement.
3. Do NOT introduce any new conditional statements anywhere in the essay. Forbidden
   constructions: "only when", "unless", "except when", "but only if", "whenever X then Y",
   "only after", "only if", or any other conditional phrasing beyond what was already in the
   original fact.
4. The essay should feel natural — like an excerpt from a chat conversation, personal blog,
   or journal entry, not a formal report or list.
5. The conditional fact may appear anywhere in the essay, surrounded by unrelated context
   before and after it.
6. 5-10 sentences total.

Return strict JSON with key "rows", a list of:
  row_id (int, same as input), essay (string)

Output ONLY valid JSON.

Input:
{json.dumps(items, ensure_ascii=False)}
""".strip()


def wrap_facts_in_essays(
    client: OpenAI,
    rows: List[Dict[str, str]],
    batch_size: int,
) -> List[Dict[str, str]]:
    """Replace entity_facts[0] with a short essay embedding the fact. Returns updated rows."""
    result = list(rows)
    num_batches = (len(rows) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(rows))
        batch = rows[start:end]

        items = []
        for i, row in enumerate(batch):
            try:
                fact = json.loads(row["entity_facts"])[0]
            except Exception:
                fact = row["entity_facts"]
            items.append({
                "row_id": start + i,
                "entity": row["entity"],
                "entity_category": row["entity_category"],
                "conditional_fact": fact,
            })

        print(f"  Wrapping essays batch {batch_idx + 1}/{num_batches} (rows {start}-{end - 1})...")
        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                parsed = _chat_json(client, _make_essay_batch_prompt(items))
                essay_rows = parsed.get("rows")
                if not isinstance(essay_rows, list) or len(essay_rows) == 0:
                    raise ValueError("Response missing 'rows' list or empty.")
                by_id = {r["row_id"]: r["essay"] for r in essay_rows if isinstance(r.get("row_id"), int)}
                for item in items:
                    essay = by_id.get(item["row_id"], "").strip()
                    if not essay:
                        raise ValueError(f"row_id {item['row_id']}: empty essay returned.")
                    result[item["row_id"]] = {
                        **result[item["row_id"]],
                        "entity_facts": json.dumps([essay]),
                    }
                break
            except Exception as e:
                last_err = e
                print(f"    Essay batch attempt {attempt + 1} failed: {e}")
        else:
            raise ValueError(f"Essay batch failed after {MAX_RETRIES} retries: {last_err}")

    return result


def build_specs(
    num_rows: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    specs = []
    # Alternate condition_met to ensure balanced dataset
    condition_met_cycle = (["yes"] * 5 + ["no"] * 5) * (num_rows // 10 + 1)
    rng.shuffle(condition_met_cycle)

    for i in range(num_rows):
        condition_type = rng.choice(CONDITION_TYPES)
        entity_category = rng.choice(ENTITY_CATEGORIES)
        # Pets can't have abstract internal states — use observable conditions only
        if entity_category == "pet":
            condition_type = rng.choice([
                "time_of_day", "weather", "temperature", "location",
                "noise_level", "lighting", "food_or_drink_present",
                "prior_activity", "company",
            ])
        specs.append({
            "row_id": i,
            "entity_category": entity_category,
            "condition_type": condition_type,
            "target_condition_met": condition_met_cycle[i],  # hint to LLM for balance
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

    print("Wrapping facts in essays...")
    all_rows = wrap_facts_in_essays(client, all_rows, batch_size)
    print("Essay wrapping complete.")

    fieldnames = [
        "entity",
        "entity_category",
        "behavior",
        "condition_type",
        "condition",
        "entity_facts",
        "question",
        "question_context",
        "condition_met",
        "ground_truth_answer",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Save raw dataset before deduplication
    raw_path = output_csv.parent / (output_csv.stem + "_raw" + output_csv.suffix)
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    len_raw = len(all_rows)
    print(f"Saved raw ({len_raw} rows): {raw_path}")

    # Deduplicate on the essay text (entity_facts[0])
    def _essay_key(row: Dict) -> str:
        try:
            return json.loads(row["entity_facts"])[0]
        except Exception:
            return row.get("entity_facts", "")

    deduped_rows = deduplicate(all_rows, key=_essay_key, threshold=0.8)
    removed = len(all_rows) - len(deduped_rows)
    print(f"Deduplication: removed {removed} near-duplicates, kept {len(deduped_rows)} rows.")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped_rows)

    all_rows = deduped_rows
    print(f"Saved deduplicated ({len(all_rows)} rows): {output_csv}")
    print(f"Model used: {MODEL_NAME}")

    # Save generation config for reproducibility
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
        "batch_size": batch_size,
        "seed": seed,
        "dedup_threshold": 0.8,
        "dedup_key_field": "entity_facts[0] (essay text)",
        "output_csv": str(output_csv),
        "raw_csv": str(raw_path),
    }
    config_path = output_csv.parent / f"generation_config_{ts}.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")

    # Condition type distribution
    ctype_counts: Dict[str, int] = {}
    for r in all_rows:
        ctype_counts[r["condition_type"]] = ctype_counts.get(r["condition_type"], 0) + 1
    print("\nCondition type distribution:")
    for ct, count in sorted(ctype_counts.items(), key=lambda x: -x[1]):
        print(f"  {ct}: {count}")

    # condition_met distribution
    met_counts: Dict[str, int] = {}
    for r in all_rows:
        met_counts[r["condition_met"]] = met_counts.get(r["condition_met"], 0) + 1
    print("\nCondition-met distribution:")
    for k, count in sorted(met_counts.items()):
        print(f"  {k}: {count}")

    # Entity category distribution
    cat_counts: Dict[str, int] = {}
    for r in all_rows:
        cat_counts[r["entity_category"]] = cat_counts.get(r["entity_category"], 0) + 1
    print("\nEntity category distribution:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    # Sanity check: entity_facts has exactly 1 statement
    wrong_facts = 0
    for r in all_rows:
        try:
            facts = json.loads(r["entity_facts"])
            if not isinstance(facts, list) or len(facts) != 1:
                wrong_facts += 1
        except Exception:
            wrong_facts += 1
    if wrong_facts:
        print(f"\nWARNING: {wrong_facts} rows have entity_facts with != 1 statement.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate conditional-facts dataset for memory evaluation."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "conditional_facts" / "conditional_facts_dataset.csv"),
        help="Path to output CSV.",
    )
    parser.add_argument("--num-rows", type=int, default=50, help="Number of datapoints to generate.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Rows per LLM call.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key.")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required via --api-key, OPENAI_API_KEY, or OPENAI_API_KEY.")

    generate_dataset(
        output_csv=Path(args.output_csv),
        api_key=api_key,
        num_rows=args.num_rows,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
