"""
Generate a coexisting-facts dataset for testing memory storage of multiple preferences.

Pipeline:
1) Generate first-person preference scenarios across diverse categories
2) For each scenario, produce one SHORT natural statement per preference
   (each will be stored in its own isolated chat conversation)
3) Generate a first-person question requiring knowledge of ALL preferences
4) Write output CSV

CSV columns:
  preference_category   -- e.g. "foods"
  preferences           -- JSON list of preference names, e.g. ["pizza","sushi","ramen"]
  preference_facts      -- JSON list of individual natural statements, one per preference
                           e.g. ["I love pizza.", "Sushi is my go-to.", "I also enjoy ramen."]
  question              -- scenario question requiring all preferences
  ground_truth_answer   -- comma-separated list of all preferences
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

PREFERENCE_CATEGORIES = [
    # Food & drink
    "foods",
    "cuisines",
    "types of desserts",
    "drinks",
    "coffee types",
    "tea varieties",
    "breakfast foods",
    "street foods",
    "snack types",
    "cocktail types",
    "cooking methods",
    "restaurant types",

    # Entertainment & media
    "music genres",
    "types of movies",
    "book genres",
    "tv show genres",
    "video game genres",
    "podcast topics",
    "comedy styles",
    "documentary subjects",
    "types of concerts",
    "types of museums",
    "sports to watch on TV",
    "types of theater",

    # Physical activity
    "sports",
    "outdoor activities",
    "physical exercises",
    "fitness class types",
    "martial arts styles",
    "dance styles",
    "cycling types",
    "swimming styles",
    "types of yoga",
    "hiking styles",

    # Lifestyle & personal style
    "fashion styles",
    "home decor styles",
    "makeup styles",
    "jewelry types",
    "shoe styles",
    "hair care styles",
    "skincare routines",
    "bag styles",
    "hat styles",

    # Travel & places
    "travel destinations",
    "vacation types",
    "nature environments",
    "city neighborhood types",
    "outdoor dining settings",

    # Hobbies & creativity
    "hobbies",
    "crafts",
    "art styles",
    "photography styles",
    "creative writing styles",
    "musical instruments",
    "types of puzzles",
    "journaling styles",
    "home improvement projects",
    "garden types",
    "houseplants",

    # Social & leisure
    "weekend activities",
    "party activities",
    "dating activities",
    "types of social gatherings",
    "board games",
    "card games",
    "conversation topics",

    # Tech & learning
    "programming languages",
    "tech interests",
    "academic subjects",
    "skills to learn",
    "languages to learn",
    "language learning methods",
    "study techniques",
    "note-taking methods",
    "productivity methods",

    # Wellness & self-care
    "mental health activities",
    "meditation types",
    "self-care activities",
    "sleep habits",
    "morning routine habits",
    "evening routine habits",

    # Career & finance
    "career fields",
    "types of volunteering",
    "investment types",
    "budget planning methods",

    # Media consumption
    "social media content types",
    "news formats",
    "podcast formats",
    "types of reading material",
    "audio listening formats",
    "content creation types",

    # Misc
    "animals",
    "dog breeds",
    "collectibles",
    "subscription box types",
    "car types",
    "home organization methods",
    "types of naps",
    "camping styles",
    "astrology interests",
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
    return f"""Generate coexisting-facts datapoints. Each datapoint represents a user (first-person "I") with MULTIPLE preferences in the same category. Each preference will be stored as a completely SEPARATE, ISOLATED memory — so each fact statement must make sense entirely on its own, with no reference to the other preferences.

For each spec below, generate:
1. "preferences": list of exactly num_preferences distinct preferences in the category
   (e.g. for foods: ["pizza", "sushi", "ramen"])

2. "preference_facts": list of exactly num_preferences short, natural first-person statements —
   ONE statement per preference, in the same order as "preferences".
   - Each statement must stand alone as a complete, self-contained fact
   - Each statement must mention ONLY that single preference (not the others)
   - Use varied, natural phrasing — not a template ("I love X", "I enjoy X", "X is my favorite", etc.)
   - 1-2 sentences max per fact
   - Examples for foods:
       "I love pizza — it's my default Friday night meal."
       "Sushi is my go-to whenever I want something fresh and light."
       "I'm a huge ramen fan, especially on cold days."

3. "question": a natural first-person scenario question that REQUIRES knowing ALL preferences.
   - Must NOT be a direct "list all my X" request — make it a realistic scenario
   - Good: "I'm going grocery shopping — what should I pick up for dinners this week?"
   - Good: "My friend wants to plan an outing I'd enjoy — what are some solid options?"
   - The question should have a clearly better answer if ALL preferences are known vs. only one

4. "ground_truth_answer": a concise comma-separated list of all preference names
   Example: "pizza, sushi, ramen"

Return strict JSON with key "rows", a list of objects:
- row_id (int)
- preference_category (string, same as input)
- preferences (list of strings)
- preference_facts (list of strings, same length as preferences, one fact per preference)
- question (string)
- ground_truth_answer (string)

Rules:
1) preference_facts must have exactly the same length as preferences
2) Each fact covers exactly ONE preference and stands alone — no cross-references
3) The question must be a realistic first-person scenario, NOT "list all my X"
4) Ground truth must include every preference, comma-separated
5) Output ONLY valid JSON

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
                prefs = row.get("preferences", [])
                facts = row.get("preference_facts", [])
                if not isinstance(prefs, list):
                    prefs = []
                if not isinstance(facts, list):
                    facts = []
                result.append({
                    "preference_category": str(row.get("preference_category", spec["preference_category"])).strip(),
                    "preferences": json.dumps(prefs),
                    "preference_facts": json.dumps(facts),
                    "question": str(row.get("question", "")).strip(),
                    "ground_truth_answer": str(row.get("ground_truth_answer", "")).strip(),
                })
            return result
        except Exception as e:
            last_err = e
            print(f"  Batch attempt {attempt + 1} failed: {e}")

    raise ValueError(f"Batch generation failed after {MAX_RETRIES} retries: {last_err}")


def build_specs(
    categories: List[str],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """One spec per category, in shuffled order."""
    shuffled = list(categories)
    rng.shuffle(shuffled)
    return [
        {
            "row_id": i,
            "preference_category": cat,
            "num_preferences": rng.randint(2, 5),
        }
        for i, cat in enumerate(shuffled)
    ]


def generate_dataset(
    output_csv: Path,
    api_key: str,
    batch_size: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    client = OpenAI(api_key=api_key)

    all_specs = build_specs(PREFERENCE_CATEGORIES, rng)
    num_rows = len(all_specs)
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

    fieldnames = [
        "preference_category",
        "preferences",
        "preference_facts",
        "question",
        "ground_truth_answer",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Save raw dataset
    raw_path = output_csv.parent / (output_csv.stem + "_raw" + output_csv.suffix)
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    len_raw = len(all_rows)
    print(f"\nGenerated {len_raw} rows (raw)")
    print(f"Saved raw: {raw_path}")

    # Deduplicate on question field
    deduped_rows = deduplicate(all_rows, key=lambda r: r["question"], threshold=0.8)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped_rows)
    print(f"After deduplication: {len(deduped_rows)} rows")
    print(f"Saved: {output_csv}")
    all_rows = deduped_rows
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
        "num_categories": len(PREFERENCE_CATEGORIES),
        "num_rows_raw": len_raw,
        "num_rows_deduped": len(all_rows),
        "rows_removed_by_dedup": len_raw - len(all_rows),
        "batch_size": batch_size,
        "seed": seed,
        "dedup_threshold": 0.7,
        "dedup_key_field": "question",
        "output_csv": str(output_csv),
        "raw_csv": str(raw_path),
    }
    config_path = output_csv.parent / f"generation_config_{ts}.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")

    # Category distribution
    cat_counts: Dict[str, int] = {}
    for r in all_rows:
        cat_counts[r["preference_category"]] = cat_counts.get(r["preference_category"], 0) + 1
    print("\nCategory distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Preference count distribution
    pref_count_dist: Dict[int, int] = {}
    for r in all_rows:
        try:
            n = len(json.loads(r["preferences"]))
        except Exception:
            n = 0
        pref_count_dist[n] = pref_count_dist.get(n, 0) + 1
    print("\nPreference count distribution:")
    for n, count in sorted(pref_count_dist.items()):
        print(f"  {n} preferences: {count}")

    # Sanity check: facts length matches preferences length
    mismatches = 0
    for r in all_rows:
        try:
            if len(json.loads(r["preferences"])) != len(json.loads(r["preference_facts"])):
                mismatches += 1
        except Exception:
            mismatches += 1
    if mismatches:
        print(f"\nWARNING: {mismatches} rows have mismatched preferences/preference_facts lengths.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate coexisting-facts dataset for memory evaluation."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "coexisting_facts" / "coexisting_facts_dataset.csv"),
        help="Path to output CSV.",
    )
    parser.add_argument("--num-rows", type=int, default=None, help="(ignored) rows are now fixed at one per category.")
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
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
