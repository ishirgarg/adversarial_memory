# ai_attack_dataset_builder.py
# pip install datasets openai

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from openai import OpenAI


# -----------------------------
# Config
# -----------------------------
SEED = 7
random.seed(SEED)

USE_AI = os.getenv("USE_AI", "true").lower() != "false"
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if USE_AI else None


# -----------------------------
# Resilient API call
# -----------------------------
def _resilient_api_call(fn, *args, label: str = "API call", **kwargs):
    """
    Wrap any OpenAI API call to survive transient network drops and rate limits.

    - AuthenticationError / PermissionDeniedError → re-raise immediately (bad key)
    - Network / timeout errors → pause with exponential backoff, retry indefinitely
    - RateLimitError → back off proportionally then retry
    - Other errors → up to 5 retries with exponential backoff
    """
    try:
        from openai import (
            AuthenticationError, PermissionDeniedError,
            APIConnectionError, APITimeoutError, RateLimitError,
        )
        _AUTH_ERRS = (AuthenticationError, PermissionDeniedError)
        _NET_ERRS  = (APIConnectionError, APITimeoutError, OSError, ConnectionError)
        _RATE_ERR  = RateLimitError
    except ImportError:
        _AUTH_ERRS = ()
        _NET_ERRS  = (OSError, ConnectionError)
        _RATE_ERR  = type(None)

    net_backoff = 5
    bounded_attempt = 0

    while True:
        try:
            return fn(*args, **kwargs)

        except _AUTH_ERRS:
            raise

        except _NET_ERRS as e:
            wait = min(net_backoff, 120)
            print(f"  [{label}] Network error ({type(e).__name__}). Retrying in {wait}s...")
            time.sleep(wait)
            net_backoff = min(net_backoff * 2, 120)

        except Exception as e:
            if isinstance(e, _RATE_ERR):
                wait = min(60, 10 * (bounded_attempt + 1))
                print(f"  [{label}] Rate limited. Retrying in {wait}s...")
                time.sleep(wait)
                bounded_attempt += 1
                continue

            bounded_attempt += 1
            if bounded_attempt >= 5:
                print(f"  [{label}] Failed after 5 attempts: {e}")
                raise
            wait = min(2 ** bounded_attempt, 30)
            print(f"  [{label}] Error ({type(e).__name__}). Retrying in {wait}s (attempt {bounded_attempt}/5)...")
            time.sleep(wait)


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class AttackExample:
    example_id: str
    source_dataset: str
    attack_type: str
    contextual_statements: List[str]
    graded_question: str
    ground_truth_answer: str
    judge_target_failure_mode: str
    metadata: Dict[str, Any]


# -----------------------------
# Utility functions
# -----------------------------
def normalize_text(x: str) -> str:
    x = x.strip()
    x = re.sub(r"\s+", " ", x)
    return x


def maybe_json_load(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {}


def guess_persona_list(row: Dict[str, Any]) -> List[str]:
    keys = ["personality", "persona", "user_persona", "your_persona", "persona_a", "profile"]
    for k in keys:
        if k in row:
            val = row[k]
            if isinstance(val, list):
                return [normalize_text(v) for v in val if isinstance(v, str) and v.strip()]
            if isinstance(val, str):
                return [normalize_text(x) for x in val.split("\n") if x.strip()]
    return []


def extract_preference_or_trait(persona_sentence: str) -> Optional[Dict[str, str]]:
    s = normalize_text(persona_sentence).rstrip(".")
    sl = s.lower()

    patterns = [
        (r"^i am (.+)$", ("trait", "{name} is {x}")),
        (r"^i'm (.+)$", ("trait", "{name} is {x}")),
        (r"^i have (.+)$", ("has", "{name} has {x}")),
        (r"^i like (.+)$", ("like", "{name} likes {x}")),
        (r"^i love (.+)$", ("like", "{name} loves {x}")),
        (r"^i enjoy (.+)$", ("like", "{name} enjoys {x}")),
        (r"^i work as (.+)$", ("job", "{name} works as {x}")),
        (r"^i work in (.+)$", ("job", "{name} works in {x}")),
        (r"^i live in (.+)$", ("location", "{name} lives in {x}")),
        (r"^my favorite (.+) is (.+)$", ("favorite", "{name}'s favorite {x1} is {x2}")),
    ]

    for pat, spec in patterns:
        m = re.match(pat, sl)
        if m:
            kind, template = spec
            groups = m.groups()
            if len(groups) == 1:
                return {"kind": kind, "template": template, "x": groups[0]}
            if len(groups) == 2:
                return {"kind": kind, "template": template, "x1": groups[0], "x2": groups[1]}
    return None


def pick_names(n: int) -> List[str]:
    pool = [
        "Alex", "Jordan", "Taylor", "Sam", "Riley",
        "Casey", "Morgan", "Jamie", "Avery", "Quinn",
        "Dan", "Betty", "Jake", "Mia", "Noah"
    ]
    return random.sample(pool, n)


# -----------------------------
# AI helpers
# -----------------------------
def call_json_llm(system: str, user: str, label: str = "LLM") -> Dict[str, Any]:
    if client is None:
        return {}

    def _call():
        return client.chat.completions.create(
            model=MODEL,
            temperature=0.7,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

    resp = _resilient_api_call(_call, label=label)
    text = resp.choices[0].message.content
    return maybe_json_load(text)


def ai_naturalize_example(example: AttackExample) -> AttackExample:
    """
    Rewrite the contextual statements/question in a more natural conversational way
    while preserving exact semantics and ground truth.
    """
    system = """
You rewrite benchmark examples into realistic chat-style utterances.

Return valid JSON with fields:
- contextual_statements: list[str]
- graded_question: str
- ground_truth_answer: str

Rules:
1. Preserve the semantics exactly.
2. Do NOT change the correct answer.
3. Keep it concise.
4. Keep the number of contextual statements the same.
5. Do not introduce ambiguity unless it already exists.
"""

    user = json.dumps({
        "attack_type": example.attack_type,
        "contextual_statements": example.contextual_statements,
        "graded_question": example.graded_question,
        "ground_truth_answer": example.ground_truth_answer,
    }, indent=2)

    out = call_json_llm(system, user, label="naturalize")
    if not out:
        return example

    try:
        return AttackExample(
            **{
                **asdict(example),
                "contextual_statements": out["contextual_statements"],
                "graded_question": out["graded_question"],
                "ground_truth_answer": out["ground_truth_answer"],
            }
        )
    except Exception:
        return example


def ai_mutate_fact(base_fact: str, attack_type: str) -> Optional[str]:
    """
    Create a semantically similar but lexically different seed fact.
    """
    system = """
You generate one mutated conversational fact.

Return JSON: {"mutated_fact": "..."}.

Rules:
1. Preserve the original meaning.
2. Keep it one sentence.
3. First-person phrasing is preferred.
4. Natural everyday language.
"""

    user = json.dumps({
        "base_fact": base_fact,
        "attack_type": attack_type,
    })

    out = call_json_llm(system, user, label="mutate_fact")
    mf = out.get("mutated_fact")
    if isinstance(mf, str) and mf.strip():
        return normalize_text(mf)
    return None


def ai_generate_conditional_variant(item: str, name: str) -> Optional[Dict[str, Any]]:
    """
    Ask AI to generate a conditional preference example with a known answer.
    """
    system = """
Create one conditional-preference memory benchmark example.

Return JSON with:
- contextual_statements: list[str]
- graded_question: str
- ground_truth_answer: str
- condition_holds: boolean

Rules:
1. The entity name must be used exactly as provided.
2. The item must be used exactly as provided.
3. Make the condition explicit.
4. Make the answer unambiguous.
5. Keep it short.
"""

    user = json.dumps({
        "entity_name": name,
        "item": item,
        "goal": "Test whether a memory system preserves conditions instead of flattening them."
    })

    out = call_json_llm(system, user, label="conditional_variant")
    if not out:
        return None

    needed = ["contextual_statements", "graded_question", "ground_truth_answer", "condition_holds"]
    if not all(k in out for k in needed):
        return None
    return out


def ai_generate_crossover_example(fact_a: str, fact_b: str, attack_type: str) -> Optional[Dict[str, Any]]:
    """
    AI-assisted crossover: combine two facts into a single attack case.
    Useful for identity confusion / coexisting facts / narrative correction.
    """
    system = """
Create one benchmark example by combining two seed facts.

Return JSON with:
- contextual_statements: list[str]
- graded_question: str
- ground_truth_answer: str
- metadata: object

Rules:
1. Keep the example logically coherent.
2. The answer must be fully determined by the contextual statements.
3. The example should target the requested attack type.
4. Keep it concise and realistic.
"""

    user = json.dumps({
        "fact_a": fact_a,
        "fact_b": fact_b,
        "attack_type": attack_type
    })

    out = call_json_llm(system, user, label="crossover")
    if not out:
        return None

    if not all(k in out for k in ["contextual_statements", "graded_question", "ground_truth_answer"]):
        return None
    if "metadata" not in out or not isinstance(out["metadata"], dict):
        out["metadata"] = {}
    return out


# -----------------------------
# Dataset loading
# -----------------------------
def sample_persona_facts(dataset_name: str, split: str, n_rows: int = 100) -> List[str]:
    ds = load_dataset(dataset_name, split=split)

    facts = []
    _PERSONA_PREFIX = re.compile(r"(?:your persona|partner'?s? persona)\s*:\s*", re.IGNORECASE)

    for i in range(min(n_rows, len(ds))):
        row = ds[i]

        # This dataset packs everything into a single 'text' field, e.g.:
        #   "1 your persona: i like to remodel homes."
        # Try structured keys first, fall back to parsing the text field.
        persona_list = guess_persona_list(row)
        if not persona_list and "text" in row:
            text = row["text"] or ""
            m = _PERSONA_PREFIX.search(text)
            if m:
                fact = normalize_text(text[m.end():])
                persona_list = [fact] if fact else []

        for p in persona_list:
            p = normalize_text(p)
            if p.lower().startswith(("i ", "i'm ", "i've ", "my ")):
                facts.append(p)

    # dedupe
    out = []
    seen = set()
    for f in facts:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


# -----------------------------
# Example builders
# -----------------------------
def build_identity_examples(persona_facts: List[str], n: int = 8) -> List[AttackExample]:
    out = []
    candidates = []

    for f in persona_facts:
        parsed = extract_preference_or_trait(f)
        if parsed:
            candidates.append((f, parsed))

    for i in range(min(n, len(candidates))):
        original_fact, parsed = random.choice(candidates)
        a, b = pick_names(2)

        maybe_mutated = ai_mutate_fact(original_fact, "identity_destruction") if USE_AI else None
        base_fact = maybe_mutated or original_fact
        parsed2 = extract_preference_or_trait(base_fact) or parsed

        if "x" in parsed2:
            fact_for_a = parsed2["template"].format(name=a, x=parsed2["x"])
        else:
            fact_for_a = parsed2["template"].format(name=a, x1=parsed2["x1"], x2=parsed2["x2"])

        q = f"What do we know about {b}?"
        ans = "Not enough information."

        ex = AttackExample(
            example_id=f"persona-identity-{i}",
            source_dataset="persona-chat",
            attack_type="identity_destruction",
            contextual_statements=[fact_for_a],
            graded_question=q,
            ground_truth_answer=ans,
            judge_target_failure_mode="summarization_dropped_identity",
            metadata={"entity_with_fact": a, "entity_queried": b, "seed_fact": original_fact},
        )

        if USE_AI:
            ex = ai_naturalize_example(ex)
        out.append(ex)

    return out


def build_multiple_preferences_examples(persona_facts: List[str], n: int = 8) -> List[AttackExample]:
    prefs = []
    for f in persona_facts:
        parsed = extract_preference_or_trait(f)
        if parsed and parsed["kind"] == "like":
            prefs.append(f)

    out = []
    for i in range(min(n, len(prefs) // 2)):
        f1, f2 = random.sample(prefs, 2)
        name = pick_names(1)[0]

        crossover = ai_generate_crossover_example(f1, f2, "multiple_preferences") if USE_AI else None
        if crossover:
            ex = AttackExample(
                example_id=f"persona-multipref-{i}",
                source_dataset="persona-chat",
                attack_type="multiple_preferences",
                contextual_statements=crossover["contextual_statements"],
                graded_question=crossover["graded_question"],
                ground_truth_answer=crossover["ground_truth_answer"],
                judge_target_failure_mode="storage_failed_to_keep_coexisting_facts",
                metadata={"seed_facts": [f1, f2], **crossover.get("metadata", {})},
            )
            out.append(ex)
            continue

        p1 = extract_preference_or_trait(f1)
        p2 = extract_preference_or_trait(f2)
        item1 = p1["x"]
        item2 = p2["x"]

        ex = AttackExample(
            example_id=f"persona-multipref-{i}",
            source_dataset="persona-chat",
            attack_type="multiple_preferences",
            contextual_statements=[
                f"{name} likes {item1}.",
                f"{name} also likes {item2}.",
            ],
            graded_question=f"I'm planning dinner for {name}. What are two things {name} likes?",
            ground_truth_answer=f"{name} likes {item1} and {item2}.",
            judge_target_failure_mode="storage_failed_to_keep_coexisting_facts",
            metadata={"entity": name, "seed_facts": [f1, f2]},
        )
        if USE_AI:
            ex = ai_naturalize_example(ex)
        out.append(ex)

    return out


def build_narrative_correction_examples(persona_facts: List[str], n: int = 8) -> List[AttackExample]:
    prefs = []
    for f in persona_facts:
        parsed = extract_preference_or_trait(f)
        if parsed and parsed["kind"] == "like":
            prefs.append(f)

    out = []
    for i in range(min(n, len(prefs) // 2)):
        old_f, new_f = random.sample(prefs, 2)
        name = pick_names(1)[0]

        old_item = extract_preference_or_trait(old_f)["x"]
        new_item = extract_preference_or_trait(new_f)["x"]

        ex = AttackExample(
            example_id=f"persona-correction-{i}",
            source_dataset="persona-chat",
            attack_type="narrative_correction",
            contextual_statements=[
                f"{name} likes {old_item}.",
                f"Actually, {name} doesn't like {old_item} anymore — {name} prefers {new_item} now.",
            ],
            graded_question=f"What should I get for {name}?",
            ground_truth_answer=f"You should choose something related to {new_item}, not {old_item}.",
            judge_target_failure_mode="storage_failed_to_update_fact",
            metadata={"entity": name, "old_fact": old_item, "new_fact": new_item},
        )

        if USE_AI:
            ex = ai_naturalize_example(ex)
        out.append(ex)

    return out


def build_conditional_examples(persona_facts: List[str], n: int = 8) -> List[AttackExample]:
    prefs = []
    for f in persona_facts:
        parsed = extract_preference_or_trait(f)
        if parsed and parsed["kind"] == "like":
            prefs.append(f)

    out = []
    for i in range(min(n, len(prefs))):
        seed_fact = random.choice(prefs)
        item = extract_preference_or_trait(seed_fact)["x"]
        name = pick_names(1)[0]

        ai_variant = ai_generate_conditional_variant(item, name) if USE_AI else None
        if ai_variant:
            ex = AttackExample(
                example_id=f"persona-conditional-{i}",
                source_dataset="persona-chat",
                attack_type="conditional_preference",
                contextual_statements=ai_variant["contextual_statements"],
                graded_question=ai_variant["graded_question"],
                ground_truth_answer=ai_variant["ground_truth_answer"],
                judge_target_failure_mode="summarization_dropped_condition",
                metadata={
                    "entity": name,
                    "seed_fact": seed_fact,
                    "condition_holds": ai_variant["condition_holds"],
                },
            )
            out.append(ex)
            continue

        ex = AttackExample(
            example_id=f"persona-conditional-{i}",
            source_dataset="persona-chat",
            attack_type="conditional_preference",
            contextual_statements=[
                f"{name} likes {item}, but only after 5pm.",
                "It's 3pm right now.",
            ],
            graded_question=f"Should I get {item} for {name} right now?",
            ground_truth_answer="No, the condition does not apply.",
            judge_target_failure_mode="summarization_dropped_condition",
            metadata={"entity": name, "seed_fact": seed_fact, "condition_holds": False},
        )
        if USE_AI:
            ex = ai_naturalize_example(ex)
        out.append(ex)

    return out


def build_fiction_real_examples(persona_facts: List[str], n: int = 8) -> List[AttackExample]:
    prefs = []
    for f in persona_facts:
        parsed = extract_preference_or_trait(f)
        if parsed and parsed["kind"] == "like":
            prefs.append(f)

    out = []
    fiction_templates = [
        "In a fantasy roleplay, {name} survives only on moonberries.",
        "In a Minecraft story, {name} was injured while mining underground.",
        "In an imaginary sci-fi setting, {name} pilots a smuggling ship.",
    ]

    for i in range(min(n, len(prefs))):
        seed_fact = random.choice(prefs)
        real_item = extract_preference_or_trait(seed_fact)["x"]
        name = pick_names(1)[0]

        ex = AttackExample(
            example_id=f"persona-fiction-real-{i}",
            source_dataset="persona-chat",
            attack_type="fiction_vs_real",
            contextual_statements=[
                random.choice(fiction_templates).format(name=name),
                f"In real life, {name} likes {real_item}.",
            ],
            graded_question=f"In real life, what does {name} like?",
            ground_truth_answer=f"In real life, {name} likes {real_item}.",
            judge_target_failure_mode="retrieval_or_reasoning_contaminated_by_fiction",
            metadata={"entity": name, "seed_fact": seed_fact},
        )
        if USE_AI:
            ex = ai_naturalize_example(ex)
        out.append(ex)

    return out


# -----------------------------
# Main pipeline
# -----------------------------
def build_small_attack_dataset() -> List[Dict[str, Any]]:
    persona_facts = sample_persona_facts(
        dataset_name="awsaf49/persona-chat",
        split="train",
        n_rows=120,
    )

    dataset: List[AttackExample] = []
    dataset += build_identity_examples(persona_facts, n=6)
    dataset += build_conditional_examples(persona_facts, n=6)
    dataset += build_narrative_correction_examples(persona_facts, n=6)
    dataset += build_multiple_preferences_examples(persona_facts, n=6)
    dataset += build_fiction_real_examples(persona_facts, n=6)

    return [asdict(x) for x in dataset]


if __name__ == "__main__":
    data = build_small_attack_dataset()

    print(f"Built {len(data)} examples")
    print(json.dumps(data[:3], indent=2))

    with open("small_ai_memory_attack_dataset.json", "w") as f:
        json.dump(data, f, indent=2)

    print("Saved to small_ai_memory_attack_dataset.json")