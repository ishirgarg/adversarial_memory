"""
Generate the long-hop chain dataset.

A K-hop chain is a sequence of K+1 short factual statements that link K+2
distinct anchors A → B → C → ... in a strict transitive chain. Statement i
introduces exactly one new edge (anchor i → anchor i+1):
  K=1:  2 statements, 3 anchors
  K=2:  3 statements, 4 anchors
  K=3:  4 statements, 5 anchors

Anchors can be proper-noun entities OR generic concepts (states, actions,
objects, preferences, places, attributes). First-person, conditional, causal,
temporal, and preference sentences are encouraged. Example general-reasoning
chain:
  "I eat apples when I'm bored."
  "When I'm bored I go to sleep."
  "When I sleep I have a dream."

The graded question references only A and asks for the terminal anchor, so the
question can be answered ONLY by chaining all K+1 statements end-to-end.

Pipeline:
  1) Generate chains in batches with gpt-5-mini, accumulating one-line
     "prior chain summaries" so the model picks novel narratives.
  2) Validate each chain locally: shape, anchor uniqueness within the chain,
     all anchors appear in their respective facts.
  3) Cross-chain conflict / similarity check via gpt-5-mini to drop chains
     whose narrative paraphrases or contradicts another chain.
  4) MinHash LSH deduplication at threshold=0.7 across every fact in the
     dataset. If any fact in a chain duplicates a fact already kept, the
     entire chain is dropped.
  5) Truncate to exactly --target-per-hop chains per hop count and write:
       datasets/long_hop/long_hop_chains.csv       (one row per chain)
       datasets/long_hop/long_hop_chains_meta.json (generation config)
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from dataset_utils.dedup import deduplicate  # noqa: E402

MODEL_NAME = "gpt-5-mini"
MAX_RETRIES = 4
RETRY_BACKOFF_S = 1.0

DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "datasets" / "long_hop" / "long_hop_chains.csv"
DEFAULT_TARGET_PER_HOP = 40
DEFAULT_OVERSAMPLE_PER_HOP = 55
DEFAULT_BATCH_SIZE = 8
DEFAULT_DEDUP_THRESHOLD = 0.7

MAX_FACTS_PER_CHAIN = 4   # K=3 → 4 facts
MAX_CHAIN_ANCHORS = 5     # K=3 → 5 anchors

CSV_COLUMNS = (
    ["id", "hop_count"]
    + [f"fact_{i}" for i in range(1, MAX_FACTS_PER_CHAIN + 1)]
    + [f"chain_{i}" for i in range(1, MAX_CHAIN_ANCHORS + 1)]
    + ["graded_question", "ground_truth_answer"]
)

# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------


def _chat_json(client: OpenAI, system: str, user: str) -> Tuple[Any, int, int]:
    """Call gpt-5-mini with JSON object response_format and basic retry."""
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            data = json.loads(content)
            usage = resp.usage
            in_tok = usage.prompt_tokens if usage else 0
            out_tok = usage.completion_tokens if usage else 0
            return data, in_tok, out_tok
        except Exception as exc:
            last_err = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * attempt)
    raise RuntimeError(f"gpt-5-mini call failed after {MAX_RETRIES} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Generation prompts
# ---------------------------------------------------------------------------

GEN_SYSTEM = """You are constructing a benchmark of multi-hop reasoning chains.

Each chain consists of K+1 short factual statements that strictly link K+2
distinct anchors A -> B -> C -> ... in a transitive chain. Statement i must
relate anchor i to anchor i+1 (and no other anchors). The chain must support a
single multi-hop question: starting from A, answering the question requires
chaining through every statement to reach the terminal anchor.

Hard rules — every chain must satisfy ALL of these:
1. EXACTLY K+1 statements per chain. Each statement is a single declarative
   English sentence, max ~16 words, no commas-separated multi-claims.
2. Statement i mentions ONLY anchor i and anchor i+1, plus an explicit relation
   word — a verb ("founded", "is owned by"), a conditional ("when", "whenever",
   "if"), a causal ("because", "leads to", "makes me"), a temporal ("after",
   "before"), or a preference ("I do X when Y"). It must not mention any other
   anchor from the chain.
2b. EVERY STATEMENT MUST BE SELF-CONTAINED. A reader will encounter each
   statement in isolation, with no access to the other statements in the chain,
   so each statement must be fully interpretable on its own. Concretely:
     - Do NOT use third-person pronouns ("he", "she", "it", "they", "him",
       "her", "them", "his", "hers", "their", "theirs", "its") to refer back
       to an anchor that only appears in a different statement. If you need to
       refer to a proper-noun anchor again, repeat its name.
     - Do NOT use referential phrases like "the company", "that town", "this
       book", "the same person" that depend on another statement to resolve.
     - Do NOT use pronouns inside a statement to refer to an anchor that does
       not appear literally in that same statement.
     - First-person ("I", "me", "my", "myself") IS allowed everywhere — the
       speaker is implicit and shared across the dataset.
     - Within one statement, a pronoun whose referent appears literally in
       that same statement is fine ("Watering succulents keeps them healthy"
       is OK because "succulents" appears in the same sentence).
3. K+2 anchors total per chain. Anchors can be ANY of:
     - Proper-noun entities — invented names like "Yermil Vasque",
       "Korolan Foundry", "the Tirvana Basin".
     - Generic concepts — states ("bored", "tired"), actions ("eat apples",
       "go for a run"), objects ("a dream", "popcorn"), preferences, places,
       attributes, moods, routines.
   First-person ("I ...") sentences are encouraged. Mix the two styles freely.
4. Within a single chain, all K+2 anchors must be distinct (case-insensitive).
5. Vary the relation patterns across the K+1 statements within one chain — do
   not reuse the same conditional or verb template back-to-back.
6. The graded question must reference ONLY anchor 1 (the head) and ask about
   the terminal anchor (the last in the chain), without ever naming any
   intermediate anchor. The question should be a single natural English
   sentence and have a unique correct answer given the K+1 statements.
7. ground_truth_answer must equal the terminal anchor exactly (or its shortest
   natural form — e.g. drop a leading "the" only if the canonical phrase has
   no article).
8. Across chains in this batch, AVOID retelling the same narrative as anything
   in PRIOR CHAIN SUMMARIES (provided in the user message). Generic words like
   "sleep" or "bored" may repeat across chains, but a chain that paraphrases
   another chain's storyline must not be produced.

Output JSON only — no commentary."""


def _examples_block(hop_count: int) -> str:
    """Return four in-context examples for the given hop count.

    Mix: 1 entity-style example + 3 general-reasoning examples (causal /
    conditional / temporal / preference, often first-person).
    """
    if hop_count == 1:  # K=1 → 2 facts, 3 anchors
        examples = [
            {
                "tag": "entity",
                "facts": [
                    "Marenza Velloux founded Strophien Atelier.",
                    "Strophien Atelier operates from the town of Treskellin.",
                ],
                "answer_chain": ["Marenza Velloux", "Strophien Atelier", "Treskellin"],
                "graded_question": "In what town does the atelier founded by Marenza Velloux operate?",
                "ground_truth_answer": "Treskellin",
            },
            {
                "tag": "general (causal, first-person)",
                "facts": [
                    "I eat apples when I'm bored.",
                    "When I'm bored I go to sleep.",
                ],
                "answer_chain": ["eat apples", "bored", "go to sleep"],
                "graded_question": "What do I do after I eat apples?",
                "ground_truth_answer": "go to sleep",
            },
            {
                "tag": "general (conditional)",
                "facts": [
                    "When it rains I make tea.",
                    "Whenever I make tea I read a book.",
                ],
                "answer_chain": ["rains", "make tea", "read a book"],
                "graded_question": "What do I end up doing when it rains?",
                "ground_truth_answer": "read a book",
            },
            {
                "tag": "general (temporal)",
                "facts": [
                    "After my morning run I take a cold shower.",
                    "A cold shower puts me in a focused mood.",
                ],
                "answer_chain": ["morning run", "cold shower", "focused mood"],
                "graded_question": "What mood am I in after my morning run?",
                "ground_truth_answer": "focused mood",
            },
        ]
    elif hop_count == 2:  # K=2 → 3 facts, 4 anchors
        examples = [
            {
                "tag": "entity",
                "facts": [
                    "Quenn Idriguay is a senior brewer at Hollomere Cask.",
                    "Hollomere Cask operates from the town of Treskellin.",
                    "Treskellin sits along the river Vellimar.",
                ],
                "answer_chain": ["Quenn Idriguay", "Hollomere Cask", "Treskellin", "river Vellimar"],
                "graded_question": "Along which river lies the town containing the brewery where Quenn Idriguay works?",
                "ground_truth_answer": "river Vellimar",
            },
            {
                "tag": "general (causal, first-person)",
                "facts": [
                    "I eat apples when I'm bored.",
                    "When I'm bored I go to sleep.",
                    "When I sleep I have vivid dreams.",
                ],
                "answer_chain": ["eat apples", "bored", "sleep", "vivid dreams"],
                "graded_question": "What do I end up having when I eat apples?",
                "ground_truth_answer": "vivid dreams",
            },
            {
                "tag": "general (conditional)",
                "facts": [
                    "On rainy days I brew strong coffee.",
                    "Strong coffee makes me restless.",
                    "When I'm restless I rearrange my bookshelf.",
                ],
                "answer_chain": ["rainy days", "strong coffee", "restless", "rearrange my bookshelf"],
                "graded_question": "What do I do on rainy days?",
                "ground_truth_answer": "rearrange my bookshelf",
            },
            {
                "tag": "general (temporal)",
                "facts": [
                    "Long flights leave me dehydrated.",
                    "When I'm dehydrated I crave watermelon.",
                    "Eating watermelon always cheers me up.",
                ],
                "answer_chain": ["long flights", "dehydrated", "crave watermelon", "cheers me up"],
                "graded_question": "What is the eventual emotional effect long flights have on me?",
                "ground_truth_answer": "cheers me up",
            },
        ]
    elif hop_count == 3:  # K=3 → 4 facts, 5 anchors
        examples = [
            {
                "tag": "entity",
                "facts": [
                    "Vespan Khorrid wrote the novel Pale Equinox of Mirru.",
                    "Pale Equinox of Mirru was published by Drannot House.",
                    "Drannot House is headquartered in the city of Yepelmir.",
                    "Yepelmir lies in the province of Korunda.",
                ],
                "answer_chain": [
                    "Vespan Khorrid",
                    "Pale Equinox of Mirru",
                    "Drannot House",
                    "Yepelmir",
                    "Korunda",
                ],
                "graded_question": "In what province is the publisher of the novel Vespan Khorrid wrote ultimately based?",
                "ground_truth_answer": "Korunda",
            },
            {
                "tag": "general (causal, first-person)",
                "facts": [
                    "I eat apples when I'm bored.",
                    "When I'm bored I go to sleep.",
                    "When I sleep I have a dream.",
                    "Every dream I have leaves me curious about the future.",
                ],
                "answer_chain": ["eat apples", "bored", "sleep", "dream", "curious about the future"],
                "graded_question": "What does eating apples eventually leave me feeling?",
                "ground_truth_answer": "curious about the future",
            },
            {
                "tag": "general (conditional)",
                "facts": [
                    "When the deadline is near I skip lunch.",
                    "Skipping lunch leaves me lightheaded.",
                    "When I'm lightheaded I take a long walk.",
                    "Long walks make me sentimental.",
                ],
                "answer_chain": [
                    "deadline is near",
                    "skip lunch",
                    "lightheaded",
                    "long walk",
                    "sentimental",
                ],
                "graded_question": "What mood overtakes me when a deadline is near?",
                "ground_truth_answer": "sentimental",
            },
            {
                "tag": "general (preference)",
                "facts": [
                    "If a movie is a horror film I watch it alone.",
                    "Watching alone I always make popcorn.",
                    "Popcorn makes me extra thirsty.",
                    "When I'm extra thirsty I finally drink water.",
                ],
                "answer_chain": [
                    "horror film",
                    "watch alone",
                    "popcorn",
                    "extra thirsty",
                    "drink water",
                ],
                "graded_question": "What do I end up doing whenever I see a horror film?",
                "ground_truth_answer": "drink water",
            },
        ]
    else:
        raise ValueError(f"unsupported hop_count: {hop_count}")

    lines: List[str] = []
    for idx, ex in enumerate(examples, start=1):
        facts_block = "\n    ".join(f"\"{f}\"" for f in ex["facts"])
        chain_block = ", ".join(f"\"{a}\"" for a in ex["answer_chain"])
        lines.append(
            f"Example {idx} ({ex['tag']}):\n"
            f"  facts: [\n    {facts_block}\n  ]\n"
            f"  answer_chain: [{chain_block}]\n"
            f"  graded_question: \"{ex['graded_question']}\"\n"
            f"  ground_truth_answer: \"{ex['ground_truth_answer']}\""
        )
    return "\n\n".join(lines)


def gen_user_prompt(
    hop_count: int,
    batch_size: int,
    prior_summaries: List[str],
) -> str:
    if prior_summaries:
        summaries_block = "\n".join(f"- {s}" for s in prior_summaries)
    else:
        summaries_block = "(none yet)"

    n_facts = hop_count + 1
    n_anchors = hop_count + 2

    return f"""Produce a JSON object with exactly {batch_size} chains, each of HOP COUNT K = {hop_count}.

Each chain must follow ALL hard rules from the system prompt. To recap for K = {hop_count}:
- Exactly {n_facts} statements.
- Exactly {n_anchors} distinct anchors, listed in answer_chain in chain order
  (head first, terminal last).
- Statement i links anchor i and anchor i+1 only.
- Every anchor must appear LITERALLY (case-insensitive substring) in the
  fact(s) it belongs to.
- EVERY STATEMENT MUST BE SELF-CONTAINED — a reader will see each fact in
  isolation. No third-person pronouns (he/she/it/they/him/her/them/his/her/
  their/its) referring to anchors from other statements; repeat proper-noun
  anchors verbatim instead. No referential phrases ("the company", "that
  town") that depend on another fact to resolve. First-person ("I", "me",
  "my") is always fine.
  BAD:  "Lina buys a ticket and travels to the coast." / "Traveling to the
        coast means she visits Seabright."   ← "she" needs fact 1 to resolve.
  GOOD: "Lina buys a ticket and travels to the coast." / "Traveling to the
        coast means Lina visits Seabright."
- The graded_question references only anchor 1 (the head) and asks about the
  terminal anchor. The question must be unanswerable from any single statement
  alone.
- ground_truth_answer is the canonical written form of the terminal anchor.

In-context examples (mix of styles — produce a similar mix):

{_examples_block(hop_count)}

PRIOR CHAIN SUMMARIES (avoid retelling these storylines; pick fresh
narratives — generic words like "sleep" or "tea" may repeat, but the chain
arc must be new):
{summaries_block}

Output schema (JSON object):
{{
  "chains": [
    {{
      "facts": ["..."],
      "answer_chain": ["..."],
      "graded_question": "...",
      "ground_truth_answer": "..."
    }},
    ...  // exactly {batch_size} chains
  ]
}}

Generate now. Aim for diversity — across the {batch_size} chains, vary the
sentence patterns (causal / conditional / temporal / preference / entity-
relation) and the topic domains (food, mood, weather, routine, work, hobbies,
people↔organizations, books↔authors, places↔regions, etc.). Roughly 1 in 4
chains may be entity-style; the rest should be general-reasoning."""


# ---------------------------------------------------------------------------
# Local validation
# ---------------------------------------------------------------------------


_PUNCT_RE = re.compile(r"[\"'.,!?;:()\[\]]")

# Singular third-person human pronouns. In our chain format every fact is read
# in isolation, so these must never appear — proper nouns must be repeated
# verbatim instead. ("it"/"its"/"they"/"them"/"their" are NOT flagged: they
# legitimately refer to in-fact objects like "succulents" or "popcorn".)
_BANNED_PRONOUNS = {
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
}
_TOKEN_RE = re.compile(r"\b([a-z']+)\b")


def _norm_for_match(s: str) -> str:
    """Lowercase + strip punctuation + collapse whitespace, for substring checks."""
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _has_banned_pronoun(fact: str) -> bool:
    """True if `fact` contains a singular third-person human pronoun.

    Self-containment requires repeating the proper noun instead — a reader
    who sees this fact in isolation cannot resolve "she"/"he"/etc.
    """
    return any(tok in _BANNED_PRONOUNS for tok in _TOKEN_RE.findall(fact.lower()))


def _validate_chain(raw: Dict[str, Any], hop_count: int) -> Optional[Dict[str, Any]]:
    """Return a normalized chain dict if it passes structural checks, else None.

    A K-hop chain has K+1 facts and K+2 anchors.
    """
    try:
        facts = raw["facts"]
        chain = raw["answer_chain"]
        question = raw["graded_question"]
        gt = raw["ground_truth_answer"]
    except (KeyError, TypeError):
        return None

    n_facts = hop_count + 1
    n_anchors = hop_count + 2

    if not isinstance(facts, list) or len(facts) != n_facts:
        return None
    if not isinstance(chain, list) or len(chain) != n_anchors:
        return None
    if not all(isinstance(f, str) and f.strip() for f in facts):
        return None
    if not all(isinstance(e, str) and e.strip() for e in chain):
        return None
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(gt, str) or not gt.strip():
        return None

    norm_chain = [_norm_for_match(e) for e in chain]
    if len(set(norm_chain)) != len(norm_chain):
        return None  # duplicate anchor within the chain

    # Self-containment: each fact must be readable in isolation. Reject any
    # fact that uses a singular third-person human pronoun — those force the
    # reader to look up an antecedent in another fact.
    for f in facts:
        if _has_banned_pronoun(f):
            return None

    # Each anchor j (0-indexed) must appear in fact j-1 (if j >= 1) and fact j
    # (if j <= n_facts - 1 == hop_count). Substring match on normalized text.
    fact_norm = [_norm_for_match(f) for f in facts]
    for j, ent in enumerate(chain):
        ent_l = _norm_for_match(ent)
        appears_in: List[int] = []
        if j - 1 >= 0:
            appears_in.append(j - 1)
        if j <= hop_count:  # j-th anchor links to fact[j] when j <= K
            appears_in.append(j)
        if not any(ent_l in fact_norm[k] for k in appears_in):
            return None

    # Question must mention the head anchor and NOT any intermediate anchor.
    # Head check is lenient: strict substring OR the head's last ≥3-char token
    # must appear in the question, to tolerate morphological variants
    # (e.g., anchor "eat apples" -> question "eating apples").
    q_norm = _norm_for_match(question)
    head_l = _norm_for_match(chain[0])
    if head_l not in q_norm:
        head_tokens = [t for t in head_l.split() if len(t) >= 3]
        if not head_tokens or head_tokens[-1] not in q_norm:
            return None
    for j in range(1, hop_count + 1):  # intermediate anchors are 1..K
        if _norm_for_match(chain[j]) in q_norm:
            return None

    # Ground truth refers to the terminal anchor.
    terminal_l = _norm_for_match(chain[-1])
    gt_l = _norm_for_match(gt)
    if terminal_l not in gt_l and gt_l not in terminal_l:
        return None

    return {
        "facts": [f.strip() for f in facts],
        "answer_chain": [e.strip() for e in chain],
        "graded_question": question.strip(),
        "ground_truth_answer": gt.strip(),
    }


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------


def generate_for_hop(
    client: OpenAI,
    hop_count: int,
    target_count: int,
    batch_size: int,
    prior_summaries: List[str],
    rng: random.Random,
    token_counter: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Generate at least target_count valid chains of the given hop_count.

    Appends a one-line summary of every accepted chain to `prior_summaries`
    (mutated in place) so subsequent calls steer toward novel narratives.
    """
    accepted: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = max(8, (target_count // max(1, batch_size)) * 4 + 6)

    while len(accepted) < target_count and attempts < max_attempts:
        attempts += 1
        recent = prior_summaries[-30:]  # cap prompt size
        prompt = gen_user_prompt(hop_count, batch_size, recent)
        try:
            data, in_tok, out_tok = _chat_json(client, GEN_SYSTEM, prompt)
        except Exception as exc:
            print(f"  [hop={hop_count}] batch {attempts}: API error ({exc!r}); retrying.")
            continue
        token_counter["in"] += in_tok
        token_counter["out"] += out_tok

        chains = data.get("chains") if isinstance(data, dict) else None
        if not isinstance(chains, list):
            print(f"  [hop={hop_count}] batch {attempts}: missing 'chains' list, skipping.")
            continue

        kept_in_batch = 0
        for raw in chains:
            chain = _validate_chain(raw, hop_count)
            if chain is None:
                continue
            accepted.append(chain)
            summary = (
                f"{chain['answer_chain'][0]} -> ... -> {chain['ground_truth_answer']}"
            )
            prior_summaries.append(summary)
            kept_in_batch += 1
            if len(accepted) >= target_count:
                break
        print(
            f"  [hop={hop_count}] batch {attempts}: accepted {kept_in_batch}/"
            f"{len(chains)} (running total {len(accepted)}/{target_count})"
        )

    if len(accepted) < target_count:
        print(
            f"  WARNING: hop={hop_count} only produced {len(accepted)} valid chains "
            f"after {attempts} batches (target {target_count})."
        )
    return accepted


# ---------------------------------------------------------------------------
# Cross-chain conflict / similarity check
# ---------------------------------------------------------------------------

CONFLICT_SYSTEM = """You are auditing a small set of made-up reasoning chains for cross-chain interference.

Two chains "interfere" if storing both into a single shared memory store would
corrupt reasoning for either chain. Interference occurs when ANY of the
following hold between facts in different chains:
- CONTRADICTION: the facts make incompatible claims about the same anchor
  (e.g., one chain says "I eat apples when bored", another says "I never eat
  apples").
- SHARED DISTINCTIVE ANCHOR: a distinctive proper-noun or distinctive composite
  phrase appears in two different chains (paraphrase or near-spelling counts).
  Generic single-word concepts ("sleep", "bored", "tea") repeating across
  chains is fine and does NOT count as interference.
- OVERTLY SIMILAR NARRATIVE: two chains tell essentially the same storyline
  with the same anchors in the same role — e.g. "X is owned by Y" and
  "Y owns X", or "When I'm bored I sleep / sleep gives me a dream" appearing
  twice with only minor word swaps.

Return the list of chain IDs to DROP to eliminate all interference. When two
chains conflict, drop only ONE of them (your choice). Be precise; do not flag
chains that merely share generic concepts or common verbs.

Output JSON only."""


def conflict_check(
    client: OpenAI,
    chains: List[Dict[str, Any]],
    batch_size: int,
    token_counter: Dict[str, int],
) -> set:
    """Return the set of chain IDs to drop. Run a global pass plus overlapping
    sliding windows to catch subtle misses."""
    if len(chains) <= 1:
        return set()

    drop: set = set()

    def _as_block(subset: List[Dict[str, Any]]) -> str:
        return json.dumps(
            [
                {"id": c["id"], "facts": c["facts"]}
                for c in subset
                if c["id"] not in drop
            ],
            ensure_ascii=False,
            indent=2,
        )

    user = (
        "CHAINS (each has an `id` and a list of `facts`):\n"
        f"{_as_block(chains)}\n\n"
        "Output JSON:\n"
        "{\n"
        "  \"to_drop\": [\"<chain_id>\", ...],\n"
        "  \"reason_per_drop\": {\"<chain_id>\": \"<one-line reason>\", ...}\n"
        "}\n"
        "If there are no conflicts, return {\"to_drop\": [], \"reason_per_drop\": {}}."
    )
    try:
        data, in_tok, out_tok = _chat_json(client, CONFLICT_SYSTEM, user)
        token_counter["in"] += in_tok
        token_counter["out"] += out_tok
        ids = data.get("to_drop") or [] if isinstance(data, dict) else []
        for cid in ids:
            if isinstance(cid, str):
                drop.add(cid)
        reasons = data.get("reason_per_drop") or {} if isinstance(data, dict) else {}
        for cid, reason in (reasons or {}).items():
            if cid in drop:
                print(f"  conflict: drop {cid} — {reason}")
    except Exception as exc:
        print(f"  conflict-check global pass failed: {exc!r}; proceeding without it.")

    remaining = [c for c in chains if c["id"] not in drop]
    if len(remaining) > batch_size:
        for start in range(0, len(remaining), batch_size // 2):
            window = remaining[start : start + batch_size]
            if len(window) < 2:
                continue
            user2 = (
                "CHAINS (window — each has an `id` and a list of `facts`):\n"
                f"{json.dumps([{'id': c['id'], 'facts': c['facts']} for c in window], ensure_ascii=False, indent=2)}\n\n"
                "Output JSON:\n"
                "{\n"
                "  \"to_drop\": [\"<chain_id>\", ...],\n"
                "  \"reason_per_drop\": {\"<chain_id>\": \"<one-line reason>\", ...}\n"
                "}\n"
                "If no conflicts, return {\"to_drop\": [], \"reason_per_drop\": {}}."
            )
            try:
                data, in_tok, out_tok = _chat_json(client, CONFLICT_SYSTEM, user2)
                token_counter["in"] += in_tok
                token_counter["out"] += out_tok
                ids = data.get("to_drop") or [] if isinstance(data, dict) else []
                reasons = data.get("reason_per_drop") or {} if isinstance(data, dict) else {}
                for cid in ids:
                    if isinstance(cid, str) and cid not in drop:
                        drop.add(cid)
                        reason = reasons.get(cid, "(no reason)")
                        print(f"  conflict (window {start}): drop {cid} — {reason}")
            except Exception as exc:
                print(f"  conflict-check window {start} failed: {exc!r}; skipping.")

    return drop


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_csv(out_path: Path, chains: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for c in chains:
            row = {col: "" for col in CSV_COLUMNS}
            row["id"] = c["id"]
            row["hop_count"] = c["hop_count"]
            for i, fact in enumerate(c["facts"], start=1):
                if i > MAX_FACTS_PER_CHAIN:
                    raise ValueError(
                        f"chain {c['id']} has {len(c['facts'])} facts (>{MAX_FACTS_PER_CHAIN})"
                    )
                row[f"fact_{i}"] = fact
            for i, anc in enumerate(c["answer_chain"], start=1):
                if i > MAX_CHAIN_ANCHORS:
                    raise ValueError(
                        f"chain {c['id']} has {len(c['answer_chain'])} anchors (>{MAX_CHAIN_ANCHORS})"
                    )
                row[f"chain_{i}"] = anc
            row["graded_question"] = c["graded_question"]
            row["ground_truth_answer"] = c["ground_truth_answer"]
            writer.writerow(row)


def write_metadata(out_path: Path, metadata: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the long-hop chain dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help="Output CSV path (overwritten).",
    )
    parser.add_argument(
        "--output-meta",
        type=str,
        default=None,
        help="Output metadata JSON path (defaults to <csv-stem>_meta.json).",
    )
    parser.add_argument(
        "--target-per-hop",
        type=int,
        default=DEFAULT_TARGET_PER_HOP,
        help="Final number of chains per hop count.",
    )
    parser.add_argument(
        "--oversample-per-hop",
        type=int,
        default=DEFAULT_OVERSAMPLE_PER_HOP,
        help="Generate this many per hop count, then filter down.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Chains generated per LLM call.",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=DEFAULT_DEDUP_THRESHOLD,
        help="MinHash LSH Jaccard threshold for fact-level dedup.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        sys.exit("Error: set OPENAI_API_KEY env var or pass --api-key.")

    rng = random.Random(args.seed)
    client = OpenAI(api_key=api_key)
    token_counter: Dict[str, int] = {"in": 0, "out": 0}
    prior_summaries: List[str] = []

    out_csv = Path(args.output_csv)
    out_meta = (
        Path(args.output_meta)
        if args.output_meta
        else out_csv.with_name(out_csv.stem + "_meta.json")
    )

    print("=" * 70)
    print("LONG-HOP CHAIN DATASET GENERATION")
    print("=" * 70)
    print(f"  Model:              {MODEL_NAME}")
    print(f"  Target per hop:     {args.target_per_hop}")
    print(f"  Oversample per hop: {args.oversample_per_hop}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Dedup threshold:    {args.dedup_threshold}")
    print(f"  Seed:               {args.seed}")
    print(f"  Output CSV:         {out_csv}")
    print(f"  Output meta JSON:   {out_meta}")
    print("=" * 70)

    all_chains: List[Dict[str, Any]] = []
    for hop in (1, 2, 3):
        print(f"\n>>> Generating {args.oversample_per_hop} chains for hop_count={hop}...")
        chains = generate_for_hop(
            client,
            hop_count=hop,
            target_count=args.oversample_per_hop,
            batch_size=args.batch_size,
            prior_summaries=prior_summaries,
            rng=rng,
            token_counter=token_counter,
        )
        for i, chain in enumerate(chains):
            chain["id"] = f"longhop-{hop}hop-{i+1:03d}"
            chain["hop_count"] = hop
        all_chains.extend(chains)
        print(f">>> hop={hop}: collected {len(chains)} chains.")

    print(f"\nTotal raw chains generated: {len(all_chains)}")

    # ── Cross-chain conflict / similarity check ───────────────────────────────
    print("\n>>> Running cross-chain conflict / similarity check (gpt-5-mini)...")
    drop_ids = conflict_check(client, all_chains, args.batch_size, token_counter)
    print(f"   conflict-check dropped {len(drop_ids)} chains.")
    survivors = [c for c in all_chains if c["id"] not in drop_ids]
    print(f"   survivors after conflict check: {len(survivors)}")

    # ── MinHash LSH dedup at threshold over every individual fact ─────────────
    print(
        "\n>>> Running fact-level MinHash LSH dedup (threshold="
        f"{args.dedup_threshold})..."
    )
    fact_rows: List[Dict[str, Any]] = []
    for chain in survivors:
        for f in chain["facts"]:
            fact_rows.append({"chain_id": chain["id"], "fact": f})
    deduped = deduplicate(fact_rows, key=lambda r: r["fact"], threshold=args.dedup_threshold)
    kept_chain_facts: Dict[str, int] = {}
    for r in deduped:
        kept_chain_facts[r["chain_id"]] = kept_chain_facts.get(r["chain_id"], 0) + 1
    expected_facts = {c["id"]: len(c["facts"]) for c in survivors}
    surviving_chains = [
        c for c in survivors
        if kept_chain_facts.get(c["id"], 0) == expected_facts[c["id"]]
    ]
    print(
        f"   dedup dropped {len(survivors) - len(surviving_chains)} chains "
        f"with a near-duplicate fact (Jaccard > {args.dedup_threshold})."
    )
    print(f"   survivors after dedup: {len(surviving_chains)}")

    # ── Truncate to target_per_hop per hop_count ──────────────────────────────
    by_hop: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: []}
    for c in surviving_chains:
        by_hop.setdefault(c["hop_count"], []).append(c)
    final: List[Dict[str, Any]] = []
    deficits: Dict[int, int] = {}
    for hop in (1, 2, 3):
        bucket = by_hop.get(hop, [])
        if len(bucket) < args.target_per_hop:
            deficits[hop] = args.target_per_hop - len(bucket)
        final.extend(bucket[: args.target_per_hop])
    print("\nFinal chain counts: " + ", ".join(
        f"hop={h}: {min(len(by_hop.get(h, [])), args.target_per_hop)}" for h in (1, 2, 3)
    ))
    if deficits:
        print(f"  WARNING: deficits {deficits} — re-run with a higher --oversample-per-hop.")

    # ── Renumber IDs sequentially within each hop ─────────────────────────────
    counts: Dict[int, int] = {1: 0, 2: 0, 3: 0}
    for c in final:
        counts[c["hop_count"]] += 1
        c["id"] = f"longhop-{c['hop_count']}hop-{counts[c['hop_count']]:03d}"

    # ── Write out ─────────────────────────────────────────────────────────────
    write_csv(out_csv, final)
    metadata = {
        "model": MODEL_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hop_counts": {str(h): sum(1 for c in final if c["hop_count"] == h) for h in (1, 2, 3)},
        "dedup_threshold": args.dedup_threshold,
        "oversample_per_hop": args.oversample_per_hop,
        "target_per_hop": args.target_per_hop,
        "seed": args.seed,
        "tokens": token_counter,
        "csv_path": str(out_csv),
        "csv_columns": CSV_COLUMNS,
        "max_facts_per_chain": MAX_FACTS_PER_CHAIN,
        "max_chain_anchors": MAX_CHAIN_ANCHORS,
    }
    write_metadata(out_meta, metadata)

    print(f"\nWrote {len(final)} chains to {out_csv}")
    print(f"Wrote metadata to {out_meta}")
    print(f"\nGeneration tokens: in={token_counter['in']:,} out={token_counter['out']:,}")
    print("Done.")


if __name__ == "__main__":
    main()
