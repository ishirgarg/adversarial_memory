"""
Generate the long-hop chain dataset.

A K-hop chain is a sequence of K+1 short factual statements that link K+2
distinct anchors A → B → C → ... in a strict transitive chain. Statement i
introduces exactly one new edge (anchor i → anchor i+1):
  K=1:  2 statements, 3 anchors
  K=2:  3 statements, 4 anchors
  K=3:  4 statements, 5 anchors

Every fact must be SUBJECTIVE — a first-person opinion / preference / routine,
or a claim about a specific named person (their habits, opinions, moods).
Facts must NEVER be encyclopedic world claims that could in principle be
looked up (no made-up companies, towns, rivers, books, or geographic
relations). First-person, conditional, causal, temporal, and preference
sentences are encouraged. Example general-reasoning chain:
  "I eat apples when I'm bored."
  "When I'm bored I go to sleep."
  "When I sleep I have a dream."

Do NOT use this example in your generation.
The graded question references only A and asks for the terminal anchor, so the
question can be answered ONLY by chaining all K+1 statements end-to-end.

Pipeline:
  1) Generate chains (without answer choices) in batches with gpt-5,
     accumulating one-line "prior chain summaries" so the model picks novel
     narratives.
  2) Validate each chain locally: shape, anchor uniqueness within the chain,
     all anchors appear in their respective facts.
  3) Cross-chain conflict / similarity check via gpt-5 to drop chains whose
     narrative paraphrases or contradicts another chain.
  4) MinHash LSH deduplication at threshold=0.7 across every fact in the
     dataset. If any fact in a chain duplicates a fact already kept, the
     entire chain is dropped.
  5) For each surviving chain, generate 4 distractor options with gpt-5,
     constrained to be realistic (no absurd / surreal options) and
     orthogonal to every fact. Drop chains whose distractor generation fails
     validation after retries.
  6) Truncate to exactly --target-per-hop chains per hop count and write:
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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

MODEL_NAME = "gpt-5"
MAX_RETRIES = 4
RETRY_BACKOFF_S = 1.0

DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "datasets" / "long_hop" / "long_hop_chains.csv"
DEFAULT_TARGET_PER_HOP = 40
DEFAULT_OVERSAMPLE_PER_HOP = 55
DEFAULT_BATCH_SIZE = 8
DEFAULT_DEDUP_THRESHOLD = 0.7

MAX_FACTS_PER_CHAIN = 4   # K=3 → 4 facts
MAX_CHAIN_ANCHORS = 5     # K=3 → 5 anchors
NUM_CHOICES = 5           # 1 correct + 4 distractors
CHOICE_LETTERS = ["A", "B", "C", "D", "E"]
NUM_DISTRACTORS = NUM_CHOICES - 1

CSV_COLUMNS = (
    ["id", "hop_count"]
    + [f"fact_{i}" for i in range(1, MAX_FACTS_PER_CHAIN + 1)]
    + [f"chain_{i}" for i in range(1, MAX_CHAIN_ANCHORS + 1)]
    + ["graded_question", "ground_truth_answer"]
    + [f"choice_{letter.lower()}" for letter in CHOICE_LETTERS]
    + ["correct_choice"]
)

# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------


def _chat_json(client: OpenAI, system: str, user: str) -> Tuple[Any, int, int]:
    """Call the OpenAI chat model with JSON object response_format and basic retry."""
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
    raise RuntimeError(f"{MODEL_NAME} call failed after {MAX_RETRIES} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Generation prompts
# ---------------------------------------------------------------------------

GEN_SYSTEM = """You are constructing a benchmark of multi-hop reasoning chains.

Each chain consists of K+1 short factual statements that strictly link K+2
distinct anchors A -> B -> C -> ... in a transitive chain. Statement i must
relate anchor i to anchor i+1 (no other anchors, except the HEAD subject —
see rule 2 — which may recur as background). The chain must support a single
multi-hop question: starting from A, answering the question requires chaining
through every statement to reach the terminal anchor.

Hard rules — every chain must satisfy ALL of these:
1. EXACTLY K+1 statements per chain. Each statement is a single declarative
   English sentence, max ~16 words, no commas-separated multi-claims.
2. Statement i mentions anchor i and anchor i+1, plus an explicit relation
   word — a verb ("loves", "hates", "always picks"), a conditional ("when",
   "whenever", "if"), a causal ("because", "leads to", "makes me"), a temporal
   ("after", "before"), or a preference ("I do X when Y"). MIDDLE and TERMINAL
   anchors (anchors 2 .. K+2) must appear ONLY in the two facts that border
   them — they must not be named in any other fact.
   The HEAD subject is special and may recur as background in every fact:
     - First-person voice: "I/me/my" is the implicit speaker. The first-person
       speaker is NEVER literally listed in answer_chain — anchor 1 is instead
       a state/action/object the speaker relates to ("eat apples", "bored").
     - Named-person voice: the chain's subject is a single named person
       ("Diego", "Marisol Vega"). That person IS anchor 1, and the same name
       must appear in every subsequent fact as background. Pronouns
       ("he/she/his/her") are FINE within a single fact when the proper
       noun also appears in that same fact (see rule 2b).
   Pick ONE voice per chain (first-person OR one named person) and stay
   consistent.
2b. EVERY STATEMENT MUST BE SELF-CONTAINED. A reader will encounter each
   statement in isolation, with no access to the other statements in the
   chain, so each statement must be fully interpretable on its own.

   Pronouns INSIDE a statement are FINE — write natural English. The only
   rule is that every reference must resolve from the statement alone:
     - First-person ("I", "me", "my", "myself") is always fine — the speaker
       is implicit and shared across the dataset.
     - Third-person pronouns ("he", "she", "it", "they", "them", "his",
       "her", "their", "its", etc.) are fine when their antecedent appears
       LITERALLY in the SAME statement. Examples that are GOOD:
         "Diego loves Korean food because he finds it spicy."   (he ↔ Diego)
         "Marisol picks pop music whenever she is alone."       (she ↔ Marisol)
         "Watering succulents keeps them healthy."              (them ↔ succulents)

   What's forbidden is leaving a statement DEPENDENT on a different statement
   to resolve a reference. Do NOT:
     - Use a pronoun whose only possible antecedent appears in a DIFFERENT
       statement. If the proper noun isn't in this statement, repeat it.
     - Use referential phrases like "the company", "that town", "this book",
       "the same person" that depend on another statement to resolve.
3. K+2 anchors total per chain. Anchors should be SUBJECTIVE / PERSONAL content
   that cannot be looked up in an encyclopedia. Use anchors like:
     - States, moods, feelings ("bored", "anxious", "calm").
     - Actions, habits, routines ("eat apples", "skip lunch", "go for a run").
     - Preferences and opinions ("loves Korean food", "thinks pop music is
       overrated").
     - Concrete personal objects ("my bookshelf", "popcorn", "chamomile tea").
     - A single specific named person introduced as the chain's head subject
       ("Diego", "Marisol Vega") — only when that person is anchor 1 and the
       rest of the chain is about their preferences / habits / moods.
   Do NOT use impersonal entities (companies, towns, rivers, books,
   institutions, geographic regions). First-person ("I ...") sentences are
   encouraged. Mix first-person and named-person styles freely.

3b. Every fact must be SUBJECTIVE: an opinion, preference, mood, routine,
   habit, or relational claim about a specific person (the speaker or a
   named person). Facts must NOT be impersonal world claims.
     GOOD:
       - "I think Italian food is overrated."           (first-person opinion)
       - "Diego loves Korean food."                     (fact about a person)
       - "Marisol always feels nervous before tests."   (personal trait)
       - "When I'm tired I get grumpy."                 (first-person routine)
     BAD:
       - "Drannot House published the novel."           (impersonal fact)
       - "Yepelmir lies in the province of Korunda."    (geographic claim)
       - "Strophien Atelier operates from Treskellin."  (impersonal corporate)
4. Within a single chain, all K+2 anchors must be distinct (case-insensitive).
5. Vary the relation patterns across the K+1 statements within one chain — do
   not reuse the same conditional or verb template back-to-back.
6. The graded question must reference anchor 1 (the head) at least once by
   name and ask about the terminal anchor (the last in the chain), without
   ever naming any intermediate anchor. The question should read as a single
   natural English sentence and have a unique correct answer given the K+1
   statements. Natural pronouns are encouraged when they aid flow — e.g.,
   "What does Diego do when he is bored?" or "When I'm dehydrated, what
   mood do I end up in?" — provided every pronoun's antecedent is clear
   from the question itself.
7. ground_truth_answer must equal the terminal anchor exactly (or its shortest
   natural form — e.g. drop a leading "the" only if the canonical phrase has
   no article).
8. Across chains in this batch, AVOID retelling the same narrative as anything
   in PRIOR CHAIN SUMMARIES (provided in the user message). Generic words like
   "sleep" or "bored" may repeat across chains, but a chain that paraphrases
   another chain's storyline must not be produced.

Distractor options are produced in a separate downstream step — DO NOT include
any distractors / answer choices in your output here.

Output JSON only — no commentary."""


def _examples_block(hop_count: int) -> str:
    """Return four in-context examples for the given hop count.

    Mix: 1 named-person opinion chain + 3 first-person chains (causal /
    conditional / temporal / preference). All examples are subjective.
    """
    if hop_count == 1:  # K=1 → 2 facts, 3 anchors
        examples = [
            {
                "tag": "opinion about a named person",
                "facts": [
                    "Diego loves Korean food.",
                    "Korean food always leaves Diego thirsty.",
                ],
                "answer_chain": ["Diego", "Korean food", "thirsty"],
                "graded_question": "What physical feeling does Diego's favorite cuisine eventually cause?",
                "ground_truth_answer": "thirsty",
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
                "tag": "opinion about a named person",
                "facts": [
                    "Marisol thinks pop music is overrated.",
                    "Whenever pop music is on Marisol leaves the room.",
                    "When Marisol leaves the room Marisol ends up in a sour mood.",
                ],
                "answer_chain": ["Marisol", "pop music", "leaves the room", "sour mood"],
                "graded_question": "What mood does Marisol end up in because of the music genre Marisol dislikes?",
                "ground_truth_answer": "sour mood",
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
                "tag": "opinion about a named person",
                "facts": [
                    "Lina hates math classes.",
                    "Math classes always leave Lina frustrated.",
                    "When Lina is frustrated Lina makes chamomile tea.",
                    "Chamomile tea always makes Lina feel calmer.",
                ],
                "answer_chain": [
                    "Lina",
                    "math classes",
                    "frustrated",
                    "chamomile tea",
                    "calmer",
                ],
                "graded_question": "What feeling does Lina end up with because of the school subject Lina hates?",
                "ground_truth_answer": "calmer",
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
- DO NOT USE THE EXAMPLES I GIVE YOU IN YOUR CHAINS.
- Statement i links anchor i and anchor i+1. The HEAD subject (first-person
  "I" implicit, OR a named person at anchor 1) may also recur as background
  in every fact. Middle and terminal anchors must each appear ONLY in their
  two bordering facts.
- Every fact must be SUBJECTIVE — an opinion / preference / mood / routine /
  habit, either first-person ("I") or about a single named person
  ("Diego", "Marisol"). No encyclopedic / entity-relation facts.
- Every anchor must appear LITERALLY (case-insensitive substring) in the
  fact(s) it belongs to.
- EVERY STATEMENT MUST BE SELF-CONTAINED — a reader will see each fact in
  isolation, so each fact must be interpretable on its own. Pronouns INSIDE
  a single fact are fine when the antecedent is in the SAME fact ("Diego
  loves Korean food because he finds it spicy" — "he" ↔ Diego). What's
  forbidden is using a pronoun whose only antecedent appears in a DIFFERENT
  fact. No referential phrases ("the company", "that town") that depend on
  another fact to resolve. First-person ("I", "me", "my") is always fine.
  BAD:  "Lina buys a ticket and travels to the coast." / "Traveling to the
        coast means she visits Seabright."   ← "she" needs fact 1 to resolve.
  GOOD: "Lina buys a ticket and travels to the coast." / "Traveling to the
        coast means Lina visits Seabright."
  ALSO GOOD: "Lina buys a ticket because she loves travel."   ← "she" has
        antecedent "Lina" in the same fact.
- The graded_question references anchor 1 (the head) at least once by name
  and asks about the terminal anchor. The question must be unanswerable from
  any single statement alone. Natural pronouns referring to the head anchor
  are encouraged when they aid flow.
- ground_truth_answer is the canonical written form of the terminal anchor.
- DO NOT include any distractors / answer choices — they are produced in a
  separate downstream step.

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
sentence patterns (causal / conditional / temporal / preference / opinion)
and the topic domains (food, mood, weather, routine, work, hobbies, music,
travel, study, exercise, etc.). EVERY chain must be subjective: a first-
person chain (no named subject — implicit "I") OR a chain about a single
named person ("Diego", "Marisol Vega", "Lina") and that person's
opinions / habits / moods. Mix the two voices freely across the batch. Do
NOT produce encyclopedic / entity-relation chains (no companies, towns,
rivers, books, geographic features)."""


# ---------------------------------------------------------------------------
# Local validation
# ---------------------------------------------------------------------------


_PUNCT_RE = re.compile(r"[\"'.,!?;:()\[\]]")


def _norm_for_match(s: str) -> str:
    """Lowercase + strip punctuation + collapse whitespace, for substring checks."""
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


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
# Distractor generation (per chain, parallel)
# ---------------------------------------------------------------------------

DISTRACTOR_GEN_SYSTEM = """You write distractor options for a multi-hop reasoning multiple-choice question.

You receive:
- A list of FACTS forming a transitive reasoning chain (the graded answer
  requires chaining ALL facts).
- The GRADED_QUESTION (references only the head anchor; asks about the
  terminal anchor).
- The correct GROUND_TRUTH_ANSWER.

Produce EXACTLY 4 distractor options. Each distractor must satisfy ALL of the
following rules:

1. SAME-SHAPE PLAUSIBILITY. Match the correct answer in grammatical form,
   length range, and answer category. If the correct answer is a noun phrase
   naming a mood, every distractor is a noun phrase naming a mood. If the
   correct answer is a short verb phrase ("drink water"), every distractor is
   a short verb phrase of similar length and shape. Pronouns and articles
   that flow naturally with the question are fine — match the voice the
   question uses (e.g., if the question is "What does Diego do when he is
   bored?", distractors phrased as "he reorganizes his closet" or simply
   "reorganizes the closet" are both acceptable, as long as the distractor
   reads as a fluent answer to the question).

2. REALISTIC AND ORDINARY. Each distractor must name something a real person
   could plausibly feel, do, prefer, eat, or experience in everyday life.
   NO absurd, surreal, slapstick, joke, or comically random options. NO
   things almost no one actually does (e.g., "duel a swan", "memorize country
   capitals from memory", "argue with neighbors about constellations"). Pick
   ordinary moods, habits, hobbies, foods, or activities — the kind of
   answer a thoughtful peer might genuinely guess.

3. UNAMBIGUOUSLY WRONG. Must not be a paraphrase, synonym, sub-phrase,
   near-spelling, or otherwise overlapping with the correct answer or with
   any anchor / relation phrase that appears in any fact.

4. ORTHOGONAL TO EVERY FACT. A reader looking at any single fact in isolation
   must NOT be able to guess the distractor as a plausible "what comes next"
   or "natural consequence" via common-sense world knowledge. Avoid
   distractors that name typical effects, components, properties, or strong
   associations of any concept mentioned in any fact (e.g., if a fact
   mentions popcorn, do NOT pick a distractor about thirst, salt, or movies;
   if a fact mentions a cold shower, do NOT pick a distractor about feeling
   refreshed or shivery). Pick subject matter unrelated to every fact's
   topic.

5. DISTINCT. The four distractors must be distinct from each other
   (case-insensitive) and distinct from the correct answer.

Examples (note: realistic, ordinary, orthogonal):

CHAIN A facts:
  - "I eat apples when I'm bored."
  - "When I'm bored I go to sleep."
  - "When I sleep I have a dream."
  - "Every dream I have leaves me curious about the future."
GRADED_QUESTION: "What does eating apples eventually leave me feeling?"
CORRECT_ANSWER: "curious about the future"
GOOD distractors (ordinary moods/feelings, orthogonal to apples / sleep / dreams):
  - "nostalgic about old friendships"
  - "motivated to clean my apartment"
  - "indifferent toward upcoming holidays"
  - "satisfied with my routine"

CHAIN B facts:
  - "Marisol thinks pop music is overrated."
  - "Whenever pop music is on Marisol leaves the room."
  - "When Marisol leaves the room Marisol ends up in a sour mood."
GRADED_QUESTION: "What mood does Marisol end up in because of the music genre Marisol dislikes?"
CORRECT_ANSWER: "sour mood"
GOOD distractors (ordinary moods, no music / departure associations):
  - "a focused mood"
  - "a contemplative mood"
  - "a generous mood"
  - "a competitive mood"

Output JSON only — no commentary."""


def _validate_distractors(
    distractors: Any, chain: Dict[str, Any]
) -> Optional[List[str]]:
    """Structural + non-overlap checks. Return cleaned list or None if invalid."""
    if not isinstance(distractors, list) or len(distractors) != NUM_DISTRACTORS:
        return None
    if not all(isinstance(d, str) and d.strip() for d in distractors):
        return None
    cleaned = [d.strip() for d in distractors]
    norm = [_norm_for_match(d) for d in cleaned]
    if len(set(norm)) != len(norm):
        return None
    gt_norm = _norm_for_match(chain["ground_truth_answer"])
    for nd in norm:
        if nd == gt_norm or nd in gt_norm or gt_norm in nd:
            return None
    fact_norms = [_norm_for_match(f) for f in chain["facts"]]
    for nd in norm:
        if not nd:
            return None
        for fn in fact_norms:
            if nd in fn:
                return None
    return cleaned


def _distractor_user_prompt(chain: Dict[str, Any]) -> str:
    facts_block = "\n".join(f"- {f}" for f in chain["facts"])
    return (
        f"FACTS:\n{facts_block}\n\n"
        f"GRADED_QUESTION: {chain['graded_question']}\n\n"
        f"CORRECT_ANSWER: {chain['ground_truth_answer']}\n\n"
        "Produce 4 distractor options that satisfy every rule from the system "
        "message. Output JSON:\n"
        "{\"incorrect_options\": [\"...\", \"...\", \"...\", \"...\"]}"
    )


def generate_distractors(
    client: OpenAI,
    chain: Dict[str, Any],
    token_counter: Dict[str, int],
) -> Optional[List[str]]:
    """Generate 4 valid distractors for a chain. Return None if all retries fail."""
    user = _distractor_user_prompt(chain)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data, in_tok, out_tok = _chat_json(client, DISTRACTOR_GEN_SYSTEM, user)
        except Exception as exc:
            print(f"  distractor-gen {chain['id']} attempt {attempt}: API error ({exc!r})")
            continue
        token_counter["in"] += in_tok
        token_counter["out"] += out_tok
        if not isinstance(data, dict):
            continue
        validated = _validate_distractors(data.get("incorrect_options"), chain)
        if validated is not None:
            return validated
        print(
            f"  distractor-gen {chain['id']} attempt {attempt}: "
            "validation failed; retrying"
        )
    return None


def generate_all_distractors(
    client: OpenAI,
    chains: List[Dict[str, Any]],
    parallelism: int,
    token_counter: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Attach `incorrect_options` to each chain in place using up to `parallelism`
    concurrent OpenAI calls. Drop chains that fail after retries."""
    if not chains:
        return []
    workers = max(1, parallelism)
    token_lock = threading.Lock()

    def _work(chain: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[List[str]]]:
        local_tc: Dict[str, int] = {"in": 0, "out": 0}
        result = generate_distractors(client, chain, local_tc)
        with token_lock:
            token_counter["in"] += local_tc["in"]
            token_counter["out"] += local_tc["out"]
        return chain, result

    distractor_map: Dict[str, List[str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for chain, result in ex.map(_work, chains):
            if result is None:
                print(f"  distractor-gen drop {chain['id']}: failed after retries")
                continue
            distractor_map[chain["id"]] = result

    kept: List[Dict[str, Any]] = []
    for chain in chains:
        d = distractor_map.get(chain["id"])
        if d is None:
            continue
        chain["incorrect_options"] = d
        kept.append(chain)
    return kept


# ---------------------------------------------------------------------------
# Choice placement
# ---------------------------------------------------------------------------


def assign_choices(chain: Dict[str, Any], rng: random.Random) -> None:
    """Shuffle the four distractors with the correct answer and record choice
    columns + correct_choice letter on the chain dict in place."""
    options = list(chain["incorrect_options"]) + [chain["ground_truth_answer"]]
    rng.shuffle(options)
    correct_letter = CHOICE_LETTERS[options.index(chain["ground_truth_answer"])]
    chain["choices"] = {letter: text for letter, text in zip(CHOICE_LETTERS, options)}
    chain["correct_choice"] = correct_letter


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def render_question_with_choices(stem: str, choices: Dict[str, str]) -> str:
    """Append labelled A–E options to the question stem so the CSV's
    `graded_question` column is a self-contained multiple-choice prompt."""
    options_block = "\n".join(f"{letter}. {choices[letter]}" for letter in CHOICE_LETTERS)
    return f"{stem}\n\nOptions:\n{options_block}"


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
            row["graded_question"] = render_question_with_choices(
                c["graded_question"], c["choices"]
            )
            row["ground_truth_answer"] = c["ground_truth_answer"]
            for letter in CHOICE_LETTERS:
                row[f"choice_{letter.lower()}"] = c["choices"][letter]
            row["correct_choice"] = c["correct_choice"]
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
    print(f"\n>>> Running cross-chain conflict / similarity check ({MODEL_NAME})...")
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

    # ── Per-chain distractor generation (parallel) ───────────────────────────
    print(
        f"\n>>> Generating distractors per chain ({MODEL_NAME}, "
        f"up to {args.batch_size} parallel calls)..."
    )
    pre_count = len(surviving_chains)
    surviving_chains = generate_all_distractors(
        client, surviving_chains, args.batch_size, token_counter
    )
    print(
        f"   distractor-gen produced distractors for "
        f"{len(surviving_chains)}/{pre_count} chains."
    )

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

    # ── Randomize correct-choice placement (A..E) per chain ──────────────────
    choice_rng = random.Random(args.seed ^ 0xC401CE)
    for c in final:
        assign_choices(c, choice_rng)

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
        "num_choices": NUM_CHOICES,
        "choice_letters": CHOICE_LETTERS,
    }
    write_metadata(out_meta, metadata)

    print(f"\nWrote {len(final)} chains to {out_csv}")
    print(f"Wrote metadata to {out_meta}")
    print(f"\nGeneration tokens: in={token_counter['in']:,} out={token_counter['out']:,}")
    print("Done.")


if __name__ == "__main__":
    main()
