"""
Evaluate memory systems on the long-hop chain dataset.

Dataset structure (datasets/long_hop/long_hop_chains.csv) — one row per chain.
Generation config lives in the sidecar long_hop_chains_meta.json.

Hop semantics: K-hop chain = K+1 facts and K+2 chain anchors.
  K=1 -> 2 facts, 3 anchors    (fact_1, fact_2; chain_1..chain_3)
  K=2 -> 3 facts, 4 anchors    (fact_1..fact_3; chain_1..chain_4)
  K=3 -> 4 facts, 5 anchors    (fact_1..fact_4; chain_1..chain_5)

CSV columns:
  id, hop_count,
  fact_1, fact_2, fact_3, fact_4,           # unused slots are empty
  chain_1, chain_2, chain_3, chain_4, chain_5,
  graded_question, ground_truth_answer

Eval setup:
  Storage phase:
    - Every fact across every chain is collected, then SHUFFLED by --seed.
    - Facts are packed into storage conversations of size --num-facts (default 1).
    - HARD CONSTRAINT: a single conversation never contains two facts drawn
      from the same chain. The packer skips a fact whose chain is already in
      the current conversation and tries to place it in a later conversation.
    - Each fact in a storage conversation is an ungraded query.

  Question phase:
    - Chain order is independently shuffled by --seed.
    - One conversation per chain, with a single graded query containing the
      chain's `graded_question` (open-ended; no MCQ choices).

Output:
  traces_<timestamp>.json — full Evaluator summary with all conversations.
  graded_traces_<timestamp>.json — question conversations only, merged with
    chain metadata so analyze_errors.py can grade and classify per-chain errors.
"""

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import (  # noqa: E402
    ChatDataset,
    ConversationData,
    ConversationHistoryPromptTemplate,
    Evaluator,
    OpenAILLM,
    TiktokenTokenizer,
)
from playground.utils import (  # noqa: E402
    UUIDEncoder,
    add_api_key_arg,
    add_memory_system_args,
    create_memory_system,
    resolve_api_key,
)

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_HOP_COUNTS = [1, 2, 3]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _parse_csv_arg(raw: str) -> List[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


_MAX_FACT_COLS = 4
_MAX_CHAIN_COLS = 5


def load_dataset(
    csv_path: Path,
    hop_counts: List[int],
    limit: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    """Load the long-hop chains dataset from CSV and shuffle row order.

    Each CSV row holds one chain. Variable-length facts/anchors are spread
    across fact_1..fact_4 / chain_1..chain_5 columns; empty cells are skipped.
    """
    rows: List[Dict[str, Any]] = []
    keep = set(hop_counts)
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                hop = int(r["hop_count"])
            except (KeyError, ValueError, TypeError):
                continue
            if hop not in keep:
                continue
            facts = [
                r[f"fact_{i}"].strip()
                for i in range(1, _MAX_FACT_COLS + 1)
                if r.get(f"fact_{i}", "").strip()
            ]
            chain = [
                r[f"chain_{i}"].strip()
                for i in range(1, _MAX_CHAIN_COLS + 1)
                if r.get(f"chain_{i}", "").strip()
            ]
            rows.append({
                "example_id": r["id"],
                "hop_count": hop,
                "facts": facts,
                "answer_chain": chain,
                "graded_question": r["graded_question"].strip(),
                "ground_truth_answer": r["ground_truth_answer"].strip(),
            })

    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit is not None:
        rows = rows[:limit]
    return rows


# ---------------------------------------------------------------------------
# Storage-phase fact packing
# ---------------------------------------------------------------------------


def pack_storage_conversations(
    rows: List[Dict[str, Any]],
    num_facts: int,
    seed: int,
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Build storage conversations.

    Returns (conversations, fact_log) where:
      conversations: list of conversations; each conversation is a list of
        items {chain_id, fact_index, fact} representing the queries.
      fact_log: same items in the global emit order across all conversations,
        for trace reconstruction.

    Algorithm:
      1. Build all (chain_id, fact_index, fact) items.
      2. Shuffle them with the seed (fact-level shuffle, independent of row shuffle).
      3. Greedy bin-pack: iterate items in shuffled order; for each item, place it
         into the *first* open conversation with capacity < num_facts that does
         NOT already contain a fact from the same chain. If none exists, open a
         new conversation. The "first available" rule keeps already-open bins
         filling up, so we don't create more bins than necessary.
    """
    items: List[Dict[str, Any]] = []
    for row in rows:
        for i, fact in enumerate(row["facts"]):
            items.append({
                "chain_id": row["example_id"],
                "fact_index": i,
                "fact": fact,
            })
    rng = random.Random(seed ^ 0xF1AC)  # distinct stream from row shuffle
    rng.shuffle(items)

    conversations: List[List[Dict[str, Any]]] = []
    chain_ids_in_conv: List[set] = []  # parallel to conversations
    for item in items:
        placed = False
        for ci, conv in enumerate(conversations):
            if len(conv) >= num_facts:
                continue
            if item["chain_id"] in chain_ids_in_conv[ci]:
                continue
            conv.append(item)
            chain_ids_in_conv[ci].add(item["chain_id"])
            placed = True
            break
        if not placed:
            conversations.append([item])
            chain_ids_in_conv.append({item["chain_id"]})
    return conversations, items


# ---------------------------------------------------------------------------
# Build ChatDataset
# ---------------------------------------------------------------------------


def build_chat_dataset(
    rows: List[Dict[str, Any]],
    num_facts: int,
    seed: int,
) -> Tuple[ChatDataset, int, List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """Build a ChatDataset of storage conversations followed by question conversations.

    Returns (dataset, num_storage_convs, question_rows, storage_packing).
    `storage_packing` is the list of per-conversation item lists used in the
    storage phase, kept for the trace JSON.
    """
    storage_packing, _ = pack_storage_conversations(rows, num_facts, seed)

    conversations: List[ConversationData] = []
    for conv in storage_packing:
        conversations.append(ConversationData(queries=[(item["fact"], False) for item in conv]))
    num_storage_convs = len(conversations)

    rng = random.Random(seed ^ 0x5EED)
    question_order = list(range(len(rows)))
    rng.shuffle(question_order)
    question_rows = [rows[i] for i in question_order]
    for row in question_rows:
        conversations.append(ConversationData(queries=[(row["graded_question"], True)]))

    return ChatDataset(conversations), num_storage_convs, question_rows, storage_packing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate memory systems on the long-hop chain dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "long_hop" / "long_hop_chains.csv"),
        help="Path to long_hop_chains.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SCRIPT_DIR / "results"),
        help="Output directory for results.",
    )
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini",
                        help="Test-taker LLM model.")
    parser.add_argument(
        "--hop-counts",
        type=str,
        default=",".join(str(h) for h in DEFAULT_HOP_COUNTS),
        help="Comma-separated hop counts to evaluate (e.g. 1,2,3).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of chains to evaluate after filtering/shuffling.",
    )
    parser.add_argument(
        "--num-facts",
        type=int,
        default=1,
        help=(
            "Maximum number of facts per storage conversation. Two facts from "
            "the same chain are NEVER placed in the same conversation, "
            "regardless of this value."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for row shuffling, fact shuffling, and storage packing.",
    )
    add_api_key_arg(parser)
    add_memory_system_args(parser)
    args = parser.parse_args()
    if args.shared_user_id == "eval_user":
        args.shared_user_id = "long_hop_eval_user"
    if args.num_facts < 1:
        parser.error("--num-facts must be >= 1")

    api_key = resolve_api_key(args)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir = Path(args.output_dir) / args.memory
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    hop_counts = [int(x) for x in _parse_csv_arg(args.hop_counts)]

    print(f"Loading dataset from {dataset_path}...")
    dataset_rows = load_dataset(
        dataset_path,
        hop_counts=hop_counts,
        limit=args.limit,
        seed=args.seed,
    )
    if not dataset_rows:
        raise ValueError("No chains matched the provided filters.")
    print(f"Loaded {len(dataset_rows)} chains.")

    chat_dataset, num_storage_convs, question_rows, storage_packing = build_chat_dataset(
        dataset_rows, args.num_facts, args.seed
    )
    num_question_convs = len(dataset_rows)
    total_facts = sum(len(row["facts"]) for row in dataset_rows)
    print(
        f"Built ChatDataset: {num_storage_convs} storage conversations holding "
        f"{total_facts} facts ({args.num_facts} max per conversation), "
        f"{num_question_convs} question conversations."
    )

    memory_system = create_memory_system(args, api_key)
    llm = OpenAILLM(api_key=api_key, model=args.llm_model)

    prompt_template = ConversationHistoryPromptTemplate()
    tokenizer = TiktokenTokenizer()

    evaluator = Evaluator(memory_system, llm, chat_dataset, prompt_template, tokenizer)
    print("Running evaluation...")
    summary = evaluator.evaluate()
    print("Evaluation complete.")

    all_memories = memory_system.get_all_memories()
    print(f"Captured {len(all_memories)} memories from store.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_metadata = {
        "memory_system": args.memory,
        "llm_model": args.llm_model,
        "num_memories": args.num_memories,
        "num_facts": args.num_facts,
        "shared_user_id": args.shared_user_id,
        "seed": args.seed,
        "dataset_path": str(dataset_path),
        "hop_counts": hop_counts,
        "limit": args.limit,
        "num_storage_convs": num_storage_convs,
        "num_question_convs": num_question_convs,
        "total_facts_stored": total_facts,
        "timestamp": ts,
        "all_cli_args": {k: v for k, v in vars(args).items() if k != "api_key"},
    }

    summary_dict = asdict(summary)

    # ── Full traces ──────────────────────────────────────────────────────────
    full_path = output_dir / f"traces_{ts}.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_metadata": run_metadata,
                "all_memories_at_time_of_questions": all_memories,
                "dataset_rows": dataset_rows,
                "storage_packing": storage_packing,
                "evaluation_summary": summary_dict,
            },
            f, indent=2, ensure_ascii=False, cls=UUIDEncoder,
        )

    # ── Graded traces (questions only, dataset metadata merged) ──────────────
    question_results = summary_dict["results"][num_storage_convs:]
    graded_traces = []
    for result, row in zip(question_results, question_rows):
        traces = result["traces"]
        if not traces:
            continue
        trace = traces[0]
        graded_traces.append({
            "example_id": row["example_id"],
            "hop_count": row["hop_count"],
            "facts": row["facts"],
            "answer_chain": row["answer_chain"],
            "ground_truth_answer": row["ground_truth_answer"],
            "support_set_size": len(row["facts"]),
            "question": trace["query"],
            "retrieved_memories": trace["retrieved_memories"],
            "llm_response": trace["response"],
            "formatted_prompt": trace["formatted_prompt"],
            "conversation_id": result["conversation_id"],
            "eval_input_tokens": trace["input_tokens"],
            "eval_output_tokens": trace["output_tokens"],
            "eval_cost": trace["cost"],
        })

    graded_path = output_dir / f"graded_traces_{ts}.json"
    with open(graded_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_metadata": run_metadata,
                "all_memories_at_time_of_questions": all_memories,
                "graded_traces": graded_traces,
            },
            f, indent=2, ensure_ascii=False, cls=UUIDEncoder,
        )

    print("=" * 80)
    print("LONG-HOP EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Total conversations:  {summary.total_conversations}")
    print(f"Total queries:        {summary.total_queries}")
    print(f"Total cost:           ${summary.total_cost:.4f}")
    print(f"Total input tokens:   {summary.total_input_tokens:,}")
    print(f"Total output tokens:  {summary.total_output_tokens:,}")
    print(f"Saved full traces:    {full_path}")
    print(f"Saved graded traces:  {graded_path}")
    print("Run analyze_errors.py on the graded traces to grade and classify errors.")
    print("=" * 80)


if __name__ == "__main__":
    main()
