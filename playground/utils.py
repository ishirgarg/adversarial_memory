"""
Shared utilities for playground evaluation scripts.

Provides:
  - add_memory_system_args(parser)  -- registers all memory-system CLI arg groups
  - create_memory_system(args, api_key)  -- factory that instantiates the chosen system
  - add_api_key_arg(parser)  -- standard --api-key argument
  - resolve_api_key(args)  -- resolves API key from args or env
  - UUIDEncoder  -- JSON encoder that serialises uuid.UUID as strings
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

# Ensure the project root (parent of playground/) is on the path so that
# `from src import ...` works regardless of where the calling script lives.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import AMEMMemorySystem, Mem0MemorySystem, SimpleMemMemorySystem  # noqa: E402

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


class UUIDEncoder(json.JSONEncoder):
    """JSON encoder that serialises uuid.UUID values as strings."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Common argument helpers
# ---------------------------------------------------------------------------


def add_api_key_arg(parser: argparse.ArgumentParser) -> None:
    """Add --api-key to a parser."""
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (falls back to OPENAI_API_KEY env var).",
    )


def resolve_api_key(args: argparse.Namespace) -> str:
    """Return the API key from --api-key or the OPENAI_API_KEY env var."""
    key = getattr(args, "api_key", None) or os.getenv("OPENAI_API_KEY") or ""
    if not key:
        raise ValueError("OpenAI API key required via --api-key or OPENAI_API_KEY env var.")
    return key


# ---------------------------------------------------------------------------
# Memory-system argument groups
# ---------------------------------------------------------------------------


def add_memory_system_args(parser: argparse.ArgumentParser) -> None:
    """Register all memory-system selection and configuration args on *parser*.

    Adds:
      --memory          which memory backend to use
      --num-memories    retrieval top-k (shared across systems)
      --shared-user-id  user-id namespace (mem0 / shared-state systems)

      --mem0-*          mem0-specific options
      --amem-*          A-MEM-specific options
      --simplemem-*     SimpleMem-specific options
    """
    # ── Shared ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--memory",
        type=str,
        default="mem0",
        choices=["mem0", "simplemem", "amem"],
        help="Memory system to use.",
    )
    parser.add_argument(
        "--num-memories",
        type=int,
        default=5,
        help="Number of memories to retrieve (top-k).",
    )
    parser.add_argument(
        "--shared-user-id",
        type=str,
        default="eval_user",
        help="Shared user-id for memory systems that namespace by user.",
    )

    # ── mem0 ─────────────────────────────────────────────────────────────────
    mem0 = parser.add_argument_group("mem0 options")
    mem0.add_argument("--mem0-llm-provider", type=str, default="openai",
                      help="LLM provider for mem0 memory operations.")
    mem0.add_argument("--mem0-llm-model", type=str, default="gpt-5-mini",
                      help="LLM model for mem0 memory operations.")
    mem0.add_argument("--mem0-embedding-provider", type=str, default=None,
                      help="Embedding provider (e.g. openai, ollama). Uses mem0 default if omitted.")
    mem0.add_argument("--mem0-embedding-model", type=str, default=None,
                      help="Embedding model name. Uses mem0 default if omitted.")
    mem0.add_argument("--mem0-ollama-base-url", type=str, default=None,
                      help="Ollama base URL (only relevant when --mem0-embedding-provider=ollama).")

    # ── A-MEM ─────────────────────────────────────────────────────────────────
    amem = parser.add_argument_group("amem options")
    amem.add_argument("--amem-llm-backend", type=str, default="openai",
                      help="LLM backend for A-MEM (openai or ollama).")
    amem.add_argument("--amem-llm-model", type=str, default="gpt-4o-mini",
                      help="LLM model for A-MEM memory operations.")
    amem.add_argument("--amem-embedding-model", type=str, default="all-MiniLM-L6-v2",
                      help="Sentence-transformer embedding model for A-MEM.")
    amem.add_argument("--amem-evo-threshold", type=int, default=100,
                      help="Note-evolution threshold for A-MEM.")

    # ── SimpleMem ─────────────────────────────────────────────────────────────
    sm = parser.add_argument_group("simplemem options")
    sm.add_argument("--simplemem-model", type=str, default="gpt-4.1-mini",
                    help="LLM model for SimpleMem memory operations.")
    sm.add_argument("--simplemem-base-url", type=str, default=None,
                    help="Custom OpenAI-compatible base URL for SimpleMem.")
    sm.add_argument("--simplemem-db-path", type=str, default="./lancedb_data",
                    help="LanceDB storage path for SimpleMem.")
    sm.add_argument("--simplemem-embedding-model", type=str, default="all-MiniLM-L6-v2",
                    help="Embedding model for SimpleMem.")
    sm.add_argument("--simplemem-embedding-dimension", type=int, default=384)
    sm.add_argument("--simplemem-embedding-context-length", type=int, default=512)
    sm.add_argument("--simplemem-enable-thinking", action="store_true", default=False)
    sm.add_argument("--simplemem-use-streaming", action="store_true", default=False)
    sm.add_argument("--simplemem-use-json-format", action="store_true", default=False)
    sm.add_argument("--simplemem-window-size", type=int, default=20)
    sm.add_argument("--simplemem-overlap-size", type=int, default=2)
    sm.add_argument("--simplemem-memory-table-name", type=str, default="memory_entries")
    sm.add_argument("--simplemem-enable-parallel-processing", action="store_true", default=True)
    sm.add_argument("--simplemem-max-parallel-workers", type=int, default=16)
    sm.add_argument("--simplemem-enable-parallel-retrieval", action="store_true", default=True)
    sm.add_argument("--simplemem-max-retrieval-workers", type=int, default=8)
    sm.add_argument("--simplemem-enable-planning", action="store_true", default=True)
    sm.add_argument("--simplemem-enable-reflection", action="store_true", default=True)
    sm.add_argument("--simplemem-max-reflection-rounds", type=int, default=2)


# ---------------------------------------------------------------------------
# Memory-system factory
# ---------------------------------------------------------------------------


def create_memory_system(args: argparse.Namespace, api_key: str) -> Any:
    """Instantiate the memory system selected by *args.memory*.

    Expects *args* to contain all attributes registered by
    :func:`add_memory_system_args`.
    """
    memory = args.memory

    if memory == "mem0":
        return Mem0MemorySystem(
            num_memories=args.num_memories,
            shared_user_id=args.shared_user_id,
            llm_provider=args.mem0_llm_provider,
            llm_model=args.mem0_llm_model,
            embedding_provider=args.mem0_embedding_provider,
            embedding_model=args.mem0_embedding_model,
            ollama_base_url=args.mem0_ollama_base_url,
        )

    if memory == "simplemem":
        return SimpleMemMemorySystem(
            num_memories=args.num_memories,
            api_key=api_key,
            model=args.simplemem_model,
            base_url=args.simplemem_base_url,
            db_path=args.simplemem_db_path,
            clear_db=True,
            embedding_model=args.simplemem_embedding_model,
            embedding_dimension=args.simplemem_embedding_dimension,
            embedding_context_length=args.simplemem_embedding_context_length,
            enable_thinking=args.simplemem_enable_thinking,
            use_streaming=args.simplemem_use_streaming,
            use_json_format=args.simplemem_use_json_format,
            window_size=args.simplemem_window_size,
            overlap_size=args.simplemem_overlap_size,
            memory_table_name=args.simplemem_memory_table_name,
            enable_parallel_processing=args.simplemem_enable_parallel_processing,
            max_parallel_workers=args.simplemem_max_parallel_workers,
            enable_parallel_retrieval=args.simplemem_enable_parallel_retrieval,
            max_retrieval_workers=args.simplemem_max_retrieval_workers,
            enable_planning=args.simplemem_enable_planning,
            enable_reflection=args.simplemem_enable_reflection,
            max_reflection_rounds=args.simplemem_max_reflection_rounds,
        )

    if memory == "amem":
        return AMEMMemorySystem(
            num_memories=args.num_memories,
            llm_backend=args.amem_llm_backend,
            llm_model=args.amem_llm_model,
            embedding_model=args.amem_embedding_model,
            evo_threshold=args.amem_evo_threshold,
            api_key=api_key,
        )

    raise ValueError(f"Unknown memory system: {memory!r}. Choose mem0, simplemem, or amem.")
