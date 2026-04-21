"""
Memory system implementations for LLM conversations.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

import mem0

from .types import Conversation, ConversationID, LLMResponse, Prompt


class NoHistoryMemorySystem:
    """Baseline that ignores conversation history and returns empty context."""

    def __init__(self, num_memories: int | None = None):
        self.num_memories = num_memories

    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        return ""

    def update_memory(
        self, prompt: Prompt, response: LLMResponse, conversation_history: Conversation
    ) -> None:
        pass

    def get_all_memories(self) -> list[str]:
        return []

    def finalize_conversation(self, conversation_id: ConversationID) -> None:
        pass


class SimpleHistoryMemorySystem:
    """Prepends raw conversation history to the prompt. Stores no memory internally."""

    def __init__(self, num_memories: int | None = None):
        self.num_memories = num_memories

    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        if not conversation.messages:
            return ""

        messages_to_include = conversation.messages
        if self.num_memories is not None:
            max_messages = self.num_memories * 2
            if len(conversation.messages) > max_messages:
                messages_to_include = conversation.messages[-max_messages:]

        history_parts = []
        for msg in messages_to_include:
            history_parts.append(f"User: {msg.prompt}")
            history_parts.append(f"Assistant: {msg.response}")

        return "\n".join(history_parts)

    def update_memory(
        self, prompt: Prompt, response: LLMResponse, conversation_history: Conversation
    ) -> None:
        pass

    def get_all_memories(self) -> list[str]:
        return []

    def finalize_conversation(self, conversation_id: ConversationID) -> None:
        pass


class Mem0MemorySystem:
    """
    Memory system using mem0 (https://github.com/mem0ai/mem0).

    Uses semantic search to retrieve relevant memories and automatically
    extracts key information from conversations.
    """

    def __init__(
        self,
        num_memories: int,
        shared_user_id: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        **mem0_kwargs,
    ):
        if embedding_provider and embedding_model:
            embedder_config: dict = {
                "provider": embedding_provider,
                "config": {"model": embedding_model},
            }
            if embedding_provider == "ollama" and "ollama_base_url" in mem0_kwargs:
                embedder_config["config"]["base_url"] = mem0_kwargs.pop("ollama_base_url")
            mem0_kwargs["embedder"] = embedder_config

        self.memory = mem0.Memory(**mem0_kwargs)
        self.num_memories = num_memories
        self.shared_user_id = shared_user_id

    def _user_id(self, conversation: Conversation) -> str:
        return self.shared_user_id if self.shared_user_id is not None else str(conversation.conversation_id)

    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        user_id = self._user_id(conversation)
        results = self.memory.search(
            query=prompt, user_id=user_id, limit=self.num_memories
        )
        return "\n".join(f"- {entry['memory']}" for entry in results["results"])

    def update_memory(
        self, prompt: Prompt, response: LLMResponse, conversation_history: Conversation
    ) -> None:
        user_id = self._user_id(conversation_history)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        self.memory.add(messages, user_id=user_id)

    def finalize_conversation(self, conversation_id: ConversationID) -> None:
        pass

    def get_all_memories(self, user_id: str | None = None) -> list[str]:
        if user_id is None:
            user_id = self.shared_user_id or "shared_user"
        result = self.memory.get_all(user_id=user_id)
        return [entry["memory"] for entry in result.get("results", [])]


class AMEMMemorySystem:
    """
    Memory system using A-MEM (https://github.com/WujiangXu/A-mem).
    NeurIPS 2025: "A-Mem: Agentic Memory for LLM Agents".

    Implements Zettelkasten-style agentic memory with:
    - LLM-driven note indexing (keywords, context, tags)
    - Semantic retrieval via ChromaDB
    - Automatic memory evolution and cross-linking
    - Linked-neighbor recall via search_agentic
    """

    def __init__(
        self,
        num_memories: int,
        llm_backend: str,
        llm_model: str,
        embedding_model: str,
        evo_threshold: int,
        api_key: str | None = None,
    ):
        if api_key is None:
            api_key = os.getenv("OPENAI_KEY")

        amem_dir = str(Path(__file__).parent.parent / "a-mem")
        if amem_dir not in sys.path:
            sys.path.insert(0, amem_dir)

        from agentic_memory.memory_system import AgenticMemorySystem  # type: ignore

        self.num_memories = num_memories
        self._memory = AgenticMemorySystem(
            model_name=embedding_model,
            llm_backend=llm_backend,
            llm_model=llm_model,
            evo_threshold=evo_threshold,
            api_key=api_key,
        )

    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        results = self._memory.search_agentic(prompt, k=self.num_memories)
        if not results:
            return ""
        parts = []
        for m in results:
            label = "Memory (linked)" if m.get("is_neighbor") else "Memory"
            line = f"{label}: {m['content']}"
            if m.get("context"):
                line += f" | context: {m['context']}"
            if m.get("tags"):
                line += f" | tags: {', '.join(m['tags'])}"
            parts.append(line)
        return "\n".join(parts)

    def update_memory(
        self, prompt: Prompt, response: LLMResponse, conversation_history: Conversation
    ) -> None:
        note_content = f"User: {prompt}\nAssistant: {response}"
        self._memory.add_note(note_content)

    def finalize_conversation(self, conversation_id: ConversationID) -> None:
        pass

    def get_all_memories(self) -> list[str]:
        parts = []
        for note in self._memory.memories.values():
            line = note.content
            extras = []
            if note.context:
                extras.append(f"context: {note.context}")
            if note.tags:
                tags = note.tags if isinstance(note.tags, list) else list(note.tags)
                extras.append(f"tags: {', '.join(str(t) for t in tags)}")
            if extras:
                line = f"{line} ({'; '.join(extras)})"
            parts.append(line)
        return parts


class SimpleMemMemorySystem:
    """
    Memory system using SimpleMem's three-stage pipeline.

    1. Semantic Structured Compression: converts dialogues into atomic memory entries
    2. Online Semantic Synthesis: intra-session consolidation during write
    3. Intent-Aware Retrieval Planning: multi-view hybrid retrieval (semantic/lexical/symbolic)
    """

    def __init__(
        self,
        num_memories: int | None = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        db_path: Optional[str] = None,
        clear_db: bool = True,
        embedding_model: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        embedding_context_length: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        use_streaming: Optional[bool] = None,
        use_json_format: Optional[bool] = None,
        window_size: Optional[int] = None,
        overlap_size: Optional[int] = None,
        semantic_top_k: Optional[int] = None,
        keyword_top_k: Optional[int] = None,
        structured_top_k: Optional[int] = None,
        memory_table_name: Optional[str] = None,
        enable_parallel_processing: Optional[bool] = None,
        max_parallel_workers: Optional[int] = None,
        enable_parallel_retrieval: Optional[bool] = None,
        max_retrieval_workers: Optional[int] = None,
        enable_planning: Optional[bool] = None,
        enable_reflection: Optional[bool] = None,
        max_reflection_rounds: Optional[int] = None,
    ):
        self.num_memories = num_memories
        simplemem_dir = Path(__file__).parent.parent / "SimpleMem"
        simplemem_dir_str = str(simplemem_dir)
        if simplemem_dir_str not in sys.path:
            sys.path.insert(0, simplemem_dir_str)

        resolved_api_key = api_key or os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY") or ""
        resolved_model = model or "gpt-4.1-mini"

        # top-k: fall back to num_memories, then to SimpleMem defaults
        resolved_semantic_top_k = semantic_top_k if semantic_top_k is not None else (num_memories if num_memories is not None else 25)
        resolved_keyword_top_k = keyword_top_k if keyword_top_k is not None else (num_memories if num_memories is not None else 5)
        resolved_structured_top_k = structured_top_k if structured_top_k is not None else (num_memories if num_memories is not None else 5)

        config_path = simplemem_dir / "config.py"
        config_path.write_text(
            f"OPENAI_API_KEY = {resolved_api_key!r}\n"
            f"OPENAI_BASE_URL = {base_url!r}\n"
            f"LLM_MODEL = {resolved_model!r}\n"
            f"EMBEDDING_MODEL = {(embedding_model or 'all-MiniLM-L6-v2')!r}\n"
            f"EMBEDDING_DIMENSION = {embedding_dimension or 384}\n"
            f"EMBEDDING_CONTEXT_LENGTH = {embedding_context_length or 512}\n"
            f"ENABLE_THINKING = {enable_thinking if enable_thinking is not None else False}\n"
            f"USE_STREAMING = {use_streaming if use_streaming is not None else False}\n"
            f"USE_JSON_FORMAT = {use_json_format if use_json_format is not None else False}\n"
            f"WINDOW_SIZE = {window_size or 20}\n"
            f"OVERLAP_SIZE = {overlap_size or 2}\n"
            f"SEMANTIC_TOP_K = {resolved_semantic_top_k}\n"
            f"KEYWORD_TOP_K = {resolved_keyword_top_k}\n"
            f"STRUCTURED_TOP_K = {resolved_structured_top_k}\n"
            f"LANCEDB_PATH = {(db_path or './lancedb_data')!r}\n"
            f"MEMORY_TABLE_NAME = {(memory_table_name or 'memory_entries')!r}\n"
            f"ENABLE_PARALLEL_PROCESSING = {enable_parallel_processing if enable_parallel_processing is not None else True}\n"
            f"MAX_PARALLEL_WORKERS = {max_parallel_workers or 16}\n"
            f"ENABLE_PARALLEL_RETRIEVAL = {enable_parallel_retrieval if enable_parallel_retrieval is not None else True}\n"
            f"MAX_RETRIEVAL_WORKERS = {max_retrieval_workers or 8}\n"
            f"ENABLE_PLANNING = {enable_planning if enable_planning is not None else True}\n"
            f"ENABLE_REFLECTION = {enable_reflection if enable_reflection is not None else True}\n"
            f"MAX_REFLECTION_ROUNDS = {max_reflection_rounds or 2}\n"
        )

        from main import SimpleMemSystem  # type: ignore

        self._system = SimpleMemSystem(
            api_key=api_key,
            model=model,
            base_url=base_url,
            db_path=db_path,
            table_name=memory_table_name,
            clear_db=clear_db,
            enable_thinking=enable_thinking,
            use_streaming=use_streaming,
            enable_planning=enable_planning,
            enable_reflection=enable_reflection,
            max_reflection_rounds=max_reflection_rounds,
            enable_parallel_processing=enable_parallel_processing,
            max_parallel_workers=max_parallel_workers,
            enable_parallel_retrieval=enable_parallel_retrieval,
            max_retrieval_workers=max_retrieval_workers,
        )

    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        contexts = self._system.hybrid_retriever.retrieve(prompt)
        if not contexts:
            return ""
        if self.num_memories is not None:
            contexts = contexts[: self.num_memories]
        return "\n".join(f"- {entry.lossless_restatement}" for entry in contexts)

    def update_memory(
        self, prompt: Prompt, response: LLMResponse, conversation_history: Conversation
    ) -> None:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._system.add_dialogue("User", prompt, timestamp)
        self._system.add_dialogue("Assistant", response, timestamp)

    def finalize_conversation(self, conversation_id: ConversationID) -> None:
        self._system.finalize()

    def get_all_memories(self) -> list[str]:
        parts = []
        for e in self._system.get_all_memories():
            line = e.lossless_restatement
            extras = []
            if e.topic:
                extras.append(f"topic: {e.topic}")
            if e.timestamp:
                extras.append(f"time: {e.timestamp}")
            if extras:
                line = f"{line} ({'; '.join(extras)})"
            parts.append(line)
        return parts
