"""
Memory system implementations for LLM conversations.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

import mem0

from .amem import AgenticMemorySystem, MemoryNote
from .types import Conversation, LLMResponse, Prompt


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
            query=prompt, top_k=self.num_memories, filters={"user_id": user_id}
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

    def get_all_memories(self, user_id: str | None = None) -> list[str]:
        if user_id is None:
            user_id = self.shared_user_id or "shared_user"
        result = self.memory.get_all(filters={"user_id": user_id})
        return [entry["memory"] for entry in result.get("results", [])]


class AMEMMemorySystem:
    """
    Memory system using A-MEM (https://github.com/WujiangXu/A-mem).
    NeurIPS 2025: "A-Mem: Agentic Memory for LLM Agents".

    Implements Zettelkasten-style agentic memory with:
    - LLM-driven note indexing (keywords, context, tags)
    - Semantic + BM25 hybrid retrieval via SimpleEmbeddingRetriever
    - Automatic memory evolution and cross-linking
    - Linked-neighbor recall following the test_advanced.py pattern
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

        self.num_memories = num_memories
        self._memory = AgenticMemorySystem(
            model_name=embedding_model,
            llm_backend=llm_backend,
            llm_model=llm_model,
            evo_threshold=evo_threshold,
            api_key=api_key,
        )

    def _format_note(self, note: MemoryNote, label: str) -> str:
        lines = [f"{label}:"]
        lines.append(f"  Content:  {note.content}")
        if note.context:
            lines.append(f"  Context:  {note.context}")
        if note.tags:
            tags = note.tags if isinstance(note.tags, list) else list(note.tags)
            lines.append(f"  Tags:     {', '.join(str(t) for t in tags)}")
        if note.keywords:
            kws = note.keywords if isinstance(note.keywords, list) else list(note.keywords)
            lines.append(f"  Keywords: {', '.join(str(k) for k in kws)}")
        return "\n".join(lines)

    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        if not self._memory.memories:
            return ""

        indices = self._memory.retriever.search(prompt, k=self.num_memories)
        all_notes = list(self._memory.memories.values())

        seen: set[int] = set()
        memory_parts: list[str] = []
        mem_num = 1

        for idx in indices:
            if idx >= len(all_notes) or idx in seen:
                continue
            seen.add(idx)
            note = all_notes[idx]
            memory_parts.append(self._format_note(note, f"Memory {mem_num}"))
            mem_num += 1

            for j, neighbor_idx in enumerate(note.links):
                if j >= self.num_memories:
                    break
                if (
                    isinstance(neighbor_idx, int)
                    and neighbor_idx < len(all_notes)
                    and neighbor_idx not in seen
                ):
                    seen.add(neighbor_idx)
                    nb = all_notes[neighbor_idx]
                    memory_parts.append(self._format_note(nb, f"Memory {mem_num} (linked)"))
                    mem_num += 1

        return "\n\n".join(memory_parts)

    def update_memory(
        self, prompt: Prompt, response: LLMResponse, conversation_history: Conversation
    ) -> None:
        note_content = f"User: {prompt}\nAssistant: {response}"
        self._memory.add_note(note_content)

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
        **kwargs,
    ):
        self.num_memories = num_memories
        simplemem_dir = Path(__file__).parent.parent / "SimpleMem"
        simplemem_dir_str = str(simplemem_dir)
        if simplemem_dir_str not in sys.path:
            sys.path.insert(0, simplemem_dir_str)

        resolved_api_key = api_key or os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY") or ""
        resolved_model = model or "gpt-4.1-mini"
        resolved_base_url = base_url

        config_path = simplemem_dir / "config.py"
        config_path.write_text(
            f"OPENAI_API_KEY = {resolved_api_key!r}\n"
            f"OPENAI_BASE_URL = {resolved_base_url!r}\n"
            f"LLM_MODEL = {resolved_model!r}\n"
            "EMBEDDING_MODEL = 'all-MiniLM-L6-v2'\n"
            "EMBEDDING_DIMENSION = 384\n"
            "EMBEDDING_CONTEXT_LENGTH = 512\n"
            "ENABLE_THINKING = False\n"
            "USE_STREAMING = False\n"
            "USE_JSON_FORMAT = False\n"
            "WINDOW_SIZE = 40\n"
            "OVERLAP_SIZE = 2\n"
            "SEMANTIC_TOP_K = 25\n"
            "KEYWORD_TOP_K = 5\n"
            "STRUCTURED_TOP_K = 5\n"
            "LANCEDB_PATH = './lancedb_data'\n"
            "MEMORY_TABLE_NAME = 'memory_entries'\n"
            "ENABLE_PARALLEL_PROCESSING = True\n"
            "MAX_PARALLEL_WORKERS = 16\n"
            "ENABLE_PARALLEL_RETRIEVAL = True\n"
            "MAX_RETRIEVAL_WORKERS = 8\n"
            "ENABLE_PLANNING = True\n"
            "ENABLE_REFLECTION = True\n"
            "MAX_REFLECTION_ROUNDS = 2\n"
        )

        from main import SimpleMemSystem  # type: ignore

        self._system = SimpleMemSystem(
            api_key=api_key,
            model=model,
            base_url=base_url,
            db_path=db_path,
            clear_db=clear_db,
            **kwargs,
        )

    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        contexts = self._system.hybrid_retriever.retrieve(prompt)
        if not contexts:
            return ""
        if self.num_memories is not None:
            contexts = contexts[: self.num_memories]
        return self._system.answer_generator._format_contexts(contexts)

    def update_memory(
        self, prompt: Prompt, response: LLMResponse, conversation_history: Conversation
    ) -> None:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._system.add_dialogue("User", prompt, timestamp)
        self._system.add_dialogue("Assistant", response, timestamp)
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
