"""
Memory system implementations for LLM conversations.

This module provides base memory system implementations. Each system manages
its own internal state (both global and per-conversation memory).
"""

from typing import Optional
from .types import Conversation, LLMResponse, Prompt
from .amem import AgenticMemorySystem, MemoryNote
import mem0
import os


class NoHistoryMemorySystem:
    """
    Memory system that ignores conversation history and returns the prompt as-is.
    Stores no memory internally.
    """

    def __init__(self, num_memories: int | None = None):
        """
        Initialize NoHistoryMemorySystem.
        
        Args:
            num_memories: Maximum number of memories to return (ignored for this system).
                Can be None for uncapped.
        """
        self.num_memories = num_memories

    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        """
        Return the prompt as-is without any conversation history.
        """
        return ""

    def update_memory(
        self, prompt: Prompt, response: LLMResponse, conversation_history: Conversation
    ) -> None:
        """
        No-op: This memory system stores no memory internally.
        """
        pass


class SimpleHistoryMemorySystem:
    """
    Simple memory system that prepends conversation history to the prompt.
    Stores no memory internally - just uses the conversation history provided.
    """

    def __init__(self, num_memories: int | None = None):
        """
        Initialize SimpleHistoryMemorySystem.
        
        Args:
            num_memories: Maximum number of memories (message pairs) to return.
                Can be None for uncapped (returns all history).
        """
        self.num_memories = num_memories

    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        """
        Retrieve conversation history as memories, capped to num_memories if set.
        """
        if not conversation.messages:
            return ""

        # Cap the number of messages if num_memories is set
        messages_to_include = conversation.messages
        if self.num_memories is not None:
            # Each "memory" is a user-assistant pair, so cap at num_memories pairs
            # Since we have (user, assistant) pairs, we need 2 * num_memories messages
            max_messages = self.num_memories * 2
            if len(conversation.messages) > max_messages:
                messages_to_include = conversation.messages[-max_messages:]
        
        # Format conversation history
        history_parts = []
        for msg in messages_to_include:
            history_parts.append(f"User: {msg.prompt}")
            history_parts.append(f"Assistant: {msg.response}")

        # Return just the history (memories), not including current prompt
        history_text = "\n".join(history_parts)
        return history_text

    def update_memory(
        self, prompt: Prompt, response: LLMResponse, conversation_history: Conversation
    ) -> None:
        """
        No-op: This memory system stores no memory internally.
        All memory comes from the conversation history.
        """
        pass


class Mem0MemorySystem:
    """
    Memory system using mem0 (https://github.com/mem0ai/mem0).

    Mem0 provides intelligent memory retrieval and storage, using semantic search
    to find relevant memories and automatically extracting key information.
    """

    def __init__(
        self,
        num_memories: int,
        shared_user_id: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        **mem0_kwargs,
    ):
        """
        Initialize Mem0 memory system.

        Args:
            num_memories: Maximum number of relevant memories to retrieve (required).
            shared_user_id: Optional shared user_id to use across all conversations.
                If None, uses conversation_id as user_id (default behavior).
            embedding_provider: Provider for embeddings (e.g., "openai", "ollama", "sentence-transformers").
                If None, uses mem0's default (usually OpenAI).
            embedding_model: Model name for embeddings (e.g., "text-embedding-3-small" for OpenAI,
                "nomic-embed-text" for Ollama, "all-MiniLM-L6-v2" for sentence-transformers).
            **mem0_kwargs: Additional arguments to pass to mem0.Memory() constructor
                (e.g., vector_store, llm_config, etc.)
        """
        # Configure embedder if provided
        if embedding_provider and embedding_model:
            embedder_config = {
                "provider": embedding_provider,
                "config": {
                    "model": embedding_model,
                }
            }
            # For Ollama, we might need to pass base_url if provided
            if embedding_provider == "ollama" and "ollama_base_url" in mem0_kwargs:
                embedder_config["config"]["base_url"] = mem0_kwargs.pop("ollama_base_url")
            mem0_kwargs["embedder"] = embedder_config
        
        self.memory = mem0.Memory(**mem0_kwargs)
        self.num_memories = num_memories
        self.shared_user_id = shared_user_id

    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        """
        Retrieve relevant memories from mem0.
        """
        user_id = self.shared_user_id if self.shared_user_id is not None else str(conversation.conversation_id)
        # Search for relevant memories
        relevant_memories = self.memory.search(
            query=prompt, user_id=user_id, limit=self.num_memories
        )
        memories_str = "\n".join(
            f"- {entry['memory']}" for entry in relevant_memories["results"]
        )
        return memories_str

    def update_memory(
        self, prompt: Prompt, response: LLMResponse, conversation_history: Conversation
    ) -> None:
        """
        Update mem0 memory by adding the conversation to memory.

        Args:
            prompt: The prompt that was sent
            response: The response received
            conversation: The conversation history (includes the new message)
        """
        # Use shared_user_id if provided, otherwise use conversation_id
        user_id = self.shared_user_id if self.shared_user_id is not None else str(conversation_history.conversation_id)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        self.memory.add(messages, user_id=user_id)


class AMEMMemorySystem:
    """
    Memory system using A-MEM (https://github.com/WujiangXu/A-mem).
    NeurIPS 2025: "A-Mem: Agentic Memory for LLM Agents".

    Implements Zettelkasten-style agentic memory with:
    - LLM-driven note indexing (keywords, context, tags)
    - Semantic + BM25 hybrid retrieval via SimpleEmbeddingRetriever
    - Automatic memory evolution and cross-linking
    - Linked-neighbor recall: each retrieved memory also pulls in its
      Zettelkasten links, following the pattern in test_advanced.py
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
        """
        Initialize AMEMMemorySystem.

        Args:
            num_memories: Number of top-k memories to retrieve per query.
                          Zettelkasten-linked neighbors are also returned.
            llm_backend: LLM backend for A-MEM ("openai" or "ollama").
            llm_model: LLM model for note generation and evolution.
            embedding_model: Sentence-transformer model name for embeddings.
            evo_threshold: How many memories before triggering evolution.
            api_key: OpenAI API key (falls back to OPENAI_KEY env var).
        """
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # MemorySystem protocol
    # ------------------------------------------------------------------

    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        """
        Retrieve the k most relevant memories via semantic + BM25 search,
        then follow each memory's Zettelkasten links to pull in related notes.

        This mirrors the retrieve_memory() / find_related_memories_raw() pattern
        from test_advanced.py in the WujiangXu/A-mem repository.
        """
        if not self._memory.memories:
            return ""

        # Retrieve top-k indices from the hybrid retriever
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

            # Follow Zettelkasten links (same logic as find_related_memories_raw)
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
        """
        Store the latest exchange as a new note in A-MEM.

        The full prompt+response is passed to add_note so A-MEM can:
        - extract keywords and tags via LLM
        - generate a context summary
        - link to and evolve related existing memories
        """
        note_content = f"User: {prompt}\nAssistant: {response}"
        self._memory.add_note(note_content)


class SimpleMemMemorySystem:
    """
    Memory system using SimpleMem's three-stage pipeline.

    Wraps the SimpleMem system (from the SimpleMem/ directory) to conform to
    the MemorySystem protocol used by the evaluation framework.

    SimpleMem uses:
    1. Semantic Structured Compression: converts dialogues into atomic memory entries
    2. Online Semantic Synthesis: intra-session consolidation during write
    3. Intent-Aware Retrieval Planning: multi-view hybrid retrieval (semantic/lexical/symbolic)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        db_path: Optional[str] = None,
        clear_db: bool = True,
        **kwargs,
    ):
        """
        Initialize the SimpleMem memory system.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var via SimpleMem config)
            model: LLM model name (defaults to SimpleMem config LLM_MODEL)
            base_url: Custom API base URL (defaults to SimpleMem config OPENAI_BASE_URL)
            db_path: Path for LanceDB vector store storage
            clear_db: Whether to clear existing database on init (recommended: True)
            **kwargs: Additional arguments forwarded to SimpleMemSystem
        """
        import sys
        from pathlib import Path

        simplemem_dir = str(Path(__file__).parent.parent / "SimpleMem")
        if simplemem_dir not in sys.path:
            sys.path.insert(0, simplemem_dir)

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
        """
        Retrieve relevant memories using SimpleMem's hybrid retrieval.

        Uses the three-layer retrieval (semantic, lexical, symbolic) and formats
        the retrieved memory entries as a context string.
        """
        contexts = self._system.hybrid_retriever.retrieve(prompt)
        if not contexts:
            return ""
        return self._system.answer_generator._format_contexts(contexts)

    def update_memory(
        self, prompt: Prompt, response: LLMResponse, conversation_history: Conversation
    ) -> None:
        """
        Add the prompt and response as dialogues to SimpleMem's memory.

        Each user/assistant exchange is stored as two dialogue entries, then
        finalized so the memory is immediately available for retrieval.
        """
        import time

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._system.add_dialogue("User", prompt, timestamp)
        self._system.add_dialogue("Assistant", response, timestamp)
        self._system.finalize()
