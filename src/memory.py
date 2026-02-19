"""
Memory system implementations for LLM conversations.

This module provides base memory system implementations. Each system manages
its own internal state (both global and per-conversation memory).
"""

from .types import Conversation, LLMResponse, Prompt
import mem0


class NoHistoryMemorySystem:
    """
    Memory system that ignores conversation history and returns the prompt as-is.
    Stores no memory internally.
    """

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

    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        """
        Retrieve conversation history as memories.
        """
        if not conversation.messages:
            return ""

        # Format conversation history
        history_parts = []
        for msg in conversation.messages:
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

    def __init__(self, memory_limit: int, shared_user_id: str | None = None, **mem0_kwargs):
        """
        Initialize Mem0 memory system.

        Args:
            memory_limit: Maximum number of relevant memories to retrieve (default: 3)
            shared_user_id: Optional shared user_id to use across all conversations.
                If None, uses conversation_id as user_id (default behavior).
            **mem0_kwargs: Additional arguments to pass to mem0.Memory() constructor
                (e.g., vector_store, llm_config, etc.)
        """

        self.memory = mem0.Memory(**mem0_kwargs)
        self.memory_limit = memory_limit
        self.shared_user_id = shared_user_id

    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        """
        Retrieve relevant memories from mem0.
        """
        user_id = self.shared_user_id if self.shared_user_id is not None else str(conversation.conversation_id)
        # Search for relevant memories
        relevant_memories = self.memory.search(
            query=prompt, user_id=user_id, limit=self.memory_limit
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
