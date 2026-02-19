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

    def __init__(self, num_memories: int, shared_user_id: str | None = None, **mem0_kwargs):
        """
        Initialize Mem0 memory system.

        Args:
            num_memories: Maximum number of relevant memories to retrieve (required).
            shared_user_id: Optional shared user_id to use across all conversations.
                If None, uses conversation_id as user_id (default behavior).
            **mem0_kwargs: Additional arguments to pass to mem0.Memory() constructor
                (e.g., vector_store, llm_config, etc.)
        """
        if num_memories is None:
            raise ValueError("num_memories is required for Mem0MemorySystem")
        
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
