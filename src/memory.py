"""
Memory system implementations for LLM conversations.

This module provides base memory system implementations. Each system manages
its own internal state (both global and per-conversation memory).
"""

from .types import Conversation, LLMResponse, Prompt, PromptContext


class NoHistoryMemorySystem:
    """
    Memory system that ignores conversation history and returns the prompt as-is.
    Stores no memory internally.
    """

    def create_context(
        self, prompt: Prompt, conversation: Conversation
    ) -> PromptContext:
        """
        Return the prompt as-is without any conversation history.

        Args:
            prompt: The current prompt
            conversation: The conversation history (ignored)

        Returns:
            The prompt unchanged
        """
        return prompt

    def update_memory(
        self, prompt: Prompt, response: LLMResponse, conversation: Conversation
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

    def create_context(
        self, prompt: Prompt, conversation: Conversation
    ) -> PromptContext:
        """
        Create context by prepending conversation history to the prompt.

        Args:
            prompt: The current prompt
            conversation: The conversation history

        Returns:
            Context with conversation history prepended to the prompt
        """
        if not conversation.messages:
            # No history, just return the prompt
            return prompt

        # Format conversation history
        history_parts = []
        for msg in conversation.messages:
            history_parts.append(f"User: {msg.prompt}")
            history_parts.append(f"Assistant: {msg.response}")

        # Prepend history to current prompt
        history_text = "\n".join(history_parts)
        context = f"{history_text}\nUser: {prompt}"

        return context

    def update_memory(
        self, prompt: Prompt, response: LLMResponse, conversation: Conversation
    ) -> None:
        """
        No-op: This memory system stores no memory internally.
        All memory comes from the conversation history.
        """
        pass
