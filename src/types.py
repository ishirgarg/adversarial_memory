"""
Core type definitions for the memory evaluation framework.
"""

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable, Tuple
from uuid import UUID


# Core Types
ConversationID = UUID
LLMResponse = str
Prompt = str
PromptContext = str


@dataclass(frozen=True)
class Message:
    """Represents a message in a conversation."""

    prompt: Prompt
    response: LLMResponse
    conversation_id: ConversationID
    timestamp: Optional[float] = None


@dataclass(frozen=True)
class Conversation:
    """
    Represents a conversation with its messages.

    Memory is managed internally by the MemorySystem, not stored here.
    """

    conversation_id: ConversationID
    messages: Tuple[Message, ...] = ()


@runtime_checkable
class LLM(Protocol):
    """Protocol for LLM providers."""

    def __init__(
        self, model: str, max_tokens: int, temperature: float, **default_kwargs: Any
    ):
        """
        Initialize the LLM.
        """
        ...

    def query(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Query the LLM with a prompt.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for the LLM

        Returns:
            The LLM's response
        """
        ...


@runtime_checkable
class MemorySystem(Protocol):
    """
    Protocol for memory systems.

    Memory systems manage their own internal state (both global and per-conversation).
    The memory system tracks all memory internally and does not expose state externally.
    """

    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        """
        Create a context from the prompt using internal memory state.

        Args:
            prompt: The current prompt
            conversation: The conversation history

        Returns:
            The context string to be used for the LLM
        """
        ...

    def update_memory(
        self, prompt: Prompt, response: LLMResponse, conversation: Conversation
    ) -> None:
        """
        Update internal memory state based on the prompt and response.

        Args:
            prompt: The prompt that was sent
            response: The response received
            conversation: The conversation history
        """
        ...
