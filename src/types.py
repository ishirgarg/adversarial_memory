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
class LLMUsage:
    """Token usage reported by the LLM API."""

    prompt_tokens: int
    completion_tokens: int


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
        self, prompt: Prompt, response: LLMResponse, conversation_history: Conversation
    ) -> None:
        """
        Update internal memory state based on the prompt and response.

        Args:
            prompt: The prompt that was sent
            response: The response received
            conversation: The conversation history
        """
        ...

    def finalize_conversation(self, conversation_id: ConversationID) -> None:
        """
        Finalize and consolidate memory at the end of a conversation.

        Called once after all turns in a conversation are complete.
        Most memory systems can implement this as a no-op.

        Args:
            conversation_id: The ID of the conversation that ended
        """
        ...


@runtime_checkable
class EvaluationPromptTemplate(Protocol):
    """
    Protocol for prompt templates used in evaluation.

    Defines how to format the final prompt given query, memories, and conversation history.
    """

    def format(self, query: str, memories: str, conversation: Conversation) -> str:
        """
        Format a prompt for evaluation.

        Args:
            query: The current user query
            memories: The memories retrieved from the memory system
            conversation: The conversation history

        Returns:
            The formatted prompt string to send to the LLM
        """
        ...
