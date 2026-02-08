"""
Adversarial Memory - A unified framework for evaluating LLM memory systems.
"""

__version__ = "0.1.0"

# Core types
from .types import (
    ConversationID,
    LLMResponse,
    Prompt,
    PromptContext,
    Message,
    Conversation,
    LLM,
    MemorySystem,
)

# LLM interface
from .llm import (
    OpenAILLM,
    AnthropicLLM,
    OllamaLLM,
)

# Memory systems
from .memory import (
    NoHistoryMemorySystem,
    SimpleHistoryMemorySystem,
)

# Chat systems
from .chat import (
    ChatSystem,
)

__all__ = [
    # Types
    "ConversationID",
    "LLMResponse",
    "Prompt",
    "PromptContext",
    "Message",
    "Conversation",
    "LLM",
    "MemorySystem",
    "ChatSystem",
    # LLM
    "OpenAILLM",
    "AnthropicLLM",
    "OllamaLLM",
    # Memory
    "NoHistoryMemorySystem",
    "SimpleHistoryMemorySystem",
    # Chat
    "ChatSystem",
]
