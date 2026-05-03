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
    EvaluationPromptTemplate,
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
    Mem0MemorySystem,
    AMEMMemorySystem,
    SimpleMemMemorySystem,
    EverMemOSMemorySystem,
    StructMemMemorySystem,
    LiCoMemoryMemorySystem,
)

# Chat systems
from .chat import (
    ChatSystem,
)

# Dataset and Evaluation
from .dataset import (
    ChatDataset,
    ConversationData,
)
from .evaluation import (
    Evaluator,
    EvaluationResult,
    EvaluationSummary,
)
from .tokenizer import (
    EvaluationTokenizer,
    TiktokenTokenizer,
)
from .prompt_templates import (
    SimplePromptTemplate,
    ConversationHistoryPromptTemplate,
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
    "Mem0MemorySystem",
    "AMEMMemorySystem",
    "SimpleMemMemorySystem",
    "EverMemOSMemorySystem",
    "StructMemMemorySystem",
    "LiCoMemoryMemorySystem",
    # Chat
    "ChatSystem",
    # Dataset
    "ChatDataset",
    "ConversationData",
    # Evaluation
    "Evaluator",
    "EvaluationResult",
    "EvaluationSummary",
    # Types
    "EvaluationPromptTemplate",
    # Tokenizer
    "EvaluationTokenizer",
    "TiktokenTokenizer",
    # Prompt Templates
    "SimplePromptTemplate",
    "ConversationHistoryPromptTemplate",
]
